"""netkeiba からページを取得する。

- リクエスト間隔を必ずあける（最低2秒。既定は config の min_interval_sec）
- 並列アクセスはしない（1つの Fetcher を順番に使う。プロセス間でもロックファイルで直列化）
- 取得したページは gzip でキャッシュし、同じページを取りに行かない
- 一時的なエラーは間隔を伸ばしながら再試行する
"""
from __future__ import annotations

import gzip
import hashlib
import logging
import os
import random
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from .config import load_config, resolve_path

log = logging.getLogger(__name__)

MIN_INTERVAL_FLOOR = 2.0  # これより短い間隔は許可しない


class FetchError(RuntimeError):
    pass


class _FileLock:
    """プロセスをまたいで「1度に1リクエスト」を守るためのロック（Windows / Mac / Linux）。"""

    def __init__(self, path: Path):
        self.path = path
        self.f = None

    def __enter__(self):
        self.f = open(self.path, "a+b")
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    self.f.seek(0)
                    msvcrt.locking(self.f.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    time.sleep(0.5)
        else:
            import fcntl
            fcntl.flock(self.f.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        try:
            if os.name == "nt":
                import msvcrt
                self.f.seek(0)
                msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
        finally:
            self.f.close()


def decode_html(raw: bytes) -> str:
    """netkeiba のページは現在 UTF-8 だが、古いページは EUC-JP のことがある。"""
    head = raw[:2048].lower()
    if b"euc-jp" in head:
        return raw.decode("euc-jp", errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("euc-jp", errors="replace")


class Fetcher:
    def __init__(self, cfg: dict | None = None, *, offline: bool = False):
        cfg = cfg or load_config()
        sc = cfg["scraper"]
        self.min_interval = max(float(sc.get("min_interval_sec", 3.0)), MIN_INTERVAL_FLOOR)
        self.jitter = float(sc.get("jitter_sec", 0.0))
        self.timeout = float(sc.get("timeout_sec", 30))
        self.max_retries = int(sc.get("max_retries", 4))
        self.cache_dir = resolve_path(cfg, sc.get("cache_dir", "data/cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline  # True ならキャッシュだけを使い、ネットに出ない
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": sc.get("user_agent", "Mozilla/5.0"),
            "Accept-Language": "ja,en;q=0.8",
        })
        self._stamp = self.cache_dir / ".last_request"
        self._lock = _FileLock(self.cache_dir / ".request.lock")
        self.requests_made = 0
        self.cache_hits = 0

    # ---------- キャッシュ ----------
    def cache_path(self, url: str) -> Path:
        u = urlparse(url)
        h = hashlib.sha1(url.encode("utf-8")).hexdigest()
        return self.cache_dir / u.netloc / h[:2] / f"{h}.gz"

    def read_cache(self, url: str, max_age_sec: float | None = None) -> bytes | None:
        p = self.cache_path(url)
        if not p.exists():
            return None
        if max_age_sec is not None and time.time() - p.stat().st_mtime > max_age_sec:
            return None
        with gzip.open(p, "rb") as f:
            return f.read()

    def write_cache(self, url: str, raw: bytes) -> None:
        p = self.cache_path(url)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with gzip.open(tmp, "wb") as f:
            f.write(raw)
        os.replace(tmp, p)

    def drop_cache(self, url: str) -> None:
        p = self.cache_path(url)
        if p.exists():
            p.unlink()

    # ---------- 間隔制御 ----------
    def _wait_turn(self) -> None:
        """前回のリクエストから min_interval(+揺らぎ) 経つまで待つ。

        時刻はキャッシュフォルダのファイルに記録するので、
        別のプロセス（例：定期実行と画面からの更新）が同時に動いても間隔が守られる。
        """
        wait_for = self.min_interval + random.uniform(0, self.jitter)
        try:
            last = float(self._stamp.read_text())
        except (FileNotFoundError, ValueError):
            last = 0.0
        delta = time.time() - last
        if delta < wait_for:
            time.sleep(wait_for - delta)
        self._stamp.write_text(str(time.time()))

    # ---------- 取得 ----------
    def get_bytes(self, url: str, *, use_cache: bool = True, max_age_sec: float | None = None,
                  store: bool = True) -> bytes:
        if use_cache:
            raw = self.read_cache(url, max_age_sec)
            if raw is not None:
                self.cache_hits += 1
                return raw
        if self.offline:
            raise FetchError(f"offline mode and not cached: {url}")
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                # ロックを持ったまま「待つ→取得」するので、別プロセスと同時に取りに行くことはない
                with self._lock:
                    self._wait_turn()
                    r = self.session.get(url, timeout=self.timeout)
                    self._stamp.write_text(str(time.time()))
                self.requests_made += 1
                if r.status_code == 200:
                    raw = r.content
                    if store:
                        self.write_cache(url, raw)
                    return raw
                if r.status_code in (403, 404):
                    raise FetchError(f"HTTP {r.status_code}: {url}")
                last_err = FetchError(f"HTTP {r.status_code}: {url}")
            except requests.RequestException as e:
                last_err = e
            backoff = self.min_interval * (2 ** attempt)
            log.warning("取得失敗 (%s) %s 秒後に再試行: %s", last_err, round(backoff), url)
            time.sleep(backoff)
        raise FetchError(f"再試行の上限に達しました: {url} ({last_err})")

    def get_html(self, url: str, **kw) -> str:
        return decode_html(self.get_bytes(url, **kw))
