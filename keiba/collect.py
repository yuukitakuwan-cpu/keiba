"""データ収集の流れ（期間指定・再開可能）。

1. 開催カレンダー（月ごと）→ 開催日
2. 開催日ごとのレース一覧 → race_id（JRAのみ）
3. レース結果ページ → races / entries / payouts / horses / jockeys / trainers
4. 確定オッズ（設定した日付以降）→ odds
5. 馬のプロフィール・血統 → horses / pedigrees
6. 騎手のプロフィール → jockeys.affiliation

各段階で済んだものは scrape_progress に記録し、もう一度実行すると続きから進む。
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Callable, Iterable

from sqlalchemy import select, update

from . import db
from .config import load_config
from .constants import ODDS_TYPES
from .fetcher import FetchError, Fetcher
from .parsers.horse import parse_horse_profile, parse_jockey_profile, parse_pedigree
from .parsers.race_result import ParseError, parse_race_result
from .parsers.racecard import parse_calendar, parse_odds_json, parse_race_ids, parse_shutuba

log = logging.getLogger(__name__)

DB_BASE = "https://db.netkeiba.com"
RACE_BASE = "https://race.netkeiba.com"
FINAL_ODDS_TYPES = (1, 2, 4, 5, 7)  # 単勝・複勝・馬連・ワイド・三連複

ProgressFn = Callable[[str], None]


def _today() -> dt.date:
    return dt.date.today()


def _d(s: str | dt.date) -> dt.date:
    return s if isinstance(s, dt.date) else dt.date.fromisoformat(s)


def months_between(start: dt.date, end: dt.date) -> list[tuple[int, int]]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


class Collector:
    def __init__(self, cfg: dict | None = None, fetcher: Fetcher | None = None, engine=None,
                 progress: ProgressFn | None = None):
        self.cfg = cfg or load_config()
        self.fetcher = fetcher or Fetcher(self.cfg)
        self.engine = engine or db.get_engine(self.cfg)
        self.progress = progress or (lambda msg: log.info(msg))
        self.stop_requested = False

    # ---------- 開催日・race_id ----------
    def kaisai_dates(self, start: dt.date, end: dt.date) -> list[str]:
        dates = []
        today = _today()
        for y, m in months_between(start, end):
            url = f"{RACE_BASE}/top/calendar.html?year={y}&month={m}"
            # 過ぎた月はずっとキャッシュ、今月以降は1日で取り直す
            past = dt.date(y + (m == 12), m % 12 + 1, 1) <= today - dt.timedelta(days=7)
            html = self.fetcher.get_html(url, max_age_sec=None if past else 86400)
            dates += [d for d in parse_calendar(html) if start <= _d(d) <= end]
        return sorted(set(dates))

    def race_ids_on(self, date: str) -> list[str]:
        """過去の開催日のレースID（db.netkeiba.com のレース一覧）。"""
        url = f"{DB_BASE}/race/list/{date.replace('-', '')}/"
        settled = _d(date) <= _today() - dt.timedelta(days=3)
        html = self.fetcher.get_html(url, max_age_sec=None if settled else 3600)
        ids = parse_race_ids(html)
        if not ids:
            self.fetcher.drop_cache(url)
        return ids

    def upcoming_race_ids(self, date: str) -> list[str]:
        """当日・前日のレースID（race.netkeiba.com の一覧）。"""
        url = f"{RACE_BASE}/top/race_list_sub.html?kaisai_date={date.replace('-', '')}"
        return parse_race_ids(self.fetcher.get_html(url, max_age_sec=1800))

    # ---------- レース結果 ----------
    def collect_results(self, start: str | dt.date, end: str | dt.date,
                        limit: int | None = None) -> dict:
        start, end = _d(start), _d(end)
        end = min(end, _today())
        stats = {"dates": 0, "races_new": 0, "races_skipped": 0, "errors": 0}
        done = db.done_keys(self.engine, "race")
        dates = self.kaisai_dates(start, end)
        self.progress(f"開催日 {len(dates)} 日を確認しました（{start}〜{end}）")
        n = 0
        for date in dates:
            if self.stop_requested:
                break
            ids = self.race_ids_on(date)
            stats["dates"] += 1
            todo = [r for r in ids if r not in done]
            stats["races_skipped"] += len(ids) - len(todo)
            if todo:
                self.progress(f"{date}: {len(ids)} レース中 {len(todo)} レースを取得します")
            for rid in todo:
                if self.stop_requested or (limit is not None and n >= limit):
                    return stats
                if self.collect_one_result(rid):
                    stats["races_new"] += 1
                else:
                    stats["errors"] += 1
                n += 1
        return stats

    def collect_one_result(self, race_id: str) -> bool:
        url = f"{DB_BASE}/race/{race_id}/"
        try:
            html = self.fetcher.get_html(url)
            parsed = parse_race_result(html, race_id)
        except ParseError as e:
            # 結果がまだ出ていないページはキャッシュから消して次回取り直す
            self.fetcher.drop_cache(url)
            with self.engine.begin() as c:
                db.mark_done(c, "race", race_id, "error", str(e))
            log.warning("%s", e)
            return False
        except FetchError as e:
            with self.engine.begin() as c:
                db.mark_done(c, "race", race_id, "error", str(e))
            log.warning("%s", e)
            return False
        db.save_race_result(self.engine, parsed)
        return True

    def reparse_all(self) -> int:
        """キャッシュ済みの結果ページを読み直してDBを作り直す（パーサを直したとき用。通信しない）。"""
        ids = sorted(db.done_keys(self.engine, "race", statuses=("done",)))
        n = 0
        for rid in ids:
            raw = self.fetcher.read_cache(f"{DB_BASE}/race/{rid}/")
            if raw is None:
                continue
            from .fetcher import decode_html
            try:
                db.save_race_result(self.engine, parse_race_result(decode_html(raw), rid))
                n += 1
            except ParseError as e:
                log.warning("%s", e)
        self.progress(f"{n} レースを読み直しました")
        return n

    # ---------- 確定オッズ ----------
    def collect_final_odds(self, race_ids: Iterable[str] | None = None,
                           limit: int | None = None) -> int:
        since = self.cfg["scraper"].get("final_odds_since")
        if race_ids is None:
            q = select(db.races.c.race_id).where(db.races.c.date >= (since or "1900-01-01"))
            with self.engine.connect() as c:
                race_ids = [r[0] for r in c.execute(q.order_by(db.races.c.date.desc()))]
        done = db.done_keys(self.engine, "odds")
        todo = [r for r in race_ids if r not in done]
        if limit is not None:
            todo = todo[:limit]
        if todo:
            self.progress(f"確定オッズを {len(todo)} レース分取得します（1レース {len(FINAL_ODDS_TYPES)} 回）")
        n = 0
        for rid in todo:
            if self.stop_requested:
                break
            rows = self.fetch_odds(rid, FINAL_ODDS_TYPES, cache=True)
            final = [r for r in rows if r["status"] == "result"]
            with self.engine.begin() as c:
                db.upsert(c, db.odds, final)
                db.mark_done(c, "odds", rid, "done" if final else "empty")
            n += 1
        return n

    def fetch_odds(self, race_id: str, types: Iterable[int], cache: bool) -> list[dict]:
        """オッズAPIを呼ぶ。cache=True は確定オッズ用（ずっと保存）、False は当日の最新オッズ用。"""
        out = []
        now = dt.datetime.now()
        for t in types:
            url = f"{RACE_BASE}/api/api_get_jra_odds.html?race_id={race_id}&type={t}&action=update"
            try:
                raw = self.fetcher.get_html(url, use_cache=cache, store=cache) if cache else \
                    self.fetcher.get_html(url, max_age_sec=300)
            except FetchError as e:
                log.warning("%s", e)
                continue
            parsed = parse_odds_json(raw, t)
            if cache and parsed["status"] != "result":
                self.fetcher.drop_cache(url)
            for r in parsed["rows"]:
                out.append({**r, "race_id": race_id, "status": parsed["status"] or "NG",
                            "official_datetime": parsed["official_datetime"], "fetched_at": now})
        return out

    # ---------- 馬・騎手の詳細 ----------
    def collect_horse_details(self, limit: int | None = None,
                              horse_ids: Iterable[str] | None = None) -> int:
        if not self.cfg["scraper"].get("fetch_horse_details", True):
            return 0
        if horse_ids is None:
            # 最近走った馬から順に
            sql = """
                SELECT h.horse_id FROM horses h
                LEFT JOIN (SELECT e.horse_id, MAX(r.date) AS last_date FROM entries e
                           JOIN races r ON r.race_id = e.race_id GROUP BY e.horse_id) x
                  ON x.horse_id = h.horse_id
                WHERE h.detail_fetched IS NULL OR h.detail_fetched = 0 OR h.detail_fetched = false
                ORDER BY x.last_date DESC
            """
            horse_ids = list(db.read_sql(self.engine, sql)["horse_id"])
        else:
            horse_ids = list(horse_ids)
        if limit is not None:
            horse_ids = horse_ids[:limit]
        if horse_ids:
            self.progress(f"馬の詳細を {len(horse_ids)} 頭分取得します（1頭 2 ページ）")
        n = 0
        for hid in horse_ids:
            if self.stop_requested:
                break
            if self.collect_one_horse(hid):
                n += 1
            if n and n % 100 == 0:
                self.progress(f"  馬の詳細 {n}/{len(horse_ids)}")
        return n

    def collect_one_horse(self, horse_id: str) -> bool:
        try:
            prof = parse_horse_profile(self.fetcher.get_html(f"{DB_BASE}/horse/{horse_id}/"), horse_id)
            ped = parse_pedigree(self.fetcher.get_html(f"{DB_BASE}/horse/ped/{horse_id}/"), horse_id)
        except FetchError as e:
            log.warning("%s", e)
            with self.engine.begin() as c:
                db.mark_done(c, "horse", horse_id, "error", str(e))
            return False
        with self.engine.begin() as c:
            db.upsert(c, db.horses, [{**prof, "detail_fetched": True, "updated_at": dt.datetime.now()}],
                      keep_existing=("name",))
            db.upsert(c, db.pedigrees, [ped])
            db.mark_done(c, "horse", horse_id)
        return True

    def collect_jockey_details(self) -> int:
        if not self.cfg["scraper"].get("fetch_jockey_details", True):
            return 0
        with self.engine.connect() as c:
            ids = [r[0] for r in c.execute(select(db.jockeys.c.jockey_id).where(
                (db.jockeys.c.detail_fetched.is_(None)) | (db.jockeys.c.detail_fetched == False)))]  # noqa: E712
        if ids:
            self.progress(f"騎手の所属を {len(ids)} 人分取得します")
        for jid in ids:
            if self.stop_requested:
                break
            try:
                p = parse_jockey_profile(self.fetcher.get_html(f"{DB_BASE}/jockey/{jid}/"), jid)
            except FetchError as e:
                log.warning("%s", e)
                continue
            with self.engine.begin() as c:
                c.execute(update(db.jockeys).where(db.jockeys.c.jockey_id == jid).values(
                    affiliation=p["affiliation"], detail_fetched=True))
        return len(ids)

    # ---------- 当日の出馬表 ----------
    def fetch_racecard(self, race_id: str, date: str | None = None) -> dict:
        url = f"{RACE_BASE}/race/shutuba.html?race_id={race_id}"
        return parse_shutuba(self.fetcher.get_html(url, max_age_sec=600), race_id, date)

    # ---------- まとめて実行 ----------
    def run(self, start, end, *, race_limit=None, horse_limit=None, odds=True,
            horses=True, jockeys=True) -> dict:
        stats = self.collect_results(start, end, limit=race_limit)
        if odds and not self.stop_requested:
            stats["odds_races"] = self.collect_final_odds()
        if jockeys and not self.stop_requested:
            stats["jockeys"] = self.collect_jockey_details()
        if horses and not self.stop_requested:
            stats["horses"] = self.collect_horse_details(limit=horse_limit)
        stats["requests"] = self.fetcher.requests_made
        stats["cache_hits"] = self.fetcher.cache_hits
        return stats
