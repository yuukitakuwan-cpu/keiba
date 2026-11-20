import time
from pathlib import Path

from keiba import db
from keiba.fetcher import Fetcher
from keiba.parsers.race_result import parse_race_result

FX = Path(__file__).parent / "fixtures"


def make_cfg(tmp_path):
    return {
        "_base_dir": str(tmp_path),
        "database": {"url": f"sqlite:///{tmp_path}/t.db"},
        "scraper": {"min_interval_sec": 0.1, "jitter_sec": 0, "cache_dir": "cache",
                    "max_retries": 0},
    }


def test_save_race_and_resume(tmp_path):
    eng = db.get_engine(url=f"sqlite:///{tmp_path}/t.db")
    parsed = parse_race_result((FX / "race_result.html").read_text(encoding="utf-8"), "202508040211")
    db.save_race_result(eng, parsed)
    db.save_race_result(eng, parsed)  # 2回保存しても重複しない
    df = db.read_sql(eng, "SELECT * FROM entries")
    assert len(df) == 5
    assert db.read_sql(eng, "SELECT COUNT(*) n FROM payouts")["n"][0] == 9
    assert "202508040211" in db.done_keys(eng, "race")


def test_upsert_keep_existing(tmp_path):
    eng = db.get_engine(url=f"sqlite:///{tmp_path}/k.db")
    with eng.begin() as c:
        db.upsert(c, db.trainers, [{"trainer_id": "1", "name": "A", "affiliation": "栗東"}])
        db.upsert(c, db.trainers, [{"trainer_id": "1", "name": "A", "affiliation": None}],
                  keep_existing=("affiliation",))
    assert db.read_sql(eng, "SELECT affiliation FROM trainers")["affiliation"][0] == "栗東"


class FakeResp:
    def __init__(self, content=b"<html>ok</html>", status=200):
        self.content, self.status_code = content, status


def test_fetcher_cache_and_interval(tmp_path):
    f = Fetcher(make_cfg(tmp_path))
    f.min_interval = 0.3  # テスト用に短く（本番は最低2秒）
    calls = []
    f.session.get = lambda url, timeout: (calls.append(time.time()), FakeResp())[1]
    assert f.get_html("https://db.netkeiba.com/race/1/") == "<html>ok</html>"
    assert f.get_html("https://db.netkeiba.com/race/1/") == "<html>ok</html>"
    assert len(calls) == 1 and f.cache_hits == 1  # 2回目はキャッシュ
    f.get_html("https://db.netkeiba.com/race/2/")
    assert calls[1] - calls[0] >= 0.29  # 間隔があいている


def test_fetcher_min_interval_floor(tmp_path):
    cfg = make_cfg(tmp_path)
    assert Fetcher(cfg).min_interval == 2.0  # 0.1 を指定しても2秒に切り上げ


def test_offline_mode(tmp_path):
    f = Fetcher(make_cfg(tmp_path), offline=True)
    try:
        f.get_html("https://db.netkeiba.com/race/9/")
        assert False
    except Exception as e:
        assert "offline" in str(e)
