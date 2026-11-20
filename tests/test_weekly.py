import datetime as dt

from keiba import db
from keiba.weekly import review


def test_review(tmp_path):
    eng = db.get_engine(url=f"sqlite:///{tmp_path}/w.db")
    now = dt.datetime.now()
    with eng.begin() as c:
        db.upsert(c, db.races, [{"race_id": "r1", "date": "2026-11-14", "venue": "東京", "race_no": 1},
                                {"race_id": "r2", "date": "2026-11-15", "venue": "東京", "race_no": 2}])
        ents = []
        for rid in ("r1", "r2"):
            for no, pos, odds in ((1, 1, 2.0), (2, 2, 4.0), (3, 3, 8.0), (4, 4, 20.0)):
                ents.append({"race_id": rid, "horse_no": no, "horse_id": f"h{no}", "finish_pos": pos,
                             "odds_win": odds, "popularity": no})
        db.upsert(c, db.entries, ents)
        db.upsert(c, db.predictions, [
            {"race_id": rid, "horse_no": no, "model_version": "v1", "p_win": pw, "p_top3": pt,
             "mark": m, "created_at": now}
            for rid in ("r1", "r2")
            for no, pw, pt, m in ((1, 0.5, 0.9, "◎"), (2, 0.3, 0.8, "○"), (3, 0.15, 0.8, "▲"), (4, 0.05, 0.5, ""))])
        db.upsert(c, db.payouts, [{"race_id": "r1", "bet_type": "単勝", "combination": "1", "payout": 200},
                                  {"race_id": "r2", "bet_type": "単勝", "combination": "1", "payout": 200}])
        c.execute(db.bet_log.insert(), [
            {"race_id": "r1", "model_version": "v1", "bet_type": "単勝", "combination": "1", "amount": 300,
             "created_at": now},
            {"race_id": "r2", "model_version": "v1", "bet_type": "単勝", "combination": "2", "amount": 100,
             "created_at": now}])
    r = review(eng, "2026-11-10", "2026-11-16")
    assert r["予想レース数"] == 2
    assert r["◎の勝率"] == 1.0
    assert r["買い目 投資"] == 400 and r["買い目 払戻"] == 600
    assert abs(r["買い目 回収率"] - 1.5) < 1e-9
    assert set(r["印別成績"]["mark"]) == {"◎", "○", "▲"}


def test_review_empty(tmp_path):
    eng = db.get_engine(url=f"sqlite:///{tmp_path}/e.db")
    assert "メモ" in review(eng, "2026-01-01", "2026-01-07")
