import numpy as np
import pandas as pd
import pytest

from keiba.features import add_derived, running_style
from keiba.stats import apply_filters, summarize, upset_rate, wilson_interval


def test_running_style():
    assert running_style("1-1-1", 16) == "逃げ"
    assert running_style("3-3", 15) == "先行"  # 3 <= ceil(15/3)=5
    assert running_style("8-9", 15) == "差し"  # 9 <= 10
    assert running_style("14-13", 15) == "追込"
    assert running_style(None, 10) is None
    assert running_style(float("nan"), 10) is None


def _df():
    rows = []
    # 2レース × 3頭。馬Aは2走（乗り替わり・距離延長・昇級）
    for rid, date, dist, cls, horses in [
        ("r1", "2026-01-05", 1600, "未勝利", [("A", 1, "j1", "1-1", 1, 2.0), ("B", 2, "j2", "3-3", 2, 5.0), ("C", 3, "j3", "2-2", 3, 9.0)]),
        ("r2", "2026-02-01", 1800, "1勝クラス", [("A", 1, "j9", "3-3", 3, 3.0), ("D", 2, "j2", "1-1", 1, 4.0), ("E", 3, "j3", "2-2", 2, 8.0)]),
    ]:
        for hid, no, jid, passing, pos, odds in horses:
            rows.append(dict(race_id=rid, date=date, distance=dist, class_code=cls, horse_id=hid,
                             horse_no=no, frame_no=no, jockey_id=jid, passing=passing, finish_pos=pos,
                             field_size=3, popularity=no, odds_win=odds, venue="東京", surface="芝"))
    return add_derived(pd.DataFrame(rows))


def test_add_derived_prev_race():
    df = _df()
    a2 = df[(df.horse_id == "A") & (df.race_id == "r2")].iloc[0]
    assert a2["days_since_prev"] == 27
    assert a2["dist_change"] == "延長" and a2["class_change"] == "昇級"
    assert a2["jockey_change"] == "乗り替わり"
    b = df[df.horse_id == "B"].iloc[0]
    assert b["dist_change"] == "前走なし" and b["interval_band"] == "初出走/データなし"


def test_summarize_and_roi():
    df = _df()
    df["has_payout"] = True
    df["win_payout"] = np.where(df.finish_pos == 1, df.odds_win * 100, 0)
    df["place_payout"] = np.where(df.finish_pos <= 3, 150, 0)
    t = summarize(df, ["style"], min_runs_warning=30).set_index("style")
    assert t.loc["逃げ", "出走数"] == 2 and t.loc["逃げ", "勝率"] == 1.0
    assert t.loc["逃げ", "単勝回収率"] == pytest.approx((200 + 400) / 200)
    assert t.loc["逃げ", "注意"].startswith("参考値")
    f = apply_filters(df, {"distance": 1800, "style": ["先行", "差し"]})
    assert set(f.race_id) == {"r2"}


def test_wilson():
    lo, hi = wilson_interval(np.array([5]), np.array([10]))
    assert 0.23 < lo[0] < 0.24 and 0.76 < hi[0] < 0.77


def test_upset_rate():
    t = upset_rate(_df(), ["venue"])
    assert t.iloc[0]["レース数"] == 2
    assert t.iloc[0]["1番人気勝率"] == 0.5
