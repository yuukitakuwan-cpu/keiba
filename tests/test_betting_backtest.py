import numpy as np
import pandas as pd
import pytest

from keiba.backtest import Strategy, run_backtest, summarize_bets
from keiba.betting import BetConfig, allocate, candidates, harville_tables, kelly_fraction
from keiba.model import normalize_probs


def test_harville_sums():
    t = harville_tables({1: 0.5, 2: 0.3, 3: 0.15, 4: 0.05})
    assert sum(t["馬連"].values()) == pytest.approx(1.0)
    assert sum(t["三連複"].values()) == pytest.approx(1.0)
    assert sum(t["ワイド"].values()) == pytest.approx(3.0)
    assert sum(t["複勝(H)"].values()) == pytest.approx(3.0)
    # 1-2 の馬連 = 0.5*0.3/0.5 + 0.3*0.5/0.7
    assert t["馬連"]["1-2"] == pytest.approx(0.5 * 0.3 / 0.5 + 0.3 * 0.5 / 0.7)


def test_kelly():
    assert kelly_fraction(0.5, 3.0) == pytest.approx(0.25)
    assert kelly_fraction(0.2, 3.0) == 0.0


def test_candidates_and_allocate():
    probs = {"単勝": {"1": 0.5, "2": 0.3, "3": 0.2}}
    odds = {"単勝": {"1": 1.5, "2": 4.0, "3": 3.0}}
    bc = BetConfig(min_ev=1.05, bet_types=("単勝",))
    c = candidates(probs, odds, bc)
    assert list(c["買い目"]) == ["2"]  # 0.3*4.0=1.2 だけ
    a = allocate(c, 10000, bc)
    # ケリー (1.2-1)/3 = 0.0667 → ×1/4 ×10000 = 166 → 100円単位で100
    assert a["金額"].tolist() == [100]
    assert allocate(c, 0, bc).empty


def test_allocate_cap():
    c = pd.DataFrame({"券種": ["単勝"] * 2, "買い目": ["1", "2"], "推定確率": [0.6, 0.6],
                      "オッズ": [3.0, 3.0], "期待値": [1.8, 1.8], "ケリー": [0.4, 0.4]})
    bc = BetConfig(kelly_fraction=1.0, max_race_exposure=0.5)
    a = allocate(c, 10000, bc)
    assert a["金額"].sum() <= 5000


def test_normalize_probs():
    df = pd.DataFrame({"race_id": ["a"] * 4 + ["b"] * 2})
    w, t = normalize_probs(df, np.array([0.2, 0.1, 0.1, 0.1, 0.5, 0.5]),
                           np.array([0.9, 0.5, 0.5, 0.5, 0.8, 0.8]))
    assert w[:4].sum() == pytest.approx(1) and w[4:].sum() == pytest.approx(1)
    assert t[:4].sum() == pytest.approx(3, abs=0.05)
    assert t.max() <= 0.99 and (t >= w).all()


def test_backtest_flow():
    pred = pd.DataFrame({"race_id": ["r1"] * 3, "horse_no": [1, 2, 3], "p_win": [0.6, 0.3, 0.1],
                         "p_top3": [0.9, 0.8, 0.5]})
    ent = pd.DataFrame({"race_id": ["r1"] * 3, "horse_no": [1, 2, 3], "date": [pd.Timestamp("2026-01-01")] * 3,
                        "odds_win": [2.5, 3.0, 9.0], "popularity": [1, 2, 3]})
    odds = pd.DataFrame(columns=["race_id", "bet_type", "combination", "odds", "odds_max"])
    pays = pd.DataFrame({"race_id": ["r1"], "bet_type": ["単勝"], "combination": ["1"], "payout": [250]})
    res = run_backtest(pred, ent, odds, pays, [Strategy("top", "top_pick"), Strategy("fav", "favorite"),
                                               Strategy("ev", "ev", ("単勝",), 1.1)])
    s = summarize_bets(res["top"])
    assert s["回収率"] == pytest.approx(2.5) and s["的中率(レース)"] == 1.0
    assert list(res["ev"]["買い目"]) == ["1"]  # 0.6*2.5=1.5
    assert summarize_bets(res["fav"])["収支"] == 150
    # ワイドのオッズが無いレースではワイドを買わない
    res2 = run_backtest(pred, ent, odds, pays, [Strategy("w", "ev", ("ワイド",), 1.0)])
    assert res2["w"].empty


def test_drawdown():
    bets = pd.DataFrame({"race_id": ["a", "b", "c"], "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
                         "券種": ["単勝"] * 3, "買い目": ["1"] * 3, "金額": [100] * 3, "払戻": [300, 0, 0]})
    s = summarize_bets(bets)
    assert s["最大ドローダウン"] == 200 and s["収支"] == 0


def test_blend_with_market():
    from keiba.model import blend_with_market, fit_alpha
    df = pd.DataFrame({"race_id": ["a"] * 3, "horse_no": [1, 2, 3], "p_win": [0.2, 0.3, 0.5],
                       "odds_win": [1.5, 5.0, 20.0]})
    assert blend_with_market(df, alpha=1.0).tolist() == pytest.approx([0.2, 0.3, 0.5])
    m = blend_with_market(df, alpha=0.0)
    assert m.sum() == pytest.approx(1) and m.iloc[0] > 0.7  # 市場だけなら1番人気が高い
    # 市場がよく当たるデータなら α は小さくなる
    rows = []
    for i in range(200):
        rows.append({"race_id": str(i), "horse_no": 1, "p_win": 0.33, "odds_win": 1.5, "is_win": 1})
        rows.append({"race_id": str(i), "horse_no": 2, "p_win": 0.33, "odds_win": 5.0, "is_win": 0})
        rows.append({"race_id": str(i), "horse_no": 3, "p_win": 0.34, "odds_win": 20.0, "is_win": 0})
    d = pd.DataFrame(rows)
    a, _ = fit_alpha(d[["race_id", "horse_no", "p_win"]], d)
    assert a <= 0.1


def test_min_prob_per_type():
    probs = {"単勝": {"1": 0.03, "2": 0.3}}
    odds = {"単勝": {"1": 100.0, "2": 4.0}}
    c = candidates(probs, odds, BetConfig(bet_types=("単勝",), min_prob={"単勝": 0.04}))
    assert list(c["買い目"]) == ["2"]
