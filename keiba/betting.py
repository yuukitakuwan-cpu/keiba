"""買い目の確率・期待値・資金配分。

- 組み合わせ（馬連・ワイド・三連複）の確率は、推定勝率から Harville の式で計算する
  （1着の馬を除いた残りの馬の勝率を比例配分して2着・3着を決める考え方）
- 期待値 = 推定確率 × オッズ
- 配分は「ケリー基準 × kelly_fraction（既定 1/4）」で控えめにし、1レースの上限もかける
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd


def harville_tables(p_win: dict[int, float]) -> dict[str, dict[str, float]]:
    """馬番→勝率 から、馬連・ワイド・三連複・複勝(Harville)の確率を返す。

    キーは '3-7' のような小さい順のハイフン区切り。
    """
    nos = sorted(p_win)
    p = np.array([max(p_win[n], 1e-9) for n in nos])
    p = p / p.sum()
    n = len(nos)
    trio: dict[tuple, float] = {}
    exacta = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                exacta[i, j] = p[i] * p[j] / (1 - p[i])
    for i, j, k in itertools.permutations(range(n), 3):
        pr = exacta[i, j] * p[k] / (1 - p[i] - p[j]) if (1 - p[i] - p[j]) > 1e-9 else 0.0
        key = tuple(sorted((i, j, k)))
        trio[key] = trio.get(key, 0.0) + pr
    out = {"馬連": {}, "ワイド": {}, "三連複": {}, "複勝(H)": {}}
    place = np.zeros(n)
    for (i, j, k), pr in trio.items():
        out["三連複"][f"{nos[i]}-{nos[j]}-{nos[k]}"] = pr
        for a, b in ((i, j), (i, k), (j, k)):
            key = f"{nos[a]}-{nos[b]}"
            out["ワイド"][key] = out["ワイド"].get(key, 0.0) + pr
        place[[i, j, k]] += pr
    for i, j in itertools.combinations(range(n), 2):
        out["馬連"][f"{nos[i]}-{nos[j]}"] = exacta[i, j] + exacta[j, i]
    out["複勝(H)"] = {str(nos[i]): place[i] for i in range(n)}
    return out


def combo_probs(pred: pd.DataFrame) -> dict[str, dict[str, float]]:
    """1レース分の予測（horse_no, p_win, p_top3）→ 券種ごとの確率。

    複勝はモデルの p_top3 をそのまま使う。
    """
    pw = dict(zip(pred["horse_no"].astype(int), pred["p_win"]))
    t = harville_tables(pw)
    probs = {
        "単勝": {str(int(n)): float(p) for n, p in zip(pred["horse_no"], pred["p_win"])},
        "複勝": {str(int(n)): float(p) for n, p in zip(pred["horse_no"], pred["p_top3"])},
        "馬連": t["馬連"], "ワイド": t["ワイド"], "三連複": t["三連複"],
    }
    return probs


def kelly_fraction(p: float, odds: float) -> float:
    """1点買いのケリー基準（資金に対する割合）。odds は払戻倍率（100円→odds×100円）。"""
    b = odds - 1
    if b <= 0:
        return 0.0
    return max(0.0, (p * odds - 1) / b)


@dataclass
class BetConfig:
    kelly_fraction: float = 0.25
    min_ev: float = 1.05
    min_bet: int = 100
    max_race_exposure: float = 0.5
    max_tickets_per_type: int = 5
    bet_types: tuple = ("単勝", "複勝", "馬連", "ワイド", "三連複")
    min_prob: float = 0.0  # 当たる確率が低すぎる買い目を除く（例 0.01）

    @classmethod
    def from_cfg(cls, cfg: dict, **over) -> "BetConfig":
        b = cfg.get("betting", {})
        kw = dict(kelly_fraction=b.get("kelly_fraction", 0.25), min_ev=b.get("min_expected_value", 1.05),
                  min_bet=b.get("min_bet", 100), max_race_exposure=b.get("max_race_exposure", 0.5),
                  max_tickets_per_type=b.get("max_tickets_per_type", 5))
        kw.update(over)
        return cls(**kw)


def candidates(probs: dict[str, dict[str, float]], odds: dict[str, dict[str, float]],
               bc: BetConfig) -> pd.DataFrame:
    """確率とオッズから、期待値が min_ev を超える買い目を券種ごとに上位 max_tickets_per_type 点まで返す。

    odds は {券種: {組み合わせ: オッズ}}。複勝・ワイドは下限オッズを使う（控えめ）。
    """
    rows = []
    for bt in bc.bet_types:
        po, oo = probs.get(bt, {}), odds.get(bt, {})
        for combo, p in po.items():
            o = oo.get(combo)
            if o is None or not np.isfinite(o) or o <= 1.0 or p < bc.min_prob:
                continue
            ev = p * o
            rows.append({"券種": bt, "買い目": combo, "推定確率": p, "オッズ": o, "期待値": ev,
                         "ケリー": kelly_fraction(p, o)})
    df = pd.DataFrame(rows, columns=["券種", "買い目", "推定確率", "オッズ", "期待値", "ケリー"])
    if df.empty:
        return df
    df = df[df["期待値"] >= bc.min_ev]
    df = df.sort_values("期待値", ascending=False).groupby("券種", sort=False).head(bc.max_tickets_per_type)
    return df.reset_index(drop=True)


def allocate(cands: pd.DataFrame, budget: float, bc: BetConfig) -> pd.DataFrame:
    """候補に金額を割り当てる。金額 = 予算 × ケリー × kelly_fraction（100円単位、合計は上限まで）。"""
    if cands.empty or budget <= 0:
        return cands.iloc[0:0].assign(金額=pd.Series(dtype=int))
    c = cands.copy()
    raw = budget * c["ケリー"] * bc.kelly_fraction
    cap = budget * bc.max_race_exposure
    if raw.sum() > cap:
        raw = raw * cap / raw.sum()
    c["金額"] = (np.floor(raw / bc.min_bet) * bc.min_bet).astype(int)
    c = c[c["金額"] >= bc.min_bet]
    return c.sort_values(["券種", "期待値"], ascending=[True, False]).reset_index(drop=True)


def odds_from_rows(rows: pd.DataFrame) -> dict[str, dict[str, float]]:
    """odds テーブルの行（bet_type, combination, odds, odds_max）→ {券種: {組: オッズ}}"""
    out: dict[str, dict[str, float]] = {}
    for bt, combo, o in zip(rows["bet_type"], rows["combination"], rows["odds"]):
        if o is None or pd.isna(o):
            continue
        out.setdefault(bt, {})[combo] = float(o)
    return out


def settle(bets: pd.DataFrame, payouts: pd.DataFrame) -> pd.DataFrame:
    """買い目に払戻を付ける。payouts は (bet_type, combination, payout)。"""
    if bets.empty:
        return bets.assign(払戻=pd.Series(dtype=float))
    pay = {(b, c): p for b, c, p in zip(payouts["bet_type"], payouts["combination"], payouts["payout"])}
    b = bets.copy()
    b["払戻"] = [pay.get((bt, cb), 0) * amt / 100 for bt, cb, amt in zip(b["券種"], b["買い目"], b["金額"])]
    return b
