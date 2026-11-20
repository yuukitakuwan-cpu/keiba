"""バックテスト：過去のレースで同じ買い方をしたらどうなったか。

予測は walk_forward（その時点より前のデータだけで学習したモデル）を使うので、
未来の情報は混ざらない。

注意：買う判断に使うオッズは「確定オッズ」で代用している。実際に買う時点のオッズとはずれるため、
結果は実際より良く見える可能性がある。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import db
from .betting import BetConfig, allocate, candidates, combo_probs, odds_from_rows


@dataclass
class Strategy:
    name: str
    mode: str  # 'ev'（期待値で選ぶ） / 'top_pick'（◎の単勝） / 'favorite'（1番人気の単勝） / 'kelly'（予想画面と同じ）
    bet_types: tuple = ("単勝",)
    min_ev: float = 1.05
    min_prob: float = 0.0
    max_tickets: int = 3
    budget: float = 10000  # kelly のときの1レースあたりの資金
    extra: dict = field(default_factory=dict)


DEFAULT_STRATEGIES = [
    Strategy("◎の単勝（毎レース100円）", "top_pick"),
    Strategy("1番人気の単勝（比較用）", "favorite"),
    Strategy("単勝 期待値1.1以上", "ev", ("単勝",), 1.1, 0.04),
    Strategy("複勝 期待値1.05以上", "ev", ("複勝",), 1.05, 0.15),
    Strategy("馬連 期待値1.2以上", "ev", ("馬連",), 1.2, 0.015),
    Strategy("ワイド 期待値1.1以上", "ev", ("ワイド",), 1.1, 0.04),
    Strategy("三連複 期待値1.3以上", "ev", ("三連複",), 1.3, 0.006),
    Strategy("総合（予想画面と同じ・1/4ケリー・資金1万円）", "kelly", ("単勝", "複勝", "馬連", "ワイド", "三連複"),
             1.05, {"単勝": 0.04, "複勝": 0.15, "馬連": 0.015, "ワイド": 0.04, "三連複": 0.006}),
]


def load_market(engine, race_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """確定オッズと払戻を読み込む。"""
    if not race_ids:
        return pd.DataFrame(), pd.DataFrame()
    odds = db.read_sql(engine, "SELECT race_id, bet_type, combination, odds, odds_max FROM odds "
                               "WHERE status='result'")
    pays = db.read_sql(engine, "SELECT race_id, bet_type, combination, payout FROM payouts")
    s = set(race_ids)
    return odds[odds["race_id"].isin(s)], pays[pays["race_id"].isin(s)]


def run_backtest(pred: pd.DataFrame, entries: pd.DataFrame, odds: pd.DataFrame,
                 payouts: pd.DataFrame, strategies: list[Strategy],
                 alpha: float | None = 0.5) -> dict[str, pd.DataFrame]:
    """戦略ごとの買い目一覧（race_id, date, 券種, 買い目, 金額, 払戻）を返す。

    alpha: モデルと市場オッズを合成する比率（1 または None ならモデルだけ）

    pred: race_id, horse_no, p_win, p_top3
    entries: race_id, horse_no, date, odds_win, popularity
    """
    d = pred.merge(entries[["race_id", "horse_no", "date", "odds_win", "popularity"]],
                   on=["race_id", "horse_no"], how="left")
    # ◎はモデルの予想、期待値の判断は予想画面と同じく市場と合成した勝率で行う
    d["p_win_model"] = d["p_win"]
    if alpha is not None and alpha < 1:
        from .model import blend_with_market
        d["p_win"] = blend_with_market(d, "p_win", "odds_win", alpha)
    odds_by_race = {k: v for k, v in odds.groupby("race_id")} if len(odds) else {}
    pay_by_race = {k: v for k, v in payouts.groupby("race_id")} if len(payouts) else {}
    results: dict[str, list] = {s.name: [] for s in strategies}

    for rid, g in d.groupby("race_id", sort=False):
        date = g["date"].iloc[0]
        pays = pay_by_race.get(rid)
        if pays is None:
            continue
        pay = {(b, c): p for b, c, p in zip(pays["bet_type"], pays["combination"], pays["payout"])}
        # 判断に使うオッズ：単勝は entries の確定オッズ、他は odds テーブル
        mk = odds_from_rows(odds_by_race[rid]) if rid in odds_by_race else {}
        mk.setdefault("単勝", {str(int(n)): float(o) for n, o in zip(g["horse_no"], g["odds_win"]) if pd.notna(o)})
        probs = None
        for s in strategies:
            bets = []
            if s.mode == "top_pick":
                top = g.loc[g["p_win_model"].idxmax()]
                bets = [("単勝", str(int(top["horse_no"])), 100)]
            elif s.mode == "favorite":
                fav = g[g["popularity"] == 1]
                if len(fav):
                    bets = [("単勝", str(int(fav["horse_no"].iloc[0])), 100)]
            else:
                if any(bt not in mk for bt in s.bet_types if bt != "単勝"):
                    if s.mode == "ev":
                        continue  # このレースはオッズが無いので判定しない
                probs = probs or combo_probs(g)
                bc = BetConfig(min_ev=s.min_ev, min_prob=s.min_prob, max_tickets_per_type=s.max_tickets,
                               bet_types=tuple(bt for bt in s.bet_types if bt in mk))
                c = candidates(probs, mk, bc)
                if s.mode == "ev":
                    bets = [(r["券種"], r["買い目"], 100) for _, r in c.iterrows()]
                else:
                    a = allocate(c, s.budget, bc)
                    bets = [(r["券種"], r["買い目"], int(r["金額"])) for _, r in a.iterrows()]
            for bt, combo, amt in bets:
                results[s.name].append({"race_id": rid, "date": date, "券種": bt, "買い目": combo,
                                        "金額": amt, "払戻": pay.get((bt, combo), 0) * amt / 100})
    return {k: pd.DataFrame(v, columns=["race_id", "date", "券種", "買い目", "金額", "払戻"])
            for k, v in results.items()}


def summarize_bets(bets: pd.DataFrame) -> dict:
    if bets.empty:
        return {"購入点数": 0, "購入レース数": 0, "投資額": 0, "払戻額": 0, "回収率": np.nan,
                "的中率(点)": np.nan, "的中率(レース)": np.nan, "収支": 0, "最大ドローダウン": 0}
    by_race = bets.groupby(["date", "race_id"], sort=True)[["金額", "払戻"]].sum().reset_index()
    by_race["収支"] = by_race["払戻"] - by_race["金額"]
    cum = by_race["収支"].cumsum()
    dd = (cum.cummax().clip(lower=0) - cum).max()
    return {
        "購入点数": int(len(bets)),
        "購入レース数": int(by_race["race_id"].nunique()),
        "投資額": int(bets["金額"].sum()),
        "払戻額": int(bets["払戻"].sum()),
        "回収率": float(bets["払戻"].sum() / bets["金額"].sum()),
        "的中率(点)": float((bets["払戻"] > 0).mean()),
        "的中率(レース)": float((by_race["払戻"] > 0).mean()),
        "収支": int(by_race["収支"].sum()),
        "最大ドローダウン": int(dd),
    }


def equity_curve(bets: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame(columns=["date", "累積収支"])
    r = bets.groupby("date")[["金額", "払戻"]].sum().sort_index()
    return pd.DataFrame({"date": r.index, "累積収支": (r["払戻"] - r["金額"]).cumsum().to_numpy()})


def compare(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame([{"買い方": k, **summarize_bets(v)} for k, v in results.items()])
