"""統計分析：切り口ごとの勝率・連対率・複勝率・単勝回収率・複勝回収率・出走数。

出走数が少ない数字には信頼区間（Wilson法 95%）と「参考値」の印を付ける。
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# 画面やコマンドで選べる切り口（表示名 → 列）
DIMENSIONS: dict[str, list[str]] = {
    "コース（場×芝ダ×距離）": ["venue", "surface", "distance"],
    "枠番": ["frame_no"],
    "馬番": ["horse_no"],
    "脚質": ["style"],
    "産地": ["origin"],
    "生産者": ["breeder"],
    "所属（美浦/栗東）": ["trainer_affiliation"],
    "父": ["sire_name"],
    "母父": ["damsire_name"],
    "父×芝ダ": ["sire_name", "surface"],
    "父×距離帯": ["sire_name", "distance_band"],
    "父×馬場状態": ["sire_name", "track_condition"],
    "母父×芝ダ": ["damsire_name", "surface"],
    "騎手": ["jockey_name"],
    "騎手×人気": ["jockey_name", "pop_band"],
    "騎手×乗り替わり": ["jockey_name", "jockey_change"],
    "乗り替わり": ["jockey_change"],
    "調教師": ["trainer_name"],
    "天候": ["weather"],
    "馬場状態": ["track_condition"],
    "人気": ["pop_band"],
    "オッズ帯": ["odds_band"],
    "前走からの間隔": ["interval_band"],
    "距離の延長/短縮": ["dist_change"],
    "昇級/降級": ["class_change"],
    "クラス": ["class_code"],
    "性別": ["sex"],
    "年齢": ["age"],
}

# 絞り込みに使える列（表示名 → 列）
FILTER_COLUMNS = {
    "競馬場": "venue", "芝ダ": "surface", "距離": "distance", "距離帯": "distance_band",
    "馬場状態": "track_condition", "天候": "weather", "脚質": "style", "クラス": "class_code",
    "回り": "direction", "内外": "inout", "人気": "pop_band", "枠番": "frame_no",
    "所属": "trainer_affiliation", "騎手": "jockey_name", "父": "sire_name", "母父": "damsire_name",
}


def wilson_interval(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, k / n, np.nan)
        denom = 1 + z ** 2 / n
        centre = (p + z ** 2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def apply_filters(df: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    """filters: {列名: 値 または 値のリスト}。日付は 'date_from' / 'date_to'。"""
    if not filters:
        return df
    m = pd.Series(True, index=df.index)
    for k, v in filters.items():
        if v is None or v == [] or v == "":
            continue
        if k == "date_from":
            m &= df["date"] >= pd.Timestamp(v)
        elif k == "date_to":
            m &= df["date"] <= pd.Timestamp(v)
        elif k == "distance_min":
            m &= df["distance"] >= v
        elif k == "distance_max":
            m &= df["distance"] <= v
        elif isinstance(v, (list, tuple, set)):
            m &= df[k].isin(list(v))
        else:
            m &= df[k] == v
    return df[m]


def summarize(df: pd.DataFrame, by: Iterable[str], min_runs_warning: int = 30,
              min_runs: int = 1) -> pd.DataFrame:
    """切り口 by ごとに成績をまとめる。

    回収率は払戻データがあるレースだけで計算する（100円ずつ買った場合、%）。
    """
    by = list(by)
    d = df.dropna(subset=by)
    if d.empty:
        return pd.DataFrame()
    if "has_payout" in d:
        d = d.assign(_pay_runs=d["has_payout"].astype(int),
                     _win_pay=np.where(d["has_payout"], d.get("win_payout", 0), 0),
                     _place_pay=np.where(d["has_payout"], d.get("place_payout", 0), 0))
    else:
        d = d.assign(_pay_runs=0, _win_pay=0, _place_pay=0)
    g = d.groupby(by, observed=True, dropna=True)
    out = pd.DataFrame({
        "出走数": g.size(),
        "1着": g["is_win"].sum(),
        "2着": g["is_top2"].sum() - g["is_win"].sum(),
        "3着": g["is_top3"].sum() - g["is_top2"].sum(),
        "_pay_runs": g["_pay_runs"].sum(),
        "_win_pay": g["_win_pay"].sum(),
        "_place_pay": g["_place_pay"].sum(),
        "_top2": g["is_top2"].sum(),
        "_top3": g["is_top3"].sum(),
    })
    n = out["出走数"].to_numpy()
    out["勝率"] = out["1着"] / out["出走数"]
    out["連対率"] = out["_top2"] / out["出走数"]
    out["複勝率"] = out["_top3"] / out["出走数"]
    lo, hi = wilson_interval(out["1着"], n)
    out["勝率95%CI"] = [f"{a:.1%}〜{b:.1%}" for a, b in zip(lo, hi)]
    lo, hi = wilson_interval(out["_top3"], n)
    out["複勝率95%CI"] = [f"{a:.1%}〜{b:.1%}" for a, b in zip(lo, hi)]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["単勝回収率"] = np.where(out["_pay_runs"] > 0, out["_win_pay"] / (out["_pay_runs"] * 100), np.nan)
        out["複勝回収率"] = np.where(out["_pay_runs"] > 0, out["_place_pay"] / (out["_pay_runs"] * 100), np.nan)
    out["注意"] = np.where(out["出走数"] < min_runs_warning, f"参考値（{min_runs_warning}走未満）", "")
    out = out[out["出走数"] >= min_runs]
    out = out.drop(columns=[c for c in out.columns if c.startswith("_")])
    return out.sort_values("出走数", ascending=False).reset_index()


def format_table(t: pd.DataFrame) -> pd.DataFrame:
    """画面表示用に率を % 表記にする。"""
    t = t.copy()
    for c in ("勝率", "連対率", "複勝率", "単勝回収率", "複勝回収率"):
        if c in t:
            t[c] = t[c].map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
    return t


def upset_rate(df: pd.DataFrame, by: Iterable[str], popularity_threshold: int = 4) -> pd.DataFrame:
    """波乱率：勝ち馬の人気が threshold 番人気以下だったレースの割合。荒れ度の根拠に使う。"""
    winners = df[df["finish_pos"] == 1].drop_duplicates("race_id")
    by = list(by)
    g = winners.dropna(subset=by).groupby(by, observed=True)
    out = pd.DataFrame({
        "レース数": g.size(),
        "波乱率": g["popularity"].apply(lambda s: (s >= popularity_threshold).mean()),
        "1番人気勝率": g["popularity"].apply(lambda s: (s == 1).mean()),
        "勝ち馬平均人気": g["popularity"].mean(),
    })
    return out.reset_index()
