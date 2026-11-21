"""分析用の基本テーブルと、派生項目（脚質・間隔・距離変化・クラス変化など）。

統計分析（stats.py）と予測モデル（model.py）の両方がここを使う。
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from . import db
from .constants import CLASS_RANK

STYLES = ["逃げ", "先行", "差し", "追込"]

BASE_SQL = """
SELECT
  e.race_id, e.horse_no, e.frame_no, e.horse_id, e.sex, e.age, e.weight_carried,
  e.jockey_id, e.trainer_id, e.horse_weight, e.weight_diff, e.odds_win, e.popularity,
  e.finish_pos, e.finish_status, e.time_sec, e.margin, e.passing, e.last3f, e.prize,
  r.date, r.venue_code, r.venue, r.race_no, r.race_name, r.grade, r.class_name, r.class_code,
  r.age_cond, r.surface, r.direction, r.inout, r.distance, r.weather, r.track_condition,
  r.field_size, r.pace_front3f, r.pace_back3f,
  h.name AS horse_name, h.birth_date, h.breeder, h.origin,
  p.sire_id, p.sire_name, p.damsire_id, p.damsire_name,
  j.name AS jockey_name, j.affiliation AS jockey_affiliation,
  t.name AS trainer_name, t.affiliation AS trainer_affiliation
FROM entries e
JOIN races r ON r.race_id = e.race_id
LEFT JOIN horses h ON h.horse_id = e.horse_id
LEFT JOIN pedigrees p ON p.horse_id = e.horse_id
LEFT JOIN jockeys j ON j.jockey_id = e.jockey_id
LEFT JOIN trainers t ON t.trainer_id = e.trainer_id
WHERE (e.finish_status IS NULL OR e.finish_status NOT IN ('取消', '除外'))
"""


def running_style(passing: str | None, field_size: int | None) -> str | None:
    """通過順位から脚質を判定する。

    - 逃げ：最初のコーナーを1番手で通過
    - 先行：最後のコーナーで頭数の1/3以内
    - 差し：最後のコーナーで頭数の2/3以内
    - 追込：それより後ろ
    （通過順位が無い直線コースなどは判定しない）
    """
    if not isinstance(passing, str) or not passing or not field_size or pd.isna(field_size):
        return None
    nums = [int(x) for x in re.findall(r"\d+", passing)]
    if not nums:
        return None
    if nums[0] == 1:
        return "逃げ"
    last = nums[-1]
    if last <= math.ceil(field_size / 3):
        return "先行"
    if last <= math.ceil(field_size * 2 / 3):
        return "差し"
    return "追込"


def popularity_band(p) -> str | None:
    if p is None or pd.isna(p):
        return None
    p = int(p)
    if p <= 3:
        return f"{p}番人気"
    if p <= 5:
        return "4-5番人気"
    if p <= 9:
        return "6-9番人気"
    return "10番人気以下"


ODDS_BINS = [0, 2, 3, 5, 10, 20, 50, 100, 1e9]
ODDS_LABELS = ["〜1.9", "2.0〜2.9", "3.0〜4.9", "5.0〜9.9", "10〜19.9", "20〜49.9", "50〜99.9", "100〜"]
INTERVAL_BINS = [-1, 13, 27, 55, 90, 180, 10000]
INTERVAL_LABELS = ["連闘〜1週", "2〜3週", "4〜7週", "8〜12週", "3〜6か月", "半年以上"]


def distance_band(d) -> str | None:
    if d is None or pd.isna(d):
        return None
    d = int(d)
    if d <= 1300:
        return "短距離(〜1300)"
    if d <= 1899:
        return "マイル(1400〜1800)"
    if d <= 2100:
        return "中距離(1900〜2100)"
    if d <= 2700:
        return "中長距離(2200〜2700)"
    return "長距離(2800〜)"


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """派生項目を追加する（df は BASE_SQL の結果。日付順でなくてもよい）。"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["course"] = df["venue"].fillna("") + df["surface"].str[:1].fillna("") + df["distance"].astype("Int64").astype(str)
    df["style"] = [running_style(p, n) for p, n in zip(df["passing"], df["field_size"])]
    df["is_win"] = (df["finish_pos"] == 1).astype(int)
    df["is_top2"] = (df["finish_pos"].between(1, 2)).astype(int)
    df["is_top3"] = (df["finish_pos"].between(1, 3)).astype(int)
    df["pop_band"] = df["popularity"].map(popularity_band)
    df["odds_band"] = pd.cut(df["odds_win"], ODDS_BINS, labels=ODDS_LABELS, right=False).astype(object)
    df["distance_band"] = df["distance"].map(distance_band)
    df["class_rank"] = df["class_code"].map(CLASS_RANK)
    df["frame_no"] = df["frame_no"].astype("Int64")
    df["horse_no"] = df["horse_no"].astype("Int64")

    # 前走との比較（同じ馬の1つ前の出走）
    df = df.sort_values(["horse_id", "date", "race_id"])
    g = df.groupby("horse_id", sort=False)
    prev_date = g["date"].shift(1)
    prev_dist = g["distance"].shift(1)
    prev_class = g["class_rank"].shift(1)
    prev_jockey = g["jockey_id"].shift(1)
    df["days_since_prev"] = (df["date"] - prev_date).dt.days
    df["interval_band"] = pd.cut(df["days_since_prev"], INTERVAL_BINS, labels=INTERVAL_LABELS).astype(object)
    df.loc[prev_date.isna(), "interval_band"] = "初出走/データなし"
    diff = df["distance"] - prev_dist
    df["dist_change"] = np.select([prev_dist.isna(), diff > 0, diff < 0], ["前走なし", "延長", "短縮"], "同距離")
    cdiff = df["class_rank"] - prev_class
    df["class_change"] = np.select([prev_class.isna() | df["class_rank"].isna(), cdiff > 0, cdiff < 0],
                                   ["前走なし", "昇級", "降級"], "同クラス")
    df["jockey_change"] = np.select([prev_jockey.isna(), df["jockey_id"] != prev_jockey],
                                    ["前走なし", "乗り替わり"], "継続騎乗")
    return df.sort_values(["date", "race_id", "horse_no"]).reset_index(drop=True)


def attach_payouts(engine, df: pd.DataFrame) -> pd.DataFrame:
    """単勝・複勝の払戻金（100円あたり）を列として付ける。外れは0。"""
    pay = db.read_sql(engine, "SELECT race_id, bet_type, combination, payout FROM payouts "
                              "WHERE bet_type IN ('単勝','複勝')")
    pay = pay[pay["combination"].str.fullmatch(r"\d+")]
    pay["horse_no"] = pay["combination"].astype(int)
    out = df.copy()
    for bt, col in (("単勝", "win_payout"), ("複勝", "place_payout")):
        p = pay[pay["bet_type"] == bt].groupby(["race_id", "horse_no"])["payout"].max().rename(col)
        out = out.merge(p, left_on=["race_id", "horse_no"], right_index=True, how="left")
        out[col] = out[col].fillna(0)
    # 払戻データが無いレース（取得前など）は回収率の計算から外せるよう印を付ける
    races_with_pay = set(pay["race_id"])
    out["has_payout"] = out["race_id"].isin(races_with_pay)
    return out


def load_raw(engine, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """BASE_SQL の結果そのもの（派生項目なし）。"""
    sql = BASE_SQL
    params = {}
    if start:
        sql += " AND r.date >= :start"
        params["start"] = start
    if end:
        sql += " AND r.date <= :end"
        params["end"] = end
    return db.read_sql(engine, sql, params)


def load_base(engine, start: str | None = None, end: str | None = None,
              with_payouts: bool = True) -> pd.DataFrame:
    df = load_raw(engine, start, end)
    if df.empty:
        return df
    df = add_derived(df)
    if with_payouts:
        df = attach_payouts(engine, df)
    return df
