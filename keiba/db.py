"""データベースのスキーマと読み書き。

SQLAlchemy Core で定義しているので、config.yaml の database.url を
PostgreSQL の URL に変えればそのまま移せる（upsert は両方の方言に対応）。
"""
from __future__ import annotations

import datetime as dt
from typing import Iterable

import pandas as pd
from sqlalchemy import (Boolean, Column, Date, DateTime, Float, ForeignKey, Index, Integer,
                        MetaData, String, Table, Text, create_engine, select, text)
from sqlalchemy.engine import Engine

from .config import database_url, load_config

metadata = MetaData()

races = Table(
    "races", metadata,
    Column("race_id", String(12), primary_key=True, comment="netkeiba のレースID"),
    Column("date", String(10), index=True, comment="開催日 YYYY-MM-DD"),
    Column("venue_code", String(2), comment="競馬場コード 01〜10"),
    Column("venue", String(8), comment="競馬場名"),
    Column("kai", Integer), Column("nichi", Integer),
    Column("race_no", Integer, comment="R番号"),
    Column("race_name", String(100)),
    Column("grade", String(4), comment="G1/G2/G3/L/OP"),
    Column("class_name", String(20), comment="新馬/未勝利/1勝クラス/.../オープン"),
    Column("class_code", String(20), comment="グレードも含めたクラス（序列用）"),
    Column("age_cond", String(10)),
    Column("conditions", String(200), comment="条件の原文"),
    Column("surface", String(4), comment="芝/ダート/障害"),
    Column("direction", String(4), comment="右/左/直線"),
    Column("inout", String(8), comment="内/外/外-内 など"),
    Column("distance", Integer),
    Column("weather", String(8)),
    Column("track_condition", String(4), comment="良/稍重/重/不良"),
    Column("start_time", String(5)),
    Column("field_size", Integer, comment="出走頭数（取消・除外を除く）"),
    Column("pace_front3f", Float), Column("pace_back3f", Float),
    Column("lap_times", Text), Column("corner_orders", Text, comment="JSON"),
    Column("updated_at", DateTime),
)
Index("ix_races_course", races.c.venue_code, races.c.surface, races.c.distance)

horses = Table(
    "horses", metadata,
    Column("horse_id", String(12), primary_key=True),
    Column("name", String(50)),
    Column("birth_date", String(10)),
    Column("breeder", String(100), comment="生産者"),
    Column("origin", String(30), comment="産地"),
    Column("owner", String(100)),
    Column("detail_fetched", Boolean, default=False),
    Column("updated_at", DateTime),
)

pedigrees = Table(
    "pedigrees", metadata,
    Column("horse_id", String(12), ForeignKey("horses.horse_id"), primary_key=True),
    Column("sire_id", String(12), index=True), Column("sire_name", String(60)),
    Column("dam_id", String(12)), Column("dam_name", String(60)),
    Column("damsire_id", String(12), index=True), Column("damsire_name", String(60)),
)

jockeys = Table(
    "jockeys", metadata,
    Column("jockey_id", String(8), primary_key=True),
    Column("name", String(30)),
    Column("affiliation", String(8), comment="美浦/栗東/地方/海外"),
    Column("detail_fetched", Boolean, default=False),
)

trainers = Table(
    "trainers", metadata,
    Column("trainer_id", String(8), primary_key=True),
    Column("name", String(30)),
    Column("affiliation", String(8), comment="美浦/栗東/地方/海外"),
)

entries = Table(
    "entries", metadata,
    Column("race_id", String(12), ForeignKey("races.race_id"), primary_key=True),
    Column("horse_no", Integer, primary_key=True, comment="馬番"),
    Column("frame_no", Integer, comment="枠番"),
    Column("horse_id", String(12), ForeignKey("horses.horse_id"), index=True),
    Column("sex", String(2)), Column("age", Integer),
    Column("weight_carried", Float, comment="斤量"),
    Column("jockey_id", String(8), ForeignKey("jockeys.jockey_id"), index=True),
    Column("trainer_id", String(8), ForeignKey("trainers.trainer_id"), index=True),
    Column("horse_weight", Integer), Column("weight_diff", Integer),
    Column("odds_win", Float, comment="確定単勝オッズ"),
    Column("popularity", Integer),
    Column("finish_pos", Integer, comment="着順（中止・除外などは NULL）"),
    Column("finish_status", String(4), comment="中止/除外/取消/失格/降着"),
    Column("time_sec", Float), Column("margin", String(20)),
    Column("passing", String(30), comment="通過順位 例 4-4"),
    Column("last3f", Float, comment="上がり3F"),
    Column("prize", Float, comment="賞金（万円）"),
)

payouts = Table(
    "payouts", metadata,
    Column("race_id", String(12), ForeignKey("races.race_id"), primary_key=True),
    Column("bet_type", String(8), primary_key=True),
    Column("combination", String(20), primary_key=True),
    Column("payout", Integer, comment="100円あたり"),
    Column("popularity", Integer),
)

# 確定オッズ（status='result'）と、当日に取得したオッズのスナップショット
odds = Table(
    "odds", metadata,
    Column("race_id", String(12), primary_key=True),
    Column("bet_type", String(8), primary_key=True),
    Column("combination", String(20), primary_key=True),
    Column("status", String(8), primary_key=True, comment="result / middle / yoso"),
    Column("odds", Float), Column("odds_max", Float),
    Column("popularity", Integer),
    Column("official_datetime", String(20)),
    Column("fetched_at", DateTime),
)

# 土日の朝に自動で出した予想（日曜の答え合わせに使う）
predictions = Table(
    "predictions", metadata,
    Column("race_id", String(12), primary_key=True),
    Column("horse_no", Integer, primary_key=True),
    Column("model_version", String(40), primary_key=True),
    Column("horse_id", String(12)),
    Column("p_win", Float), Column("p_top3", Float),
    Column("mark", String(2)),
    Column("odds_win_at_pred", Float),
    Column("created_at", DateTime),
)

# 予想と一緒に保存する買い目（答え合わせで収支を出す）
bet_log = Table(
    "bet_log", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("race_id", String(12), index=True),
    Column("model_version", String(40)),
    Column("bet_type", String(8)), Column("combination", String(20)),
    Column("amount", Integer), Column("est_prob", Float), Column("odds", Float),
    Column("expected_value", Float),
    Column("created_at", DateTime),
)

# 取得の進み具合（途中で止まっても再開するため）
scrape_progress = Table(
    "scrape_progress", metadata,
    Column("kind", String(20), primary_key=True, comment="race / horse / ped / jockey / odds / date"),
    Column("key", String(40), primary_key=True),
    Column("status", String(10), comment="done / error / empty"),
    Column("message", Text),
    Column("updated_at", DateTime),
)

model_runs = Table(
    "model_runs", metadata,
    Column("version", String(40), primary_key=True),
    Column("created_at", DateTime),
    Column("train_until", String(10)), Column("valid_from", String(10)),
    Column("valid_until", String(10)),
    Column("metrics", Text, comment="JSON"),
    Column("adopted", Boolean),
    Column("note", Text),
)

_engines: dict[str, Engine] = {}


def get_engine(cfg: dict | None = None, url: str | None = None) -> Engine:
    url = url or database_url(cfg or load_config())
    if url not in _engines:
        eng = create_engine(url, future=True)
        if eng.dialect.name == "sqlite":
            with eng.begin() as c:
                c.execute(text("PRAGMA journal_mode=WAL"))
        metadata.create_all(eng)
        _engines[url] = eng
    return _engines[url]


def _insert(engine: Engine, table: Table):
    if engine.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    return insert(table)


def upsert(conn, table: Table, rows: Iterable[dict], update: bool = True,
           keep_existing: Iterable[str] = ()) -> int:
    """主キーが同じ行は更新（update=False なら無視）する。

    keep_existing に挙げた列は、新しい値が None のとき既存の値を残す。
    """
    rows = [{k: v for k, v in r.items() if k in table.c} for r in rows]
    if not rows:
        return 0
    stmt = _insert(conn.engine, table)
    pk = [c.name for c in table.primary_key.columns]
    cols = [c for c in rows[0].keys() if c not in pk]
    if update and cols:
        from sqlalchemy import func
        set_ = {}
        for c in cols:
            if c in keep_existing:
                set_[c] = func.coalesce(stmt.excluded[c], table.c[c])
            else:
                set_[c] = stmt.excluded[c]
        stmt = stmt.on_conflict_do_update(index_elements=pk, set_=set_)
    else:
        stmt = stmt.on_conflict_do_nothing(index_elements=pk)
    conn.execute(stmt, rows)
    return len(rows)


def save_race_result(engine: Engine, parsed: dict) -> None:
    """parse_race_result の結果を1つのトランザクションで保存する。"""
    now = dt.datetime.now()
    with engine.begin() as conn:
        upsert(conn, horses, parsed["horses"], keep_existing=("name",))
        upsert(conn, jockeys, parsed["jockeys"], keep_existing=("name",))
        upsert(conn, trainers, parsed["trainers"], keep_existing=("name", "affiliation"))
        upsert(conn, races, [{**parsed["race"], "updated_at": now}])
        conn.execute(entries.delete().where(entries.c.race_id == parsed["race"]["race_id"]))
        upsert(conn, entries, [e for e in parsed["entries"] if e["horse_no"] is not None])
        conn.execute(payouts.delete().where(payouts.c.race_id == parsed["race"]["race_id"]))
        upsert(conn, payouts, parsed["payouts"])
        mark_done(conn, "race", parsed["race"]["race_id"])


def mark_done(conn, kind: str, key: str, status: str = "done", message: str | None = None) -> None:
    upsert(conn, scrape_progress, [{"kind": kind, "key": key, "status": status,
                                    "message": message, "updated_at": dt.datetime.now()}])


def done_keys(engine: Engine, kind: str, statuses=("done", "empty")) -> set[str]:
    with engine.connect() as c:
        rows = c.execute(select(scrape_progress.c.key).where(
            scrape_progress.c.kind == kind, scrape_progress.c.status.in_(statuses)))
        return {r[0] for r in rows}


def read_sql(engine: Engine, sql: str, params: dict | None = None) -> pd.DataFrame:
    with engine.connect() as c:
        return pd.read_sql(text(sql), c, params=params or {})
