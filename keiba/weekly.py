"""毎週の自動処理。

- 土日の朝（auto-predict）：
    直近の結果を取り込み → その日の全レースを予想 → predictions / bet_log に保存 → レポート
- 日曜の夜（weekly）：
    今週の結果を取り込み → 予想と結果の答え合わせ（的中率・回収率・市場との比較）
    → モデルを学習し直し → 直近期間で今のモデルより良ければ採用 → レポート
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import db
from .collect import Collector
from .config import load_config, resolve_path
from .model import adopt, evaluate, load_current, load_training_frame, record_run, train
from .predict import Predictor, format_text, save_predictions

log = logging.getLogger(__name__)


def report_dir(cfg: dict) -> Path:
    p = resolve_path(cfg, cfg["weekly"]["report_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------ 土日の朝
def auto_predict(date: str | None = None, cfg: dict | None = None, budget: int = 10000) -> Path | None:
    cfg = cfg or load_config()
    date = date or dt.date.today().isoformat()
    col = Collector(cfg)
    d = dt.date.fromisoformat(date)
    # 前日までの結果（土曜の結果など）を先に取り込む
    col.run(d - dt.timedelta(days=3), d - dt.timedelta(days=1), odds=True, horses=False, jockeys=True)
    ids = col.upcoming_race_ids(date)
    if not ids:
        log.info("%s は開催がありません", date)
        return None
    cards = []
    for rid in ids:
        try:
            c = col.fetch_racecard(rid, date)
        except Exception as e:  # noqa: BLE001
            log.warning("出馬表を取得できませんでした %s: %s", rid, e)
            continue
        if c["race"]["surface"] == "障害":
            continue  # 障害レースはモデルの対象外
        cards.append(c)
    pr = Predictor(cfg, collector=col)
    feat = pr.predict_cards(cards)
    texts = []
    for c in cards:
        rid = c["race"]["race_id"]
        try:
            p = pr.explain(c, feat[feat["race_id"] == rid], budget=budget)
        except Exception as e:  # noqa: BLE001
            log.warning("予想に失敗 %s: %s", rid, e)
            continue
        save_predictions(col.engine, feat[feat["race_id"] == rid], p, pr.model.version)
        texts.append(format_text(p))
    out = report_dir(cfg) / f"predictions_{date}.txt"
    out.write_text(f"モデル {pr.model.version}\n\n" + "\n\n".join(texts), encoding="utf-8")
    log.info("%d レースを予想しました → %s", len(texts), out)
    return out


# ------------------------------------------------------------------ 答え合わせ
def review(engine, start: str, end: str) -> dict:
    """期間内の自動予想と結果を突き合わせる。"""
    pred = db.read_sql(engine, """
        SELECT p.*, r.date, r.venue, r.race_no, e.finish_pos, e.odds_win, e.popularity
        FROM predictions p JOIN races r ON r.race_id = p.race_id
        JOIN entries e ON e.race_id = p.race_id AND e.horse_no = p.horse_no
        WHERE r.date BETWEEN :s AND :e""", {"s": start, "e": end})
    out: dict = {"期間": f"{start}〜{end}"}
    if pred.empty:
        out["メモ"] = "この期間の自動予想はありません"
        return out
    # 同じレースを複数モデルで予想していたら、最新のモデルの予想を使う
    latest = pred.sort_values("created_at").groupby("race_id")["model_version"].last()
    pred = pred[pred["model_version"] == pred["race_id"].map(latest)]
    pred["is_win"] = (pred["finish_pos"] == 1).astype(int)
    pred["is_top3"] = pred["finish_pos"].between(1, 3).astype(int)
    m = evaluate(pred[["race_id", "horse_no", "p_win", "p_top3"]],
                 pred[["race_id", "horse_no", "is_win", "is_top3", "popularity", "odds_win"]])
    out["予想レース数"] = m["races"]
    out["◎の勝率"] = m["top1_win_hit"]
    out["◎の複勝率"] = m["top1_top3_hit"]
    out["1番人気の勝率"] = m.get("fav_win_hit")
    out["logloss(勝率) モデル"] = m["logloss_win"]
    out["logloss(勝率) 市場オッズ"] = m.get("logloss_win_market")
    marks = pred[pred["mark"] != ""].groupby("mark").agg(頭数=("is_win", "size"), 勝率=("is_win", "mean"),
                                                         複勝率=("is_top3", "mean"))
    out["印別成績"] = marks.reset_index()

    bets = db.read_sql(engine, """
        SELECT b.*, r.date FROM bet_log b JOIN races r ON r.race_id = b.race_id
        WHERE r.date BETWEEN :s AND :e""", {"s": start, "e": end})
    if len(bets):
        latest_b = bets.sort_values("created_at").groupby("race_id")["model_version"].last()
        bets = bets[bets["model_version"] == bets["race_id"].map(latest_b)]
        pays = db.read_sql(engine, "SELECT race_id, bet_type, combination, payout FROM payouts")
        bets = bets.merge(pays, on=["race_id", "bet_type", "combination"], how="left")
        bets["payout"] = bets["payout"].fillna(0) * bets["amount"] / 100
        settled = bets[bets["race_id"].isin(set(pays["race_id"]))]
        out["買い目 点数"] = int(len(settled))
        out["買い目 投資"] = int(settled["amount"].sum())
        out["買い目 払戻"] = int(settled["payout"].sum())
        out["買い目 回収率"] = float(settled["payout"].sum() / settled["amount"].sum()) if len(settled) else None
        out["券種別"] = settled.groupby("bet_type").agg(点数=("amount", "size"), 投資=("amount", "sum"),
                                                       払戻=("payout", "sum")).assign(
            回収率=lambda x: x["払戻"] / x["投資"]).reset_index()
    return out


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}" if v < 2 else f"{v:,.1f}"
    if isinstance(v, pd.DataFrame):
        return "\n" + v.to_markdown(index=False) if hasattr(v, "to_markdown") else "\n" + v.to_string(index=False)
    return str(v)


# ------------------------------------------------------------------ 日曜の夜
def weekly(cfg: dict | None = None, today: dt.date | None = None) -> Path:
    cfg = cfg or load_config()
    today = today or dt.date.today()
    start = today - dt.timedelta(days=cfg["weekly"]["lookback_days"])
    col = Collector(cfg)
    eng = col.engine
    lines = [f"# 週次レポート {today}", ""]

    # 1. 今週の結果を取り込む
    st = col.run(start, today)
    lines += ["## 1. データ更新", f"- 取得したレース: {st.get('races_new', 0)} / リクエスト {st.get('requests')} 回", ""]

    # 2. 答え合わせ
    rv = review(eng, start.isoformat(), today.isoformat())
    lines += ["## 2. 予想の答え合わせ"]
    for k, v in rv.items():
        if isinstance(v, pd.DataFrame):
            lines += [f"### {k}", v.to_string(index=False), ""]
        else:
            lines.append(f"- {k}: {_fmt(v)}")
    lines.append("")

    # 3. 再学習と採用判定
    lines += ["## 3. モデルの再学習"]
    feat = load_training_frame(eng, cfg)
    valid_from = (pd.Timestamp(today) - pd.DateOffset(months=cfg["model"]["valid_months"])).strftime("%Y-%m-%d")
    try:
        new, m_new = train(eng, cfg, valid_from=valid_from, feat=feat)
    except ValueError as e:
        lines.append(f"- 学習できませんでした: {e}")
        return _write(cfg, today, lines)
    cur = load_current(cfg)
    tol = cfg["model"].get("adopt_tolerance", 0.002)
    adopted, reason = True, "現在のモデルがないため採用"
    if cur is not None:
        # どちらも「学習に使っていない直近期間」での成績（検証期間は1週ずつずれるだけでほぼ同じ）
        cur_ll = cur.meta.get("metrics", {}).get("logloss_top3")
        new_ll = m_new["logloss_top3"]
        if cur_ll is not None and new_ll > cur_ll + tol:
            adopted = False
            reason = f"検証の logloss(複勝) が悪化: 新 {new_ll:.4f} / 今 {cur_ll:.4f}"
        else:
            reason = f"検証の logloss(複勝) 新 {new_ll:.4f} / 今 {cur_ll if cur_ll is None else round(cur_ll, 4)}"
    if adopted:
        adopt(new, cfg, eng, note="週次再学習: " + reason)
    else:
        record_run(new, cfg, eng, adopted=False, note="週次再学習: " + reason)
    lines += [f"- 新モデル: {new.version}（学習データ {new.meta['train_from']}〜{new.meta['data_until']}）",
              f"- 検証期間: {valid_from}〜 / {m_new['races']} レース",
              f"- 検証成績: ◎勝率 {m_new['top1_win_hit']:.3f} / ◎複勝率 {m_new['top1_top3_hit']:.3f} / "
              f"logloss(複勝) {m_new['logloss_top3']:.4f} / AUC(複勝) {m_new.get('auc_top3', float('nan')):.3f}",
              f"- 採用: {'する' if adopted else 'しない'}（{reason}）", "",
              "### 特徴量の重要度（上位15）", new.importance().head(15).to_string(index=False)]
    return _write(cfg, today, lines)


def _write(cfg, today, lines) -> Path:
    out = report_dir(cfg) / f"weekly_{today}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("レポート: %s", out)
    return out
