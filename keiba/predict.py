"""レース予想の出力（メイン機能）。

入力：日付・競馬場・R番号
出力：
 1. 展開予想（ペース・隊列・有利な脚質と枠）
 2. 各馬の評価（推定勝率・複勝率・印・根拠）
 3. 回収率重視の買い目（期待値・1/4ケリー配分・見送り判断）
 4. 荒れ度（0〜100）と穴馬候補
"""
from __future__ import annotations

import datetime as dt
import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import db
from .betting import BetConfig, allocate, candidates, combo_probs, odds_from_rows
from .collect import Collector
from .config import load_config
from .constants import CLASS_RANK, VENUE_CODE_BY_NAME, VENUES
from .features import add_derived, load_base, load_raw
from .model import ModelBundle, build_features, load_current
from .stats import summarize, upset_rate

log = logging.getLogger(__name__)

MARKS = ["◎", "○", "▲", "△", "△"]
LIVE_ODDS_TYPES = (1, 2, 4, 5, 7)


@dataclass
class RacePrediction:
    race: dict
    table: pd.DataFrame  # 各馬の評価
    pace: dict
    bets: pd.DataFrame
    bet_summary: dict
    upset: dict
    odds_status: str
    notes: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ 入力の準備
def find_race_id(col: Collector, date: str, venue: str, race_no: int) -> str:
    code = venue if venue in VENUES else VENUE_CODE_BY_NAME.get(venue)
    if code is None:
        raise ValueError(f"競馬場が分かりません: {venue}")
    # 過去のレースならDBから
    df = db.read_sql(col.engine, "SELECT race_id FROM races WHERE date=:d AND venue_code=:v AND race_no=:r",
                     {"d": date, "v": code, "r": int(race_no)})
    if len(df):
        return df["race_id"].iloc[0]
    for rid in col.upcoming_race_ids(date):
        if rid[4:6] == code and int(rid[10:12]) == int(race_no):
            return rid
    raise ValueError(f"{date} {VENUES[code]} {race_no}R が見つかりません（開催日・場・R番号を確認してください）")


def _target_rows(engine, card: dict) -> pd.DataFrame:
    """出馬表 → BASE_SQL と同じ形の行（結果の列は空）。"""
    race = card["race"]
    ents = [e for e in card["entries"] if not e["cancelled"]]
    df = pd.DataFrame(ents)
    if df.empty:
        return df
    for k in ("date", "venue_code", "venue", "race_no", "race_name", "grade", "class_name", "class_code",
              "age_cond", "surface", "direction", "inout", "distance", "weather", "track_condition",
              "field_size"):
        df[k] = race.get(k)
    ids = "','".join(df["horse_id"].dropna())
    ped = db.read_sql(engine, f"SELECT p.*, h.breeder, h.origin, h.birth_date FROM horses h LEFT JOIN pedigrees p "
                              f"ON p.horse_id=h.horse_id WHERE h.horse_id IN ('{ids}')")
    df = df.merge(ped.drop(columns=["dam_id", "dam_name"], errors="ignore"), on="horse_id", how="left")
    jk = db.read_sql(engine, "SELECT jockey_id, name AS jockey_full, affiliation AS jockey_affiliation FROM jockeys")
    df = df.merge(jk, on="jockey_id", how="left")
    df["jockey_name"] = df["jockey_full"].fillna(df["jockey_name"])
    tr = db.read_sql(engine, "SELECT trainer_id, name AS trainer_full FROM trainers")
    df = df.merge(tr, on="trainer_id", how="left")
    df["trainer_name"] = df["trainer_full"].fillna(df["trainer_name"])
    for c in ("odds_win", "popularity", "finish_pos", "finish_status", "time_sec", "margin", "passing",
              "last3f", "prize", "pace_front3f", "pace_back3f"):
        df[c] = np.nan
    df["finish_status"] = None
    df["_obs"] = 0
    return df.drop(columns=["jockey_full", "trainer_full", "cancelled"], errors="ignore")


class Predictor:
    """履歴データを1回だけ読み込んで、複数レースの予想に使い回す。"""

    def __init__(self, cfg: dict | None = None, model: ModelBundle | None = None,
                 collector: Collector | None = None, fetch_missing_horses: bool = True):
        self.cfg = cfg or load_config()
        self.col = collector or Collector(self.cfg)
        self.engine = self.col.engine
        self.model = model or load_current(self.cfg)
        if self.model is None:
            raise RuntimeError("学習済みモデルがありません。先に `python -m keiba train` を実行してください。")
        self.fetch_missing_horses = fetch_missing_horses
        self._history: pd.DataFrame | None = None
        self._raw: pd.DataFrame | None = None

    @property
    def history(self) -> pd.DataFrame:
        if self._history is None:
            h = load_base(self.engine, with_payouts=True)
            self._history = h[h["surface"] != "障害"]
        return self._history

    def _ensure_horses(self, card: dict) -> None:
        if not self.fetch_missing_horses:
            return
        ids = [e["horse_id"] for e in card["entries"] if e["horse_id"]]
        q = "','".join(ids)
        have = set(db.read_sql(self.engine, f"SELECT horse_id FROM horses WHERE detail_fetched IN (1, true) "
                                            f"AND horse_id IN ('{q}')")["horse_id"]) if ids else set()
        missing = [i for i in ids if i not in have]
        if missing:
            with self.engine.begin() as c:
                db.upsert(c, db.horses, [{"horse_id": e["horse_id"], "name": e["horse_name"]}
                                         for e in card["entries"] if e["horse_id"] in missing],
                          keep_existing=("name",))
            self.col.collect_horse_details(horse_ids=missing)

    @property
    def raw(self) -> pd.DataFrame:
        if self._raw is None:
            r = load_raw(self.engine)
            self._raw = r[r["surface"] != "障害"]
        return self._raw

    def predict_cards(self, cards: list[dict]) -> pd.DataFrame:
        """複数の出馬表をまとめて特徴量化して予測する（返り値は特徴量つきの行）。"""
        for card in cards:
            self._ensure_horses(card)
        targets = pd.concat([_target_rows(self.engine, c) for c in cards], ignore_index=True)
        if targets.empty:
            return targets
        target_ids = set(targets["race_id"])
        hist = self.raw[~self.raw["race_id"].isin(target_ids)].assign(_obs=1)
        cols = list(dict.fromkeys(list(hist.columns)))
        targets = targets.reindex(columns=cols)
        targets["_obs"] = 0
        allrows = pd.concat([hist, targets], ignore_index=True)
        allrows["_obs"] = allrows["_obs"].astype(int)
        feat = build_features(add_derived(allrows))
        tgt = feat[feat["race_id"].isin(target_ids)].copy()
        pred = self.model.predict(tgt)
        tgt["p_win"] = pred["p_win"].to_numpy()
        tgt["p_top3"] = pred["p_top3"].to_numpy()
        return tgt

    # -------------------------------------------------------------- 1レース
    def predict_race(self, date: str, venue: str, race_no: int, budget: float = 10000,
                     live_odds: bool = True, card: dict | None = None,
                     odds_rows: pd.DataFrame | None = None) -> RacePrediction:
        race_id = find_race_id(self.col, date, venue, race_no)
        card = card or self.col.fetch_racecard(race_id, date)
        feat = self.predict_cards([card])
        return self.explain(card, feat, budget, live_odds, odds_rows)

    def explain(self, card: dict, feat: pd.DataFrame, budget: float = 10000, live_odds: bool = True,
                odds_rows: pd.DataFrame | None = None) -> RacePrediction:
        race = card["race"]
        race_id = race["race_id"]
        rows = feat[feat["race_id"] == race_id].sort_values("horse_no").copy()
        notes = []

        # オッズ
        if odds_rows is None and live_odds:
            odds_rows = pd.DataFrame(self.col.fetch_odds(race_id, LIVE_ODDS_TYPES, cache=False))
        odds_status = "なし"
        mk: dict = {}
        if odds_rows is not None and len(odds_rows):
            odds_status = {"yoso": "予想オッズ（発売前）", "middle": "発売中のオッズ", "result": "確定オッズ"}.get(
                str(odds_rows["status"].iloc[0]), str(odds_rows["status"].iloc[0]))
            mk = odds_from_rows(odds_rows)
        win_odds = mk.get("単勝", {})
        rows["odds_win"] = rows["horse_no"].map(lambda n: win_odds.get(str(int(n))))
        if win_odds:
            r = pd.Series(win_odds).astype(float).rank(method="min")
            rows["popularity"] = rows["horse_no"].map(lambda n: r.get(str(int(n))))
        else:
            notes.append("オッズがまだ出ていないため、期待値と荒れ度の一部は計算できません。")

        pace = self.pace_forecast(rows, race)
        table = self.horse_table(rows, race)
        probs = combo_probs(rows[["horse_no", "p_win", "p_top3"]])
        bc = BetConfig.from_cfg(self.cfg)
        cands = candidates(probs, mk, bc) if mk else candidates(probs, {}, bc)
        bets = allocate(cands, budget, bc)
        if bets.empty:
            summary = {"判断": "見送り推奨", "理由": "期待値が基準（%.2f）を超える買い目がありません" % bc.min_ev
                       if mk else "オッズが無いため判断できません"}
        else:
            ev = (bets["期待値"] * bets["金額"]).sum() / bets["金額"].sum()
            summary = {"判断": "購入候補あり", "合計金額": int(bets["金額"].sum()), "点数": int(len(bets)),
                       "平均期待値": round(float(ev), 2)}
        upset = self.upset_score(rows, race, pace)
        return RacePrediction(race, table, pace, bets, summary, upset, odds_status, notes)

    # -------------------------------------------------------------- 展開
    def course_history(self, race: dict) -> pd.DataFrame:
        h = self.history
        return h[(h["venue_code"] == race["venue_code"]) & (h["surface"] == race["surface"]) &
                 (h["distance"] == race["distance"])]

    def pace_forecast(self, rows: pd.DataFrame, race: dict) -> dict:
        r = rows.copy()
        r["前に行く度"] = r["h_early_pos_mean5"]
        nige = r[(r["h_nige_rate"] >= 0.3)]
        front = r[(r["h_early_pos_mean5"] <= 0.25) | (r["h_nige_rate"] >= 0.3)]
        n_front = len(front)
        if len(nige) >= 2 or n_front >= max(4, len(r) // 3):
            pace = "ハイ"
        elif len(nige) == 0 and n_front <= 2:
            pace = "スロー"
        else:
            pace = "ミドル"
        known = r.dropna(subset=["前に行く度"]).sort_values(["前に行く度", "h_nige_rate"],
                                                            ascending=[True, False])
        unknown = r[r["前に行く度"].isna()]

        def names(df):
            return [f"{int(n)} {nm}" for n, nm in zip(df["horse_no"], df["horse_name"])]

        k = len(known)
        groups = {
            "先頭": names(known.head(1)),
            "好位": names(known.iloc[1:max(2, math.ceil(k / 3))]),
            "中団": names(known.iloc[max(2, math.ceil(k / 3)):max(3, math.ceil(k * 2 / 3))]),
            "後方": names(known.iloc[max(3, math.ceil(k * 2 / 3)):]),
            "データなし": names(unknown),
        }
        ch = self.course_history(race)
        fav_style = fav_frame = None
        course_stats = {}
        if len(ch) >= 30:
            st = summarize(ch, ["style"]).set_index("style")
            if len(st):
                fav_style = st["複勝率"].idxmax()
                course_stats["脚質別"] = st[["出走数", "勝率", "複勝率", "単勝回収率", "複勝回収率"]]
            ch2 = ch.assign(枠=np.where(ch["frame_no"] <= 4, "内枠(1-4)", "外枠(5-8)"))
            fr = summarize(ch2, ["枠"]).set_index("枠")
            if len(fr):
                fav_frame = fr["複勝率"].idxmax()
                course_stats["枠別"] = fr[["出走数", "勝率", "複勝率", "単勝回収率", "複勝回収率"]]
        n_races = int(ch["race_id"].nunique()) if len(ch) else 0
        comment = {
            "ハイ": "逃げ・先行したい馬が多く、前が速くなりそう。差し・追込に向く流れ。",
            "ミドル": "極端な流れにはなりにくそう。",
            "スロー": "行きたい馬が少なく、ゆったりした流れになりそう。前に行く馬が有利。",
        }[pace]
        return {
            "ペース": pace,
            "コメント": comment,
            "逃げ候補": names(nige.sort_values("h_nige_rate", ascending=False)) or names(known.head(1)),
            "先行候補": names(front[~front.index.isin(nige.index)]),
            "隊列": groups,
            "有利な脚質": fav_style,
            "有利な枠": fav_frame,
            "コース過去レース数": n_races,
            "コース成績": course_stats,
        }

    # -------------------------------------------------------------- 各馬
    def horse_table(self, rows: pd.DataFrame, race: dict) -> pd.DataFrame:
        r = rows.sort_values("p_win", ascending=False).copy()
        r["印"] = ""
        r.iloc[:3, r.columns.get_loc("印")] = MARKS[:3]
        rest = r.iloc[3:].sort_values("p_top3", ascending=False).index[:2]
        r.loc[rest, "印"] = "△"
        r["根拠"] = [self.comment(x, rows, race) for _, x in r.iterrows()]
        r["期待値(単)"] = r["p_win"] * r["odds_win"]
        out = r[["印", "horse_no", "frame_no", "horse_name", "jockey_name", "p_win", "p_top3", "odds_win",
                 "popularity", "期待値(単)", "根拠"]].rename(columns={
                     "horse_no": "馬番", "frame_no": "枠", "horse_name": "馬名", "jockey_name": "騎手",
                     "p_win": "推定勝率", "p_top3": "推定複勝率", "odds_win": "単勝オッズ", "popularity": "人気"})
        return out.sort_values("推定勝率", ascending=False).reset_index(drop=True)

    @staticmethod
    def comment(x: pd.Series, rows: pd.DataFrame, race: dict) -> str:
        good, bad = [], []
        if pd.isna(x.get("h_finish_ratio_mean5")):
            good.append("初出走/データなし")
        else:
            m = rows["h_finish_ratio_mean5"]
            if x["h_finish_ratio_mean5"] <= m.quantile(0.25):
                good.append("近走の着順が上位")
            elif x["h_finish_ratio_mean5"] >= m.quantile(0.75):
                bad.append("近走は着順が下位")
        if pd.notna(x.get("h_last3f_rank_mean3")) and x["h_last3f_rank_mean3"] <= 0.25:
            good.append("上がりが速い")
        if x.get("h_course_runs", 0) and x.get("h_course_top3", 0) >= 0.4:
            good.append("同コースで好走歴")
        if x.get("h_surface_runs", 0) and x.get("h_surface_top3", 0) <= 0.1 and x.get("h_surface_runs", 0) >= 3:
            bad.append(f"{race['surface']}で結果が出ていない")
        if pd.notna(x.get("sire_surface_top3")) and x.get("sire_runs", 0) >= 50:
            v = x["sire_surface_top3"]
            if v >= 0.30:
                good.append(f"父{x.get('sire_name')}は{race['surface']}向き(複勝率{v:.0%})")
            elif v <= 0.15:
                bad.append(f"父{x.get('sire_name')}は{race['surface']}で苦戦")
        if pd.notna(x.get("j_top3_rate")) and x["j_top3_rate"] >= 0.35 and x.get("j_runs", 0) >= 50:
            good.append("騎手の好走率が高い")
        if x.get("jockey_change") == "乗り替わり":
            good.append("乗り替わり")
        d = x.get("days_since_prev")
        if pd.notna(d) and d >= 120:
            bad.append(f"休み明け({int(d // 7)}週)")
        cd = x.get("class_diff")
        if pd.notna(cd) and cd > 0:
            bad.append("昇級初戦")
        elif pd.notna(cd) and cd < 0:
            good.append("降級")
        dd = x.get("dist_diff")
        if pd.notna(dd) and abs(dd) >= 400:
            bad.append("距離{}{}m".format("延長" if dd > 0 else "短縮", int(abs(dd))))
        return " / ".join(good[:3] + [f"△{b}" for b in bad[:2]]) or "特に目立つ点なし"

    # -------------------------------------------------------------- 荒れ度
    def upset_score(self, rows: pd.DataFrame, race: dict, pace: dict) -> dict:
        parts: dict[str, tuple[float, float, str]] = {}  # 名前 → (0〜1のスコア, 重み, 説明)
        odds = rows["odds_win"].dropna()
        if len(odds) >= 3:
            fav = float(odds.min())
            parts["1番人気のオッズ"] = (min(1, max(0, (fav - 1.5) / 4.5)), 0.25, f"{fav:.1f}倍")
            q = (1 / odds) / (1 / odds).sum()
            ent = float(-(q * np.log(q)).sum() / np.log(len(q)))
            parts["オッズの割れ具合"] = (min(1, max(0, (ent - 0.75) / 0.2)), 0.2, f"均等度 {ent:.2f}")
        pw = rows["p_win"].to_numpy()
        ent_m = float(-(pw * np.log(np.clip(pw, 1e-9, 1))).sum() / np.log(len(pw))) if len(pw) > 1 else 0
        parts["予測の割れ具合"] = (min(1, max(0, (ent_m - 0.80) / 0.17)), 0.15, f"均等度 {ent_m:.2f}")
        ch = self.course_history(race)
        hist_note = "データ不足"
        if len(ch) and ch["race_id"].nunique() >= 10:
            u = upset_rate(ch, ["surface"])
            rate = float(u["波乱率"].iloc[0])
            n = int(u["レース数"].iloc[0])
            # 全体平均へ縮める（レース数が少ないとき）
            allu = upset_rate(self.history, ["surface"]).set_index("surface")
            base = float(allu.loc[race["surface"], "波乱率"]) if race["surface"] in allu.index else 0.35
            rate = (rate * n + base * 30) / (n + 30)
            hist_note = f"同コースで4番人気以下の勝利 {rate:.0%}（{n}レース）"
            parts["コースの波乱率"] = (min(1, max(0, (rate - 0.2) / 0.3)), 0.15, hist_note)
        cl = self.history[self.history["class_code"] == race.get("class_code")]
        if cl["race_id"].nunique() >= 20:
            r = float(upset_rate(cl.assign(_k=1), ["_k"])["波乱率"].iloc[0])
            parts["クラスの波乱率"] = (min(1, max(0, (r - 0.2) / 0.3)), 0.05, f"{race.get('class_code')} {r:.0%}")
        n = len(rows)
        parts["頭数"] = (min(1, max(0, (n - 8) / 10)), 0.1, f"{n}頭")
        tc = race.get("track_condition")
        parts["馬場状態"] = ({"良": 0.0, "稍重": 0.4, "重": 0.7, "不良": 1.0}.get(tc, 0.2), 0.05, tc or "未発表")
        parts["展開"] = ({"ハイ": 0.9, "ミドル": 0.4, "スロー": 0.3}[pace["ペース"]], 0.05, f"{pace['ペース']}ペース予想")
        wsum = sum(w for _, w, _ in parts.values())
        score = 100 * sum(s * w for s, w, _ in parts.values()) / wsum
        level = "大荒れ注意" if score >= 70 else "荒れ気味" if score >= 50 else "やや平穏" if score >= 30 else "堅い"
        # 穴馬候補：人気薄（6番人気以下 or オッズ15倍以上）で、推定複勝率が上位 or 単勝期待値>1
        r = rows.copy()
        r["rank_top3"] = r["p_top3"].rank(ascending=False)
        cond = (r["popularity"] >= 6) | (r["odds_win"] >= 15)
        holes = r[cond & ((r["rank_top3"] <= 5) | (r["p_win"] * r["odds_win"] >= 1.2))]
        hole_list = []
        for _, x in holes.sort_values("p_top3", ascending=False).head(3).iterrows():
            reason = self.comment(x, rows, race)
            hole_list.append({"馬番": int(x["horse_no"]), "馬名": x["horse_name"],
                              "人気": None if pd.isna(x["popularity"]) else int(x["popularity"]),
                              "単勝オッズ": x["odds_win"], "推定複勝率": x["p_top3"],
                              "理由": f"推定複勝率{x['p_top3']:.0%}は{int(x['rank_top3'])}位。{reason}"})
        return {
            "スコア": round(score),
            "判定": level,
            "根拠": pd.DataFrame([{"項目": k, "荒れ寄与(0-1)": round(s, 2), "重み": w, "内容": d}
                                  for k, (s, w, d) in parts.items()]),
            "穴馬候補": hole_list,
        }


def save_predictions(engine, feat: pd.DataFrame, pred: RacePrediction, version: str) -> None:
    """自動予想の結果を predictions / bet_log に保存する（日曜の答え合わせ用）。"""
    now = dt.datetime.now()
    rid = pred.race["race_id"]
    t = pred.table
    rows = [{"race_id": rid, "horse_no": int(r["馬番"]), "model_version": version,
             "horse_id": feat.loc[feat["horse_no"] == r["馬番"], "horse_id"].iloc[0],
             "p_win": float(r["推定勝率"]), "p_top3": float(r["推定複勝率"]), "mark": r["印"],
             "odds_win_at_pred": None if pd.isna(r["単勝オッズ"]) else float(r["単勝オッズ"]),
             "created_at": now} for _, r in t.iterrows()]
    with engine.begin() as c:
        db.upsert(c, db.predictions, rows)
        c.execute(db.bet_log.delete().where((db.bet_log.c.race_id == rid) &
                                            (db.bet_log.c.model_version == version)))
        if len(pred.bets):
            c.execute(db.bet_log.insert(), [{
                "race_id": rid, "model_version": version, "bet_type": b["券種"], "combination": b["買い目"],
                "amount": int(b["金額"]), "est_prob": float(b["推定確率"]), "odds": float(b["オッズ"]),
                "expected_value": float(b["期待値"]), "created_at": now} for _, b in pred.bets.iterrows()])


def format_text(p: RacePrediction) -> str:
    """コマンドライン用のテキスト出力。"""
    r = p.race
    L = []
    L.append(f"■ {r['date']} {r['venue']}{r['race_no']}R {r['race_name']}  "
             f"{r['surface']}{r['distance']}m {r.get('direction') or ''}{r.get('inout') or ''} "
             f"{r.get('class_code') or ''} {r.get('field_size')}頭  馬場:{r.get('track_condition') or '未発表'}")
    L.append(f"  オッズ: {p.odds_status}")
    for n in p.notes:
        L.append(f"  ※ {n}")
    pc = p.pace
    L.append("\n【1. 展開予想】")
    L.append(f"  ペース: {pc['ペース']} … {pc['コメント']}")
    L.append(f"  逃げ候補: {', '.join(pc['逃げ候補']) or 'なし'}")
    L.append(f"  先行候補: {', '.join(pc['先行候補']) or 'なし'}")
    for k, v in pc["隊列"].items():
        if v:
            L.append(f"  {k}: {', '.join(v)}")
    L.append(f"  このコースで有利: 脚質={pc['有利な脚質'] or 'データ不足'} / 枠={pc['有利な枠'] or 'データ不足'}"
             f"（過去{pc['コース過去レース数']}レース）")
    L.append("\n【2. 各馬の評価】")
    for _, x in p.table.iterrows():
        odds = "" if pd.isna(x["単勝オッズ"]) else f"{x['単勝オッズ']:.1f}倍"
        L.append(f"  {x['印'] or '  '} {int(x['馬番']):>2} {x['馬名']:<10} 勝率{x['推定勝率']:6.1%} "
                 f"複勝率{x['推定複勝率']:6.1%} {odds:>7}  {x['根拠']}")
    L.append("\n【3. 買い目（回収率重視）】")
    s = p.bet_summary
    L.append(f"  {s['判断']}" + (f"：{s['理由']}" if "理由" in s else
                                  f"：{s['点数']}点 合計{s['合計金額']}円 平均期待値{s['平均期待値']}"))
    for _, b in p.bets.iterrows():
        L.append(f"   {b['券種']:<4} {b['買い目']:<8} {int(b['金額']):>6}円  確率{b['推定確率']:6.1%} "
                 f"オッズ{b['オッズ']:7.1f} 期待値{b['期待値']:.2f}")
    u = p.upset
    L.append(f"\n【4. 荒れ度】 {u['スコア']} / 100（{u['判定']}）")
    for _, x in u["根拠"].iterrows():
        L.append(f"   {x['項目']}: {x['内容']}")
    if u["穴馬候補"]:
        L.append("  穴馬候補:")
        for h in u["穴馬候補"]:
            L.append(f"   {h['馬番']} {h['馬名']}（{h['人気']}番人気）: {h['理由']}")
    return "\n".join(L)
