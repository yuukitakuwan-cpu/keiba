"""予測モデル：特徴量づくり・学習・予測（LightGBM）。

未来の情報が混ざらないように：
- 馬の過去成績は「その馬の1つ前までの出走」だけから計算
- 騎手・調教師・血統などの成績は「そのレースの日より前の日」までの累計だけから計算
- 当日のオッズ・人気は特徴量に入れない
- 学習と検証は日付で分ける（検証期間より前のデータだけで学習）
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from . import db
from .config import load_config, resolve_path
from .features import add_derived, load_base

log = logging.getLogger(__name__)

CATEGORICAL = ["venue_code", "surface", "track_condition", "sex", "direction", "weather"]

FEATURES = [
    # レース条件
    "venue_code", "surface", "distance", "track_condition", "weather", "direction", "field_size",
    "class_rank", "is_inner_course",
    # 馬の基本
    "sex", "age", "weight_carried", "horse_weight", "weight_diff", "frame_no", "horse_no",
    "horse_no_ratio",
    # 前走との比較
    "days_since_prev", "dist_diff", "class_diff", "jockey_changed",
    # 馬の過去成績
    "h_runs", "h_win_rate", "h_top3_rate", "h_finish_ratio_last", "h_finish_ratio_mean3",
    "h_finish_ratio_mean5", "h_time_behind_last", "h_time_behind_mean3", "h_last3f_rank_mean3",
    "h_last3f_rank_last", "h_early_pos_mean5", "h_late_pos_mean5", "h_nige_rate",
    "h_surface_runs", "h_surface_top3", "h_course_runs", "h_course_top3", "h_dist_runs",
    "h_dist_top3", "h_track_top3", "h_prize_mean3",
    # 騎手・調教師・血統（前日までの累計）
    "j_win_rate", "j_top3_rate", "j_runs", "j_course_top3", "t_win_rate", "t_top3_rate",
    "jt_top3", "sire_surface_top3", "sire_dist_top3", "sire_track_top3", "damsire_surface_top3",
    "sire_runs",
    # レース内での相対値・展開
    "rel_finish_ratio_mean5", "rel_time_behind_mean3", "rel_early_pos", "rank_early_pos",
    "n_front_runners", "rel_h_top3_rate", "rel_j_top3_rate",
]


# ------------------------------------------------------------------ 特徴量
def _prior_rate(df: pd.DataFrame, keys: list[str], name: str, prior_n: float = 20.0,
                rates=("win", "top3")) -> pd.DataFrame:
    """keys ごとに「前日までの」出走数と勝率・複勝率（事前分布で平滑化）を付ける。"""
    obs = df["_obs"]
    tmp = pd.DataFrame({k: df[k] for k in keys})
    tmp["date"] = df["date"]
    tmp["n"] = obs
    tmp["w"] = df["is_win"] * obs
    tmp["t3"] = df["is_top3"] * obs
    agg = tmp.groupby(keys + ["date"], dropna=False, sort=True)[["n", "w", "t3"]].sum().reset_index()
    agg = agg.sort_values(keys + ["date"])
    g = agg.groupby(keys, dropna=False, sort=False)
    for c in ("n", "w", "t3"):
        agg["c" + c] = g[c].cumsum() - agg[c]
    gw = df.loc[obs == 1, "is_win"].mean() if (obs == 1).any() else 0.08
    gt = df.loc[obs == 1, "is_top3"].mean() if (obs == 1).any() else 0.24
    out = pd.DataFrame(index=agg.index)
    out[f"{name}_runs"] = agg["cn"]
    if "win" in rates:
        out[f"{name}_win_rate"] = (agg["cw"] + prior_n * gw) / (agg["cn"] + prior_n)
    if "top3" in rates:
        out[f"{name}_top3"] = (agg["ct3"] + prior_n * gt) / (agg["cn"] + prior_n)
    out[keys + ["date"]] = agg[keys + ["date"]]
    merged = df[keys + ["date"]].merge(out, on=keys + ["date"], how="left")
    merged.index = df.index
    cols = [c for c in out.columns if c not in keys + ["date"]]
    res = merged[cols]
    # キーが欠けている行（血統不明など）は NaN にする
    missing = df[keys].isna().any(axis=1)
    res.loc[missing, :] = np.nan
    return res


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """load_base / add_derived 済みの df から特徴量を作る。

    結果がまだ無い行（予測したいレース）は `_obs=0` として渡すと、
    集計には使われず、特徴量だけが計算される。
    """
    df = df.copy()
    if "_obs" not in df:
        df["_obs"] = 1
    df["_obs"] = df["_obs"].astype(int)
    df = df.sort_values(["date", "race_id", "horse_no"]).reset_index(drop=True)

    # 1レース内の値
    first_time = df.groupby("race_id")["time_sec"].transform("min")
    df["time_behind"] = df["time_sec"] - first_time
    df["finish_ratio"] = df["finish_pos"] / df["field_size"]
    df["last3f_rank"] = df.groupby("race_id")["last3f"].rank(method="min") / df["field_size"]

    def corner_pos(p, idx):
        if not isinstance(p, str) or not p:
            return np.nan
        parts = [x for x in p.split("-") if x.strip().isdigit()]
        return float(parts[idx]) if parts else np.nan

    df["early_pos"] = [corner_pos(p, 0) for p in df["passing"]] / df["field_size"]
    df["late_pos"] = [corner_pos(p, -1) for p in df["passing"]] / df["field_size"]
    df["is_nige"] = (df["style"] == "逃げ").astype(float)
    df.loc[df["passing"].isna(), "is_nige"] = np.nan
    for c in ("time_behind", "finish_ratio", "last3f_rank", "early_pos", "late_pos", "is_nige"):
        df.loc[df["_obs"] == 0, c] = np.nan

    df["is_inner_course"] = (df["inout"] == "内").astype(float)
    df["horse_no_ratio"] = df["horse_no"].astype(float) / df["field_size"]
    df["dist_diff"] = df["distance"] - df.groupby("horse_id")["distance"].shift(1)
    df["class_diff"] = df["class_rank"] - df.groupby("horse_id")["class_rank"].shift(1)
    df["jockey_changed"] = (df["jockey_change"] == "乗り替わり").astype(float)
    df.loc[df["jockey_change"] == "前走なし", "jockey_changed"] = np.nan

    # 馬の過去成績（1つ前までの出走）
    df = df.sort_values(["horse_id", "date", "race_id"])
    g = df.groupby("horse_id", sort=False)

    def past_mean(col, n):
        return g[col].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())

    def past_last(col):
        return g[col].shift(1)

    obs = df["_obs"]
    df["h_runs"] = g["_obs"].cumsum() - obs
    cw = g["is_win"].cumsum() - df["is_win"] * obs
    ct = g["is_top3"].cumsum() - df["is_top3"] * obs
    df["h_win_rate"] = (cw + 0.5 * 0.08) / (df["h_runs"] + 0.5)
    df["h_top3_rate"] = (ct + 0.5 * 0.24) / (df["h_runs"] + 0.5)
    df["h_finish_ratio_last"] = past_last("finish_ratio")
    df["h_finish_ratio_mean3"] = past_mean("finish_ratio", 3)
    df["h_finish_ratio_mean5"] = past_mean("finish_ratio", 5)
    df["h_time_behind_last"] = past_last("time_behind")
    df["h_time_behind_mean3"] = past_mean("time_behind", 3)
    df["h_last3f_rank_last"] = past_last("last3f_rank")
    df["h_last3f_rank_mean3"] = past_mean("last3f_rank", 3)
    df["h_early_pos_mean5"] = past_mean("early_pos", 5)
    df["h_late_pos_mean5"] = past_mean("late_pos", 5)
    df["h_nige_rate"] = past_mean("is_nige", 5)
    df["_prize"] = df["prize"].fillna(0) if "prize" in df else 0.0
    df.loc[df["_obs"] == 0, "_prize"] = np.nan
    df["h_prize_mean3"] = past_mean("_prize", 3)
    df = df.sort_values(["date", "race_id", "horse_no"]).reset_index(drop=True)

    # 条件別の馬の成績（前日まで）
    df["h_dist_band"] = df["distance_band"]
    for keys, name in (
        (["horse_id", "surface"], "h_surface"),
        (["horse_id", "venue_code", "surface"], "h_course"),
        (["horse_id", "distance_band"], "h_dist"),
        (["horse_id", "track_condition"], "h_track"),
    ):
        r = _prior_rate(df, keys, name, prior_n=2.0, rates=("top3",))
        df[name + "_top3"] = r[name + "_top3"]
        if name + "_runs" in FEATURES:
            df[name + "_runs"] = r[name + "_runs"]

    # 騎手・調教師・血統
    r = _prior_rate(df, ["jockey_id"], "j")
    df["j_win_rate"], df["j_top3_rate"], df["j_runs"] = r["j_win_rate"], r["j_top3"], r["j_runs"]
    df["j_course_top3"] = _prior_rate(df, ["jockey_id", "venue_code", "surface"], "jc", rates=("top3",))["jc_top3"]
    r = _prior_rate(df, ["trainer_id"], "t")
    df["t_win_rate"], df["t_top3_rate"] = r["t_win_rate"], r["t_top3"]
    df["jt_top3"] = _prior_rate(df, ["jockey_id", "trainer_id"], "jt", rates=("top3",))["jt_top3"]
    r = _prior_rate(df, ["sire_id", "surface"], "ss", prior_n=30, rates=("top3",))
    df["sire_surface_top3"], df["sire_runs"] = r["ss_top3"], r["ss_runs"]
    df["sire_dist_top3"] = _prior_rate(df, ["sire_id", "distance_band"], "sd", 30, ("top3",))["sd_top3"]
    df["sire_track_top3"] = _prior_rate(df, ["sire_id", "track_condition"], "st", 30, ("top3",))["st_top3"]
    df["damsire_surface_top3"] = _prior_rate(df, ["damsire_id", "surface"], "ds", 30, ("top3",))["ds_top3"]

    # レース内の相対値・展開
    gr = df.groupby("race_id")
    for col, out in (("h_finish_ratio_mean5", "rel_finish_ratio_mean5"),
                     ("h_time_behind_mean3", "rel_time_behind_mean3"),
                     ("h_early_pos_mean5", "rel_early_pos"),
                     ("h_top3_rate", "rel_h_top3_rate"),
                     ("j_top3_rate", "rel_j_top3_rate")):
        df[out] = df[col] - gr[col].transform("mean")
    df["rank_early_pos"] = gr["h_early_pos_mean5"].rank(method="min")
    front = ((df["h_nige_rate"] >= 0.3) | (df["h_early_pos_mean5"] <= 0.15)).astype(int)
    df["n_front_runners"] = front.groupby(df["race_id"]).transform("sum")

    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


# ------------------------------------------------------------------ 学習
def _lgb_params(cfg: dict) -> dict:
    p = dict(cfg["model"]["lightgbm"])
    p.update(objective="binary", verbose=-1, random_state=42)
    return p


def _to_matrix(df: pd.DataFrame, categories: dict | None = None) -> tuple[pd.DataFrame, dict]:
    X = df[FEATURES].copy()
    cats = categories or {c: list(X[c].astype(str).dropna().unique()) for c in CATEGORICAL}
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c].astype(str), categories=cats[c])
    for c in X.columns:
        if c not in CATEGORICAL:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    return X, cats


def normalize_probs(df: pd.DataFrame, raw_win: np.ndarray, raw_top3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """勝率は1レースの合計が1、複勝率は合計が3（頭数が3未満なら頭数）になるように揃える。"""
    tmp = pd.DataFrame({"race_id": df["race_id"].to_numpy(), "w": raw_win, "t": raw_top3})
    w = tmp["w"] / tmp.groupby("race_id")["w"].transform("sum")
    n = tmp.groupby("race_id")["t"].transform("size").clip(upper=3)
    t = tmp["t"] * n / tmp.groupby("race_id")["t"].transform("sum")
    # 1 を超えた分は丸める（数回まわして合計を保つ）
    for _ in range(5):
        t = t.clip(upper=0.99)
        t = t * n / t.groupby(tmp["race_id"]).transform("sum")
    t = np.minimum(np.maximum(t, w), 0.99)
    return w.to_numpy(), t.to_numpy()


class ModelBundle:
    def __init__(self, win_model, top3_model, categories, meta):
        self.win_model = win_model
        self.top3_model = top3_model
        self.categories = categories
        self.meta = meta

    @property
    def version(self) -> str:
        return self.meta["version"]

    def predict(self, feat: pd.DataFrame) -> pd.DataFrame:
        X, _ = _to_matrix(feat, self.categories)
        rw = self.win_model.predict_proba(X)[:, 1]
        rt = self.top3_model.predict_proba(X)[:, 1]
        pw, pt = normalize_probs(feat, rw, rt)
        out = feat[["race_id", "horse_no", "horse_id"]].copy()
        out["p_win"] = pw
        out["p_top3"] = pt
        return out

    def importance(self) -> pd.DataFrame:
        imp = pd.DataFrame({
            "特徴量": FEATURES,
            "重要度(勝率)": self.win_model.booster_.feature_importance("gain"),
            "重要度(複勝率)": self.top3_model.booster_.feature_importance("gain"),
        })
        for c in ("重要度(勝率)", "重要度(複勝率)"):
            imp[c] = imp[c] / imp[c].sum()
        return imp.sort_values("重要度(複勝率)", ascending=False).reset_index(drop=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "ModelBundle":
        with open(path, "rb") as f:
            return pickle.load(f)


def fit_models(train: pd.DataFrame, cfg: dict, n_estimators: int | None = None,
               valid: pd.DataFrame | None = None):
    params = _lgb_params(cfg)
    if n_estimators:
        params["n_estimators"] = n_estimators
    X, cats = _to_matrix(train)
    models, best = {}, {}
    for target in ("is_win", "is_top3"):
        m = lgb.LGBMClassifier(**params)
        kw = {}
        if valid is not None and len(valid):
            Xv, _ = _to_matrix(valid, cats)
            kw = dict(eval_set=[(Xv, valid[target])], callbacks=[lgb.early_stopping(50, verbose=False)])
        m.fit(X, train[target], **kw)
        models[target] = m
        best[target] = int(m.best_iteration_ or params["n_estimators"]) if kw else params["n_estimators"]
    return models, cats, best


def evaluate(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    d = pred.merge(truth[["race_id", "horse_no", "is_win", "is_top3", "popularity", "odds_win"]],
                   on=["race_id", "horse_no"])
    d = d.dropna(subset=["is_win"])
    m = {"races": int(d["race_id"].nunique()), "rows": int(len(d))}
    eps = 1e-6
    m["logloss_win"] = float(log_loss(d["is_win"], d["p_win"].clip(eps, 1 - eps), labels=[0, 1]))
    m["logloss_top3"] = float(log_loss(d["is_top3"], d["p_top3"].clip(eps, 1 - eps), labels=[0, 1]))
    if d["is_win"].nunique() > 1:
        m["auc_win"] = float(roc_auc_score(d["is_win"], d["p_win"]))
        m["auc_top3"] = float(roc_auc_score(d["is_top3"], d["p_top3"]))
    top = d.loc[d.groupby("race_id")["p_win"].idxmax()]
    m["top1_win_hit"] = float(top["is_win"].mean())
    m["top1_top3_hit"] = float(top["is_top3"].mean())
    # 市場（単勝オッズ）との比較：オッズから出した勝率の logloss
    mk = d.dropna(subset=["odds_win"]).copy()
    if len(mk):
        mk["p_mkt"] = (1 / mk["odds_win"]) / (1 / mk["odds_win"]).groupby(mk["race_id"]).transform("sum")
        m["logloss_win_market"] = float(log_loss(mk["is_win"], mk["p_mkt"].clip(eps, 1 - eps), labels=[0, 1]))
        fav = d[d["popularity"] == 1]
        m["fav_win_hit"] = float(fav["is_win"].mean()) if len(fav) else None
    return m


def load_training_frame(engine, cfg: dict | None = None) -> pd.DataFrame:
    base = load_base(engine, with_payouts=False)
    base = base[base["surface"] != "障害"]  # 障害は別物なので除外
    return build_features(base)


def train(engine=None, cfg: dict | None = None, valid_from: str | None = None,
          until: str | None = None, save: bool = True, feat: pd.DataFrame | None = None) -> tuple[ModelBundle, dict]:
    """時系列で分けて学習・検証し、最後に全期間で学習し直したモデルを返す。

    - 学習：valid_from より前
    - 検証：valid_from 〜 until
    - 本番用：until までの全データで、検証で決めた木の数を使って学習し直す
    """
    cfg = cfg or load_config()
    engine = engine or db.get_engine(cfg)
    if feat is None:
        feat = load_training_frame(engine, cfg)
    feat = feat[feat["_obs"] == 1]
    if until:
        feat = feat[feat["date"] <= pd.Timestamp(until)]
    last = feat["date"].max()
    if valid_from is None:
        valid_from = (last - pd.DateOffset(months=cfg["model"]["valid_months"])).strftime("%Y-%m-%d")
    vf = pd.Timestamp(valid_from)
    tr, va = feat[feat["date"] < vf], feat[feat["date"] >= vf]
    if tr["race_id"].nunique() < cfg["model"].get("min_train_races", 300):
        log.warning("学習データが少なすぎます（%d レース）。結果は参考程度にしてください。", tr["race_id"].nunique())
    if tr.empty or va.empty:
        raise ValueError("学習期間か検証期間のデータがありません。期間を見直してください。")
    log.info("学習 %s〜%s (%d レース) / 検証 %s〜%s (%d レース)", tr["date"].min().date(),
             tr["date"].max().date(), tr["race_id"].nunique(), va["date"].min().date(),
             va["date"].max().date(), va["race_id"].nunique())

    models, cats, best = fit_models(tr, cfg, valid=va)
    vb = ModelBundle(models["is_win"], models["is_top3"], cats, {})
    metrics = evaluate(vb.predict(va), va)
    metrics["best_iteration"] = best

    # 本番用：全期間で学習し直す
    n_est = max(50, int(np.mean(list(best.values())) * 1.1))
    full_models, full_cats, _ = fit_models(feat, cfg, n_estimators=n_est)
    version = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    meta = {
        "version": version,
        "train_from": str(tr["date"].min().date()),
        "valid_from": str(vf.date()),
        "data_until": str(last.date()),
        "n_estimators": n_est,
        "metrics": metrics,
        "features": FEATURES,
    }
    bundle = ModelBundle(full_models["is_win"], full_models["is_top3"], full_cats, meta)
    if save:
        bundle.save(model_dir(cfg) / f"model-{version}.pkl")
    return bundle, metrics


def model_dir(cfg: dict) -> Path:
    return resolve_path(cfg, cfg["model"]["dir"])


def current_model_path(cfg: dict) -> Path:
    return model_dir(cfg) / "current.pkl"


def adopt(bundle: ModelBundle, cfg: dict, engine=None, note: str = "") -> None:
    """本番用モデルとして current.pkl に保存し、model_runs に記録する。"""
    bundle.save(current_model_path(cfg))
    record_run(bundle, cfg, engine, adopted=True, note=note)


def record_run(bundle: ModelBundle, cfg: dict, engine=None, adopted: bool = False, note: str = "") -> None:
    engine = engine or db.get_engine(cfg)
    with engine.begin() as c:
        db.upsert(c, db.model_runs, [{
            "version": bundle.version, "created_at": dt.datetime.now(),
            "train_until": bundle.meta["data_until"], "valid_from": bundle.meta["valid_from"],
            "valid_until": bundle.meta["data_until"],
            "metrics": json.dumps(bundle.meta["metrics"], ensure_ascii=False),
            "adopted": adopted, "note": note,
        }])


def load_current(cfg: dict | None = None) -> ModelBundle | None:
    cfg = cfg or load_config()
    p = current_model_path(cfg)
    return ModelBundle.load(p) if p.exists() else None


def walk_forward(feat: pd.DataFrame, cfg: dict, start: str, end: str | None = None,
                 step_months: int = 1) -> pd.DataFrame:
    """バックテスト用：期間を区切って「その時点より前のデータだけで学習→次の期間を予測」を繰り返す。"""
    feat = feat[feat["_obs"] == 1]
    s = pd.Timestamp(start)
    e = pd.Timestamp(end) if end else feat["date"].max()
    outs = []
    while s <= e:
        nxt = s + pd.DateOffset(months=step_months)
        tr = feat[feat["date"] < s]
        te = feat[(feat["date"] >= s) & (feat["date"] < nxt) & (feat["date"] <= e)]
        if len(te) and tr["race_id"].nunique() >= 50:
            models, cats, _ = fit_models(tr, cfg, n_estimators=cfg["model"].get("walk_forward_trees", 300))
            p = ModelBundle(models["is_win"], models["is_top3"], cats, {}).predict(te)
            outs.append(p)
            log.info("walk-forward %s: 学習 %d レース → 予測 %d レース", s.date(),
                     tr["race_id"].nunique(), te["race_id"].nunique())
        s = nxt
    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()
