"""競馬予想・分析アプリの画面（Streamlit）。

起動：  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from keiba import db
from keiba.config import load_config, resolve_path
from keiba.constants import VENUES
from keiba.stats import DIMENSIONS, FILTER_COLUMNS, apply_filters, summarize

st.set_page_config(page_title="競馬予想・分析", page_icon="🏇", layout="wide")

# 検証済みの配色（カテゴリ1・2番目）
BLUE, ORANGE = "#2a78d6", "#eb6834"

# スマホでも読みやすく：余白を詰め、表の文字を少し小さく
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-left: 1rem; padding-right: 1rem; max-width: 1200px;}
@media (max-width: 640px) {
  .block-container {padding-left: .5rem; padding-right: .5rem;}
  h1 {font-size: 1.5rem;} h2 {font-size: 1.25rem;} h3 {font-size: 1.1rem;}
}
</style>
""", unsafe_allow_html=True)

CFG = load_config()
ENGINE = db.get_engine(CFG)
ROOT = Path(__file__).resolve().parent
PCT = st.column_config.NumberColumn(format="%.1f%%")


def pct_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df:
            df[c] = df[c] * 100
    return df


@st.cache_data(ttl=600, show_spinner="データを読み込んでいます…")
def base_data() -> pd.DataFrame:
    from keiba.features import load_base
    return load_base(ENGINE)


@st.cache_resource(show_spinner="モデルと履歴データを準備しています…")
def predictor():
    from keiba.predict import Predictor
    return Predictor(CFG)


st.title("🏇 競馬予想・分析")
tab_data, tab_stats, tab_pred, tab_bt = st.tabs(["データ更新", "統計分析", "レース予想", "バックテスト"])

# ======================================================================= データ更新
with tab_data:
    c = db.read_sql(ENGINE, """
        SELECT (SELECT COUNT(*) FROM races) AS races, (SELECT COUNT(*) FROM entries) AS entries,
               (SELECT COUNT(*) FROM horses) AS horses,
               (SELECT COUNT(*) FROM horses WHERE detail_fetched IN (1, true)) AS horses_detail,
               (SELECT MIN(date) FROM races) AS first, (SELECT MAX(date) FROM races) AS last,
               (SELECT COUNT(DISTINCT race_id) FROM odds WHERE status='result') AS odds_races
    """).iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("レース数", f"{int(c['races']):,}")
    m2.metric("出走データ", f"{int(c['entries']):,}")
    m3.metric("馬（詳細取得済み）", f"{int(c['horses_detail']):,} / {int(c['horses']):,}")
    st.caption(f"期間：{c['first'] or '-'} 〜 {c['last'] or '-'}　／　確定オッズのあるレース：{int(c['odds_races']):,}")

    st.subheader("過去データの取得")
    st.caption("取得はバックグラウンドで動きます（間隔3秒以上・1本ずつ）。止めても次回は続きから再開します。")
    col1, col2 = st.columns(2)
    d_from = col1.date_input("開始日", dt.date.today() - dt.timedelta(days=30))
    d_to = col2.date_input("終了日", dt.date.today())
    opts = st.multiselect("取得するもの", ["確定オッズ", "馬の詳細（血統・生産者）", "騎手の所属"],
                          default=["確定オッズ", "馬の詳細（血統・生産者）", "騎手の所属"])
    log_path = resolve_path(CFG, "data/logs/app_collect.log")
    if st.button("取得を開始", type="primary"):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        args = [sys.executable, "-m", "keiba", "collect", "--start", str(d_from), "--end", str(d_to)]
        if "確定オッズ" not in opts:
            args.append("--no-odds")
        if "馬の詳細（血統・生産者）" not in opts:
            args.append("--no-horses")
        if "騎手の所属" not in opts:
            args.append("--no-jockeys")
        with open(log_path, "a", encoding="utf-8") as f:
            subprocess.Popen(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        st.success("開始しました。下の『進み具合』で確認できます。")
    with st.expander("進み具合（ログの最後）"):
        if log_path.exists():
            st.code("".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines(True)[-15:]))
        st.button("更新")

    st.subheader("モデル")
    runs = db.read_sql(ENGINE, "SELECT version, created_at, train_until, valid_from, adopted, note, metrics "
                               "FROM model_runs ORDER BY created_at DESC LIMIT 10")
    if len(runs):
        import json
        mt = pd.json_normalize(runs["metrics"].map(json.loads))
        show = pd.concat([runs[["version", "adopted", "train_until", "note"]], mt[[
            c for c in ("races", "top1_win_hit", "top1_top3_hit", "logloss_top3", "auc_top3") if c in mt]]], axis=1)
        show.columns = [{"version": "版", "adopted": "採用", "train_until": "データ最終日", "note": "メモ",
                         "races": "検証レース", "top1_win_hit": "◎勝率", "top1_top3_hit": "◎複勝率",
                         "logloss_top3": "logloss(複勝)", "auc_top3": "AUC(複勝)"}.get(x, x) for x in show.columns]
        st.dataframe(show, hide_index=True, use_container_width=True)
    else:
        st.info("まだ学習したモデルがありません。")
    if st.button("今のデータで学習し直す（数分かかります）"):
        from keiba.model import adopt, train
        with st.spinner("学習中…"):
            b, m = train(ENGINE, CFG)
            adopt(b, CFG, ENGINE, note="画面から学習")
            predictor.clear()
        st.success(f"モデル {b.version} を採用しました。")
        st.dataframe(b.importance().head(20), hide_index=True)

    st.subheader("週次レポート・自動予想")
    rdir = resolve_path(CFG, CFG["weekly"]["report_dir"])
    files = sorted(rdir.glob("*"), reverse=True) if rdir.exists() else []
    if files:
        f = st.selectbox("ファイル", files, format_func=lambda p: p.name)
        st.code(Path(f).read_text(encoding="utf-8"), language="markdown")
    else:
        st.caption("まだありません（土日の朝の自動予想・日曜夜の週次処理で作られます）。")

# ======================================================================= 統計分析
with tab_stats:
    df = base_data()
    if df.empty:
        st.info("データがありません。『データ更新』で取得してください。")
    else:
        dim = st.selectbox("切り口", list(DIMENSIONS), index=list(DIMENSIONS).index("脚質"))
        with st.expander("絞り込み（組み合わせ可：例 東京 × ダート × 1600 × 重 × 差し）", expanded=False):
            fc = st.columns(3)
            filters = {}
            for i, (label, col) in enumerate(FILTER_COLUMNS.items()):
                vals = sorted(df[col].dropna().unique().tolist(), key=lambda x: (str(type(x)), x))
                if len(vals) > 300:
                    vals = df[col].value_counts().head(300).index.tolist()
                sel = fc[i % 3].multiselect(label, vals, key=f"f_{col}")
                if sel:
                    filters[col] = sel
            dc = st.columns(2)
            dfrom = dc[0].date_input("期間（から）", df["date"].min().date(), key="s_from")
            dto = dc[1].date_input("期間（まで）", df["date"].max().date(), key="s_to")
            filters["date_from"], filters["date_to"] = str(dfrom), str(dto)
        min_runs = st.slider("表示する最小出走数", 1, 200, 10 if len(DIMENSIONS[dim]) > 1 or dim in (
            "父", "母父", "騎手", "調教師", "生産者") else 1)
        d = apply_filters(df, filters)
        t = summarize(d, DIMENSIONS[dim], CFG["stats"]["min_runs_warning"], min_runs=min_runs)
        st.caption(f"対象：{d['race_id'].nunique():,} レース / {len(d):,} 頭。"
                   f"出走数 {CFG['stats']['min_runs_warning']} 未満は「参考値」。95%CI は Wilson 法。"
                   "回収率は100円ずつ買った場合。")
        if t.empty:
            st.warning("該当するデータがありません。")
        else:
            key = DIMENSIONS[dim]
            t["項目"] = t[key].astype(str).agg(" × ".join, axis=1)
            show = pct_cols(t, ["勝率", "連対率", "複勝率", "単勝回収率", "複勝回収率"])
            st.dataframe(show[["項目", "出走数", "1着", "2着", "3着", "勝率", "連対率", "複勝率", "勝率95%CI",
                               "複勝率95%CI", "単勝回収率", "複勝回収率", "注意"]], hide_index=True,
                         use_container_width=True,
                         column_config={c: PCT for c in ("勝率", "連対率", "複勝率", "単勝回収率", "複勝回収率")})
            top = t.head(25)
            metric = st.radio("グラフ", ["複勝率", "勝率", "単勝回収率", "複勝回収率"], horizontal=True)
            fig = px.bar(top, x="項目", y=metric, color_discrete_sequence=[BLUE],
                         hover_data={"出走数": True, metric: ":.1%"})
            fig.update_traces(marker_line_width=0)
            if "回収率" in metric:
                fig.add_hline(y=1.0, line_dash="dot", line_color="gray", annotation_text="100%")
            fig.update_layout(yaxis_tickformat=".0%", height=380, margin=dict(l=10, r=10, t=30, b=10),
                              xaxis_title=None, title=f"{dim}別の{metric}（出走数の多い順 上位25）")
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("CSVで保存", t.to_csv(index=False).encode("utf-8-sig"), "stats.csv")

# ======================================================================= レース予想
with tab_pred:
    c1, c2, c3, c4 = st.columns([2, 2, 1, 2])
    today = dt.date.today()
    nxt = today + dt.timedelta(days=(5 - today.weekday()) % 7)  # 次の土曜（今日が土日なら今日）
    if today.weekday() == 6:
        nxt = today
    pdate = c1.date_input("日付", nxt, key="p_date")
    venue = c2.selectbox("競馬場", list(VENUES.values()), index=4)
    rno = c3.number_input("R", 1, 12, 11)
    budget = c4.number_input("予算（円）", 1000, 1_000_000, 10000, step=1000,
                             help="1/4ケリーで配分します。1レースで使うのは最大で予算の半分です。")
    if st.button("予想する", type="primary"):
        try:
            pr = predictor()
            with st.spinner("出馬表・オッズを取得して予想しています…（初めての馬は血統の取得に時間がかかります）"):
                st.session_state["pred"] = pr.predict_race(str(pdate), venue, int(rno), budget=budget)
        except Exception as e:  # noqa: BLE001
            st.error(f"予想できませんでした：{e}")
    p = st.session_state.get("pred")
    if p is not None:
        r = p.race
        st.subheader(f"{r['date']} {r['venue']}{r['race_no']}R {r['race_name']}")
        st.caption(f"{r['surface']}{r['distance']}m {r.get('direction') or ''}{r.get('inout') or ''}　"
                   f"{r.get('class_code') or ''}　{r.get('field_size')}頭　馬場：{r.get('track_condition') or '未発表'}"
                   f"　／　オッズ：{p.odds_status}")
        for n in p.notes:
            st.warning(n)

        u = p.upset
        k1, k2, k3 = st.columns(3)
        k1.metric("荒れ度", f"{u['スコア']} / 100", u["判定"], delta_color="off")
        k2.metric("予想ペース", p.pace["ペース"])
        k3.metric("買い目", p.bet_summary["判断"],
                  f"{p.bet_summary.get('点数', 0)}点 {p.bet_summary.get('合計金額', 0):,}円" if "点数" in p.bet_summary else None,
                  delta_color="off")

        st.markdown("### 1. 展開予想")
        pc = p.pace
        st.write(f"**{pc['ペース']}ペース**：{pc['コメント']}")
        st.write(f"逃げ候補：{', '.join(pc['逃げ候補']) or 'なし'}　／　先行候補：{', '.join(pc['先行候補']) or 'なし'}")
        st.table(pd.DataFrame([(k, "、".join(v)) for k, v in pc["隊列"].items() if v], columns=["位置", "馬"]))
        st.write(f"このコースで有利：脚質 **{pc['有利な脚質'] or 'データ不足'}** ／ 枠 **{pc['有利な枠'] or 'データ不足'}**"
                 f"（過去 {pc['コース過去レース数']} レース）")
        for name, tbl in pc["コース成績"].items():
            with st.expander(f"コース成績：{name}"):
                st.dataframe(pct_cols(tbl, ["勝率", "複勝率", "単勝回収率", "複勝回収率"]), use_container_width=True,
                             column_config={c: PCT for c in ("勝率", "複勝率", "単勝回収率", "複勝回収率")})

        st.markdown("### 2. 各馬の評価")
        tb = pct_cols(p.table, ["推定勝率", "推定複勝率", "買い目用勝率"])
        st.caption("推定勝率・複勝率＝モデルの予想。買い目の期待値は、モデルと単勝オッズを合成した『買い目用勝率』で計算します。")
        st.dataframe(tb, hide_index=True, use_container_width=True, column_config={
            "推定勝率": PCT, "推定複勝率": PCT, "買い目用勝率": PCT,
            "単勝オッズ": st.column_config.NumberColumn(format="%.1f"),
            "期待値(単)": st.column_config.NumberColumn(format="%.2f"),
            "根拠": st.column_config.TextColumn(width="large")})

        st.markdown("### 3. 回収率重視の買い目")
        s = p.bet_summary
        if s["判断"] == "見送り推奨":
            st.info(f"見送り推奨：{s['理由']}")
        else:
            st.success(f"{s['点数']}点・合計 {s['合計金額']:,} 円・平均期待値 {s['平均期待値']}")
            st.dataframe(pct_cols(p.bets, ["推定確率"]).drop(columns=["ケリー"]), hide_index=True,
                         use_container_width=True, column_config={
                             "推定確率": PCT, "オッズ": st.column_config.NumberColumn(format="%.1f"),
                             "期待値": st.column_config.NumberColumn(format="%.2f"),
                             "金額": st.column_config.NumberColumn(format="%d 円")})
        st.caption("期待値＝推定確率×オッズ。複勝・ワイドは下限オッズで計算。組み合わせの確率は推定勝率から"
                   "Harville法で計算。オッズは締切までに変わるので、直前に見直してください。")

        st.markdown("### 4. 荒れ度")
        fig = go.Figure(go.Indicator(mode="gauge+number", value=u["スコア"],
                                     gauge={"axis": {"range": [0, 100]}, "bar": {"color": BLUE},
                                            "steps": [{"range": [0, 30], "color": "rgba(128,128,128,.08)"},
                                                      {"range": [30, 50], "color": "rgba(128,128,128,.16)"},
                                                      {"range": [50, 70], "color": "rgba(128,128,128,.24)"},
                                                      {"range": [70, 100], "color": "rgba(128,128,128,.32)"}]},
                                     title={"text": u["判定"]}))
        fig.update_layout(height=230, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(u["根拠"], hide_index=True, use_container_width=True)
        if u["穴馬候補"]:
            st.write("**穴馬候補**")
            for h in u["穴馬候補"]:
                st.write(f"- {h['馬番']} {h['馬名']}（{h['人気']}番人気）：{h['理由']}")
        else:
            st.caption("穴馬候補：なし")

# ======================================================================= バックテスト
with tab_bt:
    from keiba.backtest import DEFAULT_STRATEGIES, Strategy, compare, equity_curve, load_market, run_backtest
    st.caption("過去の各月について「その月より前のデータだけで学習したモデル」で予想し、買い方ごとの成績を比べます。"
               "買う判断には確定オッズを使うので、実際より良く見える可能性があります。")
    wf_path = resolve_path(CFG, "data/backtest/walk_forward_predictions.csv")
    bc1, bc2 = st.columns(2)
    bt_from = bc1.date_input("バックテスト開始日", dt.date.today() - dt.timedelta(days=180), key="bt_from")
    step = bc2.selectbox("学習し直す間隔（月）", [1, 2, 3, 6], index=0)
    if st.button("予測を作り直す（データ量によって数分〜数十分）"):
        from keiba.model import load_training_frame, walk_forward
        with st.spinner("walk-forward で予測しています…"):
            feat = load_training_frame(ENGINE, CFG)
            pred = walk_forward(feat, CFG, str(bt_from), step_months=step)
            wf_path.parent.mkdir(parents=True, exist_ok=True)
            pred.to_csv(wf_path, index=False)
    if not wf_path.exists():
        st.info("まだ予測がありません。上のボタンで作ってください。")
    else:
        pred = pd.read_csv(wf_path, dtype={"race_id": str, "horse_id": str})
        ent = db.read_sql(ENGINE, "SELECT e.race_id, e.horse_no, r.date, e.odds_win, e.popularity FROM entries e "
                                  "JOIN races r ON r.race_id=e.race_id")
        ent["date"] = pd.to_datetime(ent["date"])
        st.caption(f"予測済み：{pred['race_id'].nunique():,} レース")
        with st.expander("自分の買い方を追加"):
            a1, a2, a3 = st.columns(3)
            btype = a1.selectbox("券種", ["単勝", "複勝", "馬連", "ワイド", "三連複"])
            mev = a2.number_input("期待値の下限", 0.5, 5.0, 1.2, 0.05)
            mpr = a3.number_input("推定確率の下限", 0.0, 0.9, 0.02, 0.01)
            custom = Strategy(f"自分：{btype} 期待値{mev}以上 確率{mpr:.0%}以上", "ev", (btype,), mev, mpr)
        names = [s.name for s in DEFAULT_STRATEGIES]
        chosen = st.multiselect("比べる買い方", names, default=names[:4])
        strategies = [s for s in DEFAULT_STRATEGIES if s.name in chosen] + [custom]
        from keiba.model import load_current
        _cur = load_current(CFG)
        _a = CFG["betting"].get("market_alpha")
        if not isinstance(_a, (int, float)):
            _a = _cur.meta.get("market_alpha", 0.5) if _cur else 0.5
        alpha = st.slider("モデルと市場オッズの合成比率 α（1=モデルだけ、0=オッズだけ）", 0.0, 1.0, float(_a), 0.05,
                          help="期待値の判断に使う勝率 ∝ モデル^α × 市場^(1-α)。初期値は学習時に検証データで決めた値です。")
        odds, pays = load_market(ENGINE, list(pred["race_id"].unique()))
        res = run_backtest(pred, ent, odds, pays, strategies, alpha=alpha)
        cmp_ = compare(res)
        st.dataframe(pct_cols(cmp_, ["回収率", "的中率(点)", "的中率(レース)"]), hide_index=True,
                     use_container_width=True,
                     column_config={c: PCT for c in ("回収率", "的中率(点)", "的中率(レース)")} |
                     {c: st.column_config.NumberColumn(format="%d 円") for c in ("投資額", "払戻額", "収支", "最大ドローダウン")})
        curves = pd.concat([equity_curve(v).assign(買い方=k) for k, v in res.items() if len(v)], ignore_index=True)
        if len(curves):
            fig = px.line(curves, x="date", y="累積収支", color="買い方",
                          color_discrete_sequence=[BLUE, ORANGE, "#1baf7a", "#eda100", "#e87ba4", "#008300",
                                                   "#6250d6", "#e34948"])
            fig.update_traces(line_width=2)
            fig.add_hline(y=0, line_color="gray", line_width=1)
            fig.update_layout(height=420, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
                              legend=dict(orientation="h", y=-0.2), xaxis_title=None, yaxis_title="累積収支（円）",
                              title="収支の推移")
            st.plotly_chart(fig, use_container_width=True)
        if any(len(v) for v in res.values()) and odds.empty:
            st.warning("確定オッズ（馬連・ワイド・三連複・複勝）が無いため、それらの券種は計算できません。"
                       "『データ更新』で確定オッズを取得してください。")
