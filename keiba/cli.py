"""コマンドラインの入口。  python -m keiba <コマンド> --help"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_collect(a) -> None:
    from .collect import Collector
    from .config import load_config
    cfg = load_config(a.config)
    end = dt.date.fromisoformat(a.end) if a.end else dt.date.today()
    if a.start:
        start = dt.date.fromisoformat(a.start)
    else:
        years = a.years or cfg["collect"]["default_years"]
        start = end.replace(year=end.year - years)
    col = Collector(cfg)
    stats = col.run(start, end, race_limit=a.race_limit, horse_limit=a.horse_limit,
                    odds=not a.no_odds, horses=not a.no_horses, jockeys=not a.no_jockeys)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def _parse_filters(items):
    """--filter venue=東京 --filter distance=1600 --filter style=差し,追込"""
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        vals = [int(x) if x.isdigit() else x for x in v.split(",")]
        out[k] = vals if len(vals) > 1 else vals[0]
    return out


def cmd_stats(a) -> None:
    import pandas as pd

    from . import db
    from .config import load_config
    from .features import load_base
    from .stats import DIMENSIONS, apply_filters, format_table, summarize
    cfg = load_config(a.config)
    if a.list:
        print("切り口:", " / ".join(DIMENSIONS))
        return
    by = DIMENSIONS.get(a.by, a.by.split(","))
    df = load_base(db.get_engine(cfg))
    df = apply_filters(df, _parse_filters(a.filter))
    t = summarize(df, by, cfg["stats"]["min_runs_warning"], min_runs=a.min_runs)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.unicode.east_asian_width", True)
    print(format_table(t).head(a.top).to_string(index=False))
    if a.csv:
        t.to_csv(a.csv, index=False, encoding="utf-8-sig")


def cmd_train(a) -> None:
    from . import db
    from .config import load_config
    from .model import adopt, load_current, record_run, train
    cfg = load_config(a.config)
    eng = db.get_engine(cfg)
    bundle, m = train(eng, cfg, valid_from=a.valid_from, until=a.until)
    print("検証結果:", json.dumps(m, ensure_ascii=False, indent=2))
    print("\n特徴量の重要度（上位20）:")
    print(bundle.importance().head(20).to_string(index=False))
    cur = load_current(cfg)
    if a.force or cur is None:
        adopt(bundle, cfg, eng, note="手動学習")
        print(f"\nモデル {bundle.version} を採用しました。")
    else:
        record_run(bundle, cfg, eng, adopted=False, note="手動学習（未採用）")
        print(f"\nモデル {bundle.version} を保存しました（採用は --force で）。")


def cmd_backtest(a) -> None:
    import pandas as pd

    from . import db
    from .backtest import DEFAULT_STRATEGIES, compare, load_market, run_backtest
    from .config import load_config, resolve_path
    from .features import load_base
    from .model import load_training_frame, walk_forward
    cfg = load_config(a.config)
    eng = db.get_engine(cfg)
    feat = load_training_frame(eng, cfg)
    pred = walk_forward(feat, cfg, a.start, a.end, step_months=a.step)
    if pred.empty:
        print("予測できる期間がありません（学習データが足りない可能性があります）")
        return
    ent = feat[["race_id", "horse_no", "date", "odds_win", "popularity"]]
    odds, pays = load_market(eng, list(pred["race_id"].unique()))
    from .model import load_current
    cur = load_current(cfg)
    alpha = cfg["betting"].get("market_alpha")
    if not isinstance(alpha, (int, float)):
        alpha = cur.meta.get("market_alpha", 0.5) if cur else 0.5
    print(f"モデルと市場の合成比率 α={alpha}（1=モデルだけ）")
    res = run_backtest(pred, ent, odds, pays, DEFAULT_STRATEGIES, alpha=alpha)
    out = compare(res)
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    print(out.to_string(index=False))
    d = resolve_path(cfg, "data/backtest")
    d.mkdir(parents=True, exist_ok=True)
    pred.to_csv(d / "walk_forward_predictions.csv", index=False)
    print(f"\n予測を {d / 'walk_forward_predictions.csv'} に保存しました（画面のバックテストで使います）")


def cmd_predict(a) -> None:
    from .predict import Predictor, format_text
    p = Predictor()
    r = p.predict_race(a.date, a.venue, a.race, budget=a.budget)
    print(format_text(r))


def cmd_auto_predict(a) -> None:
    from .weekly import auto_predict
    out = auto_predict(a.date, budget=a.budget)
    print(f"保存しました: {out}" if out else "開催がありません")


def cmd_weekly(a) -> None:
    from .weekly import weekly
    print(f"レポート: {weekly()}")


def cmd_reparse(a) -> None:
    from .collect import Collector
    from .config import load_config
    from .fetcher import Fetcher
    cfg = load_config(a.config)
    Collector(cfg, fetcher=Fetcher(cfg, offline=True)).reparse_all()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m keiba", description="競馬予想・分析アプリ")
    p.add_argument("--config", help="config.yaml のパス")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="過去レースのデータを取得してDBに保存する（再開可能）")
    c.add_argument("--start", help="開始日 YYYY-MM-DD（省略時は --years 年前）")
    c.add_argument("--end", help="終了日 YYYY-MM-DD（省略時は今日）")
    c.add_argument("--years", type=int, help="何年分取るか（--start を省略したとき）")
    c.add_argument("--race-limit", type=int, help="取得するレース数の上限（試し用）")
    c.add_argument("--horse-limit", type=int, help="詳細を取得する馬の数の上限（試し用）")
    c.add_argument("--no-odds", action="store_true", help="確定オッズを取らない")
    c.add_argument("--no-horses", action="store_true", help="馬の詳細を取らない")
    c.add_argument("--no-jockeys", action="store_true", help="騎手の所属を取らない")
    c.set_defaults(func=cmd_collect)

    s = sub.add_parser("stats", help="切り口ごとの成績を集計する")
    s.add_argument("--by", default="脚質", help="切り口（--list で一覧）または列名をカンマ区切り")
    s.add_argument("--filter", action="append", help="絞り込み 例: venue=東京 surface=ダート distance=1600")
    s.add_argument("--min-runs", type=int, default=1)
    s.add_argument("--top", type=int, default=50)
    s.add_argument("--csv", help="結果をCSVに保存")
    s.add_argument("--list", action="store_true", help="切り口の一覧を表示")
    s.set_defaults(func=cmd_stats)

    t = sub.add_parser("train", help="予測モデルを学習する（時系列で検証）")
    t.add_argument("--valid-from", help="検証期間の開始日（省略時は直近 model.valid_months か月）")
    t.add_argument("--until", help="この日までのデータで学習")
    t.add_argument("--force", action="store_true", help="成績に関係なく本番モデルとして採用する")
    t.set_defaults(func=cmd_train)

    b = sub.add_parser("backtest", help="walk-forward で予測して、買い方ごとの回収率を比べる")
    b.add_argument("--start", required=True, help="バックテスト開始日")
    b.add_argument("--end", help="終了日")
    b.add_argument("--step", type=int, default=1, help="何か月ごとに学習し直すか")
    b.set_defaults(func=cmd_backtest)

    r = sub.add_parser("predict", help="レース予想を出す")
    r.add_argument("--date", required=True, help="開催日 YYYY-MM-DD")
    r.add_argument("--venue", required=True, help="競馬場（例 東京）")
    r.add_argument("--race", type=int, required=True, help="R番号")
    r.add_argument("--budget", type=int, default=10000, help="予算（円）")
    r.set_defaults(func=cmd_predict)

    ap = sub.add_parser("auto-predict", help="その日の全レースを自動で予想して保存する（土日の朝用）")
    ap.add_argument("--date", help="開催日（省略時は今日）")
    ap.add_argument("--budget", type=int, default=10000)
    ap.set_defaults(func=cmd_auto_predict)

    w = sub.add_parser("weekly", help="結果の取り込み→答え合わせ→再学習（日曜の夜用）")
    w.set_defaults(func=cmd_weekly)

    rp = sub.add_parser("reparse", help="キャッシュ済みの結果ページを読み直す（通信しない）")
    rp.set_defaults(func=cmd_reparse)

    a = p.parse_args(argv)
    _setup_logging(a.verbose)
    try:
        a.func(a)
    except KeyboardInterrupt:
        print("\n中断しました。もう一度同じコマンドを実行すると続きから再開します。", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
