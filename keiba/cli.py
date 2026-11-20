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

    a = p.parse_args(argv)
    _setup_logging(a.verbose)
    try:
        a.func(a)
    except KeyboardInterrupt:
        print("\n中断しました。もう一度同じコマンドを実行すると続きから再開します。", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
