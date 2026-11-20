# keiba — 競馬予想・分析アプリ（JRA）

netkeiba の過去レースデータを集めて SQLite に保存し、統計分析・予測モデル・
回収率重視の買い目提案・バックテストを行う個人用アプリです。

> **利用上の注意**：netkeiba の利用規約を確認し、個人で使う範囲にとどめてください。
> このアプリはリクエストの間隔を最低2秒（既定3秒＋揺らぎ）あけ、並列アクセスをせず、
> 取得したページをキャッシュして同じページを取り直さないように作っています。
> 取得したデータやキャッシュを他の人に配らないでください。

## セットアップ（Windows）

1. Python 3.11 以上をインストール（インストーラで「Add python.exe to PATH」にチェック）
2. このフォルダで PowerShell を開いて：

```powershell
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 使い方

### 1. データの取得

```powershell
# まず1か月分で試す
python -m keiba collect --start 2026-10-01 --end 2026-10-31

# 直近5年分（config.yaml の collect.default_years）
python -m keiba collect
```

- 途中で止めても（Ctrl+C、PCの再起動など）、同じコマンドをもう一度実行すれば続きから再開します。
- 5年分の取得には時間がかかります（目安：レース約17,000件で約20時間、馬の詳細 約3〜4万頭で約2〜3日、
  確定オッズは1レース5回の取得が必要）。数日かけてバックグラウンドで動かしてください。
- `--no-horses` `--no-odds` で、馬の詳細や確定オッズを後回しにできます。

取得の流れ・テーブルの構成は [docs/schema.md](docs/schema.md) を見てください。

### テスト

```powershell
python -m pytest -q
```

## フォルダ構成

```
config.yaml          設定（間隔・DB・モデル・買い方）
keiba/
  fetcher.py         取得（間隔制御・キャッシュ・再試行）
  parsers/           ページの解析（結果・馬・血統・出馬表・オッズ）
  db.py              テーブル定義と保存
  collect.py         期間指定の収集（再開可能）
  cli.py             コマンドライン
docs/schema.md       ER図とスキーマ
tests/               テスト（自作の小さなHTMLを使用）
data/                DB・キャッシュ・モデル（git には入れません）
```
