# データベース設計

定義の本体は `keiba/db.py`（SQLAlchemy Core）です。既定は SQLite（`data/keiba.db`）で、
`config.yaml` の `database.url` を PostgreSQL の URL に変えればそのまま移せます
（upsert は SQLite と PostgreSQL の両方の書き方に対応しています）。

## ER図

```mermaid
erDiagram
    races ||--o{ entries : "出走"
    races ||--o{ payouts : "払戻"
    races ||--o{ odds : "オッズ"
    horses ||--o{ entries : "出走"
    horses ||--o| pedigrees : "血統"
    jockeys ||--o{ entries : "騎乗"
    trainers ||--o{ entries : "管理"
    races ||--o{ predictions : "予想"
    races ||--o{ bet_log : "買い目"

    races {
        string race_id PK "YYYY+場+回+日+R"
        string date "YYYY-MM-DD"
        string venue_code "01-10"
        string venue
        int kai
        int nichi
        int race_no
        string race_name
        string grade "G1/G2/G3/L/OP"
        string class_name "新馬/未勝利/1勝/2勝/3勝/オープン"
        string class_code "グレード込み"
        string surface "芝/ダート/障害"
        string direction "右/左/直線"
        string inout "内/外"
        int distance
        string weather
        string track_condition "良/稍重/重/不良"
        int field_size
        float pace_front3f
        float pace_back3f
        text lap_times
        text corner_orders "JSON"
    }
    entries {
        string race_id PK,FK
        int horse_no PK
        int frame_no
        string horse_id FK
        string sex
        int age
        float weight_carried
        string jockey_id FK
        string trainer_id FK
        int horse_weight
        int weight_diff
        float odds_win
        int popularity
        int finish_pos
        string finish_status "中止/除外/取消/失格/降着"
        float time_sec
        string margin
        string passing "通過順位"
        float last3f
        float prize
    }
    horses {
        string horse_id PK
        string name
        string birth_date
        string breeder "生産者"
        string origin "産地"
        string owner
        bool detail_fetched
    }
    pedigrees {
        string horse_id PK,FK
        string sire_id
        string sire_name
        string dam_id
        string dam_name
        string damsire_id
        string damsire_name
    }
    jockeys {
        string jockey_id PK
        string name
        string affiliation "美浦/栗東/地方/海外"
    }
    trainers {
        string trainer_id PK
        string name
        string affiliation "美浦/栗東/地方/海外"
    }
    payouts {
        string race_id PK,FK
        string bet_type PK
        string combination PK
        int payout "100円あたり"
        int popularity
    }
    odds {
        string race_id PK
        string bet_type PK
        string combination PK
        string status PK "result/middle/yoso"
        float odds
        float odds_max "複勝・ワイドの上限"
        int popularity
    }
    predictions {
        string race_id PK
        int horse_no PK
        string model_version PK
        float p_win
        float p_top3
        string mark
    }
    bet_log {
        int id PK
        string race_id
        string bet_type
        string combination
        int amount
        float est_prob
        float odds
        float expected_value
    }
```

## テーブルの説明

| テーブル | 主キー | 内容 | 取得元 |
|---|---|---|---|
| races | race_id | レースの条件（日付・場・距離・芝ダ・回り・内外・天候・馬場・クラス・頭数・前後半3F・ラップ・コーナー通過順） | db.netkeiba.com/race/{id}/ |
| entries | race_id, horse_no | 出走馬ごとの結果（枠・馬番・性齢・斤量・騎手・調教師・馬体重・単勝オッズ・人気・着順・タイム・着差・通過順・上がり3F） | 同上 |
| horses | horse_id | 馬名・生年月日・生産者・産地・馬主 | db.netkeiba.com/horse/{id}/ |
| pedigrees | horse_id | 父・母・母父 | db.netkeiba.com/horse/ped/{id}/ |
| jockeys | jockey_id | 騎手名・所属 | db.netkeiba.com/jockey/{id}/ |
| trainers | trainer_id | 調教師名・所属（結果表の [東]/[西]/[地] から） | 結果ページ |
| payouts | race_id, bet_type, combination | 全券種の払戻金（同着で複数行あり） | 結果ページ |
| odds | race_id, bet_type, combination, status | 確定オッズ（status=result）と当日の最新オッズ | race.netkeiba.com のオッズAPI |
| predictions | race_id, horse_no, model_version | 土日の朝に自動で出した予想（推定勝率・複勝率・印） | アプリ |
| bet_log | id | 自動予想で提案した買い目（日曜の答え合わせ用） | アプリ |
| scrape_progress | kind, key | 取得済みの記録（途中から再開するため） | アプリ |
| model_runs | version | 学習したモデルの成績と、採用したかどうか | アプリ |

## 補足
- `finish_pos` は着順の数字。中止・除外・取消・失格は NULL で、理由は `finish_status` に入ります。降着は着順（降着後の順位）が入り、`finish_status='降着'` になります。
- 脚質（逃げ/先行/差し/追込）は保存せず、`passing`（通過順位）と頭数から集計時に計算します（`keiba/features.py`）。
- 確定オッズ（odds テーブル）は取得量が多いため、`config.yaml` の `scraper.final_odds_since` 以降のレースだけを取得します。
