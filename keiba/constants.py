"""競馬場コードなどの定数。"""

# netkeiba の race_id の5〜6桁目（JRA）
VENUES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}
VENUE_CODE_BY_NAME = {v: k for k, v in VENUES.items()}

SURFACES = {"芝": "芝", "ダ": "ダート", "障": "障害"}

TRACK_CONDITIONS = ["良", "稍重", "重", "不良"]

# クラスの序列（昇級・降級の判定に使う）
CLASS_RANK = {
    "新馬": 1, "未勝利": 1,
    "1勝クラス": 2, "2勝クラス": 3, "3勝クラス": 4,
    "オープン": 5, "L": 5, "G3": 6, "G2": 7, "G1": 8,
}

# 払戻の券種（netkeiba の th の class → 日本語）
BET_TYPES = {
    "tan": "単勝", "fuku": "複勝", "waku": "枠連", "uren": "馬連",
    "wide": "ワイド", "utan": "馬単", "sanfuku": "三連複", "santan": "三連単",
}

# オッズAPIの type 番号 → 券種
ODDS_TYPES = {1: "単勝", 2: "複勝", 3: "枠連", 4: "馬連", 5: "ワイド", 6: "馬単", 7: "三連複", 8: "三連単"}

# 調教師の所属表記
AFFILIATION = {"東": "美浦", "西": "栗東", "地": "地方", "外": "海外"}


def is_jra_race_id(race_id: str) -> bool:
    return len(race_id) == 12 and race_id[4:6] in VENUES
