from pathlib import Path

import pytest

from keiba.parsers.common import (normalize_class, normalize_grade, parse_course, parse_horse_weight,
                                  parse_time_sec)
from keiba.parsers.horse import parse_horse_profile, parse_pedigree
from keiba.parsers.race_result import ParseError, parse_race_result
from keiba.parsers.racecard import parse_calendar, parse_odds_json, parse_race_ids, parse_shutuba

FX = Path(__file__).parent / "fixtures"


def read(name):
    return (FX / name).read_text(encoding="utf-8")


def test_race_info():
    r = parse_race_result(read("race_result.html"), "202508040211")
    race = r["race"]
    assert race["date"] == "2025-10-05"
    assert race["venue"] == "京都" and race["venue_code"] == "08"
    assert (race["kai"], race["nichi"], race["race_no"]) == (4, 2, 11)
    assert race["surface"] == "芝" and race["direction"] == "右" and race["inout"] == "外"
    assert race["distance"] == 1800
    assert race["weather"] == "曇" and race["track_condition"] == "稍重"
    assert race["grade"] == "G3" and race["class_code"] == "G3" and race["class_name"] == "オープン"
    assert race["pace_front3f"] == 35.1 and race["pace_back3f"] == 34.4
    assert race["field_size"] == 4  # 取消を除く


def test_entries():
    r = parse_race_result(read("race_result.html"), "202508040211")
    e = {x["horse_no"]: x for x in r["entries"]}
    assert e[3]["finish_pos"] == 1 and e[3]["odds_win"] == 3.2 and e[3]["popularity"] == 1
    assert e[3]["time_sec"] == pytest.approx(106.5)
    assert e[3]["horse_weight"] == 480 and e[3]["weight_diff"] == 4
    assert e[3]["passing"] == "2-2-2" and e[3]["last3f"] == 34.1
    assert e[1]["sex"] == "牝" and e[1]["age"] == 5
    assert e[2]["finish_pos"] == 3 and e[2]["finish_status"] == "降着"
    assert e[4]["finish_pos"] is None and e[4]["finish_status"] == "中止"
    assert e[4]["horse_weight"] is None
    assert e[5]["finish_status"] == "取消" and e[5]["odds_win"] is None
    tr = {t["trainer_id"]: t for t in r["trainers"]}
    assert tr["02001"]["affiliation"] == "栗東"
    assert tr["02002"]["affiliation"] == "美浦"
    assert tr["02003"]["affiliation"] == "地方"


def test_payouts():
    r = parse_race_result(read("race_result.html"), "202508040211")
    p = {(x["bet_type"], x["combination"]): x for x in r["payouts"]}
    assert p[("単勝", "3")]["payout"] == 320
    assert p[("複勝", "1")]["payout"] == 310 and p[("複勝", "1")]["popularity"] == 5
    assert p[("馬連", "1-3")]["payout"] == 2150
    assert p[("ワイド", "2-3")]["payout"] == 250
    assert p[("三連複", "1-2-3")]["payout"] == 3010
    assert len([k for k in p if k[0] == "ワイド"]) == 3


def test_no_result_table_raises():
    with pytest.raises(ParseError):
        parse_race_result("<html><body>まだ</body></html>", "202508040211")


def test_pedigree_and_profile():
    p = parse_pedigree(read("pedigree.html"), "x")
    assert p["sire_id"] == "000a000001" and p["sire_name"] == "父ウマ"
    assert p["dam_name"] == "母ウマ"  # (米) を除く
    assert p["damsire_id"] == "000a000005" and p["damsire_name"] == "母父ウマ"
    h = parse_horse_profile(read("horse.html"), "x")
    assert h["birth_date"] == "2021-03-04" and h["breeder"] == "テスト牧場" and h["origin"] == "日高町"


def test_shutuba():
    r = parse_shutuba(read("shutuba.html"), "202608050511", "2026-11-21")
    race = r["race"]
    assert race["surface"] == "ダート" and race["distance"] == 1900 and race["direction"] == "右"
    assert race["weather"] == "晴" and race["track_condition"] == "稍重"
    assert race["grade"] == "L" and race["class_code"] == "L"
    assert len(r["entries"]) == 2  # 関係ない表の行は入らない
    a, b = r["entries"]
    assert a["horse_id"] == "2022100001" and a["weight_carried"] == 57.0 and a["horse_weight"] == 482
    assert a["trainer_affiliation"] == "栗東" and a["jockey_id"] == "01102"
    assert b["cancelled"] and race["field_size"] == 1


def test_odds_json():
    j = '{"status":"result","data":{"official_datetime":"x","odds":{"5":{"0103":["7.6","9.1","6"]}}}}'
    r = parse_odds_json(j, 5)
    assert r["status"] == "result"
    assert r["rows"] == [{"bet_type": "ワイド", "combination": "1-3", "odds": 7.6, "odds_max": 9.1,
                          "popularity": 6}]
    assert parse_odds_json('{"status":"NG","data":""}', 1)["rows"] == []
    j = '{"status":"result","data":{"odds":{"7":{"010203":["1,012.4","0.0","199"]}}}}'
    r = parse_odds_json(j, 7)["rows"][0]
    assert r["combination"] == "1-2-3" and r["odds"] == 1012.4 and r["odds_max"] is None


def test_small_helpers():
    assert parse_calendar('<a href="?kaisai_date=20251101">') == ["2025-11-01"]
    ids = parse_race_ids('<a href="/race/202505050511/"></a><a href="/race/202546112301/"></a>')
    assert ids == ["202505050511"]  # 地方は除外
    assert parse_time_sec("2:01.5") == pytest.approx(121.5)
    assert parse_horse_weight("計不") == (None, None)
    assert normalize_grade("ジャパンC(GI)") == "G1"
    assert normalize_grade("中山大障害(J.GI)") == "G1"
    assert normalize_class("3歳以上1000万下") == "2勝クラス"
    c = parse_course("障芝 外3000m")
    assert c["surface"] == "障害" and c["distance"] == 3000
    assert parse_course("芝直線1000m")["direction"] == "直線"


def test_odds_placeholders_are_dropped():
    import json
    from keiba.parsers.racecard import parse_odds_json
    j = {"status": "middle", "data": {"official_datetime": "2026-11-20 18:37:55", "odds": {
        "1": {"01": ["999.9", "", "8"], "02": ["3.2", "", "1"]},
        "2": {"01": ["0", "999.9", "1"], "02": ["1.1", "1.6", "2"]}}}}
    win = {r["combination"]: r["odds"] for r in parse_odds_json(json.dumps(j), 1)["rows"]}
    assert win == {"1": None, "2": 3.2}
    place = {r["combination"]: (r["odds"], r["odds_max"]) for r in parse_odds_json(json.dumps(j), 2)["rows"]}
    assert place == {"1": (None, None), "2": (1.1, 1.6)}
    # 確定オッズの 999.9 は本物として残す
    j["status"] = "result"
    win = {r["combination"]: r["odds"] for r in parse_odds_json(json.dumps(j), 1)["rows"]}
    assert win["1"] == 999.9
