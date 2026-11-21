"""db.netkeiba.com/race/<race_id>/ （レース結果ページ）のパーサ。"""
from __future__ import annotations

import json
import re

from ..constants import AFFILIATION, BET_TYPES, VENUE_CODE_BY_NAME, VENUES
from .common import (class_code, id_from_href, norm, normalize_class, normalize_grade,
                     parse_course, parse_horse_weight, parse_sex_age, parse_time_sec, soup,
                     to_float, to_int)

# 着順欄の表記（netkeiba は「中」「除」「取」「失」と1文字のことが多い）
FINISH_STATUS = {"中": "中止", "除": "除外", "取": "取消", "失": "失格"}


class ParseError(ValueError):
    pass


def _header_key(th) -> str:
    return norm(th.get_text("", strip=True)).replace(" ", "")


def parse_race_result(html: str, race_id: str) -> dict:
    """結果ページを解析して {race, entries, payouts, horses, jockeys, trainers} を返す。

    結果がまだ無いページ（未確定）の場合は ParseError を投げる。
    """
    s = soup(html)
    table = s.select_one("table.race_table_01")
    if table is None:
        raise ParseError(f"結果の表がありません: {race_id}")

    race = _parse_race_info(s, race_id)

    # ---- 着順表（列は見出しの文字で対応づける）----
    rows = table.find_all("tr")
    headers = [_header_key(th) for th in rows[0].find_all("th")]
    entries, horses, jockeys, trainers = [], {}, {}, {}
    for tr in rows[1:]:
        tds = tr.find_all("td")
        if len(tds) != len(headers):
            continue
        cell = dict(zip(headers, tds))

        def text(key):
            td = cell.get(key)
            return norm(td.get_text(" ", strip=True)) if td is not None else ""

        def link(key, kind):
            td = cell.get(key)
            a = td.find("a") if td is not None else None
            return (id_from_href(a.get("href"), kind), norm(a.get_text())) if a else (None, None)

        pos_text = text("着順")
        finish_pos = to_int(pos_text) if re.match(r"^\d", pos_text) else None
        status = None
        for k, w in FINISH_STATUS.items():
            if pos_text.startswith(k):
                status = w
        if "降" in pos_text:
            status = "降着"

        horse_id, horse_name = link("馬名", "horse")
        jockey_id, jockey_name = link("騎手", "jockey")
        trainer_id, trainer_name = link("調教師", "trainer")
        tr_text = text("調教師")
        m = re.match(r"\[(.)\]", tr_text)
        trainer_aff = AFFILIATION.get(m.group(1)) if m else None
        sex, age = parse_sex_age(text("性齢"))
        hw, hw_diff = parse_horse_weight(text("馬体重"))
        passing = text("通過") or None

        entries.append({
            "race_id": race_id,
            "horse_no": to_int(text("馬番")),
            "frame_no": to_int(text("枠番")),
            "horse_id": horse_id,
            "sex": sex,
            "age": age,
            "weight_carried": to_float(text("斤量")),
            "jockey_id": jockey_id,
            "trainer_id": trainer_id,
            "horse_weight": hw,
            "weight_diff": hw_diff,
            "odds_win": to_float(text("単勝")),
            "popularity": to_int(text("人気")),
            "finish_pos": finish_pos,
            "finish_status": status,
            "time_sec": parse_time_sec(text("タイム")),
            "margin": text("着差") or None,
            "passing": passing,
            "last3f": to_float(text("上り")),
            "prize": to_float(text("賞金(万円)")),
        })
        if horse_id:
            horses[horse_id] = {"horse_id": horse_id, "name": horse_name}
        if jockey_id:
            jockeys[jockey_id] = {"jockey_id": jockey_id, "name": jockey_name}
        if trainer_id:
            trainers[trainer_id] = {"trainer_id": trainer_id, "name": trainer_name,
                                    "affiliation": trainer_aff}

    race["field_size"] = sum(1 for e in entries if e["finish_status"] not in ("取消", "除外"))
    _parse_lap_and_corners(s, race)

    return {
        "race": race,
        "entries": entries,
        "payouts": parse_payouts(s, race_id),
        "horses": list(horses.values()),
        "jockeys": list(jockeys.values()),
        "trainers": list(trainers.values()),
    }


def _parse_race_info(s, race_id: str) -> dict:
    intro = s.select_one(".data_intro")
    if intro is None:
        raise ParseError(f"レース情報がありません: {race_id}")
    name = norm(intro.select_one("h1").get_text(" ", strip=True)) if intro.select_one("h1") else ""
    span = intro.select_one("dd p span")
    cond = norm(span.get_text(" ", strip=True)) if span else ""
    small = intro.select_one("p.smalltxt")
    smalltxt = norm(small.get_text(" ", strip=True)) if small else ""

    course = parse_course(cond.split("/")[0])
    weather = None
    m = re.search(r"天候\s*:\s*(\S+)", cond)
    if m:
        weather = m.group(1)
    # 馬場状態：「芝 : 良」「ダート : 稍重」（障害は芝の状態を採用）
    tc = {}
    for k, v in re.findall(r"(芝|ダート)\s*:\s*(良|稍重|重|不良)", cond):
        tc[k] = v
    if course["surface"] == "ダート":
        track = tc.get("ダート")
    else:
        track = tc.get("芝") or tc.get("ダート")
    m = re.search(r"発走\s*:\s*(\d{1,2}:\d{2})", cond)
    start_time = m.group(1) if m else None

    date = None
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", smalltxt)
    if m:
        date = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    venue_code = race_id[4:6]
    m = re.search(r"(\d+)回(\S+?)(\d+)日目", smalltxt)
    kai, nichi = (int(m.group(1)), int(m.group(3))) if m else (int(race_id[6:8]), int(race_id[8:10]))
    venue = VENUES.get(venue_code)
    if m and m.group(2) in VENUE_CODE_BY_NAME:
        venue = m.group(2)
    rest = smalltxt
    if m:
        rest = smalltxt[m.end():]
    grade = normalize_grade(name)
    cls = normalize_class(rest)
    m_age = re.search(r"(\d歳(?:以上)?)", rest)

    return {
        "race_id": race_id,
        "date": date,
        "venue_code": venue_code,
        "venue": venue,
        "kai": kai,
        "nichi": nichi,
        "race_no": int(race_id[10:12]),
        "race_name": name,
        "grade": grade,
        "class_name": cls,
        "class_code": class_code(cls, grade),
        "age_cond": m_age.group(1) if m_age else None,
        "conditions": rest.strip() or None,
        "surface": course["surface"],
        "direction": course["direction"],
        "inout": course["inout"],
        "distance": course["distance"],
        "weather": weather,
        "track_condition": track,
        "start_time": start_time,
    }


def _parse_lap_and_corners(s, race: dict) -> None:
    corners = {}
    lap = pace = None
    for tr in s.select("table.result_table_02 tr"):
        th, td = tr.find("th"), tr.find("td")
        if th is None or td is None:
            continue
        k = norm(th.get_text(strip=True))
        v = norm(td.get_text(strip=True))
        if "コーナー" in k:
            corners[k] = v
        elif k == "ラップ":
            lap = v
        elif k == "ペース":
            pace = v
    race["corner_orders"] = json.dumps(corners, ensure_ascii=False) if corners else None
    race["lap_times"] = lap
    race["pace_front3f"] = race["pace_back3f"] = None
    if pace:
        m = re.search(r"\((\d+\.\d)-(\d+\.\d)\)", pace)
        if m:
            race["pace_front3f"] = float(m.group(1))
            race["pace_back3f"] = float(m.group(2))


def parse_payouts(s, race_id: str) -> list[dict]:
    out = []
    for tr in s.select("table.pay_table_01 tr"):
        th = tr.find("th")
        tds = tr.find_all("td")
        if th is None or len(tds) < 2:
            continue
        cls = (th.get("class") or [""])[0]
        bet = BET_TYPES.get(cls) or norm(th.get_text())

        def split(td):
            return [norm(x) for x in td.get_text("\n").split("\n") if norm(x)]

        combos = split(tds[0])
        pays = split(tds[1])
        pops = split(tds[2]) if len(tds) > 2 else []
        for i, combo in enumerate(combos):
            if i >= len(pays):
                break
            out.append({
                "race_id": race_id,
                "bet_type": bet,
                "combination": combo.replace(" ", ""),
                "payout": to_int(pays[i]),
                "popularity": to_int(pops[i]) if i < len(pops) else None,
            })
    return out
