"""当日の出馬表・オッズ・開催カレンダー・レース一覧のパーサ（race.netkeiba.com）。"""
from __future__ import annotations

import json
import re

from ..constants import ODDS_TYPES, VENUES, is_jra_race_id
from .common import (class_code, id_from_href, norm, normalize_class, normalize_grade,
                     parse_course, parse_horse_weight, parse_sex_age, soup, to_float, to_int)


def parse_shutuba(html: str, race_id: str, date: str | None = None) -> dict:
    """race.netkeiba.com/race/shutuba.html?race_id=... → {race, entries}"""
    s = soup(html)
    name_el = s.select_one(".RaceName")
    name = norm(name_el.get_text(" ", strip=True)) if name_el else ""
    d1 = norm(s.select_one(".RaceData01").get_text(" ", strip=True)) if s.select_one(".RaceData01") else ""
    d2 = norm(s.select_one(".RaceData02").get_text(" ", strip=True)) if s.select_one(".RaceData02") else ""
    # レース名横のグレードアイコン（Icon_GradeType1=G1 ...）
    grade = normalize_grade(name)
    icon = s.select_one(".RaceName .Icon_GradeType") or s.select_one(".RaceList_Item02 .Icon_GradeType")
    if grade is None and icon is not None:
        cls = " ".join(icon.get("class", []))
        m = re.search(r"Icon_GradeType(\d+)", cls)
        if m:
            grade = {"1": "G1", "2": "G2", "3": "G3", "15": "L", "5": "OP"}.get(m.group(1))

    m = re.search(r"((?:芝|ダ|障)[^/]*?\d{3,4}m[^/]*)", d1)
    course = parse_course(m.group(1)) if m else parse_course(d1)
    weather = track = None
    m = re.search(r"天候\s*:\s*(\S+)", d1)
    if m:
        weather = m.group(1)
    m = re.search(r"馬場\s*:\s*(良|稍重|重|不良|稍)", d1)
    if m:
        track = {"稍": "稍重"}.get(m.group(1), m.group(1))
    m = re.search(r"(\d{1,2}:\d{2})発走", d1)
    start_time = m.group(1) if m else None
    cls = normalize_class(d2)
    m_age = re.search(r"(\d歳(?:以上)?)", d2)

    race = {
        "race_id": race_id,
        "date": date,
        "venue_code": race_id[4:6],
        "venue": VENUES.get(race_id[4:6]),
        "kai": int(race_id[6:8]),
        "nichi": int(race_id[8:10]),
        "race_no": int(race_id[10:12]),
        "race_name": name,
        "grade": grade,
        "class_name": cls,
        "class_code": class_code(cls, grade),
        "age_cond": m_age.group(1) if m_age else None,
        "conditions": d2 or None,
        "surface": course["surface"],
        "direction": course["direction"],
        "inout": course["inout"],
        "distance": course["distance"],
        "weather": weather,
        "track_condition": track,
        "start_time": start_time,
    }

    entries = []
    seen = set()
    table = s.select_one("table.Shutuba_Table")
    for tr in (table.select("tr.HorseList") if table else []):
        def cell(prefix):
            for td in tr.find_all("td"):
                if any(c.startswith(prefix) for c in td.get("class", [])):
                    return td
            return None

        horse_td = cell("HorseInfo")
        a = horse_td.find("a") if horse_td else None
        horse_id = id_from_href(a.get("href"), "horse") if a else None
        jt, tt = cell("Jockey"), cell("Trainer")
        ja = jt.find("a") if jt else None
        ta = tt.find("a") if tt else None
        sex, age = parse_sex_age(cell("Barei").get_text() if cell("Barei") else "")
        # 斤量：Barei の次の Txt_C セル
        wc = None
        tds = tr.find_all("td")
        for i, td in enumerate(tds):
            if "Barei" in td.get("class", []) and i + 1 < len(tds):
                wc = to_float(tds[i + 1].get_text())
        hw, hwd = parse_horse_weight(cell("Weight").get_text() if cell("Weight") else "")
        trainer_text = norm(tt.get_text(" ", strip=True)) if tt else ""
        aff = None
        for k in ("美浦", "栗東", "地方", "海外"):
            if trainer_text.startswith(k):
                aff = k
        cancelled = "Cancel" in " ".join(tr.get("class", []))
        horse_no = to_int(cell("Umaban").get_text()) if cell("Umaban") else None
        if horse_no is None or horse_id is None or horse_no in seen:
            continue
        seen.add(horse_no)
        entries.append({
            "race_id": race_id,
            "frame_no": to_int(cell("Waku").get_text()) if cell("Waku") else None,
            "horse_no": horse_no,
            "horse_id": horse_id,
            "horse_name": norm(a.get_text()) if a else None,
            "sex": sex,
            "age": age,
            "weight_carried": wc,
            "jockey_id": id_from_href(ja.get("href"), "jockey") if ja else None,
            "jockey_name": norm(ja.get_text()) if ja else None,
            "trainer_id": id_from_href(ta.get("href"), "trainer") if ta else None,
            "trainer_name": norm(ta.get_text()) if ta else None,
            "trainer_affiliation": aff,
            "horse_weight": hw,
            "weight_diff": hwd,
            "cancelled": cancelled,
        })
    race["field_size"] = sum(1 for e in entries if not e["cancelled"])
    return {"race": race, "entries": entries}


# 票が入っていない組み合わせに出る仮の値（発売直後に多い）
PLACEHOLDER_ODDS = {999.9, 9999.9, 99999.9}


def _odds_value(v: str, status: str | None) -> float | None:
    if v in ("", "---.-"):
        return None
    o = to_float(v)
    if o is None or o <= 0:
        return None
    if status != "result" and o in PLACEHOLDER_ODDS:
        return None
    return o


def parse_odds_json(text: str, odds_type: int) -> dict:
    """オッズAPIの JSON → {status, official_datetime, rows:[{combination, odds, odds_max, popularity}]}

    status: 'yoso'（前売り前の予想オッズ） / 'middle'（発売中） / 'result'（確定） / 'NG'
    combination は '04' や '0415' のような2桁区切りを '4' / '4-15' に直す。
    発売中の 999.9 / 9999.9 / 99999.9（票が入っていない仮の値）と 0 はオッズなし（None）にする。
    """
    try:
        j = json.loads(text)
    except json.JSONDecodeError:
        return {"status": "NG", "official_datetime": None, "rows": []}
    status = j.get("status")
    data = j.get("data") or {}
    if not isinstance(data, dict):
        return {"status": status, "official_datetime": None, "rows": []}
    odds = (data.get("odds") or {}).get(str(odds_type)) or {}
    rows = []
    for combo, vals in odds.items():
        nums = [str(int(combo[i:i + 2])) for i in range(0, len(combo), 2)]
        o = _odds_value(vals[0], status) if vals else None
        o2 = _odds_value(vals[1], status) if len(vals) > 1 else None
        rows.append({
            "bet_type": ODDS_TYPES[odds_type],
            "combination": "-".join(nums),
            "odds": o,
            "odds_max": o2,
            "popularity": to_int(vals[2]) if len(vals) > 2 else None,
        })
    return {"status": status, "official_datetime": data.get("official_datetime"), "rows": rows}


def parse_calendar(html: str) -> list[str]:
    """開催カレンダー → ['2025-11-01', ...]"""
    dates = sorted(set(re.findall(r"kaisai_date=(\d{8})", html)))
    return [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in dates]


def parse_race_ids(html: str) -> list[str]:
    """レース一覧（db / race どちらも）から JRA の race_id を取り出す。"""
    ids = set(re.findall(r"race/(\d{12})", html)) | set(re.findall(r"race_id=(\d{12})", html))
    return sorted(i for i in ids if is_jra_race_id(i))
