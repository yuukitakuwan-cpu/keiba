"""馬のプロフィール・血統ページ、騎手プロフィールのパーサ。"""
from __future__ import annotations

import re

from .common import id_from_href, norm, soup


def parse_horse_profile(html: str, horse_id: str) -> dict:
    """db.netkeiba.com/horse/<id>/ → 生年月日・生産者・産地・馬主など。"""
    s = soup(html)
    out = {"horse_id": horse_id, "name": None, "birth_date": None, "breeder": None,
           "origin": None, "owner": None}
    title = s.select_one(".horse_title h1")
    if title:
        out["name"] = norm(title.get_text(" ", strip=True))
    prof = s.select_one("table.db_prof_table")
    if prof is None:
        return out
    for tr in prof.find_all("tr"):
        th, td = tr.find("th"), tr.find("td")
        if not th or not td:
            continue
        k = norm(th.get_text(strip=True))
        v = norm(td.get_text(" ", strip=True))
        if k == "生年月日":
            m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", v)
            if m:
                out["birth_date"] = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        elif k == "生産者":
            out["breeder"] = v or None
        elif k == "産地":
            out["origin"] = v or None
        elif k == "馬主":
            out["owner"] = v or None
    return out


def _first_link(td):
    a = td.find("a") if td is not None else None
    if a is None:
        return None, None
    name = re.sub(r"\s*\([^)]*\)\s*$", "", norm(a.get_text()))  # 末尾の (米) などを除く
    return id_from_href(a.get("href"), "horse"), name


def parse_pedigree(html: str, horse_id: str) -> dict:
    """db.netkeiba.com/horse/ped/<id>/ の5代血統表から父・母・母父を取り出す。

    表は32行。父は1行目の rowspan=16 のセル、母は17行目の rowspan=16 のセル、
    母父は17行目の rowspan=8 のセル。
    """
    s = soup(html)
    out = {"horse_id": horse_id, "sire_id": None, "sire_name": None, "dam_id": None,
           "dam_name": None, "damsire_id": None, "damsire_name": None}
    t = s.select_one("table.blood_table")
    if t is None:
        return out
    rows = t.find_all("tr")
    if not rows:
        return out
    half = len(rows) // 2

    def by_span(tr, span):
        for td in tr.find_all("td"):
            if td.get("rowspan") == str(span):
                return td
        return None

    big = str(half)
    out["sire_id"], out["sire_name"] = _first_link(by_span(rows[0], big))
    if len(rows) > half:
        out["dam_id"], out["dam_name"] = _first_link(by_span(rows[half], big))
        out["damsire_id"], out["damsire_name"] = _first_link(by_span(rows[half], half // 2))
    return out


def parse_jockey_profile(html: str, jockey_id: str) -> dict:
    """騎手プロフィール → 名前と所属（美浦/栗東/地方/海外）。"""
    s = soup(html)
    name = None
    if s.title:
        m = re.match(r"(.+?)のプロフィール", norm(s.title.get_text()))
        if m:
            name = m.group(1)
    text = norm(s.get_text(" ", strip=True))
    aff = None
    m = re.search(r"\[(美浦|栗東|地方|海外|外国)\]", text)
    if m:
        aff = "海外" if m.group(1) == "外国" else m.group(1)
    elif "フリー" in text and "引退" not in text:
        aff = None
    return {"jockey_id": jockey_id, "name": name, "affiliation": aff}
