"""パーサ共通の小さな関数。"""
from __future__ import annotations

import re
import unicodedata

from bs4 import BeautifulSoup


def soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def norm(s: str | None) -> str:
    """全角英数を半角に、空白をまとめる。"""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s).strip()


def to_int(s) -> int | None:
    if s is None:
        return None
    m = re.search(r"-?\d+", str(s).replace(",", ""))
    return int(m.group()) if m else None


def to_float(s) -> float | None:
    if s is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(s).replace(",", ""))
    return float(m.group()) if m else None


def id_from_href(href: str | None, kind: str) -> str | None:
    """'https://db.netkeiba.com/horse/2020110024/' → '2020110024'。
    騎手・調教師は '/jockey/result/recent/05386/' の形もある。"""
    if not href:
        return None
    m = re.search(rf"/{kind}/(?:result/recent/|profile/)?([0-9A-Za-z]+)", href)
    return m.group(1) if m else None


def parse_time_sec(s: str | None) -> float | None:
    """'1:23.3' → 83.3、'59.8' → 59.8"""
    s = norm(s)
    if not s:
        return None
    m = re.fullmatch(r"(?:(\d+):)?(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    return (int(m.group(1)) * 60 if m.group(1) else 0) + float(m.group(2))


def parse_horse_weight(s: str | None) -> tuple[int | None, int | None]:
    """'546(+6)' → (546, 6)、'計不' → (None, None)"""
    s = norm(s)
    m = re.match(r"(\d+)\s*\(([+-]?\d+)\)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"(\d+)", s)
    if m:
        return int(m.group(1)), None
    return None, None


def parse_sex_age(s: str | None) -> tuple[str | None, int | None]:
    s = norm(s)
    m = re.match(r"(牡|牝|セ|騸)(\d+)", s)
    if not m:
        return None, None
    sex = "セ" if m.group(1) in ("セ", "騸") else m.group(1)
    return sex, int(m.group(2))


def normalize_grade(text: str) -> str | None:
    """レース名の '(GI)' '(G3)' '(L)' '(OP)' などからグレードを取り出す。"""
    t = norm(text).upper()
    m = re.search(r"\((J\.)?G(I{1,3}|[123])\)", t)
    if m:
        g = m.group(2)
        g = {"I": "1", "II": "2", "III": "3"}.get(g, g)
        return f"G{g}"
    if "(L)" in t:
        return "L"
    if "(OP)" in t:
        return "OP"
    return None


def normalize_class(text: str) -> str | None:
    """'3歳以上2勝クラス' や '3歳以上1000万下' などからクラスを取り出す。"""
    t = norm(text)
    if "新馬" in t:
        return "新馬"
    if "未勝利" in t:
        return "未勝利"
    for old, new in (("500万", "1勝クラス"), ("1000万", "2勝クラス"), ("1600万", "3勝クラス")):
        if old in t:
            return new
    m = re.search(r"([123])勝クラス", t)
    if m:
        return f"{m.group(1)}勝クラス"
    if "オープン" in t or "OP" in t:
        return "オープン"
    return None


def class_code(class_name: str | None, grade: str | None) -> str | None:
    """集計・序列用のクラス表記（新馬/未勝利/1勝クラス/.../オープン/L/G3/G2/G1）。"""
    if grade in ("G1", "G2", "G3", "L"):
        return grade
    if grade == "OP":
        return "オープン"
    return class_name


def parse_course(text: str) -> dict:
    """'芝右 外1600m' 'ダ左1400m' '障芝 外3000m' '芝直線1000m' などを分解する。"""
    t = norm(text)
    out = {"surface": None, "direction": None, "inout": None, "distance": None}
    m = re.search(r"(芝|ダ|障)", t)
    if m:
        out["surface"] = {"芝": "芝", "ダ": "ダート", "障": "障害"}[m.group(1)]
    if "直線" in t:
        out["direction"] = "直線"
    else:
        m = re.search(r"(右|左)", t)
        if m:
            out["direction"] = m.group(1)
    m = re.search(r"(外-内|内-外|外内|内外|内2周|外|内)\s*\d", t)
    if m:
        out["inout"] = m.group(1)
    m = re.search(r"(\d{3,4})\s*m", t)
    if m:
        out["distance"] = int(m.group(1))
    return out
