# -*- coding: utf-8 -*-
"""Cross-Shelf 공통 유틸리티.

GitHub 공개본은 원본 API 캐시(data_raw)를 포함하지 않는다.
API 재수집이 필요한 단계는 환경변수로 인증키를 읽고, 이미 포함된 CSV만
확인하는 단계는 인증키 없이도 실행할 수 있다.
"""
from __future__ import annotations

import math
import os
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

DATA4LIBRARY_KEY = "DATA4LIBRARY_AUTH_KEY"
NL_SEOJI_KEY = "NL_SEOJI_CERT_KEY"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}

KDC_MAIN = {
    "0": "총류", "1": "철학", "2": "종교", "3": "사회과학", "4": "자연과학",
    "5": "기술과학", "6": "예술", "7": "언어", "8": "문학", "9": "역사",
}

SHELF_NAMES = {
    1: "인간과 삶을 응시하는 소설",
    2: "아이와 함께 자라는 부모의 서가",
    3: "부와 성공, 나를 바꾸는 서가",
    4: "위로가 필요한 날의 소설과 교양",
    5: "웃으며 읽는 어린이 교양",
}
SELECTED_CLUSTERS = {2, 4, 5}
NOTABLE_CLUSTERS = {1, 3}

READER = {
    "0": "교양", "1": "실용", "2": "여성", "3": "(예비)", "4": "청소년",
    "5": "중고학습참고", "6": "초등학습참고", "7": "아동", "8": "(예비)", "9": "전문",
}
FORM = {
    "0": "문고본", "1": "사전", "2": "신서판", "3": "단행본", "4": "전집·다권본",
    "5": "(예비)", "6": "도감", "7": "그림책·만화", "8": "혼합자료", "9": "(예비)",
}

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def env_key(name: str) -> str:
    """환경변수에서 인증키를 읽는다. .env 파일이 있으면 선택적으로 읽는다."""
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(ROOT / ".env")
    except Exception:
        pass
    return os.getenv(name, "").strip()


def require_key(name: str) -> str:
    key = env_key(name)
    if key:
        return key
    guide = {
        DATA4LIBRARY_KEY: "정보나루 Open API",
        NL_SEOJI_KEY: "국립중앙도서관 ISBN 서지정보 Open API",
    }.get(name, name)
    raise RuntimeError(f"{guide} 인증키가 필요합니다. 환경변수 {name} 를 설정하세요.")


def data4library_get(endpoint: str, params: dict, *, sleep: float = 0.25) -> dict:
    p = dict(params)
    p.setdefault("authKey", require_key(DATA4LIBRARY_KEY))
    p.setdefault("format", "json")
    r = requests.get(
        f"http://data4library.kr/api/{endpoint}", params=p, headers=HEADERS, timeout=30
    )
    r.raise_for_status()
    data = r.json()
    err = data.get("response", {}).get("error")
    if err:
        raise RuntimeError(f"정보나루 API 오류({endpoint}): {err}")
    if sleep:
        time.sleep(sleep)
    return data


def seoji_get(isbn: str, *, sleep: float = 0.20) -> dict:
    key = require_key(NL_SEOJI_KEY)
    r = requests.get(
        "https://seoji.nl.go.kr/landingPage/SearchApi.do",
        params={
            "cert_key": key, "result_style": "json", "page_no": 1,
            "page_size": 10, "isbn": str(isbn),
        },
        headers=HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    if sleep:
        time.sleep(sleep)
    return data


def to_bool(series):
    return series.astype(str).str.lower().isin(["true", "1", "y", "yes"])


def kdc1(class_no) -> str | None:
    if class_no is None or (isinstance(class_no, float) and math.isnan(class_no)):
        return None
    m = re.match(r"\s*(\d)", str(class_no))
    return m.group(1) if m else None


def norm_title(title: str) -> str:
    base = re.split(r"[:;=/]", str(title or ""))[0]
    return re.sub(r"\s+", "", base).lower()


GENERIC_TAGS = {
    "개정판", "완전판", "특별판", "최신판", "개정증보판", "전면개정판",
    "큰글자책", "양장본", "리커버",
}


def norm_series(title: str) -> str:
    base = re.split(r"[:：=/]", str(title))[0]
    return re.sub(r"\s+", "", base).lower()


def paren_tag(title: str):
    m = re.match(r"^\s*\(([^)]{2,})\)", str(title))
    if not m:
        return None
    tag = re.sub(r"\s+", "", m.group(1)).lower()
    return None if tag in GENERIC_TAGS else tag


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def _common_suffix(a: str, b: str) -> int:
    return _common_prefix(a[::-1], b[::-1])


def same_series(a: str, b: str) -> bool:
    ta, tb = paren_tag(a), paren_tag(b)
    if ta and tb and _common_prefix(ta, tb) >= 4:
        return True
    na, nb = norm_series(a), norm_series(b)
    if len(na) < 3 or len(nb) < 3:
        return False
    if na in nb or nb in na:
        return True
    return _common_prefix(na, nb) >= 3 and _common_suffix(na, nb) >= 2


def addition_code_labels(code: str | None) -> tuple[str, str, str]:
    code = "" if code is None else str(code).strip()
    if code.endswith(".0"):
        code = code[:-2]
    if len(code) < 2:
        return code, "", ""
    return code, READER.get(code[0], "?"), FORM.get(code[1], "?")


# 시각화 공통
A4_TEXT_W = 150 / 25.4


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    candidates = [
        "Malgun Gothic", "AppleGothic", "Apple SD Gothic Neo", "NanumGothic",
        "NanumBarunGothic", "Noto Sans CJK KR", "Noto Sans KR", "Gulim",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["savefig.dpi"] = 300
    return plt
