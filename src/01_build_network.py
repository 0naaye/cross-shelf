# -*- coding: utf-8 -*-
"""1단계: 정보나루 함께대출/추천 데이터를 이용해 책-책 방향 네트워크를 구축한다.

출력: data/network_edges.csv

GitHub 공개본에는 API 원본 JSON 캐시를 포함하지 않는다. 이미 계산된
network_edges.csv가 저장소에 포함되어 있으므로 결과 확인에는 인증키가 필요 없다.
재수집하려면 DATA4LIBRARY_AUTH_KEY 환경변수를 설정한 뒤 실행한다.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import pandas as pd

from common import DATA, KDC_MAIN, data4library_get, kdc1, norm_title

PERIOD = {"startDt": "2024-01-01", "endDt": "2024-12-31"}
N_PER_CLASS = 10
CLASS_POOL = 60
DICT_PAGES = 3
MAX_INDIV_LOOKUP = 250
EDGE_TYPES = {
    "coLoanBooks": "함께대출",
    "maniaRecBooks": "마니아",
    "readerRecBooks": "다독자",
}


def build_kdc_dict(pages: int) -> dict:
    isbn2info = {}
    for page in range(1, pages + 1):
        data = data4library_get(
            "loanItemSrch",
            {"pageNo": page, "pageSize": 1000, **PERIOD},
        )
        for item in data.get("response", {}).get("docs", []) or []:
            doc = item.get("doc", {})
            isbn = str(doc.get("isbn13", "")).strip()
            if isbn:
                isbn2info[isbn] = {
                    "bookname": str(doc.get("bookname", "")).strip(),
                    "class_no": doc.get("class_no", ""),
                    "class_nm": doc.get("class_nm", ""),
                }
    return isbn2info


def pick_stratified_sources(isbn2info: dict, n_per_class: int) -> list[dict]:
    sources = []
    for c in range(10):
        data = data4library_get(
            "loanItemSrch",
            {"pageNo": 1, "pageSize": CLASS_POOL, "kdc": c, **PERIOD},
        )
        seen, picked = set(), 0
        for item in data.get("response", {}).get("docs", []) or []:
            doc = item.get("doc", {})
            if kdc1(doc.get("class_no")) != str(c):
                continue
            title = str(doc.get("bookname", "")).strip()
            key = norm_title(title)
            if not key or key in seen:
                continue
            seen.add(key)
            isbn = str(doc.get("isbn13", "")).strip()
            if not isbn:
                continue
            src = {
                "isbn13": isbn,
                "bookname": title,
                "class_no": doc.get("class_no", ""),
                "class_nm": doc.get("class_nm", ""),
                "kdc": str(c),
            }
            sources.append(src)
            isbn2info[isbn] = {
                "bookname": title,
                "class_no": doc.get("class_no", ""),
                "class_nm": doc.get("class_nm", ""),
            }
            picked += 1
            if picked >= n_per_class:
                break
        print(f"KDC {c} {KDC_MAIN[str(c)]}: {picked}권")
    return sources


def collect_edges(sources: list[dict], isbn2info: dict, max_lookup: int) -> pd.DataFrame:
    seed_isbns = {s["isbn13"] for s in sources}
    lookup_cache = {}
    lookup_count = 0
    rows = []

    def lookup_info(isbn: str, title: str) -> dict:
        nonlocal lookup_count
        info = isbn2info.get(isbn) or lookup_cache.get(isbn)
        if info is None and lookup_count < max_lookup:
            lookup_count += 1
            d = data4library_get("usageAnalysisList", {"isbn13": isbn})
            book = d.get("response", {}).get("book", {}) or {}
            info = {
                "bookname": str(book.get("bookname", title)).strip(),
                "class_no": book.get("class_no", ""),
                "class_nm": book.get("class_nm", ""),
            }
            lookup_cache[isbn] = info
        return info or {"bookname": title, "class_no": "", "class_nm": ""}

    for i, src in enumerate(sources, 1):
        src_kdc = kdc1(src["class_no"])
        data = data4library_get("usageAnalysisList", {"isbn13": src["isbn13"]})
        response = data.get("response", {})
        counts = defaultdict(int)
        for api_name, label in EDGE_TYPES.items():
            for rank, item in enumerate(response.get(api_name, []) or [], 1):
                book = item.get("book", {}) or {}
                dst_isbn = str(book.get("isbn13", "")).strip()
                dst_title = str(book.get("bookname", "")).strip()
                if not dst_isbn or norm_title(dst_title) == norm_title(src["bookname"]):
                    continue
                info = lookup_info(dst_isbn, dst_title)
                dst_kdc = kdc1(info.get("class_no"))
                rows.append({
                    "source_title": src["bookname"],
                    "source_isbn": src["isbn13"],
                    "source_kdc": src_kdc,
                    "source_category": KDC_MAIN.get(src_kdc, "미상"),
                    "target_title": info.get("bookname") or dst_title,
                    "target_isbn": dst_isbn,
                    "target_kdc": dst_kdc,
                    "target_category": KDC_MAIN.get(dst_kdc, "미상") if dst_kdc else "미상",
                    "edge_type": label,
                    "rank": rank,
                    "cross_category": bool(src_kdc and dst_kdc and src_kdc != dst_kdc),
                    "target_is_seed": dst_isbn in seed_isbns,
                })
                counts[label] += 1
        print(f"[{i:>3}/{len(sources)}] {src['bookname'][:25]} -> {dict(counts)}")

    df = pd.DataFrame(rows)
    pairs = set(zip(df["source_isbn"], df["target_isbn"]))
    df["reciprocal"] = [
        (dst, src) in pairs for src, dst in zip(df["source_isbn"], df["target_isbn"])
    ]
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="KDC별 2권만 수집")
    args = parser.parse_args()

    n_per_class = 2 if args.test else N_PER_CLASS
    pages = 1 if args.test else DICT_PAGES
    max_lookup = 40 if args.test else MAX_INDIV_LOOKUP

    print("[1/3] KDC 사전 구축")
    isbn2info = build_kdc_dict(pages)
    print("[2/3] KDC 층화 씨앗 선정")
    sources = pick_stratified_sources(isbn2info, n_per_class)
    print("[3/3] 함께대출/추천 연결 수집")
    edges = collect_edges(sources, isbn2info, max_lookup)

    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "network_edges.csv"
    edges.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"저장: {out}")
    print(f"씨앗 {len(sources)}권 / 방향 엣지 {len(edges)}개")


if __name__ == "__main__":
    main()
