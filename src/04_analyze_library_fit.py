# -*- coding: utf-8 -*-
"""4단계: 선정 35권의 서울 도서관 적용 가능성을 분석한다.

입력 : data/selected_books.csv
출력 : data/library_fit.csv
갱신 : data/selected_books.csv의 seoul_library_count

방법
  1) ISBN별 서울 소장 도서관을 수집해 35권과의 겹침 순위를 계산
  2) 대표 소형/대형 도서관 2곳에 대해 bookExist로 대출가능 여부를 확인
     - 소장O + 대출가능O = 전시형
     - 소장O + 대출중    = 안내형
     - 소장X             = 미소장

저장소에는 이미 계산된 스냅샷이 포함되어 있다. 새로 수집하려면 --refresh.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import pandas as pd

from common import DATA, DATA4LIBRARY_KEY, data4library_get, env_key

REGION = "11"  # 서울특별시
TARGET_LIBRARIES = [
    ("강남구립개포하늘꿈도서관", "본사례(소형 4만권)"),
    ("성동구립도서관", "대조군(대형 44.7만권)"),
]


def validate_existing() -> None:
    f = DATA / "library_fit.csv"
    if not f.exists():
        raise FileNotFoundError(f)
    df = pd.read_csv(f)
    print(f"기존 적용성 데이터: {len(df)}개 서울 도서관")
    cases = df[df.case_role.fillna("") != ""]
    for _, r in cases.iterrows():
        print(
            f"  {r.library_name}: 전시형 {int(r.display_count)} / "
            f"안내형 {int(r.guide_count)} / 미소장 {int(r.missing_count)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="정보나루 API로 소장/대출가능 상태를 새로 수집")
    args = parser.parse_args()

    if not args.refresh:
        validate_existing()
        print("API 재수집은 생략했습니다. 새로 갱신하려면 --refresh 를 사용하세요.")
        return
    if not env_key(DATA4LIBRARY_KEY):
        raise RuntimeError("--refresh에는 DATA4LIBRARY_AUTH_KEY가 필요합니다.")

    book_path = DATA / "selected_books.csv"
    books = pd.read_csv(book_path, dtype={"isbn": str})

    # 서울 도서관 기본정보 및 장서수
    lib_data = data4library_get(
        "libSrch", {"region": REGION, "pageNo": 1, "pageSize": 1000}
    )
    lib_entries = lib_data.get("response", {}).get("libs", []) or []
    base_info = {
        str(e["lib"].get("libCode")): e["lib"]
        for e in lib_entries if e.get("lib", {}).get("libCode")
    }

    lib_books = defaultdict(set)
    per_book_count = {}
    for i, r in books.iterrows():
        isbn = str(r.isbn)
        data = data4library_get(
            "libSrchByBook",
            {"isbn": isbn, "region": REGION, "pageNo": 1, "pageSize": 1000},
        )
        response = data.get("response", {}) or {}
        libs = response.get("libs", []) or []
        per_book_count[isbn] = int(response.get("numFound", len(libs)) or len(libs))
        for entry in libs:
            lib = entry.get("lib", {}) or {}
            code = str(lib.get("libCode", "")).strip()
            if not code:
                continue
            base_info.setdefault(code, lib)
            lib_books[code].add(isbn)
        print(f"[{i+1:>2}/{len(books)}] {str(r.title)[:32]} -> 서울 {len(libs)}곳")

    books["seoul_library_count"] = books.isbn.map(per_book_count).astype("Int64")

    by_cluster = {
        int(cid): set(g.isbn.astype(str)) for cid, g in books.groupby("cluster_id")
    }
    total = len(books)
    rows = []
    for code, owned in lib_books.items():
        info = base_info.get(code, {})
        row = {
            "library_code": code,
            "library_name": info.get("libName", ""),
            "address": info.get("address", ""),
            "owned_books": len(owned),
            "coverage_pct": round(len(owned) / total * 100, 1),
            "collection_size": info.get("BookCount", ""),
        }
        for cid in [2, 4, 5]:
            members = by_cluster.get(cid, set())
            row[f"cluster_{cid}_owned"] = len(owned & members)
            row[f"cluster_{cid}_total"] = len(members)
        rows.append(row)

    ranking = pd.DataFrame(rows).sort_values(
        ["owned_books", "collection_size"], ascending=[False, False]
    ).reset_index(drop=True)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    ranking["case_role"] = ""
    ranking["display_count"] = 0
    ranking["guide_count"] = 0
    ranking["missing_count"] = 0

    # 대표 2개 관의 현재 대출가능 상태
    for library_name, role in TARGET_LIBRARIES:
        hit = ranking[ranking.library_name.str.strip() == library_name]
        if hit.empty:
            print(f"[주의] 대표 도서관을 순위표에서 찾지 못함: {library_name}")
            continue
        idx = hit.index[0]
        code = str(hit.iloc[0].library_code)
        counts = {"전시형": 0, "안내형": 0, "미소장": 0}
        for _, book in books.iterrows():
            data = data4library_get(
                "bookExist", {"isbn13": str(book.isbn), "libCode": code}
            )
            result = data.get("response", {}).get("result", {}) or {}
            has = result.get("hasBook", "N")
            available = result.get("loanAvailable", "N")
            if has != "Y":
                counts["미소장"] += 1
            elif available == "Y":
                counts["전시형"] += 1
            else:
                counts["안내형"] += 1
        ranking.at[idx, "case_role"] = role
        ranking.at[idx, "display_count"] = counts["전시형"]
        ranking.at[idx, "guide_count"] = counts["안내형"]
        ranking.at[idx, "missing_count"] = counts["미소장"]

    columns = [
        "rank", "library_code", "library_name", "address", "owned_books", "coverage_pct",
        "cluster_2_owned", "cluster_2_total", "cluster_4_owned", "cluster_4_total",
        "cluster_5_owned", "cluster_5_total", "collection_size", "case_role",
        "display_count", "guide_count", "missing_count",
    ]
    ranking[columns].to_csv(DATA / "library_fit.csv", index=False, encoding="utf-8-sig")
    books.to_csv(book_path, index=False, encoding="utf-8-sig")
    print(f"저장: library_fit.csv ({len(ranking)}개 도서관), selected_books.csv 갱신")


if __name__ == "__main__":
    main()
