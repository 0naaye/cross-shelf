# -*- coding: utf-8 -*-
"""3단계: 선정된 3개 테마 서가에 해석용 메타데이터를 결합한다.

입력 : data/selected_books.csv, data/cluster_summary.csv
갱신 : data/selected_books.csv, data/cluster_summary.csv

결합 정보
  - 정보나루 usageAnalysisList: 전국 누적 대출건수, 키워드
  - 국가서지 ISBN 서지정보: 부가기호 -> 독자대상/발행형태

저장소에는 이미 보강된 결과가 포함되어 있다. 인증키가 없으면 기존 데이터를
검증만 하고 종료한다. API로 새로 갱신하려면 --refresh 옵션을 사용한다.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import pandas as pd

from common import (
    DATA, DATA4LIBRARY_KEY, NL_SEOJI_KEY, addition_code_labels, data4library_get,
    env_key, seoji_get,
)


def validate_existing(books: pd.DataFrame) -> None:
    required = ["loan_count", "addition_symbol", "reader_group", "publication_form", "top_keywords"]
    missing = [c for c in required if c not in books.columns]
    if missing:
        raise RuntimeError(f"selected_books.csv에 필요한 열이 없습니다: {missing}")
    filled = {c: int(books[c].notna().sum()) for c in required}
    print(f"기존 보강 데이터 확인: {filled}")


def fetch_usage(isbn: str) -> tuple[dict, list[tuple[str, int]]]:
    data = data4library_get("usageAnalysisList", {"isbn13": isbn})
    response = data.get("response", {}) or {}
    book = response.get("book", {}) or {}
    keywords = []
    for item in response.get("keywords", []) or []:
        k = item.get("keyword", {}) or {}
        word = str(k.get("word", "")).strip()
        if not word:
            continue
        try:
            weight = int(k.get("weight", 0) or 0)
        except Exception:
            weight = 0
        keywords.append((word, weight))
    return book, keywords


def fetch_addition_symbol(isbn: str, fallback: str = "") -> str:
    # 국가서지를 우선 사용하되, 키가 없으면 정보나루의 addition_symbol로 대체 가능.
    if env_key(NL_SEOJI_KEY):
        data = seoji_get(isbn)
        docs = data.get("docs", []) or []
        if docs:
            code = str(docs[0].get("EA_ADD_CODE", "")).strip()
            if code:
                return code
    return str(fallback or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="API를 다시 호출해 35권 메타데이터 갱신")
    args = parser.parse_args()

    book_path = DATA / "selected_books.csv"
    summary_path = DATA / "cluster_summary.csv"
    books = pd.read_csv(book_path, dtype={"isbn": str})
    summary = pd.read_csv(summary_path)

    if not args.refresh:
        validate_existing(books)
        print("API 재수집은 생략했습니다. 새로 갱신하려면 --refresh 를 사용하세요.")
        return

    if not env_key(DATA4LIBRARY_KEY):
        raise RuntimeError("--refresh에는 DATA4LIBRARY_AUTH_KEY가 필요합니다.")

    cluster_kw_score = defaultdict(lambda: defaultdict(int))
    cluster_kw_books = defaultdict(lambda: defaultdict(set))

    for i, r in books.iterrows():
        isbn = str(r["isbn"])
        book, keywords = fetch_usage(isbn)
        loan_count = book.get("loanCnt")
        addition = fetch_addition_symbol(isbn, book.get("addition_symbol", ""))
        addition, reader_group, publication_form = addition_code_labels(addition)

        books.at[i, "loan_count"] = loan_count
        books.at[i, "addition_symbol"] = addition
        books.at[i, "reader_group"] = reader_group
        books.at[i, "publication_form"] = publication_form
        books.at[i, "top_keywords"] = "; ".join(w for w, _ in sorted(keywords, key=lambda x: -x[1])[:5])

        cid = int(r["cluster_id"])
        for word, weight in keywords:
            cluster_kw_score[cid][word] += weight
            cluster_kw_books[cid][word].add(isbn)
        print(f"[{i+1:>2}/{len(books)}] {str(r['title'])[:34]}")

    for cid in sorted(set(books.cluster_id.astype(int))):
        shared = [
            (w, score, len(cluster_kw_books[cid][w]))
            for w, score in cluster_kw_score[cid].items()
            if len(cluster_kw_books[cid][w]) >= 2
        ]
        if not shared:
            shared = [
                (w, score, len(cluster_kw_books[cid][w]))
                for w, score in cluster_kw_score[cid].items()
            ]
        shared.sort(key=lambda x: (-x[1], -x[2], x[0]))
        text = "; ".join(f"{w}({score}/{n}권)" for w, score, n in shared[:12])
        summary.loc[summary.cluster_id.astype(int) == cid, "top_keywords"] = text

    books.to_csv(book_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("갱신 완료: selected_books.csv, cluster_summary.csv")


if __name__ == "__main__":
    main()
