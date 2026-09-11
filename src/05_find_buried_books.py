# -*- coding: utf-8 -*-
"""5단계: 네트워크 연결은 강하지만 대출은 상대적으로 적은 '묻힌 책'을 찾는다.

입력 : data/selected_books.csv, data/weighted_edges.csv
출력 : data/buried_books.csv

묻힘점수 = 서가 내부 연결강도 백분위 - 서가 내부 대출건수 백분위
양수일수록 '테마 안에서는 잘 연결되지만 상대적으로 덜 빌린 책'이다.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from common import DATA


def main() -> None:
    books = pd.read_csv(DATA / "selected_books.csv", dtype={"isbn": str})
    edges = pd.read_csv(DATA / "weighted_edges.csv", dtype={"isbn_a": str, "isbn_b": str})

    books["loan_count"] = pd.to_numeric(books["loan_count"], errors="coerce")
    if books["loan_count"].isna().any():
        missing = int(books["loan_count"].isna().sum())
        raise RuntimeError(
            f"loan_count가 비어 있는 책이 {missing}권 있습니다. "
            "03_enrich_clusters.py --refresh 또는 포함된 selected_books.csv를 확인하세요."
        )

    results = []
    for cid, group in books.groupby("cluster_id"):
        members = set(group.isbn)
        inner = edges[edges.isbn_a.isin(members) & edges.isbn_b.isin(members)]

        strength = defaultdict(float)
        degree = defaultdict(int)
        for _, e in inner.iterrows():
            strength[e.isbn_a] += float(e.weight)
            strength[e.isbn_b] += float(e.weight)
            degree[e.isbn_a] += 1
            degree[e.isbn_b] += 1

        g = group.copy()
        g["internal_strength"] = g.isbn.map(strength).fillna(0).round(3)
        g["internal_degree"] = g.isbn.map(degree).fillna(0).astype(int)
        g["strength_percentile"] = (g.internal_strength.rank(pct=True) * 100).round(0)
        g["loan_percentile"] = (g.loan_count.rank(pct=True) * 100).round(0)
        g["buried_score"] = (g.strength_percentile - g.loan_percentile).round(0)
        results.append(g)

    out = pd.concat(results, ignore_index=True)
    out = out[[
        "cluster_id", "shelf_name", "title", "isbn", "is_seed", "internal_strength",
        "internal_degree", "loan_count", "strength_percentile", "loan_percentile", "buried_score",
    ]].sort_values(["cluster_id", "buried_score"], ascending=[True, False])
    out["loan_count"] = out["loan_count"].astype(int)
    out.to_csv(DATA / "buried_books.csv", index=False, encoding="utf-8-sig")

    for cid, g in out.groupby("cluster_id"):
        print(f"\n[{g.iloc[0].shelf_name}]")
        for _, r in g[(g.buried_score > 0) & (g.strength_percentile >= 50)].head(3).iterrows():
            print(
                f"  {r.title[:42]} | 연결 {r.strength_percentile:.0f} / "
                f"대출 {r.loan_percentile:.0f} / 묻힘 +{r.buried_score:.0f}"
            )
    print("\n저장: data/buried_books.csv")


if __name__ == "__main__":
    main()
