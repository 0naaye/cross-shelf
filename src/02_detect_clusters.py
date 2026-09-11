# -*- coding: utf-8 -*-
"""2단계: 함께대출 네트워크를 보정하고 안정적인 테마 클러스터를 추출한다.

입력
  data/network_edges.csv
출력
  data/weighted_edges.csv
  data/cluster_summary.csv
  data/selected_books.csv  (기본 멤버십; 03 단계가 메타데이터를 보강)

핵심 방법
  1) 함께대출(coLoan)만 클러스터링에 사용, 마니아/다독자 추천은 검증용
  2) 양방향 순위점수 합산 × 역연관빈도 기반 허브 편향 보정
  3) 제목 패턴으로 같은 시리즈 노드를 합치고 시리즈 내부 링크 제거
  4) Louvain 3개 resolution × 10개 seed = 30회 반복
  5) 70% 이상 함께 묶인 쌍의 연결성분을 안정적 클러스터로 정의
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict

import networkx as nx
import pandas as pd

from common import (
    DATA, NOTABLE_CLUSTERS, SELECTED_CLUSTERS, SHELF_NAMES, same_series, to_bool,
)

RESOLUTIONS = [0.8, 1.0, 1.2]
SEEDS = list(range(10))
STABLE_THRESHOLD = 0.70
CRITERIA = {
    "size_min": 8,
    "size_max": 20,
    "kdc_min": 3,
    "hub_max": 0.5,
}


def load_network_edges() -> pd.DataFrame:
    f = DATA / "network_edges.csv"
    if not f.exists():
        raise FileNotFoundError(f"먼저 01_build_network.py를 실행하거나 {f.name}을 준비하세요.")
    df = pd.read_csv(f, dtype={"source_isbn": str, "target_isbn": str})
    for c in ["cross_category", "target_is_seed", "reciprocal"]:
        if c in df:
            df[c] = to_bool(df[c])
    return df


def compute_hub_correction(co: pd.DataFrame):
    n_sources = co["source_isbn"].nunique()
    doc_freq = co.groupby("target_isbn")["source_isbn"].nunique().to_dict()
    ln_n = math.log(n_sources)
    corr = {
        isbn: math.log(n_sources / freq) / ln_n
        for isbn, freq in doc_freq.items()
        if freq > 0
    }
    return defaultdict(lambda: 1.0, corr), n_sources


def build_node_merge(df: pd.DataFrame):
    title, category = {}, {}
    for _, r in df.iterrows():
        title[r.source_isbn] = r.source_title
        title[r.target_isbn] = r.target_title
        category[r.source_isbn] = r.source_category
        category[r.target_isbn] = r.target_category

    co = df[df.edge_type == "함께대출"]
    degree = Counter()
    for _, r in co.iterrows():
        degree[r.source_isbn] += 1
        degree[r.target_isbn] += 1
    nodes = list(degree)

    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if same_series(title[nodes[i]], title[nodes[j]]):
                union(nodes[i], nodes[j])

    groups = defaultdict(list)
    for n in nodes:
        groups[find(n)].append(n)

    rep, meta = {}, {}
    for members in groups.values():
        representative = max(members, key=lambda m: (degree[m], -len(str(title[m]))))
        for m in members:
            rep[m] = representative
        meta[representative] = (title[representative], category[representative])
    return rep, meta


def build_weighted_network(df: pd.DataFrame):
    df = df.copy()
    rep, meta = build_node_merge(df)
    original_seeds = set(df.source_isbn)
    seeds = {rep.get(s, s) for s in original_seeds}

    df["source_isbn"] = df.source_isbn.map(lambda x: rep.get(x, x))
    df["target_isbn"] = df.target_isbn.map(lambda x: rep.get(x, x))
    df = df[df.source_isbn != df.target_isbn].copy()

    for _, r in df.iterrows():
        meta.setdefault(r.source_isbn, (r.source_title, r.source_category))
        meta.setdefault(r.target_isbn, (r.target_title, r.target_category))

    co = df[df.edge_type == "함께대출"].copy()
    other = df[df.edge_type != "함께대출"]

    other_pairs = defaultdict(set)
    for _, r in other.iterrows():
        other_pairs[tuple(sorted((r.source_isbn, r.target_isbn)))].add(r.edge_type)

    correction, n_sources = compute_hub_correction(co)

    # 병합 후 동일 방향 중복은 가장 높은 순위(가장 작은 rank)만 유지
    dir_best = {}
    for _, r in co.iterrows():
        key = (r.source_isbn, r.target_isbn)
        rank = int(r["rank"])
        if key not in dir_best or rank < dir_best[key]:
            dir_best[key] = rank

    pairs = defaultdict(lambda: {"score": 0.0, "dirs": set()})
    for (a, b), rank in dir_best.items():
        key = tuple(sorted((a, b)))
        pairs[key]["score"] += (11 - rank) / 10
        pairs[key]["dirs"].add((a, b))

    rows = []
    for (a, b), p in pairs.items():
        title_a, cat_a = meta.get(a, ("?", "미상"))
        title_b, cat_b = meta.get(b, ("?", "미상"))
        if same_series(title_a, title_b):
            continue
        reciprocal = len(p["dirs"]) == 2
        hub = min(correction[a], correction[b])
        score = p["score"]
        rows.append({
            "isbn_a": a,
            "isbn_b": b,
            "title_a": title_a,
            "title_b": title_b,
            "category_a": cat_a,
            "category_b": cat_b,
            "cross_category": bool(cat_a != cat_b and cat_a != "미상" and cat_b != "미상"),
            "reciprocal": reciprocal,
            "rank_score": round(score, 3),
            "hub_correction": round(hub, 3),
            "weight": round(score * hub, 4),
            "recommendation_validation": len(other_pairs.get((a, b), set())),
        })
    weighted = pd.DataFrame(rows).sort_values("weight", ascending=False).reset_index(drop=True)
    return weighted, meta, seeds, n_sources


def to_graph(edges: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    for _, r in edges.iterrows():
        g.add_edge(r.isbn_a, r.isbn_b, weight=float(r.weight))
    return g


def stable_clusters(g: nx.Graph):
    runs = []
    for resolution in RESOLUTIONS:
        for seed in SEEDS:
            runs.append(
                nx.community.louvain_communities(
                    g, weight="weight", resolution=resolution, seed=seed
                )
            )

    together = Counter()
    for communities in runs:
        for community in communities:
            members = sorted(community)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    together[(members[i], members[j])] += 1

    stable = nx.Graph()
    stable.add_nodes_from(g.nodes())
    n_runs = len(runs)
    for (a, b), count in together.items():
        if count / n_runs >= STABLE_THRESHOLD:
            stable.add_edge(a, b)

    clusters = [set(c) for c in nx.connected_components(stable) if len(c) >= 2]
    clusters.sort(key=len, reverse=True)
    return clusters


def evaluate_cluster(members, edges, meta, seeds, baseline):
    members = set(members)
    inner = edges[edges.isbn_a.isin(members) & edges.isbn_b.isin(members)]
    categories = [meta.get(i, ("?", "미상"))[1] for i in members]
    known_categories = [c for c in categories if c != "미상"]

    node_weight = defaultdict(float)
    for _, r in inner.iterrows():
        node_weight[r.isbn_a] += float(r.weight)
        node_weight[r.isbn_b] += float(r.weight)
    total_weight = float(inner.weight.sum())
    hub = max(node_weight.values()) / total_weight if total_weight > 0 else 1.0

    k = len(members)
    density = len(inner) / (k * (k - 1) / 2) if k > 1 else 0.0
    avg_weight = float(inner.weight.mean()) if len(inner) else 0.0
    cross_rate = float(to_bool(inner.cross_category).mean() * 100) if len(inner) else 0.0
    validation_rate = float((inner.recommendation_validation.astype(int) > 0).mean() * 100) if len(inner) else 0.0

    reasons = []
    if not (CRITERIA["size_min"] <= k <= CRITERIA["size_max"]):
        reasons.append(f"size {k} outside {CRITERIA['size_min']}-{CRITERIA['size_max']}")
    n_kdc = len(set(known_categories))
    if n_kdc < CRITERIA["kdc_min"]:
        reasons.append(f"KDC categories {n_kdc} < {CRITERIA['kdc_min']}")
    if hub > CRITERIA["hub_max"]:
        reasons.append(f"hub concentration {hub:.3f} > {CRITERIA['hub_max']}")
    if avg_weight < baseline:
        reasons.append(f"avg weight {avg_weight:.3f} < core mean {baseline:.3f}")

    return {
        "size": k,
        "kdc_count": n_kdc,
        "kdc_distribution": dict(Counter(categories)),
        "cross_rate_pct": round(cross_rate, 1),
        "internal_edges": len(inner),
        "avg_weight": round(avg_weight, 3),
        "density": round(density, 3),
        "hub_concentration": round(hub, 3),
        "validation_rate_pct": round(validation_rate, 1),
        "selected": len(reasons) == 0,
        "reasons": reasons,
        "inner": inner,
        "members": members,
        "seed_count": sum(i in seeds for i in members),
    }


def existing_map(path, key: str) -> dict:
    path = pd.io.common.stringify_path(path)
    try:
        old = pd.read_csv(path, dtype=str)
    except Exception:
        return {}
    return {str(r[key]): r.to_dict() for _, r in old.iterrows() if key in old.columns}


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    raw = load_network_edges()
    weighted, meta, seeds, n_sources = build_weighted_network(raw)
    weighted.to_csv(DATA / "weighted_edges.csv", index=False, encoding="utf-8-sig")

    degree = Counter()
    for _, r in weighted.iterrows():
        degree[r.isbn_a] += 1
        degree[r.isbn_b] += 1
    core_nodes = {n for n, d in degree.items() if d >= 2}
    core_edges = weighted[weighted.isbn_a.isin(core_nodes) & weighted.isbn_b.isin(core_nodes)].copy()
    core_graph = to_graph(core_edges)
    baseline = float(core_edges.weight.mean())
    clusters = stable_clusters(core_graph)

    # 기존 보강값은 재실행 시 보존
    old_summary = existing_map(DATA / "cluster_summary.csv", "cluster_id")
    old_books = existing_map(DATA / "selected_books.csv", "isbn")

    summary_rows = []
    selected_rows = []
    for cluster_id, members in enumerate(clusters, 1):
        ev = evaluate_cluster(members, weighted, meta, seeds, baseline)
        status = "selected" if ev["selected"] else (
            "notable" if cluster_id in NOTABLE_CLUSTERS else "rejected"
        )
        prev = old_summary.get(str(cluster_id), {})
        summary_rows.append({
            "cluster_id": cluster_id,
            "status": status,
            "size": ev["size"],
            "kdc_count": ev["kdc_count"],
            "kdc_distribution": json.dumps(ev["kdc_distribution"], ensure_ascii=False, separators=(",", ":")),
            "cross_rate_pct": ev["cross_rate_pct"],
            "internal_edges": ev["internal_edges"],
            "avg_weight": ev["avg_weight"],
            "density": ev["density"],
            "hub_concentration": ev["hub_concentration"],
            "validation_rate_pct": ev["validation_rate_pct"],
            "shelf_name": SHELF_NAMES.get(cluster_id, ""),
            "top_keywords": prev.get("top_keywords", "") if isinstance(prev, dict) else "",
            "rejection_reason": " / ".join(ev["reasons"]),
        })

        if ev["selected"]:
            strength = defaultdict(float)
            idegree = defaultdict(int)
            for _, e in ev["inner"].iterrows():
                strength[e.isbn_a] += float(e.weight)
                strength[e.isbn_b] += float(e.weight)
                idegree[e.isbn_a] += 1
                idegree[e.isbn_b] += 1
            for isbn in sorted(members, key=lambda x: -strength[x]):
                prev_b = old_books.get(str(isbn), {})
                row = {
                    "cluster_id": cluster_id,
                    "shelf_name": SHELF_NAMES.get(cluster_id, ""),
                    "isbn": isbn,
                    "title": meta.get(isbn, ("?", "미상"))[0],
                    "kdc_category": meta.get(isbn, ("?", "미상"))[1],
                    "is_seed": isbn in seeds,
                    "internal_strength": round(strength[isbn], 3),
                    "internal_degree": idegree[isbn],
                    "addition_symbol": prev_b.get("addition_symbol", ""),
                    "reader_group": prev_b.get("reader_group", ""),
                    "publication_form": prev_b.get("publication_form", ""),
                    "loan_count": prev_b.get("loan_count", ""),
                    "seoul_library_count": prev_b.get("seoul_library_count", ""),
                    "top_keywords": prev_b.get("top_keywords", ""),
                }
                selected_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    selected = pd.DataFrame(selected_rows)
    summary.to_csv(DATA / "cluster_summary.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(DATA / "selected_books.csv", index=False, encoding="utf-8-sig")

    selected_ids = summary.loc[summary.status == "selected", "cluster_id"].tolist()
    print(f"coLoan source nodes after series merge: {n_sources}")
    print(f"weighted network: {len(weighted)} edges / {len(set(weighted.isbn_a) | set(weighted.isbn_b))} nodes")
    print(f"core: {len(core_nodes)} nodes / {len(core_edges)} edges / mean weight {baseline:.3f}")
    print(f"stable clusters: {len(clusters)} / selected: {selected_ids} / selected books: {len(selected)}")
    print("saved: weighted_edges.csv, cluster_summary.csv, selected_books.csv")


if __name__ == "__main__":
    main()
