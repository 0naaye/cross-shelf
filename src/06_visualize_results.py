# -*- coding: utf-8 -*-
"""6단계: 최종 결과 시각화 2종을 생성한다.

입력 : data/weighted_edges.csv, data/selected_books.csv
출력 : outputs/network_overview.png, outputs/parent_shelf_detail.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx

from common import A4_TEXT_W, DATA, OUTPUTS, setup_matplotlib

CLUSTER_STYLE = {
    2: {"name": "아이와 함께 자라는 부모의 서가", "color": "#20456B", "marker": "o"},
    4: {"name": "위로가 필요한 날의 소설과 교양", "color": "#C1502D", "marker": "s"},
    5: {"name": "웃으며 읽는 어린이 교양", "color": "#D3A017", "marker": "^"},
}


def plot_network(plt):
    from matplotlib.lines import Line2D

    edges = pd.read_csv(DATA / "weighted_edges.csv", dtype={"isbn_a": str, "isbn_b": str})
    books = pd.read_csv(DATA / "selected_books.csv", dtype={"isbn": str})
    membership = dict(zip(books.isbn, books.cluster_id.astype(int)))

    g = nx.Graph()
    for _, r in edges.iterrows():
        g.add_edge(r.isbn_a, r.isbn_b, weight=float(r.weight))
    nx.set_node_attributes(g, {n: membership.get(n, 0) for n in g}, "cluster")
    for _, _, d in g.edges(data=True):
        d["layout_weight"] = 0.2 + d["weight"] / 2

    components = sorted(nx.connected_components(g), key=len, reverse=True)
    main = g.subgraph(components[0]).copy()
    pos = nx.spring_layout(main, weight="layout_weight", k=1.6, iterations=800, seed=7)
    p = np.array(list(pos.values()))
    lo, hi = p.min(axis=0), p.max(axis=0)
    span = hi - lo

    x = lo[0] + span[0] * 0.02
    y = lo[1] - span[1] * 0.13
    for comp in components[1:]:
        sub = g.subgraph(comp)
        small = nx.spring_layout(sub, seed=7, k=0.9)
        small = {n: np.array(v) * span[0] * 0.03 for n, v in small.items()}
        for n, v in small.items():
            pos[n] = v + np.array([x, y])
        x += span[0] * 0.105

    weighted_degree = {n: sum(d["weight"] for _, _, d in g.edges(n, data=True)) for n in g}
    values = np.array(list(weighted_degree.values()))
    denom = max(values.max() - values.min(), 1e-9)
    sizes = {n: 12 + 205 * (weighted_degree[n] - values.min()) / denom for n in g}

    fig, ax = plt.subplots(figsize=(A4_TEXT_W, A4_TEXT_W * 0.92))
    ax.set_axis_off()
    ax.set_aspect("equal")

    for u, v, d in g.edges(data=True):
        cu, cv = g.nodes[u]["cluster"], g.nodes[v]["cluster"]
        inside = cu == cv and cu in CLUSTER_STYLE
        color = CLUSTER_STYLE[cu]["color"] if inside else "#E7E7E7"
        ax.plot(*zip(pos[u], pos[v]), color=color,
                lw=(0.5 + 1.5 * d["weight"] / 2) if inside else 0.35,
                alpha=0.7 if inside else 0.55, zorder=3 if inside else 1)

    others = [n for n in g if g.nodes[n]["cluster"] not in CLUSTER_STYLE]
    ax.scatter([pos[n][0] for n in others], [pos[n][1] for n in others],
               s=[sizes[n] * 0.75 for n in others], c="#CFCFCF", alpha=0.28,
               linewidths=0.35, edgecolors="#FFFFFF", zorder=2)

    for cid, style in CLUSTER_STYLE.items():
        nodes = [n for n in g if g.nodes[n]["cluster"] == cid]
        ax.scatter([pos[n][0] for n in nodes], [pos[n][1] for n in nodes],
                   s=[sizes[n] for n in nodes], c=style["color"], marker=style["marker"],
                   linewidths=0.8, edgecolors="#FFFFFF", zorder=6)

    handles = [
        Line2D([], [], marker=s["marker"], color="none", markerfacecolor=s["color"],
               markeredgecolor="#FFFFFF", markersize=8, label=s["name"])
        for s in CLUSTER_STYLE.values()
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=8)
    ax.set_title("함께대출 네트워크에서 추출한 융합형 테마 서가 3개",
                 fontsize=12, fontweight="bold", pad=10)
    ax.text(0.5, -0.02,
            f"도서 {g.number_of_nodes()}종 · 함께대출 연결 {g.number_of_edges()}쌍 | "
            "노드 크기=가중 연결도 · 선 굵기=허브 보정 후 함께대출 강도",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS / "network_overview.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("저장:", out)


def _short_title(title: str, n: int = 24) -> str:
    t = str(title).split(":")[0].strip()
    if t.startswith("(") and ")" in t:
        t = t[t.find(")") + 1:].strip() or t
    return t if len(t) <= n else t[: n - 1] + "…"


def plot_parent_shelf(plt):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    books = pd.read_csv(DATA / "selected_books.csv", dtype={"isbn": str})
    d = books[books.cluster_id.astype(int) == 2].copy()
    adult_groups = {"교양", "실용", "청소년"}
    d["audience_side"] = np.where(d.reader_group.isin(adult_groups), "어른책", "아이책")
    d["label"] = d.title.map(_short_title)

    kdc_order = ["기술과학", "문학", "철학", "예술", "총류", "사회과학"]
    d["order"] = d.kdc_category.map({k: i for i, k in enumerate(kdc_order)})
    d = d.sort_values(["order", "audience_side", "reader_group"]).reset_index(drop=True)

    styles = {
        "어른책": {"color": "#20456B", "marker": "o", "size": 95},
        "아이책": {"color": "#D3A017", "marker": "^", "size": 115},
    }
    groups = [(k, d[d.kdc_category == k]) for k in kdc_order if (d.kdc_category == k).any()]
    n_adult = int((d.audience_side == "어른책").sum())
    n_child = int((d.audience_side == "아이책").sum())

    h_head, h_book, h_gap = 1.45, 1.0, 0.45
    total = sum(h_head + len(g) * h_book + h_gap for _, g in groups)
    unit = (0.955 - 0.035) / total
    y_book, headers = {}, []
    cur = 0.955
    for k, g in groups:
        cur -= h_head * unit
        headers.append((k, len(g), cur))
        for idx in g.index:
            cur -= h_book * unit
            y_book[idx] = cur + h_book * unit * 0.35
        cur -= h_gap * unit

    bins = [("어른책", n_adult, 0.930), ("아이책", n_child, 0.440)]
    y_slot, boxes = {}, {}
    for name, n, top in bins:
        ys = [top - 0.075 - i * 0.048 for i in range(n)]
        for idx, y in zip(d.index[d.audience_side == name], ys):
            y_slot[idx] = y
        boxes[name] = (top, ys[-1] - 0.05)

    fig, ax = plt.subplots(figsize=(A4_TEXT_W, A4_TEXT_W * 1.10))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    x_head, x_node, x_title, x_tag, x_rib, x_bin, bin_w = 0.005, 0.045, 0.075, 0.475, 0.495, 0.70, 0.295

    for name, n, top in bins:
        bottom = boxes[name][1]
        style = styles[name]
        ax.add_patch(FancyBboxPatch((x_bin, bottom), bin_w, top - bottom,
                                    boxstyle="round,pad=0.006,rounding_size=0.012",
                                    fc=style["color"], ec="none", alpha=0.07))
        ax.text(x_bin + bin_w / 2, top - 0.028, f"{name} {n}권", ha="center",
                va="center", fontsize=9.5, fontweight="bold", color=style["color"])

    for i, r in d.iterrows():
        style = styles[r.audience_side]
        ax.add_patch(FancyArrowPatch((x_rib, y_book[i]), (x_bin + 0.028, y_slot[i]),
                                     connectionstyle="arc3,rad=0.10", arrowstyle="-",
                                     color=style["color"], lw=1.0, alpha=0.42))
        ax.scatter(x_bin + 0.028, y_slot[i], s=style["size"] * 0.55, c=style["color"],
                   marker=style["marker"], linewidths=0.6, edgecolors="#FFFFFF")

    for k, n, y in headers:
        ax.text(x_head, y, k, ha="left", va="center", fontsize=9.5, fontweight="bold")
        ax.text(0.185, y, f"{n}권", ha="right", va="center", fontsize=8, color="#6E6E6E")
        ax.plot([0.205, x_rib - 0.01], [y, y], color="#D5D5D5", lw=0.8)

    for i, r in d.iterrows():
        style = styles[r.audience_side]
        ax.scatter(x_node, y_book[i], s=style["size"], c=style["color"], marker=style["marker"],
                   linewidths=0.8, edgecolors="#FFFFFF")
        ax.text(x_title, y_book[i], r.label, ha="left", va="center", fontsize=8.2)
        ax.text(x_tag, y_book[i], r.reader_group, ha="right", va="center", fontsize=7.2, color="#6E6E6E")

    ax.text(x_head, 0.995, "KDC 대분류 — 현재의 서가 배치", ha="left", va="top",
            fontsize=9.5, fontweight="bold")
    ax.text(x_head, 0.972, f"{len(groups)}개 서가로 흩어진다", ha="left", va="top",
            fontsize=7.8, color="#6E6E6E")
    ax.text(x_bin + bin_w / 2, 0.995, "부가기호 독자대상", ha="center", va="top",
            fontsize=9.5, fontweight="bold")
    ax.text(x_bin + bin_w / 2, 0.972, "어른책과 아이책이 한 테마에서 만난다", ha="center", va="top",
            fontsize=7.8, color="#6E6E6E")
    ax.set_title("「아이와 함께 자라는 부모의 서가」 — KDC 6갈래, 독자대상 2갈래",
                 fontsize=11.5, fontweight="bold", pad=12)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS / "parent_shelf_detail.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("저장:", out)


def main() -> None:
    plt = setup_matplotlib()
    plot_network(plt)
    plot_parent_shelf(plt)


if __name__ == "__main__":
    main()
