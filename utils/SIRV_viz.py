# sirv_viz.py
# -----------------------------------------
# 依賴：numpy, pandas, matplotlib, networkx
# 功能：畫左上 100×100 網格的多時點快照
# -----------------------------------------

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# ─────────────────────────────────────────
# 基本工具
# ─────────────────────────────────────────
def filter_left_top(df: pd.DataFrame, sub_m=100, sub_n=100):
    """保留左上 sub_m × sub_n 子網格"""
    idx = df.index
    mask = [(r < sub_m and c < sub_n) for r, c in idx]
    return df.loc[mask]


# ─────────────────────────────────────────
# snapshot → DataFrame（支援舊 dict 與新 uint8）
# ─────────────────────────────────────────
def get_status(
    iterations: list[dict],
    *,
    snap_num: int = -1,
    M: int,
    N: int,
    filt: bool = False,
    sub_m: int = 100,
    sub_n: int = 100,
) -> pd.DataFrame:
    """從 iterations 取出第 snap_num 步的節點狀態並轉成 DataFrame"""
    def _snap_to_dict(snap):
        # 支援 uint8 或舊 dict
        if isinstance(snap, np.ndarray):
            lin = np.arange(snap.size, dtype=np.int32)
            r, c = divmod(lin, N)
            mask = (r < sub_m) & (c < sub_n)
            return {(int(rr), int(cc)): int(st)
                    for rr, cc, st in zip(r[mask], c[mask], snap[mask])}
        else:
            return snap                           # 已是 dict

    # ── 1. 初始狀態 ───────────────────────────────────────────
    init_dict = _snap_to_dict(iterations[0]["status"])
    init_df   = pd.Series(init_dict).to_frame(0)          # col 0

    # ── 2. 指定 snapshot 狀態 ────────────────────────────────
    snap_dict = _snap_to_dict(iterations[snap_num]["status"])
    snap_df   = pd.Series(snap_dict).to_frame("Final_Status")

    # ── 3. 合併並裁左上角 ───────────────────────────────────
    status_df = pd.concat([init_df, snap_df], axis=1, sort=False)
    status_df = filter_left_top(status_df, sub_m, sub_n).astype(int)

    conditions = [
        (status_df[0] == 0) & (status_df["Final_Status"] == 0),
        (status_df[0] == 0) & (status_df["Final_Status"] >= 2),
        (status_df[0] == 1) & (status_df["Final_Status"] == 1),
        (status_df[0] == 1) & (status_df["Final_Status"] >= 2),
        (status_df[0] == 2) & (status_df["Final_Status"] >= 2),
    ]
    labels = ["SFR", "FFR", "HV", "IV", "II"]
    status_df["group"] = np.select(conditions, labels, default="Unknown")
    return status_df


# ─────────────────────────────────────────
# DataFrame → networkx 與顏色
# ─────────────────────────────────────────
def get_nw_pos(nodes_data: pd.DataFrame):
    G = nx.Graph()
    for pos, row in nodes_data.iterrows():
        G.add_node(pos, pos=pos, group=row["group"])

    colors = {
        "SFR": "blue",
        "FFR": "red",
        "HV": "green",
        "IV": "black",
        "II": "gray",
        "Unknown": "white",
    }
    node_colors = [colors[G.nodes[n]["group"]] for n in G.nodes]
    pos = nx.get_node_attributes(G, "pos")
    return G, pos, node_colors


# ─────────────────────────────────────────
# 主繪圖：四張快照並排
# ─────────────────────────────────────────
def get_nw_graph(
    iterations: list[dict],
    params: dict,
    *,
    M: int,
    N: int,
    stage_num: int = 4,
    sub_m: int = 100,
    sub_n: int = 100,
):
    fig, axes = plt.subplots(
        1, stage_num, figsize=(40, 10), squeeze=False
    )
    axes = axes[0]

    it_len = len(iterations)
    local_times = [0]
    if stage_num > 1:
        step = it_len // (stage_num - 1)
        local_times += [step * k for k in range(1, stage_num - 1)]
        local_times.append(it_len - 1)

    for ax, snap_id in zip(axes, local_times):
        status_df = get_status(
            iterations,
            snap_num=snap_id,
            M=M,
            N=N,
            sub_m=sub_m,
            sub_n=sub_n,
        )
        G, pos, node_colors = get_nw_pos(status_df)
        nx.draw(
            G,
            pos,
            ax=ax,
            node_color=node_colors,
            node_shape="s",
            node_size=35,
        )
        ax.set_title(f"$t={snap_id}$", y=-0.05, fontsize=32)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"(b) $x={params['fraction_vaccinated']}$", fontsize=48,
                 x=0.5, y=0.98)
    fig.text(
        0.51,
        0.07,
        s="Local time steps",
        horizontalalignment="center",
        fontsize=48,
    )
    plt.annotate(
        "",
        xy=(0.7, 0.12),
        xycoords="figure fraction",
        xytext=(0.1, 0.12),
        arrowprops=dict(
            arrowstyle="-|>,head_length=1.0,head_width=0.5",
            facecolor="black",
            lw=10,
            mutation_scale=50,
        ),
    )
    fig.subplots_adjust(wspace=0, bottom=0.22)
    plt.show()
    return
