"""
Fast SIRV model on a 2-D grid (or any undirected graph in CSR form).

核心特色
--------
1. 以 4 個 bool 向量 (S, I, R, V) 表節點狀態；記憶體極小。
2. 鄰居掃描改用 CSR adjacency + Numba 向量化 `sirv_step()`。
3. `run_until_eq()` 自動迭代至無感染者或達 `max_iter`。
4. 可重複呼叫 `set_initial_status()` 以便在 global stage 中多季重用。

依賴
----
pip install numpy scipy numba
"""

from __future__ import annotations
import numpy as np
import numba as nb
import tqdm
import math
import networkx as nx
import time
import matplotlib.pyplot as plt
from utils.DiffusionTrend import DiffusionTrend

from utils.DiffusionModel import DiffusionModel


# ----------------------------------------------------------------------
# 1. 核心：單步狀態更新（Numba JIT）
# ----------------------------------------------------------------------
@nb.njit(parallel=True, fastmath=True)
def sirv_step(S, I, R, V, indptr, indices, beta, eta, gamma):
    """
    Parameters
    ----------
    S, I, R, V : 1-D bool ndarray
    indptr, indices : CSR 鄰接矩陣兩個欄位
    beta, eta, gamma : float
    Returns
    -------
    done : bool    # 若已無感染者 ⇒ True
    """
    n = S.size
    new_inf = np.zeros(n, np.bool_)             # 疫情擴散暫存
    p_rec = 1.0 - math.exp(-gamma)               # 只算一次
    
    # idx=0→deg=2, etc.
    p_edge_S = (
        1.0 - math.exp(-beta / 2),
        1.0 - math.exp(-beta / 3),
        1.0 - math.exp(-beta / 4),  # k=4
    )
    p_edge_V = (
        1.0 - math.exp(-beta * (1.0 - eta) / 2),
        1.0 - math.exp(-beta * (1.0 - eta) / 3),
        1.0 - math.exp(-beta * (1.0 - eta) / 4),
    )

    for u in nb.prange(n):                      # C 級多核心並行
        if I[u]:
            deg_u = indptr[u+1] - indptr[u]          # 真實度數
            
            # 把 ODE 速率換成這個節點對鄰邊的 per‑step 機率
            # 掃鄰居
            for k in range(indptr[u], indptr[u + 1]):
                v = indices[k]
                if S[v]:
                    if np.random.random() < p_edge_S[deg_u-2]:
                        new_inf[v] = True
                elif V[v]:
                    if np.random.random() < p_edge_V[deg_u-2]:
                        new_inf[v] = True
            # 自身康復？
            if np.random.random() < p_rec:
                I[u] = False
                R[u] = True

    # 一次性批次更新
    S[new_inf] = False
    V[new_inf] = False
    I[new_inf] = True

    return I.sum() == 0                         # 收斂判定


# ----------------------------------------------------------------------
# 2. 類別封裝：FastSIRV
# ----------------------------------------------------------------------
class FastSIRV(DiffusionModel):
    """
    Fast SIRV on a fixed graph (CSR).

    Parameters
    ----------
    indptr, indices : 1-D int ndarray
        CSR adjacency obtained from scipy.sparse.csr_matrix.indptr / .indices
    """

    def __init__(self, indptr: np.ndarray, indices: np.ndarray):
        self.indptr = indptr.astype(np.int32)
        self.indices = indices.astype(np.int32)
        self.nnodes = indptr.size - 1

        # 預先配置狀態向量，後續季節重複利用
        self.S = np.empty(self.nnodes, np.bool_)
        self.I = np.empty(self.nnodes, np.bool_)
        self.R = np.empty(self.nnodes, np.bool_)
        self.V = np.empty(self.nnodes, np.bool_)
        
        self.available_statuses = {"Susceptible": 0, "Vaccinated": 1, "Infected": 2, "Recovered": 3}

    # ------------------------------------------------------------------
    # 2-1 初始化
    # ------------------------------------------------------------------
    def set_initial_status(
        self,
        frac_infected: float,
        frac_vaccinated: float,
        rng: np.random.Generator | None = None,
    ):
        """
        依給定比例隨機產生初始 S/I/R/V。

        Notes
        -----
        • 不重設 RNG 時，每次呼叫都用全域亂數。
        • R 在此假設為 0。
        """
        if rng is None:
            rng = np.random.default_rng()

        self.S[:] = True
        self.I[:] = False
        self.R[:] = False
        self.V[:] = False

        # print(self.nnodes, frac_vaccinated)

        # 隨機挑感染者
        n_inf = int(frac_infected * self.nnodes)
        if n_inf > 0:
            idx_inf = rng.choice(self.nnodes, n_inf, replace=False)
            self.I[idx_inf] = True
            self.S[idx_inf] = False
        else:
            n_inf = 1

        # 隨機挑接種者（不可重疊感染者）
        susceptible_pool = np.flatnonzero(self.S)
        n_vac_target = int(round(frac_vaccinated * self.nnodes))
        n_vac = min(n_vac_target, susceptible_pool.size)  # <‑‑ safety cap
        if n_vac > 0:
            pool = np.flatnonzero(self.S)       # 目前仍為 S
            idx_vac = rng.choice(pool, n_vac, replace=False)
            self.V[idx_vac] = True
            self.S[idx_vac] = False

    # ------------------------------------------------------------------
    # 2-2 主迴圈：跑到收斂或達上限
    # ------------------------------------------------------------------
    def run_until_eq(
        self,
        beta: float,
        eta: float,
        gamma: float,
        max_iter: int = 1000,
    ):
        """
        Returns
        -------
        counts : tuple[int]  (S, I, R, V 數量)
        n_iter : int         實際跑了幾步
        """
        for it in range(max_iter):
            done = sirv_step(
                self.S, self.I, self.R, self.V,
                self.indptr, self.indices,
                beta, eta, gamma
            )
            if done:
                break
        return (int(self.S.sum()), int(self.I.sum()),
                int(self.R.sum()), int(self.V.sum())), it + 1

    def iteration_bunch(
            self,
            max_iter: int = 1000,
            progress_bar: bool = False,
            *,
            beta: float,
            eta: float,
            gamma: float,
            keep_node_status: bool = False,
    ):
        """
        連續執行直到收斂或達 max_iter，回傳可供 build_trends() 使用的 list。

        Parameters
        ----------
        keep_node_status : bool
            True 時額外儲存 'status' (uint8 向量)，只在回放動畫才開。
        """
        out = []
        last_S = int(self.S.sum())
        last_I = int(self.I.sum())
        last_R = int(self.R.sum())
        last_V = int(self.V.sum())

        it_range = range(max_iter)
        if progress_bar:
            it_range = tqdm.tqdm(it_range)

        for it in it_range:
            done = sirv_step(
                self.S, self.I, self.R, self.V,
                self.indptr, self.indices,
                beta, eta, gamma
            )

            S_cnt = int(self.S.sum())
            I_cnt = int(self.I.sum())
            R_cnt = int(self.R.sum())
            V_cnt = int(self.V.sum())

            record = {
                "iteration": it,
                "node_count": {
                    0: S_cnt, 1: V_cnt, 2: I_cnt, 3: R_cnt
                },
                "status_delta": {
                    0: S_cnt - last_S,
                    1: V_cnt - last_V,
                    2: I_cnt - last_I,
                    3: R_cnt - last_R,
                }
            }

            if keep_node_status:               # 只有需要動畫時才開
                # 將四個布林向量打包成 uint8：0=S,1=V,2=I,3=R
                record["status"] = (
                    self.S.astype(np.uint8)*0 +
                    self.V.astype(np.uint8)*1 +
                    self.I.astype(np.uint8)*2 +
                    self.R.astype(np.uint8)*3
                )

            out.append(record)
            last_S, last_I, last_R, last_V = S_cnt, I_cnt, R_cnt, V_cnt

            if done:
                break

        return out



# ----------------------------------------------------------------------
# 3. 小示範：800×800 方格，跑到均衡
# ----------------------------------------------------------------------
if __name__ == "__main__":
    G = nx.grid_2d_graph(m=100, n=100)   
    row, col = zip(*G.edges())
    N = len(G)
    # csr = csr_matrix(
    #     (np.ones(len(row) * 2, np.bool_),
    #      (row + col, col + row)),
    #     shape=(N, N)
    # )
    csr = nx.to_scipy_sparse_array(G, dtype=bool, format="csr")
    indptr  = csr.indptr.astype("int32")
    indices = csr.indices.astype("int32")


    # 實例化
    model = FastSIRV(csr.indptr, csr.indices)

    # 參數
    # beta, eta, gamma = 0.3, 0.8, 0.1
    beta = 5/6 # transmission rate
    gamma = 1/3 # recovery rate
    eta = 0.5 # vaccine efficacy
    epsilon = 0.01 # fraction infected
    x = 0.5 # fraction vaccinated

    params = {
        "beta": beta, 
        "gamma": gamma, 
        "eta": eta, 
        "fraction_infected": epsilon, 
        "fraction_vaccinated": x
    }

    model.set_initial_status(epsilon, x)

    t0 = time.perf_counter()
    counts, n_iter = model.run_until_eq(beta, eta, gamma, max_iter=1000)
    t1 = time.perf_counter()

    print(f"收斂花 {n_iter} 步, 耗時 {t1 - t0:.3f} s")
    print("最後各狀態數量 (S, I, R, V) =", counts)

    iterations = model.iteration_bunch(
        max_iter=1000,
        progress_bar=False,
        beta=params["beta"],
        eta=params["eta"],
        gamma=params["gamma"],
        keep_node_status=False    # 只想畫趨勢曲線就設 False
    )

    trends = model.build_trends(iterations)   # 舊版 build_trends 不需改
    DiffusionTrend(model, trends).plot()
    plt.show()

