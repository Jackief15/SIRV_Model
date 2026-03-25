# sweep.py ──────────────────────────────────────────────────────────
import itertools, os, sys, traceback
from typing import Sequence, Tuple, List, Iterable

import numpy as np
import pandas as pd
import networkx as nx
from joblib import Parallel, delayed

from utils.FastSIRV import FastSIRV
# from utils.Gloabl_stage import GlobalStage
from utils.newGlobalStage import GlobalStage


# key: (M, N) → (indptr, indices)
_CSR_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

def get_sirv_core(M: int = 100, N: int = 100) -> FastSIRV:
    key = (M, N)
    if key not in _CSR_CACHE:
        G   = nx.grid_2d_graph(M, N)
        csr = nx.to_scipy_sparse_array(G, format='csr', dtype=bool)
        _CSR_CACHE[key] = (csr.indptr.astype(np.int32),
                           csr.indices.astype(np.int32))
    indptr, indices = _CSR_CACHE[key]
    return FastSIRV(indptr, indices)


# 2) ────────── 單一組 (C,η,β,γ,θ) 的處理函式 ────────────────────
def _process_one(
    C: float, eta: float, beta: float, gamma: float, 
    theta: float = 0.5, 
    Dr = -1, Dg = -1,
    M = 100, N = 100,
    epsilon: float = 0.01,
    x0 = 0.5, n0 = 0.5
) -> List[float]:
    try:
        sirv = get_sirv_core(M, N)
        gst  = GlobalStage(
            sirv,
            C=C, eta=eta, beta=beta, gamma=gamma,
            theta=theta,                     # ← 可變參數
            alpha=10,                       # Robustness check
            epsilon=epsilon,
            Dr=Dr, Dg=Dg,
            x0=x0, n0=n0,
            max_seasons=2000,
            keep_iterations=False,
        )
        x_eq, n_eq, pC, pD, epi = gst.run(return_equi=True)
        return [x_eq, n_eq, pC, pD, epi, C, eta, beta, gamma, theta, x0, n0]

    except Exception as e:
        print(f"[ERR] C={C} eta={eta} beta={beta} gamma={gamma}"
              f"theta={theta}: {e}", file=sys.stderr)
        traceback.print_exc()
        return [np.nan]*5 + [C, eta, beta, gamma, theta]


# 3) ─────────── 外部呼叫的高階函式 ────────────────────────────────
def run_sweep(
    theta_values: Sequence[float],
    Dr: float = -1,
    Dg: float = -1,
    M: int = 100,
    N: int = 100,
    epsilon: float = 0.01,
    *,
    grid_step: float = 0.1,
    n_jobs: int | None = None,
    outfile: str | None = None
) -> pd.DataFrame:
    """平行掃描 10 000 個 (C,η,β,γ) × |θ|。

    Parameters
    ----------
    theta_values : iterable of theta you want to test, e.g. [0.1, 0.2, 0.3]
    grid_step    : step size for C,η,β,γ (default 0.1 → 0.1..1.0)
    n_jobs       : how many worker processes (default = cores-1)
    outfile      : csv path to save; if None, just return DataFrame
    """
    # param grid for C,η,β,γ
    vals = np.arange(grid_step, 1.0 + 1e-8, grid_step)
    base_grid = list(itertools.product(vals, vals, vals, vals))

    tasks: List[Tuple[float,...]] = [
        (C, eta, beta, gamma, th, Dr, Dg, M, N, epsilon)
        for (C, eta, beta, gamma) in base_grid
        for th in theta_values
    ]

    # parallel settings
    if n_jobs is None:
        n_jobs = max(1, os.cpu_count() - 1)
    os.environ['OMP_NUM_THREADS'] = '1'

    results = Parallel(
        n_jobs=n_jobs,
        backend='loky',
        mmap_mode='r',
        # verbose=10
    )(
        delayed(_process_one)(C, eta, beta, gamma, th, Dr, Dg, M, N, epsilon)
        for (C, eta, beta, gamma, th, Dr, Dg, M, N, epsilon) in tasks
    )

    df = pd.DataFrame(results, columns=[
        'x', 'n', 'payoff_C', 'payoff_D', 'epi_size',
        'C', 'eta', 'beta', 'gamma', 'theta', 'x0', 'n0'
    ])

    df['L'] = M
    df['Dr'] = Dr
    df['Dg'] = Dg
    if Dr < 0:
        if Dg < 0:
            df['Game'] = 'H'
        else:
            df['Game'] = 'CH'
    else:
        if Dg > 0:
            df['Game'] = 'PD'
        else:
            df['Game'] = 'SH'

    if outfile:
        df.to_csv(outfile, index=False)
    return df

def run_init_sweep(
    theta_values: Sequence[float],
    *,
    C: float,
    eta: float,
    beta: float,
    gamma: float,
    Dr: float = -1,
    Dg: float = -1,
    M: int = 100,
    N: int = 100,
    epsilon: float = 0.01,
    init_step: float = 0.1,
    n_jobs: int | None = None
) -> pd.DataFrame:
    """掃 (x0, n0)×θ；其他參數固定。"""
    vals = np.arange(init_step, 1.0 + 1e-8, init_step)
    init_grid = list(itertools.product(vals, vals))

    tasks = [
        (C, eta, beta, gamma, th, Dr, Dg, M, N, epsilon, x0, n0)
        for (x0, n0) in init_grid
        for th        in theta_values
    ]

    if n_jobs is None:
        n_jobs = max(1, os.cpu_count() - 1)
    os.environ['OMP_NUM_THREADS'] = '1'

    results = Parallel(n_jobs=n_jobs, backend='loky')(delayed(_process_one)(*t)
                                                     for t in tasks)

    cols = ['x', 'n', 'payoff_C', 'payoff_D', 'epi_size',
            'C', 'eta', 'beta', 'gamma', 'theta', 'x0', 'n0']
    
    df = pd.DataFrame(results, columns=cols)

    df['L'] = M
    df['Dr'] = Dr
    df['Dg'] = Dg
    if Dr < 0:
        if Dg < 0:
            df['Game'] = 'H'
        else:
            df['Game'] = 'CH'
    else:
        if Dg > 0:
            df['Game'] = 'PD'
        else:
            df['Game'] = 'SH'

    return df

# ──────────────────────────────────────────────────────────────
def run_sweep_flex(
    theta_values      : Sequence[float],
    *,
    # ── 可選：直接給 param_grid ─────────────────────────────
    param_grid        : Iterable[Tuple[float,float,float,float]] | None = None,
    # ── 或逐參數指定 ────────────────────────────────────
    C_vals            : Iterable[float] | None = None,
    eta_vals          : Iterable[float] | None = None,
    beta_vals         : Iterable[float] | None = None,
    gamma_vals        : Iterable[float] | None = None,
    # ── 共用超參數 ────────────────────────────────────
    Dr: float = 0.5,
    Dg: float = 0.5,
    M : int   = 100,
    N : int   = 100,
    epsilon: float = 0.001,
    # ── 備用：若上述都沒給，才用 grid_step ───────────────
    grid_step: float = 0.1,
    # ── 併行設定與輸出 ─────────────────────────────────
    n_jobs : int | None = None,
    outfile: str | None = None,
) -> pd.DataFrame:
    """
    Flexible parameter sweep over (C, η, β, γ) × θ.

    優先順序：
      1. param_grid 若非 None → 直接使用
      2. 否則檢查各 *_vals，未指定者以 grid_step 補足
      3. 都沒指定 → 四參數均用 grid_step 等距掃描

    其餘引數沿用舊版 run_sweep 行為。
    """

    # ---------- 1. 構造 base_grid ----------------------------------
    if param_grid is not None:
        base_grid = list(param_grid)
    else:
        def _default_vals():
            return np.round(np.arange(grid_step, 1.0 + 1e-8, grid_step), 3)

        C_vals     = list(C_vals)     if C_vals  is not None else _default_vals()
        eta_vals   = list(eta_vals)   if eta_vals is not None else _default_vals()
        beta_vals  = list(beta_vals)  if beta_vals is not None else _default_vals()
        gamma_vals = list(gamma_vals) if gamma_vals is not None else _default_vals()

        base_grid = list(itertools.product(C_vals, eta_vals, beta_vals, gamma_vals))

    # ---------- 2. 組合所有任務 -----------------------------------
    tasks = [
        (C, eta, beta, gamma, th, Dr, Dg, M, N, epsilon)
        for (C, eta, beta, gamma) in base_grid
        for th                    in theta_values
    ]

    # ---------- 3. 設定併行 ---------------------------------------
    if n_jobs is None:
        n_jobs = max(1, os.cpu_count() - 1)
    os.environ['OMP_NUM_THREADS'] = '1'

    results = Parallel(n_jobs=n_jobs, backend="loky", mmap_mode="r")(
        delayed(_process_one)(*t) for t in tasks
    )

    df = pd.DataFrame(results, columns=[
        'x', 'n', 'payoff_C', 'payoff_D', 'epi_size',
        'C', 'eta', 'beta', 'gamma', 'theta', 'x0', 'n0'
    ])

    df['L'] = M
    df['Dr'] = Dr
    df['Dg'] = Dg
    if Dr < 0:
        if Dg < 0:
            df['Game'] = 'H'
        else:
            df['Game'] = 'CH'
    else:
        if Dg > 0:
            df['Game'] = 'PD'
        else:
            df['Game'] = 'SH'

    if outfile:
        df.to_csv(outfile, index=False)
    return df

# # ────── small demo ──────
# if __name__ == "__main__":          # python sweep.py 會跑這段 demo
#     thetas = [0.1, 0.3, 0.5]
#     df_out = run_sweep(thetas, grid_step=0.2, n_jobs=2,
#                        outfile="demo_sweep.csv")
#     print(df_out.head())
