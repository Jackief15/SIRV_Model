import itertools, os, sys, traceback
from typing import Sequence, Tuple, List, Iterable

import numpy as np
import pandas as pd
import networkx as nx
from joblib import Parallel, delayed

from utils.FastSIRV import FastSIRV
from utils.newGlobalStage import GlobalStage


# Cache for CSR format: key (M, N) → (indptr, indices)
_CSR_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

def get_sirv_core(M: int = 100, N: int = 100) -> FastSIRV:
    """
    Returns an instance of FastSIRV for a grid graph. 
    Uses a cache to avoid redundant graph generation and CSR conversion.
    """
    key = (M, N)
    if key not in _CSR_CACHE:
        G   = nx.grid_2d_graph(M, N)
        csr = nx.to_scipy_sparse_array(G, format='csr', dtype=bool)
        _CSR_CACHE[key] = (csr.indptr.astype(np.int32),
                           csr.indices.astype(np.int32))
    indptr, indices = _CSR_CACHE[key]
    return FastSIRV(indptr, indices)


# Processing Function for Single Parameter Set
def _process_one(
    C: float, eta: float, beta: float, gamma: float, 
    theta: float = 0.5, 
    Dr = -1, Dg = -1,
    M = 100, N = 100,
    epsilon: float = 0.01,
    x0 = 0.5, n0 = 0.5
) -> List[float]:
    """
    Simulates the vaccination dynamics for a single set of parameters (C, η, β, γ, θ).
    """
    try:
        sirv = get_sirv_core(M, N)
        gst  = GlobalStage(
            sirv,
            C=C, eta=eta, beta=beta, gamma=gamma,
            theta=theta,
            alpha=0,                   # Robustness check parameter
            epsilon=epsilon,
            Dr=Dr, Dg=Dg,
            x0=x0, n0=n0,
            max_seasons=2000,
            keep_iterations=False,
        )
        # Obtain equilibrium states: fraction vaccinated, cognitive environment, payoffs, and epidemic size
        x_eq, n_eq, pC, pD, epi = gst.run(return_equi=True)
        return [x_eq, n_eq, pC, pD, epi, C, eta, beta, gamma, theta, x0, n0]

    except Exception as e:
        print(f"[ERR] C={C} eta={eta} beta={beta} gamma={gamma}"
              f"theta={theta}: {e}", file=sys.stderr)
        traceback.print_exc()
        return [np.nan]*5 + [C, eta, beta, gamma, theta]


# High-level Functions for Large-scale Sweeps
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
    """
    Performs a parallel parameter sweep over (C, η, β, γ) combinations crossed with θ.

    Parameters
    ----------
    theta_values : iterable of theta to test, e.g., [0.1, 0.2, 0.3]
    grid_step    : step size for C, η, β, γ (default 0.1 generates a 10^4 grid)
    n_jobs       : number of worker processes (defaults to CPU cores - 1)
    outfile      : path to save the resulting CSV; returns DataFrame if None
    """
    # Construct parameter grid for (C, η, β, γ)
    vals = np.arange(grid_step, 1.0 + 1e-8, grid_step)
    base_grid = list(itertools.product(vals, vals, vals, vals))

    tasks: List[Tuple[float,...]] = [
        (C, eta, beta, gamma, th, Dr, Dg, M, N, epsilon)
        for (C, eta, beta, gamma) in base_grid
        for th in theta_values
    ]

    # Parallel execution settings
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

    # Classify the Game Type based on Dilemma Strength (Dr, Dg)
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
    """
    Sweep over initial conditions (x0, n0) across various θ values while keeping other parameters fixed.
    """
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

def run_sweep_flex(
    theta_values      : Sequence[float],
    *,
    # -- Optional: Direct provision of param_grid ------------------
    param_grid        : Iterable[Tuple[float,float,float,float]] | None = None,
    # -- Or specify values per-parameter ---------------------------
    C_vals            : Iterable[float] | None = None,
    eta_vals          : Iterable[float] | None = None,
    beta_vals         : Iterable[float] | None = None,
    gamma_vals        : Iterable[float] | None = None,
    # -- Shared hyperparameters ------------------------------------
    Dr: float = 0.5,
    Dg: float = 0.5,
    M : int   = 100,
    N : int   = 100,
    epsilon: float = 0.001,
    # -- Fallback: grid_step used only if the above are omitted ---
    grid_step: float = 0.1,
    # -- Parallel settings and output ------------------------------
    n_jobs : int | None = None,
    outfile: str | None = None,
) -> pd.DataFrame:
    """
    Flexible parameter sweep over (C, η, β, γ) × θ.

    Precedence Order:
      1. If param_grid is provided -> used directly.
      2. Otherwise, check individual *_vals; missing ones are filled using grid_step.
      3. If no specific values are provided -> all four parameters are swept using grid_step.
    """

    # ---------- Construct base_grid ----------------------------
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

    # ---------- Assemble all tasks -----------------------------
    tasks = [
        (C, eta, beta, gamma, th, Dr, Dg, M, N, epsilon)
        for (C, eta, beta, gamma) in base_grid
        for th                    in theta_values
    ]

    # ---------- Execute Parallel Processing --------------------
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
