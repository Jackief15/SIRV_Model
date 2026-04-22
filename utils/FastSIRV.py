"""
Fast SIRV model on a 2-D grid (or any undirected graph in CSR form).
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


# 1. One-step update
@nb.njit(parallel=True, fastmath=True)
def sirv_step(S, I, R, V, indptr, indices, beta, eta, gamma):
    """
    Perform a single time-step update using Numba acceleration.

    Parameters
    ----------
    S, I, R, V : 1-D bool ndarray
        Boolean arrays representing node statuses.
    indptr, indices : 1-D int ndarray
        Two fields of the Compressed Sparse Row (CSR) adjacency matrix.
    beta, eta, gamma : float
        Transmission rate, vaccine efficacy, and recovery rate.

    Returns
    -------
    is_done : bool
        Returns True if the infection has cleared (I.sum() == 0).
    """
    n = S.size
    new_inf = np.zeros(n, np.bool_)
    p_rec = 1.0 - math.exp(-gamma)
    
    # Pre-calculate probabilities for specific degrees (e.g., grid nodes)
    # Mapping: idx=0 -> deg=2, etc.
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

    for u in nb.prange(n):
        if I[u]:
            deg_u = indptr[u+1] - indptr[u]     # Actual degree of node u
            
            # Convert ODE rates to per-step edge probabilities
            # Iterate through neighbors
            for k in range(indptr[u], indptr[u + 1]):
                v = indices[k]
                if S[v]:
                    if np.random.random() < p_edge_S[deg_u-2]:
                        new_inf[v] = True
                elif V[v]:
                    if np.random.random() < p_edge_V[deg_u-2]:
                        new_inf[v] = True
            
            # Check for self-recovery
            if np.random.random() < p_rec:
                I[u] = False
                R[u] = True

    # Batch update statuses
    S[new_inf] = False
    V[new_inf] = False
    I[new_inf] = True

    return I.sum() == 0                           # Convergence check



# 2. Class Wrapper: FastSIRV
class FastSIRV(DiffusionModel):
    """
    Fast SIRV implementation on a fixed graph using CSR format.

    Parameters
    ----------
    indptr, indices : 1-D int ndarray
        CSR adjacency obtained from scipy.sparse.csr_matrix.indptr / .indices
    """

    def __init__(self, indptr: np.ndarray, indices: np.ndarray):
        self.indptr = indptr.astype(np.int32)
        self.indices = indices.astype(np.int32)
        self.nnodes = indptr.size - 1

        # Pre-allocate status vectors for reuse across multiple seasons/simulations
        self.S = np.empty(self.nnodes, np.bool_)
        self.I = np.empty(self.nnodes, np.bool_)
        self.R = np.empty(self.nnodes, np.bool_)
        self.V = np.empty(self.nnodes, np.bool_)
        
        self.available_statuses = {"Susceptible": 0, "Vaccinated": 1, "Infected": 2, "Recovered": 3}

    # Initialization
    def set_initial_status(
        self,
        frac_infected: float,
        frac_vaccinated: float,
        rng: np.random.Generator | None = None,
    ):
        """
        Randomly initialize S/I/R/V statuses based on given fractions.

        Notes
        -----
        - Uses the global random generator if 'rng' is not provided.
        - Initial Recovered (R) count is assumed to be 0.
        """
        if rng is None:
            rng = np.random.default_rng()

        self.S[:] = True
        self.I[:] = False
        self.R[:] = False
        self.V[:] = False

        # print(self.nnodes, frac_vaccinated)

        # Randomly select vaccinated nodes (must not overlap with infected)
        n_inf = int(frac_infected * self.nnodes)
        if n_inf > 0:
            idx_inf = rng.choice(self.nnodes, n_inf, replace=False)
            self.I[idx_inf] = True
            self.S[idx_inf] = False
        else:
            n_inf = 1

        # Randomly select vaccinated nodes (must not overlap with infected)
        susceptible_pool = np.flatnonzero(self.S)
        n_vac_target = int(round(frac_vaccinated * self.nnodes))
        n_vac = min(n_vac_target, susceptible_pool.size)  # safety cap
        if n_vac > 0:
            pool = np.flatnonzero(self.S)                 # # Nodes still in S status
            idx_vac = rng.choice(pool, n_vac, replace=False)
            self.V[idx_vac] = True
            self.S[idx_vac] = False

    # Main Loop: Run until equilibrium or max iterations
    def run_until_eq(
        self,
        beta: float,
        eta: float,
        gamma: float,
        max_iter: int = 1000,
    ):
        """
        Run simulation until no infected nodes remain or max_iter is reached.

        Returns
        -------
        counts : tuple[int]  
            The final counts of (S, I, R, V).
        n_iter : int         
            Actual number of iterations performed.
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
        Run until convergence and return a list of records for build_trends().

        Parameters
        ----------
        keep_node_status : bool
            If True, saves the 'status' (uint8 vector) for each step. 
            Enable this only for generating playback animations.
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

            if keep_node_status:
                # Pack boolean vectors into uint8: 0=S, 1=V, 2=I, 3=R
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




# 3. Demo: 100x100 Grid run to equilibrium
if __name__ == "__main__":
    G = nx.grid_2d_graph(m=100, n=100)   
    row, col = zip(*G.edges())
    N = len(G)

    # Convert to CSR format for fast access
    csr = nx.to_scipy_sparse_array(G, dtype=bool, format="csr")
    indptr  = csr.indptr.astype("int32")
    indices = csr.indices.astype("int32")


    # Instantiate model
    model = FastSIRV(csr.indptr, csr.indices)

    # Parameters
    # beta, eta, gamma = 0.3, 0.8, 0.1
    beta = 5/6      # transmission rate
    gamma = 1/3     # recovery rate
    eta = 0.5       # vaccine efficacy
    epsilon = 0.01  # fraction infected
    x = 0.5         # fraction vaccinated

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

    print(f"Converged in {n_iter} steps, Time taken: {t1 - t0:.3f} s")
    print("Final status counts (S, I, R, V) =", counts)

    # Re-run to capture iteration data for plotting
    iterations = model.iteration_bunch(
        max_iter=1000,
        progress_bar=False,
        beta=params["beta"],
        eta=params["eta"],
        gamma=params["gamma"],
        keep_node_status=False                # Set to False if only trend curves are needed
    )

    # Visualization
    trends = model.build_trends(iterations)
    DiffusionTrend(model, trends).plot()
    plt.show()

