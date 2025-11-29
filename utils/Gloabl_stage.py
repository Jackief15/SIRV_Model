# global_stage.py
"""A lightweight wrapper that reproduces the behaviour of the old
`global_time_GPT()` **but** is written as a class and plugs a *FastSIRVCore*
instance inside.  Nothing about the graphical style is touched – you can still
feed the returned ``iterations`` list to the old visualisation pipeline.

Usage example
-------------
sirv = FastSIRV(indptr, indices)
gst  = GlobalStage(
        sirv,
        C=0.7, eta=0.8, beta=0.3, gamma=0.1,
        epsilon=0.01, x0=0.5, n0=0.5,
        max_seasons=1000)
x_seq, n_seq, iterations = gst.run(return_iterations=True)
# plot trend
import matplotlib.pyplot as plt
plt.plot(x_seq); plt.plot(n_seq)
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from FastSIRV import FastSIRV



class GlobalStage:
    """Repeated‑season (global‑time) dynamics around a *FastSIRVCore* kernel."""

    # ────────────────────────────────────────────────────────────────────
    # construction
    # ────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        sirv,                   # FastSIRVcore
        *,
        C: float,
        eta: float,
        beta: float,
        gamma: float,
        # global parameters
        Dr: float = -1.0,
        Dg: float = -1.0,
        kappa: float = 0.1,
        theta: float = 0.5,
        epsilon: float = 0.01,   # initial infected fraction
        x0: float = 0.5,         # initial vaccination
        n0: float = 0.5,         # initial social norm
        max_seasons: int = 1000,
        keep_iterations: bool = False,
    ) -> None:
        self.sirv = sirv
        
        # store local SIRV parameters in a dict for convenience
        self.local_params: Dict[str, float] = {
            "beta": beta,
            "gamma": gamma,
            "eta": eta,
        }
        
        # global parameters
        self.C, self.Dr, self.Dg = C, Dr, Dg
        self.kappa, self.theta   = kappa, theta
        self.epsilon             = epsilon
        
        # mutable global state
        self.x: float = x0
        self.n: float = n0
        self.max_seasons = max_seasons
        self.keep_iter = keep_iterations

        # storage – in the same shape you used before
        self.x_seq: List[float] = [x0]
        self.n_seq: List[float] = [n0]
        self.payoff_C_seq: List[float] = []
        self.payoff_D_seq: List[float] = []
        self.HV_seq:   List[float] = []
        self.IV_seq:   List[float] = []
        self.SFR_seq:  List[float] = []
        self.FFR_seq:  List[float] = []
        self.epi_size_seq: List[float] = []
        self.it_seq: List[List[dict]] = []  # if requested

    # ────────────────────────────────────────────────────────────────────
    # public API
    # ────────────────────────────────────────────────────────────────────
    def run(self, *, return_equi: bool = False):
        """Run season by season until `max_seasons` or equilibrium.

        Returns
        -------
        (x_seq, n_seq, iterations)  if *return_iterations* == True
        (x_seq, n_seq)              otherwise
        """
        print(self.x, self.n, self.C, self.local_params["eta"], self.local_params["beta"], self.local_params["gamma"])
        
        for season in range(self.max_seasons):
            it_list = self._single_season()
            if self.keep_iter:
                self.it_seq.append(it_list)
            if self._at_equilibrium():
                break

        # --- prepare outputs --------------------------------------
        if return_equi:
            last_pc = self.payoff_C_seq[-1]
            last_pd = self.payoff_D_seq[-1]
            last_epi = self.epi_size_seq[-1]
            return self.x_seq[-1], self.n_seq[-1], last_pc, last_pd, last_epi

        color_seq = np.linspace(0.0, 1.0, len(self.x_seq))
        return (
            np.asarray(self.x_seq),
            np.asarray(self.n_seq),
            color_seq,
            np.asarray(self.payoff_C_seq),
            np.asarray(self.payoff_D_seq),
            np.asarray(self.HV_seq),
            np.asarray(self.IV_seq),
            np.asarray(self.SFR_seq),
            np.asarray(self.FFR_seq),
            np.asarray(self.epi_size_seq),
            self.it_seq if self.keep_iter else [],
        )

    # ────────────────────────────────────────────────────────────────────
    # internal helpers
    # ────────────────────────────────────────────────────────────────────
    def _single_season(self) -> List[dict]:
        """Run Fast‑SIRV until local equilibrium, then update *(x, n)*."""
        # 1) initialise grid for this season
        self.sirv.set_initial_status(self.epsilon, self.x)
        # print(self.x, self.n)

        # 2) run to local equilibrium – iterations **not** stored by default
        iterations = self.sirv.iteration_bunch(
            max_iter=1000,
            beta=self.local_params["beta"],
            eta=self.local_params["eta"],
            gamma=self.local_params["gamma"],
            keep_node_status=False,
            progress_bar=False,
        )
        initial_counts = iterations[0]["node_count"]
        last_counts = iterations[-1]["node_count"]  # dict {0:S 1:V 2:I 3:R}
        N = self.sirv.nnodes * (1 - self.epsilon)
        HV  = last_counts[1] / N  # Healthy & Vaccinated
        IV  = (initial_counts[1] - last_counts[1]) / N # Infected & Vaccinated = V_0 - V_T
        SFR = last_counts[0] / N  # Healthy & Non-vaccinated
        FFR = (initial_counts[0] - last_counts[0]) / N  # Infected & Non-vaccinated = S_0 - S_T
        epi_size = (last_counts[3] - (self.sirv.nnodes * self.epsilon)) / N

        # 3) payoff & strategy update
        payoff_C = HV * (self.n - self.C * (1 - self.n)) + \
                   IV * (-self.Dr * self.n + (1 - self.n) * (-1 - self.C))
        payoff_D = SFR * (1 + self.Dg) * self.n - \
                   FFR * (1 - self.n)
        pd2c = 1 / (1 + np.exp(-(payoff_C - payoff_D) / self.kappa))
        pc2d = 1 / (1 + np.exp(-(payoff_D - payoff_C) / self.kappa))

        self.x += self.x * (1 - self.x) * (pd2c - pc2d) / 20.0
        self.n += self.n * (1 - self.n) * (-1 + (1 + self.theta) * self.x) / 20.0
        self.x = float(np.clip(self.x, 1e-5, 0.99999))
        self.n = float(np.clip(self.n, 1e-5, 0.99999))

        # 4) store history
        self.x_seq.append(self.x)
        self.n_seq.append(self.n)
        self.payoff_C_seq.append(payoff_C)
        self.payoff_D_seq.append(payoff_D)
        self.epi_size_seq.append(epi_size)

        return iterations

    def _at_equilibrium(self, tol=1e-5):
        if len(self.x_seq) < 2:
            return False
        dx = abs(self.x_seq[-1] - self.x_seq[-2])
        dn = abs(self.n_seq[-1] - self.n_seq[-2])
        return dx < tol and dn < tol

    # ────────────────────────────────────────────────────────────────────
    # convenience getters
    # ────────────────────────────────────────────────────────────────────
    @property
    def x_sequence(self):
        return np.asarray(self.x_seq)

    @property
    def n_sequence(self):
        return np.asarray(self.n_seq)

    @property
    def epi_size_sequence(self):
        return np.asarray(self.epi_size_seq)

    # simple trend plot
    def plot_trend(self, *, show=True, savefile: str | None = None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.x_seq, label="x (vaccinated)")
        ax.plot(self.n_seq, label="n (norm)")
        ax.set_xlabel("Season")
        ax.legend()
        fig.tight_layout()
        if savefile:
            fig.savefig(savefile, dpi=150)
        if show:
            plt.show()
        return fig
