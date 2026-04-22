import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt

class GlobalStage:
    """Repeated‑season (global‑time) dynamics around a *FastSIRVCore* kernel."""

    # construction
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
        theta: float = 0.5,      # acts as baseline theta_0
        alpha: float = 0.5,      # sensitivity to disease prevalence (I_t)
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
        self.kappa = kappa
        self.theta_0 = theta     # store as baseline theta_0
        self.alpha = alpha       # store prevalence sensitivity
        self.epsilon = epsilon
        
        # mutable global state
        self.x: float = x0
        self.n: float = n0
        self.max_seasons = max_seasons
        self.keep_iter = keep_iterations

        # storage – in the same shape you used before
        self.x_seq: List[float] = [x0]
        self.n_seq: List[float] = [n0]
        self.theta_seq: List[float] = []    # Track dynamic theta
        self.payoff_C_seq: List[float] = []
        self.payoff_D_seq: List[float] = []
        self.HV_seq:   List[float] = []
        self.IV_seq:   List[float] = []
        self.SFR_seq:  List[float] = []
        self.FFR_seq:  List[float] = []
        self.epi_size_seq: List[float] = []
        self.it_seq: List[List[dict]] = []  # if requested

    # public API
    def run(self, *, return_equi: bool = False):
        """Run season by season until `max_seasons` or equilibrium."""
        print(self.x, self.n, self.C, self.local_params["eta"], self.local_params["beta"], self.local_params["gamma"])
        
        for season in range(self.max_seasons):
            it_list = self._single_season()
            if self.keep_iter:
                self.it_seq.append(it_list)
            if self._at_equilibrium():
                break

        # prepare outputs
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
            np.asarray(self.theta_seq)  # optionally return theta history
        )

    # internal helpers
    def _single_season(self) -> List[dict]:
        """Run Fast‑SIRV until local equilibrium, then update *(x, n)*."""
        # 1) initialise grid for this season
        self.sirv.set_initial_status(self.epsilon, self.x)

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
        HV  = last_counts[1] / N  
        IV  = (initial_counts[1] - last_counts[1]) / N 
        SFR = last_counts[0] / N  
        FFR = (initial_counts[0] - last_counts[0]) / N  
        epi_size = (last_counts[3] - (self.sirv.nnodes * self.epsilon)) / N

        # 3) payoff & strategy update
        payoff_C = HV * (self.n - self.C * (1 - self.n)) + \
                   IV * (-self.Dr * self.n + (1 - self.n) * (-1 - self.C))
        payoff_D = SFR * (1 + self.Dg) * self.n - \
                   FFR * (1 - self.n)
        
        pd2c = 1 / (1 + np.exp(-(payoff_C - payoff_D) / self.kappa))
        pc2d = 1 / (1 + np.exp(-(payoff_D - payoff_C) / self.kappa))

        # Calculate dynamic theta_t based on epidemic size and vaccine effectiveness
        # Formula: theta_t = theta_0 * (1 + alpha * I_t) * eta
        current_theta = self.theta_0 * (1 + self.alpha * epi_size) * self.local_params["eta"]

        self.x += self.x * (1 - self.x) * (pd2c - pc2d) / 20.0
        # Use current_theta instead of self.theta
        self.n += self.n * (1 - self.n) * (-1 + (1 + current_theta) * self.x) / 20.0 
        
        self.x = float(np.clip(self.x, 1e-5, 0.99999))
        self.n = float(np.clip(self.n, 1e-5, 0.99999))

        # 4) store history
        self.x_seq.append(self.x)
        self.n_seq.append(self.n)
        self.theta_seq.append(current_theta)  # Store current theta
        self.payoff_C_seq.append(payoff_C)
        self.payoff_D_seq.append(payoff_D)
        self.epi_size_seq.append(epi_size)

        return iterations

    def _at_equilibrium(self, tol=1e-5):
        if len(self.x_seq) < 2:
            return False
        dx = abs(self.x_seq[-1] - self.x_seq[-2])
        dn = abs(self.n_seq[-1] - self.n_seq[-2])
        # can also add a check for dtheta if needed, 
        # but dx and dn are usually sufficient to determine eco-evolutionary equilibrium.
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
        
    @property
    def theta_sequence(self):
        return np.asarray(self.theta_seq)

    # simple trend plot
    def plot_trend(self, *, show=True, savefile: str | None = None):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.x_seq, label="x (vaccinated)", color="blue")
        ax.plot(self.n_seq, label="n (norm)", color="green")
        
        # Add theta to the plot on a secondary y-axis to observe its dynamics
        ax2 = ax.twinx()
        # Since theta_seq is updated at the END of season 0, its length is len(x_seq)-1.
        # We can pad it with the initial theta to match lengths for plotting.
        initial_theta = self.theta_0 * (1 + self.alpha * 0) * self.local_params["eta"]
        full_theta_seq = [initial_theta] + self.theta_seq
        ax2.plot(full_theta_seq, label="θ (enhancement)", color="red", linestyle="--", alpha=0.5)
        ax2.set_ylabel("θ value")
        
        ax.set_xlabel("Season")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        fig.tight_layout()
        if savefile:
            fig.savefig(savefile, dpi=150)
        if show:
            plt.show()
        return fig