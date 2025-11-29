import future.utils
import numpy as np

from DiffusionModel import DiffusionModel

__author__ = "Giulio Rossetti"
__license__ = "BSD-2-Clause"
__email__ = "giulio.rossetti@gmail.com"

class SIRVModel(DiffusionModel):
    """
    Model Parameters to be specified via ModelConfig

    :param beta: The infection rate (float value in [0,1])
    :param gamma: The recovery rate (float value in [0,1])
    """

    def __init__(self, graph, seed=None):
        """
        Model Constructor

        :param graph: A networkx graph object
        """
        super(self.__class__, self).__init__(graph, seed)
        self.available_statuses = {"Susceptible": 0, "Vaccinated": 1, "Infected": 2, "Recovered": 3}
        self.nnodes = self.graph.number_of_nodes()

        self.parameters = {
            "model": {
                "beta": {"descr": "Infection rate", "range": [0, 1], "optional": False},
                "gamma": {"descr": "Recovery rate", "range": [0, 1], "optional": False},
                "tp_rate": {
                    "descr": "Whether if the infection rate depends on the number of infected neighbors",
                    "range": [0, 1],
                    "optional": True,
                    "default": 1,
                },
            },
            "nodes": {},
            "edges": {},
        }

        self.active = []
        self.name = "SIRV"
        

    def iteration(self, node_status=True):
        """
        Execute a single model iteration

        :return: Iteration_id, Incremental node status (dictionary node->status)
        """
        self.clean_initial_status(self.available_statuses.values())

        actual_status = {
            node: nstatus for node, nstatus in future.utils.iteritems(self.status)
        }
        # ====================
        # Infected / Recovered nodes
        # ====================
        self.active = [
            node
            for node, nstatus in future.utils.iteritems(self.status)
            if nstatus > self.available_statuses["Vaccinated"]
        ]

        if self.actual_iteration == 0:
            self.actual_iteration += 1
            delta, node_count, status_delta = self.status_delta(actual_status)
            if node_status:
                return {
                    "iteration": 0,
                    "status": actual_status.copy(),
                    "node_count": node_count.copy(),
                    "status_delta": status_delta.copy(),
                }
            else:
                return {
                    "iteration": 0,
                    "status": {},
                    "node_count": node_count.copy(),
                    "status_delta": status_delta.copy(),
                }

        for u in self.active:

            u_status = self.status[u]

            # infected nodes
            if u_status == 2:

                # susceptible neighbors
                if self.graph.directed:
                    susceptible_neighbors = [
                        v for v in self.graph.successors(u) if self.status[v] == 0
                    ]
                else:
                    susceptible_neighbors = [
                        v for v in self.graph.neighbors(u) if self.status[v] == 0
                    ]
                for v in susceptible_neighbors:
                    eventp = np.random.random_sample()
                    if eventp < self.params["model"]["beta"]:
                        actual_status[v] = 2
                        # print("susceptible", eventp, '<', self.params["model"]["beta"])

                # ====================
                # vaccinated neighbors
                # ====================
                vaccinated_neighbors = [
                    v for v in self.graph.neighbors(u) if self.status[v] == 1
                ]
                for v in vaccinated_neighbors:
                    eventp = np.random.random_sample()
                    if eventp < self.params["model"]["beta"] * (1 - self.params["model"]["eta"]):
                        actual_status[v] = 2
                        # print("vaccinated", eventp, '<', self.params["model"]["beta"] * (1 - self.params["model"]["eta"]))
                
                eventp = np.random.random_sample()
                if eventp < self.params["model"]["gamma"]:
                    actual_status[u] = 3

        delta, node_count, status_delta = self.status_delta(actual_status)
        self.status = actual_status
        self.actual_iteration += 1

        if node_status:
            return {
                "iteration": self.actual_iteration - 1,
                "status": delta.copy(),
                "node_count": node_count.copy(),
                "status_delta": status_delta.copy(),
            }
        else:
            return {
                "iteration": self.actual_iteration - 1,
                "status": {},
                "node_count": node_count.copy(),
                "status_delta": status_delta.copy(),
            }