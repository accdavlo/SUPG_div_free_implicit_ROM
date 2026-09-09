"""Deferred Correction (DeC) time-quadrature coefficients"""

import numpy as np

from ..quadr import lagrange_basis, nodes_weights


class DeC:
    """Deferred Correction (DeC) temporal quadrature coefficients.

    Attributes
    ----------
    theta:
        Integration coefficients mapping sub-step residuals to each node.
    beta:
        Sub-node positions in the normalized time interval [0, 1].
    """

    def __init__(self, M_sub, n_iter, nodes_type):
        self.n_subNodes = M_sub + 1
        self.M_sub = M_sub
        self.n_iter = n_iter
        self.nodes_type = nodes_type
        self.compute_theta_DeC()
        self.name = "DeC_" + self.nodes_type
    
    def compute_theta_DeC(self):
        nodes, w = nodes_weights(self.n_subNodes,self.nodes_type)
        int_nodes, int_w = nodes_weights(self.n_subNodes,"gaussLobatto")
        # generate theta coefficients 
        self.theta = np.zeros((self.n_subNodes,self.n_subNodes))
        self.beta = np.zeros(self.n_subNodes)
        for m in range(self.n_subNodes):
            self.beta[m] = nodes[m]
            nodes_m = int_nodes*(nodes[m])
            w_m = int_w*(nodes[m])
            for r in range(self.n_subNodes):
                self.theta[m,r] = np.sum(lagrange_basis(nodes,nodes_m,r)*w_m)
        return self.theta, self.beta
    
