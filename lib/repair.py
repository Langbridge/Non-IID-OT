import numpy as np
import pandas as pd
from ot import dist, emd_1d

from lib.recursive_stopping import RecursiveStoppingRule

# import sys
# sys.path.insert(0, "/Users/al4518/Desktop/PhD/FPD-OT/FPD-OT/COT")
# from repair import StoppingRepair

class RecursiveStoppingRepair():
    # use vertices[1:-1] from current stopping rule as mu_0 and mu_1
    # use mu_0 and mu_1 to calculate the optimal map, T
    # coerce point as in StoppingRepair
        
    def __init__(self, rules: dict, u_vals=[0,1], centres=False):
        self.u_vals = u_vals
        self.centres = centres

        self.T = self.design_repair(rules)

    def _dist_approx(self, rules: dict):
        # test with cell centres, too??
        if self.centres:
            self.mu = {u: {s: self._find_centres(rules[u][s].curr_rule.verts[1:-1]) for s in [0,1]} for u in self.u_vals} # get centres from corresponding verts
        else:
            self.mu = {u: {s: rules[u][s].curr_rule.verts[1:-1] for s in [0,1]} for u in self.u_vals} # get vertices from corresponding stopping rule

        self.weights = {u: {s: rules[u][s].curr_rule.verts_weights[1:-1] / np.sum(rules[u][s].curr_rule.verts_weights[1:-1]) for s in [0,1]} for u in self.u_vals} # get weights for each vertex

    def _find_centres(self, verts):
        ub = verts[-1] + (verts[-1] - verts[-2])
        verts = np.insert(verts, len(verts), ub, axis=0)

        return 0.5*(verts[1:] + verts[:-1])

    def design_repair(self, rules: dict):
        self._dist_approx(rules)

        T = {u: emd_1d(self.mu[u][0], self.mu[u][1], self.weights[u][0], self.weights[u][1]) for u in self.u_vals} # accelerate solving with 1D EMD
        return T

    def repair_data(self, data: pd.DataFrame):
        assert len(data.columns) == 3, 'only one feature can be repaired at one time. data columns should be [feat, u, s]'
        assert np.all(data.columns[1:] == ['u', 's']), 'u and s features must be in data in this order [feat, u, s]'

        repaired_data = data.copy()
        for i,(x, u, s) in data.iterrows():
            repaired_data.loc[i, data.columns[0]] = self._repair_point(x, u, s)

        return repaired_data

    def _repair_point(self, x, u, s):

        idx = np.searchsorted(self.mu[u][s], x, side='left')
        if idx == 0 or idx == len(self.mu[u][s]):
            i = min(idx, len(self.mu[u][s])-1)
        else:
            interp = (x - self.mu[u][s][idx-1]) / np.diff(self.mu[u][s])[idx-1]
            if np.round(interp, 4) == 1.0:
                i = idx
            else:
                i = np.random.choice([idx-1, idx], p=[1-interp, interp]) # stochastic decision about which marginal entry to transport from

        if s:
            row = self.T[u][:,i]
        else:
            row = self.T[u][i,:]
            
        if np.sum(row) > 0.0:
            j = np.random.choice(self.T[u].shape[int(not(s))], p=(row / np.sum(row))) # stochastic choice of which marginal entry to transport to
        else:
            j = i
        return (0.5)*self.mu[u][int(s)][i] + (0.5)*self.mu[u][int(not(s))][j]