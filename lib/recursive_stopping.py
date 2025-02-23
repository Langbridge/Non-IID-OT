import sys
sys.path.insert(0, "/Users/al4518/Desktop/PhD/FPD-OT/FPD-OT")

import numpy as np

from COT.stopping import StoppingRule, WeightedStoppingRule

class RecursiveStoppingRule():
    def __init__(self, N_bar, x_min, x_max, epsilon, v_init, g_hat_init=0, g_hat_2_init=0, a_init=0.01, b_init=0.01, F_hat=None):
        self.curr_rule = StoppingRule(N_bar, x_min, x_max, epsilon, v_init, g_hat_init, g_hat_2_init, a_init, b_init, F_hat)

        self.N_bar = N_bar
        self.x_min = x_min
        self.x_max = x_max
        self.epsilon = epsilon

        self.v_init = v_init # small constant (prior knowledge, should be << 1)
        self.g_hat_init = g_hat_init
        self.g_hat_2_init = g_hat_2_init
        self.a_init = a_init
        self.b_init = b_init
        self.F_hat_init = F_hat

        self.rules = []

    def __str__(self):
        return f"Recursive Stopping Rule with range [{self.x_min}, {self.x_max}], epsilon = {self.epsilon}, nu_init = {self.v_init}"
    
    def __repr__(self):
        return f"RecursiveStoppingRule({self.N_bar}, {self.x_min}, {self.x_max}, {self.epsilon}, {self.v_init})"

    def sample(self, x):
        return self.curr_rule.sample(x)

    def increment_t(self, l):
        # set v, h_hat, a, b
        v = l * self.curr_rule.v + (1-l) * self.v_init
        g_hat = 1 / v * ( l * self.curr_rule.v * self.curr_rule.g_hat + (1-l) * self.v_init * self.g_hat_init )
        g_hat_2 = 1 / v * ( l * self.curr_rule.v * self.curr_rule.g_hat_2 + (1-l) * self.v_init * self.g_hat_2_init )
        a = self.curr_rule.a
        b = self.curr_rule.b

        # decimate vertices from previous rule as start for next rule
        new_rule = StoppingRule(self.N_bar, self.x_min, self.x_max, self.epsilon, v, g_hat, g_hat_2, a, b, self.F_hat_init)
        new_rule.verts = np.hstack([self.x_min,
                                    np.sort(np.random.choice(self.curr_rule.verts[1:-1], int(np.floor(l * len(self.curr_rule.verts))), replace=False)),
                                    self.x_max])

        # redirect sample to current rule
        self.rules.append(self.curr_rule)
        self.curr_rule = new_rule