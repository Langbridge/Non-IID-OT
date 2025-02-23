import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

def _eval_kld(x_0: np.array, x_1: np.array, n=500, bw_method='silverman'):
    """
    Compute KLD (x_0 || x_1) (from x_1 to x_0) 
    """
    support = np.linspace(np.min([np.min(x_0), np.min(x_1)]), np.max([np.max(x_0), np.max(x_1)]), n)

    kde_0 = gaussian_kde(x_0, bw_method=bw_method)
    pmf_0 = kde_0.evaluate(support)
    pmf_0 /= np.sum(pmf_0)

    kde_1 = gaussian_kde(x_1, bw_method=bw_method)
    pmf_1 = kde_1.evaluate(support)
    pmf_1 /= np.sum(pmf_1)

    return - np.sum(pmf_0 * np.log(pmf_1 / pmf_0))

def eval_damage(x: np.array, x_prime: np.array, n=500):
    return _eval_kld(x, x_prime, n)

def eval_invariance(x_0: np.array, x_1: np.array, n=500):
    return _eval_kld(x_0, x_1, n)

def eval_report(data: pd.DataFrame, repaired_data: pd.DataFrame):
    # data = df with features, u and s columns
    feats = data.drop(columns=['s', 'u']).columns

    report = {
        f: {'damage': {}, 'invariance': {}} for f in feats
    }

    for f in feats:

        for u_val in data['u'].unique():
            # data damage, \in {0, \inf}
            report[f]['damage'][u_val] = eval_damage(data[(data['u'] == u_val)][f].to_numpy(),
                                                     repaired_data[(data['u'] == u_val)][f].to_numpy())
            
            # s-invariance scaled according to original damage, \in {0,1}
            report[f]['invariance'][u_val] = eval_invariance(repaired_data[(repaired_data['s'] == 0) & (repaired_data['u'] == u_val)][f].to_numpy(),
                                                             repaired_data[(repaired_data['s'] != 0) & (repaired_data['u'] == u_val)][f].to_numpy())
            report[f]['invariance'][u_val] /= eval_invariance(data[(data['s'] == 0) & (data['u'] == u_val)][f].to_numpy(),
                                                             data[(data['s'] != 0) & (data['u'] == u_val)][f].to_numpy())
            
    return report