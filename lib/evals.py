import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

def __eval_kld(x_0: np.array, x_1: np.array, n=500, bw_method='silverman'):
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

def eval_damage(x: np.array, x_prime: np.array, n=500, method='KLD', supp=None):
    assert method in ['KLD', 'TV', 'Wasserstein'], 'Invalid Method.'

    if method == 'KLD':
        metric = _eval_kld
    elif method == 'TV':
        metric = _eval_TV
    else: raise NotImplementedError('Wasserstein distance has not yet been implemented.')
    
    return metric(x, x_prime, n, supp)

def eval_invariance(x_0: np.array, x_1: np.array, n=500, method='KLD', supp=None):
    assert method in ['KLD', 'TV', 'Wasserstein'], 'Invalid Method.'

    if method == 'KLD':
        metric = _eval_kld
    elif method == 'TV':
        metric = _eval_TV

    return metric(x_0, x_1, n, supp)

def eval_report(data: pd.DataFrame, repaired_data: pd.DataFrame, n=500, method='KLD'):
    # data = df with features, u and s columns
    feats = data.drop(columns=['s', 'u']).columns

    report = {
        f: {'damage': {}, 'invariance': {}, 'original_invariance': {}} for f in feats
    }

    for f in feats:

        supp = [np.min(np.hstack([data[f], repaired_data[f]])), np.max(np.hstack([data[f], repaired_data[f]]))]

        for u_val in data['u'].unique():
            # data damage, \in {0, \inf}
            report[f]['damage'][u_val] = eval_damage(data[(data['u'] == u_val)][f].to_numpy(),
                                                     repaired_data[(data['u'] == u_val)][f].to_numpy(),
                                                     n=n, method=method, supp=supp)
            
            # s-invariance scaled according to original damage, \in {0,1}
            report[f]['invariance'][u_val] = eval_invariance(repaired_data[(repaired_data['s'] == 0) & (repaired_data['u'] == u_val)][f].to_numpy(),
                                                             repaired_data[(repaired_data['s'] != 0) & (repaired_data['u'] == u_val)][f].to_numpy(),
                                                             n=n, method=method, supp=supp)
            report[f]['original_invariance'][u_val] = eval_invariance(data[(data['s'] == 0) & (data['u'] == u_val)][f].to_numpy(),
                                                                      data[(data['s'] != 0) & (data['u'] == u_val)][f].to_numpy(),
                                                                      n=n, method=method, supp=supp)
            
    return report

def _eval_TV(x_0: np.array, x_1: np.array, n=500, supp=None):
    """
    Compute TV (x_0, x_1)
    """
    if supp:
        bins = np.linspace(supp[0], supp[1], num=n)
    else:
        bins = np.linspace(np.min(np.hstack([x_0, x_1])), np.max(np.hstack([x_0, x_1])), num=n)

    p, _ = np.histogram(x_0, bins, density=True)
    q, _ = np.histogram(x_1, bins, density=True)

    return 0.5 * np.sum(np.abs(p - q))

def _eval_kld(x_0: np.array, x_1: np.array, n=500, supp=None, bw_method=None):
    """
    Compute KLD (x_0 || x_1) (from x_1 to x_0) 
    """
    if supp:
        bins = np.linspace(supp[0], supp[1], num=n)
    else:
        bins = np.linspace(np.min(np.hstack([x_0, x_1])), np.max(np.hstack([x_0, x_1])), num=n)

    p, _ = np.histogram(x_0, bins, density=True)
    q, _ = np.histogram(x_1, bins, density=True)


    return - np.sum(p * np.log(q / p))