import numpy as np

from kde import *

from utils import TIMER

##

histogram_kde_ = lambda x, x_range=None, n_bins=60, param=300, periodic=True, weights=None, verbose=True :\
                 KDE(x, periodic=periodic, x_range=x_range, n_bins=n_bins, param=param, weights=weights, verbose=verbose).histogram

def histogram_np_(x, x_range = None, n_bins=60, param=300, periodic=True, weights=None, verbose=True):
    dim = x.shape[1]
    if dim > 1 and type(n_bins) is not list: n_bins = [n_bins]*dim
    else: pass
    hist = np.histogramdd(x, range=x_range, bins=n_bins, weights=weights)[0].astype(float)
    return hist / hist.sum()

def get_histogram_axis_(x_range : list, n_bins : int):
    """ For plotting.
    """
    Min, Max = x_range
    half_bin_width = 0.5*(Max-Min)/n_bins
    grid = np.linspace(Min + half_bin_width, Max - half_bin_width, n_bins)
    return grid

def get_histograms_(X_list : np.ndarray or list,
                    periodic : bool or list = True,
                    x_range : list = [-np.pi,np.pi],
                    n_bins : int or list = 60,
                    use_kde : bool = True,
                    kde_parameter : float = 200.0,
                    joint : bool = True,
                    flattened_outputs : bool = True,
                    return_axes : bool = False,
                    verbose : bool = True):
    """
    Inputs:
        X_list : (N,m,d) shapes array (X), or a list of such arrays.
            N = # frames
            m = # molecules
            d = # dimensions (>=1)
        periodic : bool, or list/array (mask) of ones and zeros (ones = True). [for kde]
        x_range : list or list of lists for ranges of marginals.
        n_bins : int or list for # bins for marginals.
        use_kde : bool
        kde_parameter : larger number less smooth (adjust as heuristic).
        joint : bool : whether to compute joint histogram (for each X[i]) of just marginals.
        flattened_outputs : bool : True, unless visualising.
        return_axes : bool : whether to also return marginal axes for plotting.
        verbose : bool : whether to show loading bar.
    """
    if use_kde: histogram_ = histogram_kde_
    else: histogram_ = histogram_np_

    if type(X_list) is list: pass
    else: X_list = [X_list]
        
    n_trajs = len(X_list)
    n_features = X_list[0].shape[-1] # = dim.
    
    Ns = []
    for i in range(n_trajs): Ns.append(X_list[i].shape[0])
    Ns_sum = sum(Ns)
    
    if verbose: timer = TIMER(n_trajs)
    else: pass

    if len(np.array(x_range).shape) == 1: x_ranges = [x_range]*n_features
    else:                                 x_ranges = x_range
    if type(n_bins) not in [int, float]: pass
    else: n_bins = [n_bins]*n_features

    if joint:
        X_hists = np.zeros([Ns_sum, 1] + n_bins) # (N.., 1, n_bins, n_bins, ...)
        for i in range(n_trajs):
            for j in range(Ns[i]):
                a = sum(Ns[:i]) + j
                
                x = X_list[i][j] # X_list[i] ~ (N,m,n_torsions), X_list[i][j] ~ (m,n_torsions)
                X_hists[a,0,:] = histogram_(x, x_range = x_ranges, n_bins = n_bins, param = kde_parameter, periodic = periodic)

            if verbose: timer.check_(i)
            else: pass

        if flattened_outputs: X_hists = X_hists.reshape([Ns_sum, 1, np.product(n_bins)])  # (N.., 1, n_bins^dim)
        else: pass
    
    else:
        X_hists = np.zeros([Ns_sum, n_features, max(n_bins)]) # (N.., n_features, n_bins)
        for i in range(n_trajs):
            for j in range(Ns[i]):
                a = sum(Ns[:i]) + j
        
                x = X_list[i][j]
                for k in range(n_features):
                    X_hists[a,k,:n_bins[k]] = histogram_(x[:,k:k+1], x_range = [x_ranges[k]], n_bins = n_bins[k], param = kde_parameter, periodic = periodic)

            if verbose: timer.check_(i)
            else: pass

    if return_axes:
        axes = [get_histogram_axis_(x_ranges[i], n_bins[i]) for i in range(n_features)]
        return X_hists, axes
    else:
        return X_hists

def get_torsion_histograms_(X_list : np.ndarray or list,
                            n_bins : int = 60,
                            use_kde : bool = True,
                            kde_parameter : float = 200.0,
                            joint : bool = True,
                            flattened_outputs : bool = True,
                            return_axes : bool = False, 
                            verbose : bool = True):
    """ get_histograms_ for intra-molecular torsions only.
    """
    return get_histograms_(X_list = X_list,
                           periodic= True, x_range = [-np.pi,np.pi],
                           n_bins = n_bins,
                           use_kde = use_kde,
                           kde_parameter = kde_parameter,
                           joint = joint,
                           flattened_outputs = flattened_outputs,
                           return_axes = return_axes, 
                           verbose = verbose)

