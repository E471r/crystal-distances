import numpy as np
from utils_jit import cdist_ # not needed.
from histograms import histogram_kde_, histogram_np_
from utils import clamp_list_
##

def get_radial_distributions_traj_(R : np.ndarray,     # xyz.coordinates[i]
                                   min_box_length,     # min(xyz.min_box_lengths)
                                   boxes = None,       # xyz.boxes[i]
                                   min_distance = 1.0,
                                   masses = None,      # xyz.masses
                                   n_bins = 60,
                                   use_kde = False,
                                   kde_parameter = 700.0,
                                   ):
    """
    Inputs: [these come directly from XYZ object when data is imported] 
        R : (N,m,n,3) shaped array (not list) of Cartesian coordinates of trajectory.
            N = # frames
            m = # molecules
            n = # atoms
        boxes : (N,3,3) shaped array of Supercell box vectors (rows) in each frame.
        min_box_length : float : length of shortest box vector in whole dataset.
            right edge of histogram axis.
        masses = (n,) shaped list or array of atomic masses.

    Args:
        min_distance : float : parameter, default is 1 (Angstrom).
            left edge of histogram axis.
        n_bins : int : # histogram bins.
        use_kde : bool.
        kde_parameter : float : larger number less smooth (for 1D default here is 700, a heuristic).

    Outputs:
        hists : (N,1,n_bins) shaped array of radial distribution histograms.
    """

    if use_kde: histogram_ = histogram_kde_
    else: histogram_ = histogram_np_
    
    N, m, n = R.shape[:3]
    
    if masses is not None: ws = np.array(masses).reshape(1,1,n,1) / np.sum(masses)
    else:                  ws = np.ones([1,1,n,1]) / n
    COMs = (ws*R).sum(2) # (N,m,3)
    
    half_min_box_length = 0.5 * min_box_length
    x_range = [min_distance, half_min_box_length] # ax = get_histogram_axis_(x_range, n_bins)
    hists = np.zeros([N,1,n_bins]) # Output.
    counts = np.zeros([N,])
    triu_inds = np.triu_indices(m, k=1)
    
    list_ds = []
    if boxes is not None:
        # Correct method:
        Ls = np.linalg.norm(boxes, axis=1, keepdims=True) # (N,3) # box lengths in each frame 
        triu_inds_A, triu_inds_B = triu_inds
        for i in range(N):
            X = COMs[i] # ~ (m,3)
            dX = X[:,np.newaxis,:] - X[np.newaxis,:,:] # (m,m,3)
            dX = dX[triu_inds_A,triu_inds_B,:] # (K,3)
            
            hL = 0.5 * Ls[i] # (1,3) 
            mask = np.where(dX > hL, 1. , 0.) + np.where(dX <= - hL, -1. , 0.) # (K,3)
            dX -= np.einsum('ij,Kj->Ki', boxes[i], mask) # (K,3)
            
            ds = np.linalg.norm(dX, axis=-1) # (K,)

            ds = ds[np.where(ds <= half_min_box_length)[0]] ; counts[i] = ds.shape[0] / m
            ds = ds[np.where(ds >= min_distance)[0]]
            list_ds.append(ds)
    else:
        # Old method when ignoring pbc:
        for i in range(N):
            ds = cdist_(COMs[i], COMs[i], metric='euclidean')[triu_inds]
            ds = ds[np.where(ds <= half_min_box_length)[0]] ; counts[i] = ds.shape[0] / m
            ds = ds[np.where(ds >= min_distance)[0]]
            list_ds.append(ds)

    for i in range(N):
        ds = list_ds[i]
        hist = histogram_(ds[:,np.newaxis],
                        n_bins = n_bins,
                        x_range=[x_range],
                        periodic = False,
                        param = kde_parameter,
                        weights = 1.0/ds**2)
        # ! lengths of ds not equal for all i, but all histograms are normalised to 1.
        hists[i,0] = hist

    return hists[:,0,:] # (N,n_bins)

def get_radial_distributions_(Rs,                 # xyz.coordinates
                              min_box_length,     # min(xyz.min_box_lengths)
                              boxes = None,       # xyz.boxes
                              min_distance = 1.0, # usually ~4.0 to minimise empty bins
                              masses = None,      # xyz.masses
                              n_bins = 30,
                              use_kde = False,
                              kde_parameter = 700.0,
                              ):
    """ Looping through trajectories, pooling outputs from get_radial_distributions_traj_.
        Outputs here are not concatenated, for interpretability.
    """
    Rs = clamp_list_(Rs)
    boxes = clamp_list_(boxes)
    list_hists = []
    for i in range(len(Rs)):
        list_hists.append( get_radial_distributions_traj_(Rs[i],
                                                min_box_length = min_box_length,
                                                boxes = boxes[i],
                                                min_distance = min_distance,
                                                masses = masses,
                                                n_bins = n_bins,
                                                use_kde = use_kde,
                                                kde_parameter = kde_parameter,
                                                ) )
    return list_hists
