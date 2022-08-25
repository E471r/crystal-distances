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
                                   weight_parameter : int = 2,
                                   radius_to_return_density : float = None):
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
        weight_parameter : int : further away shells have larger volume and therefore lower density.
            Default is 2 (area of sphere 4*pi*r^2). Setting to 0 gives uniform weights.
        radius_to_return_density : float : if not None density of crystal in frames returned as 2nd output.
            ! radius_to_return_density <= min_box_length !
    Outputs:
        hists : (N,1,n_bins) shaped array of radial distribution histograms.
        crystal_density : (N,) shaped array of (# atoms / volume within radius_to_return_density) in each frame.
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
                        weights = 1.0/ds**weight_parameter)
        # ! lengths of ds not equal for all i, but all histograms are normalised to 1.
        hists[i,0] = hist

    if radius_to_return_density is not None:
        return hists, crystal_density_(counts, radius_to_return_density) # (N,1,n_bins), (N,)
    else:
        return hists # (N,1,n_bins)

def crystal_density_(counts : np.ndarray, # (N,)
                     radius : float):
    volume_of_sphere = 4.0*np.pi*radius**3 / 3.0
    return np.array(counts).astype(float) / volume_of_sphere # number of particles in fixed volume.


def get_radial_distributions_(Rs,                 # xyz.coordinates
                              min_box_length,     # min(xyz.min_box_lengths)
                              boxes = None,       # xyz.boxes
                              min_distance = 1.0, # usually ~4.0 to minimise empty bins
                              masses = None,      # xyz.masses
                              n_bins = 60,
                              use_kde = False,
                              kde_parameter = 700.0,
                              weight_parameter : int = 2,
                              radius_to_return_density : float = None):
    """ Looping through trajectories, pooling outputs from get_radial_distributions_traj_.
        Outputs here are not concatenated, for interpretability.
    """
    Rs = clamp_list_(Rs)
    boxes = clamp_list_(boxes)
    list_hists = []
    list_densities = []
    for i in range(len(Rs)):
        Output = get_radial_distributions_traj_(Rs[i],
                                                min_box_length = min_box_length,
                                                boxes = boxes[i],
                                                min_distance = min_distance,
                                                masses = masses,
                                                n_bins = n_bins,
                                                use_kde = use_kde,
                                                kde_parameter = kde_parameter,
                                                weight_parameter = weight_parameter,
                                                radius_to_return_density = radius_to_return_density,
                                                )
        if len(Output) > 1:
            list_hists.append(Output[0])
            list_densities.append(Output[1])
        else:
            list_hists.append(Output)

    if radius_to_return_density is not None: return list_hists, list_densities
    else: return list_hists

