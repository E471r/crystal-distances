import numpy as np

import matplotlib.pyplot as plt

from utils import sta_array_, TIMER

##

def prepare_FSDP_(x : np.ndarray,
                 x_is_distance_matrix : bool = True,
                 gamma : float = 0.1,
                ):
    ''' FSDP function (1/3).
    Inputs:
        x: (N,N) or (N,dim) shaped array, if x_is_distance_matrix = True or False respectively.
        x_is_distance_matrix: if False, rows of x are treated as datapoints and euclidean distance matrix computed locally.
                              if True, externally computed distance matrix (between datapoints) is used directly.
        gamma: float, bandwidth of rbf kernel applied over the distance matrix to get local densities ('D' in FSDP).
        [ gamma is the main/only hyperparameter in FSDP. ]
    Outputs:
        pd: (N,2) shaped array. For each datapoint, [local density, distance to nearest-neighbour-of-higher-density].
        nn_hd: (N,) shaped array, of index of nearest-neighbour-of-higher-density.
    '''
    c1 = - gamma
    
    if x_is_distance_matrix:
        D = x
        p = np.exp( c1 * D**2).sum(1) # D assumed from 'euclidean', check if better sqrt it before input.
        # p = np.where(D-co<0,1,0).sum(1) # another option is nearest neighbour ~adjacency matrix within a co distance.
        
        dp = p[:,np.newaxis] - p[:,np.newaxis].T
        Dsele = np.where(dp<0, 1, 0) * D
        Dsele = np.ma.masked_where(Dsele==0, Dsele)
        
        d = np.min(Dsele, axis=1)
        nn_hd = np.argmin(Dsele, axis=1)

    else: # x_is_data_matrix:
        # doing same as above but one row at a time incase N is large:
        N, dim = x.shape
        p = np.zeros([N,])
        d = np.ma.masked_array(np.zeros([N,]))
        nn_hd = np.zeros([N,]).astype(int)
        
        for i in range(N):
            Di_sq = ((x[i:i+1,:]-x)**2).sum(1) # ~ (N,) # 'sqeuclidean' # Di_sq = fast_distance_metric_of_choice(x)
            
            Gi = np.exp( c1*Di_sq )            # ~ (N,)
            p[i] = Gi.sum()                    # np.where(np.sqrt(Di_sq)-co<0,1,0).sum()

        for i in range(N):
            Di_sq = ((x[i:i+1]-x)**2).sum(1)   # ~ (N,) # Di_sq = fast_distance_metric_of_choice(x)
            Di = np.sqrt(Di_sq)                # ~ (N,)    

            dpi = (p[i,np.newaxis] - p[:,np.newaxis].T)[0,:] # ~(N,)
            Di_sele = np.where(dpi<0, 1, 0) * Di
            Di_sele = np.ma.masked_where(Di_sele==0, Di_sele)

            di = np.min(Di_sele)
            nn_hd_i = np.argmin(Di_sele)

            d[i] = di
            nn_hd[i] = nn_hd_i
            
    d = sta_array_(d)
    p = sta_array_(p) + 1e-6

    d[np.argmax(p)] = 1.
    d = d.data
    pd = np.stack([p,d], axis=1)
    
    return pd, nn_hd # shapes: (N,2), (N,)

def outputs_FSDP_(nn_hd : np.ndarray, inds_cluster_centres : np.ndarray):
    ''' FSDP function (3/3).
    Inputs:
        nn_hd: (N,) shaped array. For each datapoint, index of nearest-neighbour-of-higher-density.
        inds_cluster_centres: (n_clusters,) shaped array, of indices of points which are cluster centres.
    Outputs:
        c : (N,) shaped array, of final cluster_assignments.
    '''
    N = len(nn_hd)
    n_clusters = len(inds_cluster_centres)
    
    c = np.zeros(N)-1
    for i in range(n_clusters):
        c[inds_cluster_centres[i]] = i
        
    while -1 in c:
        placed_something =  False
        for i in range(N):
            if c[i] == -1:
                if c[nn_hd[i]] == -1: pass
                else: c[i] = c[nn_hd[i]] ; placed_something = True
            else: pass
        if placed_something is False:
            print('break. (This should not happen)')
            break
    # noise cluster not implemented here yet, all data is assigned to a cluster.
    return c.astype(int)

def plot_fsdp_(x, y, inds_above_point, cut_off):
    plt.scatter(x, y, label = 'not cluster centres')
    plt.scatter(x[inds_above_point], y[inds_above_point], label = 'cluster centres')
    plt.plot([0,x.max()], [cut_off,cut_off], color='orange')
    plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
    plt.show()

def d1_decision_(pd : np.ndarray, cut_off : float = 0.5, cut_off_up_down : list = [0.,0.], verbose : bool = True, plot_type=1):
    ''' FSDP function (2/3) {I}.
    Inputs:
        pd: (N,2) shaped array. The pd output from prepare_FSDP_.
        cut_off: float in range [0.0,1.0]. The proposed decision point/boundary. 
    Outputs:
        inds_above_point: (n_clusters,) shaped array, of indices of points which are potential cluster centres.
    '''
    a = cut_off_up_down[1]
    b = cut_off_up_down[0]
    decision_space_1d = pd[:,0]*pd[:,1] - a*pd[:,0] - b*pd[:,1]
    
    inds_above_point = np.where(decision_space_1d > cut_off)[0]
    
    if verbose:
        print(len(inds_above_point), 'orange points.')
        if plot_type == 0:
            plot_fsdp_(np.arange(pd.shape[0]), decision_space_1d, inds_above_point, cut_off)
        else:
            x = np.linspace(0,1,100)
            y = (cut_off + a*x)/(x - b)
            valid_y = np.where((y<=1.)&(y>=0.))[0]
            plt.plot(x[valid_y], y[valid_y])
            plot_fsdp_(pd[:,0], pd[:,1], inds_above_point, cut_off)           
    else: pass 
    
    return inds_above_point # shape: (n_clusters,)

def d1_decision_flat_(pd : np.ndarray, cut_off : float = 0.5, verbose: bool = True):
    ''' FSDP function (2/3) {II}.
        Similar to 'd1_decision_()' above.
    '''
    d = pd[:,1]
    inds_above_point = np.where(d > cut_off)[0]

    if verbose:
        print(len(inds_above_point), 'orange points.')
        plot_fsdp_(pd[:,0], pd[:,1], inds_above_point, cut_off) 
    else: pass

    return inds_above_point

class FSDP:
    ''' Wrapper class for the three FSDP functions above, to organise the steps:
    
        ## Minimal example: [not valid now]
        ## cluster_assignments = FSDP(x = D, x_is_distance_matrix = True, gamma = 0.01).return_cluster_assignments()
            
    '''
    def __init__(self,
                 x,
                 x_is_distance_matrix : bool = True, # if False: large dataset of raw cartesian data is also accepted.
                 gamma : float = 0.01,                  # rbf bandwidth applied over the distance matrix.
                ):
        self.x = x
        self.x_is_distance_matrix = x_is_distance_matrix

        self._ = False
        self.change_gamma(gamma, verbose = False)
        self._ = True
        
    def change_gamma(self, new_gamma : float, verbose : bool = True, **kwargs):
        ''' Recomputes the 1st step of FSDP and stores the outputs. 
        '''
        pd, nn_hd = prepare_FSDP_(x = self.x, x_is_distance_matrix = self.x_is_distance_matrix, gamma = new_gamma)
        if self._:
            self.previous_gamma = float(self.gamma)
            self.previous_pd = np.array(self.pd)
        self.pd = pd
        self.nn_hd = nn_hd
        self.gamma = new_gamma

        if verbose:
            x0 = self.previous_pd[:,0]
            y0 = self.previous_pd[:,1]
            x1 = self.pd[:,0]
            y1 = self.pd[:,1]
            plt.scatter(x0, y0, label = 'Previous decision plot (gamma = '+str(self.previous_gamma)+')', **kwargs)
            plt.scatter(x1, y1, label = 'Updated decision plot  (gamma = '+str(self.gamma)+')', **kwargs)
            for i in range(pd.shape[0]):
                plt.plot([x0[i],x1[i]],[y0[i],y1[i]],'-', color='black', alpha=0.5)
            plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
        else: pass
        
    def place_simple_decision_boundary(self, cut_off : float = 0.5, cut_off_up_down : list = [0.,0.], plot_type=1, verbose : bool = True):
        ''' Recomputes the 2np step of FSDP and stores the outputs. 
        '''
        self.inds_cluster_centres = d1_decision_(self.pd, cut_off = cut_off, cut_off_up_down=cut_off_up_down, plot_type=plot_type, verbose=verbose)
    
    #def place_simple_decision_boundary_auto(self, percentile : float = 99.5, verbose = True):
    #    ''' Recomputes the 2np step of FSDP (in unsupervised/default way) and stores the outputs. 
    #    '''
    #    cut_off = 2.5 * np.percentile(self.pd[:,0]*self.pd[:,1], percentile)
    #    self.inds_cluster_centres = interactive_FSDP_decision_(self.pd, cut_off = cut_off, verbose = verbose)
    
    def place_flat_decision_boundary(self, cut_off_d : float = 0.5, verbose : bool = True):
        ''' Recomputes the 2np step of FSDP and stores the outputs. 
        '''
        self.inds_cluster_centres = d1_decision_flat_(self.pd, cut_off = cut_off_d, verbose=verbose)

    #def place_curved_decision_bondary(self):
    #    #accept user unput, via drawing. There is a way.

    def return_cluster_assignments(self, verbose : bool = True):
        ''' Given the stored outputs from 1st and 2np steps of FSDP, all datapoints are assigned to clusters.
        '''
        self.cluster_assignments = outputs_FSDP_(self.nn_hd, self.inds_cluster_centres)
        self.n_clusters = len(set(self.cluster_assignments))

        if verbose:
            print('Returning labels for', self.n_clusters, 'clusters found.')
        else: pass 

        return self.cluster_assignments

    def evaluate_(self, y, verbose : bool = True):
        ''' not sure if this works.
        '''
        Ny = y.shape[0]
        y_cluster_assignments = []
        if self.x_is_distance_matrix:
            print('not implemented')
        else:
            if verbose: timer = TIMER(Ny) ; a = 0
            else: pass
            for i in range(Ny):
                Di_sq = ((y[i:i+1,:]-self.x)**2).sum(1)
                y_cluster_assignments.append( self.cluster_assignments[np.argmin(Di_sq)] )
                if verbose: timer.check_(a) ; a+=1
                else: pass

        return np.array(y_cluster_assignments)
