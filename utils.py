import numpy as np

import pickle

import time                              # TIMER
from IPython.display import clear_output # TIMER

##

## Saving/Loading:

def save_npy_(x,name):
    with open(name, "wb") as f: np.save(f, x) ; print('saved',name)
    
def load_npy_(name):
    with open(name, "rb") as f: x = np.load(f) ; return x

def save_pickle_(x,name):
    with open(name, "wb") as f: pickle.dump(x, f) ; print('saved',name)
    
def load_pickle_(name):
    with open(name, "rb") as f: x = pickle.load(f) ; return x

## General:

def clamp_list_(x):
    if type(x) is list: return x
    else: return [x]  

def clamp_range_(x, x_range):
    a, b = x_range
    x = np.where(x < a, a, x)  
    x = np.where(x > b, b, x)
    return x

def sta_array_(x : np.ndarray,
               input_range : list = None,
               target_range : list = [0.0, 1.0],
               return_J : bool = False):
    ''' 
    Inputs:
        x : any shaped numpy array.
        input_range : None or list of two floats.
        target_range : list of two floats (default = [0,1]).
        return_J : rescaling factor (J) returned.
    Outputs:
        y : elements of x are shifted and scaled to fit inside the target_range.
        J : rescaling factor.
    '''
    if input_range is None:
        x_min = x.min() ; x_max = x.max()
    else:
        x_min, x_max = input_range

    y_min, y_max =  target_range

    J = (y_max - y_min)/(x_max - x_min)

    y = J * (x - x_min) + y_min

    if return_J: return y, J
    else: return y

def psd_mat_power_(C : np.ndarray, power : float):
    ''' Gives square root of covariance matrix.
    '''
    U,l = np.linalg.svd(C)[:2]
    L_power = np.eye(l.shape[0]) * np.where(l<0.0,0.0,l) ** power
    return U.dot(L_power).dot(U.T)

def wrapper_(f,args):
    return f(*args)

def uniform_mesh_(dim, n_bins, x_range=[0,1]):
    n_bins = max(n_bins,2)
    a, b = x_range
    mesh = wrapper_(np.meshgrid, [np.linspace(a, b, n_bins)]*dim)
    mesh = np.stack([mesh[i].flatten() for i in range(dim)], axis=1)
    return mesh

"""
''' Not used anywhere so far.
    Was considered for GMM on torsions for fingerprints, 
    but landmark kernelizing is rough approximation of that.
'''
def gaussian_function_(x, mu, C):#, C_det, C_inv):
    C_inv = np.linalg.pinv(C)
    C_det = np.linalg.det(C)
    c = 1.0 / np.sqrt(2*np.pi)**x.shape[1]
    c /= np.sqrt(C_det+1e-10)
    delta = x-mu
    d = np.einsum('ij,jk,ik->i', delta, C_inv, delta)
    return c * np.exp(-0.5*d)

def padded_gaussian_function_(x, mu, C, periodic=False):
    if periodic:
        mesh_offset = uniform_mesh_(dim=x.shape[1], n_bins=3, x_range=[-np.pi*2.0,np.pi*2.0])
        n_images = mesh_offset.shape[0]
        return np.sum([f_(mesh_offset[i]+x, mu=mu, C=C) for i in range(n_images)], axis=0)
    else:
        return f_(x, mu=mu, C=C)
"""

## concatenate to one molecule:
''' Molecular features A ~ (N,m,dim), array reshaped to (N*m,dim) shaped array.
'''
concat_to_molecule_ = lambda _s : np.concatenate([x.reshape(x.shape[0]*x.shape[1],x.shape[2]) for x in _s],axis=0)

## Rigid body aligment: (not used anywhere here)

def least_squares_rotation_matrix_(x, z, ws):
    # x,z ~ (n,3)
    u,s,v = np.linalg.svd( x.T.dot(ws).dot(z) )
    R = u.dot(v)
    return R

def get_tripod_(x):
    # x ~ (n,3)
    v01 = x[0]-x[1]
    v02 = x[0]-x[2]
    u012 = np.cross(v01,v02)
    return np.stack([v01,v02,u012],axis=0)
"""
def rigid_allign_(X : np.ndarray,
                  z : np.ndarray,
                  subset_inds : list = None,
                  centre_on_subset0 : bool = False,
                  d3_subset_inds_planar : bool = False,
                  verbose : bool = False):

    ''' Rigid body alignment (just one iteration) of cartesian coordinates: 
    
    Inputs:
        X : (m,n,3) shaped array to be aligned to fixed reference (z)
            m = number of molecules, or frames containing one molecule.
            n = number of atoms in the molecule.
            3 = three cartesian coordinates.
        z : (n,3) shaped array : structure alignment template.

    Parameters:
        subset_inds : list (default is None : all atoms used) of indices 
                      for atoms to use in alignment (e.g., [1,2,3,..]).
        centre_on_subset0 : if True (default is False) and subset_inds is not None, 
                            the first atom in subset_inds list will be used instead 
                            of centre of mass, as the centre for the rotation fit.
        d3_subset_inds_planar : if all subset_inds are atoms which may sometimes appear on 
                                a 2D plane (e.g., all are part of a ring) set this to True.
        verbose : bool.

    Output:
        Y : (m,n,3) shaped array of aligned conformers, where
             all conformers X[i] least squares superposed to fit on z.
    '''

    X = np.array(X) ; N,n,d = X.shape # (N,n,d) 
    z = np.array(z)                   # (n,d)

    if subset_inds is None: subset_inds = np.arange(n).tolist()
    else: subset_inds = np.array(subset_inds).flatten().tolist()
 
    X_ = X[:,subset_inds,:] ; z_ = z[subset_inds,:]
    
    if centre_on_subset0: mu_z_ =  np.array(z_[0])[np.newaxis,:]
    else: mu_z_ = z_.mean(0, keepdims=True)
    z_ -= mu_z_
    
    ##
    ws_ = np.eye(z_.shape[0]) 
    ##

    Y = np.zeros([N,n,d])
    for i in range(N):
        x_ = np.array(X_[i]) # (n_s, 3)
        
        if centre_on_subset0:  mu_x_ = np.array(x_[0])[np.newaxis,:]
        else: mu_x_ = x_.mean(0, keepdims=True)

        x_ -= mu_x_

        if d3_subset_inds_planar:
            R = least_squares_rotation_matrix_(get_tripod_(x_[[0,1,2]]),
                                               get_tripod_(z_[[0,1,2]]),
                                               np.eye(3))
        else:
            R = least_squares_rotation_matrix_(x_, z_, ws_)

        err_before = np.linalg.norm(x_-z_)
        y_ = x_.dot(R)
        err_after = np.linalg.norm(y_-z_)

        if err_after < err_before:
            if verbose:
                stars = (10*(err_before-err_after)/err_before).astype(int)
                print('% err drop:  [','*'*stars,'.'*(10-stars),']')
            else: pass
            Xi_ali = (X[i] - mu_x_).dot(R) + mu_z_
            Y[i,:] = Xi_ali

        else:
            if verbose:
                print('no change at frame:', i) # rotation skipped.
            else: pass
            Y[i,:] = X[i] - mu_x_ + mu_z_

    return Y
"""

def rigid_allign_(X : np.ndarray,
                  z : np.ndarray,
                  subset_inds : list = None,
                  centre_on_subset0 : bool = False,
                  d3_subset_inds_planar : bool = False,
                  verbose : bool = False,
                  masses = None):

    ''' Rigid body alignment (just one iteration) of cartesian coordinates: 
    
    Inputs:
        X : (m,n,3) shaped array to be aligned to fixed reference (z)
            m = number of molecules, or frames containing one molecule.
            n = number of atoms in the molecule.
            3 = three cartesian coordinates.
        z : (n,3) shaped array : structure alignment template.

    Parameters:
        subset_inds : list (default is None : all atoms used) of indices 
                      for atoms to use in alignment (e.g., [1,2,3,..]).
        centre_on_subset0 : if True (default is False) and subset_inds is not None, 
                            the first atom in subset_inds list will be used instead 
                            of centre of mass, as the centre for the rotation fit.
        d3_subset_inds_planar : if all subset_inds are atoms which may sometimes appear on 
                                a 2D plane (e.g., all are part of a ring) set this to True.
        verbose : bool.
        masses : mass weighted allign.

    Output:
        Y : (m,n,3) shaped array of aligned conformers, where
             all conformers X[i] least squares superposed to fit on z.
    '''

    X = np.array(X) ; N,n,d = X.shape # (N,n,d) 
    z = np.array(z)                   # (n,d)

    if subset_inds is None: subset_inds = np.arange(n).tolist()
    else: subset_inds = np.array(subset_inds).flatten().tolist()
 
    X_ = X[:,subset_inds,:] ; z_ = z[subset_inds,:]
    if masses is not None:
        ws = np.array(masses).flatten()[subset_inds,np.newaxis]
        ws /= ws.sum()
    else: pass
        
    ##
    ws_ = np.eye(z_.shape[0]) 
    if masses is not None: ws_ *= ws
    else: pass
    ##

    if centre_on_subset0: mu_z_ =  np.array(z_[0])[np.newaxis,:]
    else: 
        if masses is not None: mu_z_ = (z_*ws).sum(0,keepdims=True)
        else:                  mu_z_ = z_.mean(0, keepdims=True)
    z_ -= mu_z_

    Y = np.zeros([N,n,d])
    for i in range(N):
        x_ = np.array(X_[i]) # (n_s, 3)
        
        if centre_on_subset0:  mu_x_ = np.array(x_[0])[np.newaxis,:]
        else: 
            if masses is not None: mu_x_ = (x_*ws).sum(0,keepdims=True)
            else:                  mu_x_ = x_.mean(0, keepdims=True)

        x_ -= mu_x_

        if d3_subset_inds_planar:
            R = least_squares_rotation_matrix_(get_tripod_(x_[[0,1,2]]),
                                               get_tripod_(z_[[0,1,2]]),
                                               np.eye(3))
        else:
            R = least_squares_rotation_matrix_(x_, z_, ws_)

        err_before = np.linalg.norm(x_-z_)
        y_ = x_.dot(R)
        err_after = np.linalg.norm(y_-z_)

        if err_after < err_before:
            if verbose:
                stars = (10*(err_before-err_after)/err_before).astype(int)
                print('% err drop:  [','*'*stars,'.'*(10-stars),']')
            else: pass
            Xi_ali = (X[i] - mu_x_).dot(R) + mu_z_
            Y[i,:] = Xi_ali

        else:
            if verbose:
                print('no change at frame:', i) # rotation skipped.
            else: pass
            Y[i,:] = X[i] - mu_x_ + mu_z_

    return Y

## Timer:

class TIMER:
    ''' Template:
    
        t0 = time.time()
        timer = TIMER(1000, n_stars = 100)
        for i in range(1000):
            time.sleep(0.01)
            timer.check_(i) # or timer.print_(i)
            
    '''        
    def __init__(self,
                 n_itter : int,
                 n_stars_show : int = 100,
                ):
        
        self.n_itter = n_itter
        self.loading_grid = np.linspace(0,n_itter-1,n_stars_show).astype(int)
        self.n_stars_show = n_stars_show
        self.it_was_zero = False
        self.t0 = time.time()
        
    def check_(self, it):
        if it in self.loading_grid:
            self.print_(it)
        else: pass
    
    def print_(self, it):
        self.refresh_loading_bar_(it)
        
    def refresh_loading_bar_(self, it):
        clear_output(wait=True)

        if it == 0: self.it_was_zero = True
        else: pass
        if self.it_was_zero is True: it += 1
        else: pass

        percent_completed = 100 * it / self.n_itter

        time_elapsed = time.time() - self.t0

        av_time_per_percent = time_elapsed / percent_completed

        expected_time_remaining = (100.0 - percent_completed)*time_elapsed / percent_completed

        #n_stars = self.n_stars ; n_spaces = self._n_stars_ - n_stars
        n_stars = int(percent_completed*self.n_stars_show/100.0) ;  n_spaces = self.n_stars_show - n_stars
        
        stars = ['*'*n_stars+'-'*n_spaces+' ']
        percent = [str(int(percent_completed)),'% ']
        time_taken = ['(time taken: ',str(round(time_elapsed, 3))+'s']
        av_time_step = ['' if av_time_per_percent is None else ' ['+str(round(av_time_per_percent, 3))+'s/%'+']']
        time_remaining = [')' if expected_time_remaining is None else ' ETA: '+str(round(expected_time_remaining, 3))+'s'+')']
    
        #info_string = ''.join(stars + percent + time_taken + av_time_step + time_remaining)
        #print(info_string)
        print(''.join(stars + percent))
        print(''.join(time_taken + av_time_step + time_remaining))


