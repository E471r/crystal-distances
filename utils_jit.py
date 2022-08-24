import numpy as np

import warnings
warnings.filterwarnings("ignore", message="First-class function type feature is experimental")
# ^ this still comes through the first time cdist_ is ran.

## General:

#@jit(nopython = True, fastmath = True)
def cdist_( X : np.ndarray,
            Y : np.ndarray = None,
            metric : str = 'sqeuclidean',
            gamma : float = 1.6 ):
    
    # first call is slow, but after initialised it is faster than scipy.cdist.
    ''' Computes distance matrix between all rows of one or two input matrices.
    
    Inputs:
        X : (N0,d) shaped array.
        Y : (N1,d) shaped array, or None (sets Y = X).
    Paramters:
        metric : string indicating which type of distance to use. Default is 'sqeuclidean'.
        gamma : relevant if metric is 'rbf' or 'p_norm'. 
                Bandwidth parameter for the rbf kernel. p in 'p_norm'.
    Outputs:
        (N0,N1) shaped distance matrix.
    '''

    if Y is None: Y = X
    else: pass
    
    N0 = X.shape[0] ; N1 = Y.shape[0]   
    
    if metric in ['sqeuclidean',
                  'L2','euclidean',
                  'rbf','gaussian']:
        
        # (Xij-Ykj)**2 = Xij**2 + Ykj**2 - 2XijYkj = Dijk -> Dijk.sum(j) = D
        p1 = (X**2).sum(1) * np.ones((1,N0))
        p2 = (Y**2).sum(1) * np.ones((1,N1))
        D = p1.T + p2 - 2*X.dot(Y.T)

        # Due to subtraction above, small -ve numbers in D can come from round-off errors, they are clamped:
        if metric == 'sqeuclidean': return np.where(D<0., 0., D)
        elif metric in ['L2','euclidean']: return np.sqrt( np.where(D<0., 0., D) )
        elif metric in ['rbf','gaussian']: return np.exp(-gamma*D)

    elif metric in ['L1','1','sparse','manhattan',
                    'Linf','inf','chebyshev', # best if missing data.
                    'Lp','p','p_norm','minkowski']:
        
        if metric in ['L1','1','sparse','manhattan']:   f = lambda x, _ : np.abs(x).sum()
        elif metric in ['Linf','inf','chebyshev']:      f = lambda x, _ : np.abs(x).max()
        elif metric in ['Lp','p','p_norm','minkowski']: f = lambda x, p : np.sum(np.abs(x)**p)**(1/p)

        D = np.zeros((N0,N1))
        for i in range(N0):
            for j in range(N1):
                D[i,j] = f(X[i] - Y[j], gamma)
        
        return D
    
    else: print('This metric is not recognised.')

## REMatch:

#@jit(nopython = True, fastmath = True)
def cdist_rbf_( X : np.ndarray,
                Y : np.ndarray = None,
                gamma : float = 1.6 ):
    ''' Just rbf. Used by REMatch_distance_().
    '''
    if Y is None: Y = X
    else: pass

    p1 = (X**2).sum(1) * np.ones((1,X.shape[0]))
    p2 = (Y**2).sum(1) * np.ones((1,Y.shape[0]))
    D = p1.T + p2 - 2.0*X.dot(Y.T)

    return np.exp(-gamma*D)

#@jit(nopython = True, fastmath = True)
def sinkhorn_( C : np.ndarray,
               alpha : float = 0.01,
               err_limit : float = 1e-6 ):
    
    ''' ## Runs Sinkhorn iterations needed for REMatch. ##
    # Inputs:
    ' C: (n,m) shaped positive (transportation cost) matrix with elements in range [0,1] (Gram matrix).       '
    
    # Default parameters:
    ' alpha: Level of entropic regularisation. [Lower alpha results in a solution which is closer to the      '
    '        exact (linear) solution, but which requires more iterations (which can also be less stable).     '
    '        At increasing alpha, the result approaches a trivial solution. Default value is 0.01.]           '
    ' err_limit: As the parameters (u and v; which are exponentials of –(Lagrange multipliers)/alpha) become  '
    '            updated by increasingly smaller increments (as a result of approaching the stationary point) '
    '            the err converges to zero. Non-zero default cut-off (1e-6) is selected for efficiency.       '
    
    # Outputs:
    ' Scalar: Sinkhorn distance between uniform 1D histograms (en and em) given transportation costs (C).     '
    '''
    n, m = C.shape
    K = np.exp( (C-1)/alpha )

    u = np.ones((n,))/n ; en = np.ones((n,))/n
    v = np.ones((m,))/m ; em = np.ones((m,))/m

    itercount = 0 ; err = 1
    while err > err_limit:
        uprev = u ; vprev = v

        v = em / K.T.dot(u)
        u = en / K.dot(v)

        if itercount % 5:
            err = ((u - uprev)**2).sum() / (u**2).sum() + ((v - vprev)**2).sum() / (v**2).sum()
        itercount += 1
        
    P = v * K * ( u + np.zeros((1,n)) ).T
    distance = (P * C).sum()
    
    return distance

## KDE:

#@jit(nopython = True, fastmath = True)
def periodic_W_F_(x, n_bins : int = 32, param : float or int = 1.0):
                 
    ' x: (N,) shaped array, with all elements in range [0,1].'
    
    N = x.shape[0]
    
    cs_x = np.stack((np.cos(x*np.pi*2), np.sin(x*np.pi*2)), axis=1) # (N,2)
    
    z = np.linspace(0.0 + 0.5/n_bins, 1.0 - 0.5/n_bins, n_bins) ; dz = z[1]-z[0]
    cs_z = np.stack((np.cos(z*np.pi*2), np.sin(z*np.pi*2)), axis=1) # (K,2)

    var = 1. / param

    c1 = np.pi / (np.sqrt(2*np.pi*var)*n_bins) ; c2 = - 0.125/var
    
    W = np.zeros((N,n_bins))
    F = np.zeros((N,n_bins))

    for j in range(n_bins):
        
        cs_zj = cs_z[j:j+1] # ~ (1,2)
        
        sum_here = (((cs_x - cs_zj)**2).sum(1)*np.ones((1,N))).T # (N,1)
        
        p_j =  c1 * np.exp(c2 * sum_here) # (N,1)
        
        W[:,j] = p_j[:,0] # (N,)
        
        dp_j_dcs_j = 2. * p_j * c2 * (cs_x - cs_zj) # (N,1)*(N,2) = (N,2)
        
        J_dcs_j_dx_j = 2. * np.pi * np.stack((-cs_x[:,1], cs_x[:,0]), axis=1) # (N,2)
        
        dp_j_dx_j = (dp_j_dcs_j * J_dcs_j_dx_j).sum(1) # (N,2)*(N,2),sum(1) = (N,)
        
        F[:,j] = dp_j_dx_j


    C = (W.sum(1)*np.ones((1,N))).T     # ; W.sum(axis=1, keepdims=True)
    W /= C                              # = W_tilde
    
    Fsum1 = (F.sum(1)*np.ones((1,N))).T
    F = (F - W * Fsum1) / C             # = F_tilde

    return W, W, F # = W_tilde, W_tilde, F_tilde

#@jit(nopython = True, fastmath = True)
def interval_W_F_(x, n_bins : int =32, param : float or int = 1.0):
    
    ' x: (N,) shaped array, with all elements in range [0,1]. '

    #'''
    N = x.shape[0]
    #z = np.linspace(0.0, 1.0, n_bins) ; dz = z[1]-z[0]
    z = np.linspace(0.0 + 0.5/n_bins, 1.0 - 0.5/n_bins, n_bins) ; dz = z[1]-z[0]

    W = np.zeros((N,n_bins))
    F = np.zeros((N,n_bins))

    s = 1. / (np.sqrt(param)*5.2)
    
    for j in range(n_bins):
        zj = z[j]
        
        Cj = n_bins*((1. / (1. + np.exp((zj-1.)/s))) - (1. / (1. + np.exp(zj/s))))
        
        a = np.exp((zj-x)/s)
        b = 1. + 2.*a + a**2
        
        f = (1./s) * (a / b)

        W[:,j] = f / Cj
        
        df_dx = 2 * ((a**2+a**3)/s**2) * (b**(-2)) - (a/s**2) * (b**(-1))
        
        F[:,j] = df_dx / Cj
    
    C = (W.sum(1)*np.ones((1,N))).T      # ; W.sum(axis=1, keepdims=True)
    W_tilde = W / C
    
    Fsum1 = (F.sum(1)*np.ones((1,N))).T
    F_tilde = (F - W_tilde * Fsum1) / C

    return W, W_tilde, F_tilde # = W, W_tilde, F_tilde

## no jump (make molecules whole and not jumping):

#@jit(nopython=True, fastmath = True)
def no_jump_(R_in : np.ndarray,
             boxes : np.ndarray, # (3,3) vecs as columns!
             fit_molecules_compactly : bool = True # R_in should include the 1st frame which is topology like.
            ):
    
    half_Ls = 0.5 * np.sqrt((boxes**2).sum(1)) # lengths of box vectors.

    n_frames, n_mols, n_atoms = R_in.shape[:3] #### ; flip = -np.arange(1,n_frames+1)

    R_out = np.zeros_like(R_in) + R_in
    
    for _ in range(2):
        for i in range(n_frames):

            for mol in range(n_mols):
                for atom1 in range(n_atoms):
                    for atom2 in range(n_atoms):
                        mask_p = np.where(R_out[i, mol, atom1] - R_out[i, mol, atom2] > half_Ls[i], 1. , 0.)
                        mask_n = np.where(R_out[i, mol, atom1] - R_out[i, mol, atom2] <= - half_Ls[i], -1. , 0.)

                        if (mask_p - mask_n).sum() == 0: pass
                        else: R_out[i, mol, atom1] -= ( boxes[i] * (mask_p + mask_n) ).sum(1) # box.dot(mask)

            
    if fit_molecules_compactly:
        for _ in range(3):
            for i in range(n_frames):

                if i == 0: ref_index = 0
                else: ref_index = i-1

                for mol in range(n_mols):
                    mask_p = np.where(R_out[i, mol, 0] - R_out[ref_index, mol, 0] > half_Ls[i], 1. , 0.)
                    mask_n = np.where(R_out[i, mol, 0] - R_out[ref_index, mol, 0] <= - half_Ls[i], -1. , 0.)

                    if (mask_p - mask_n).sum() == 0: pass
                    else: R_out[i, mol] -= ( boxes[i] * (mask_p + mask_n) ).sum(1)
    else: pass
    
    return R_out

#@jit(nopython=True, fastmath = True)
def wrap_COMs_(Rijl : np.ndarray,
               boxes : np.ndarray,
               ind_j : int):
    ''' Not used.
    '''
    # wrap_atoms_around_one_atom
    n_frames, n_mols = Rijl.shape[:2]
    R_out = np.zeros_like(Rijl) + Rijl
    half_Ls = 0.5 * np.sqrt((boxes**2).sum(1)) # lengths of box vectors.

    for _ in range(2):
        for i in range(n_frames):
            for mol_1 in range(n_mols):
                mask_p = np.where(R_out[i, mol_1] - R_out[i, ind_j] > half_Ls[i], 1. , 0.)
                mask_n = np.where(R_out[i, mol_1] - R_out[i, ind_j] <= - half_Ls[i], -1. , 0.)

                if (mask_p - mask_n).sum() == 0: pass
                else: R_out[i, mol_1] -= ( boxes[i] * (mask_p + mask_n) ).sum(1)
        

    for _ in range(3):
        for i in range(n_frames):

            if i == 0: ref_index = 0
            else: ref_index = i-1

            for mol in range(n_mols):
                mask_p = np.where(R_out[i, mol] - R_out[ref_index, mol] > half_Ls[i], 1. , 0.)
                mask_n = np.where(R_out[i, mol] - R_out[ref_index, mol] <= - half_Ls[i], -1. , 0.)

                if (mask_p - mask_n).sum() == 0: pass
                else: R_out[i, mol] -= ( boxes[i] * (mask_p + mask_n) ).sum(1)
                    
    return R_out

try:
    from numba import jit
    # If jit imported sucessfuly, decorations applied:
 
    # REMatch:
    sinkhorn_ = jit(nopython = True, fastmath = True)(sinkhorn_)
    cdist_ = jit(nopython = True, fastmath = True)(cdist_)
    cdist_rbf_ = jit(nopython = True, fastmath = True)(cdist_rbf_)
	
    # KDE:
    periodic_W_F_ = jit(nopython = True, fastmath = True)(periodic_W_F_)
    interval_W_F_ = jit(nopython = True, fastmath = True)(interval_W_F_)
	
    # no jump:
    no_jump_ =  jit(nopython = True, fastmath = True)(no_jump_)
    wrap_COMs_ = jit(nopython = True, fastmath = True)(wrap_COMs_)
    #

except:
    print('!! : utils_jit : could not import numba.jit.')



