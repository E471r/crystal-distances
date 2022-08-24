import numpy as np

from utils_jit import cdist_rbf_, sinkhorn_
#from utils_einsum import cdist_sqeuclidean_row_

from utils import clamp_list_, TIMER # this is needed here..

import math

##

import torch
def cdist_sqeuclidean_row_( X : np.ndarray,
                            Y : np.ndarray = None):

    X = torch.as_tensor(X)

    if Y is None: Y = X
    else: Y = torch.as_tensor(Y)
    
    p1 = (X**2).sum(2, keepdims=True)
    p2 = (Y**2).sum(2, keepdims=True)
    D = p1 + torch.einsum('ijk->ikj', p2) - 2.0*torch.einsum('ijk,ilk->ijl', X, Y)
    D = torch.where(D<0.0,0.0,D)
    return D.numpy()

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def info_REMatch_distance_():

    info = """
    ## REMatch_distance_() yields conformational distance matrix between supercells of molecules. ##
    
    Template: 
    D = REMatch_distance_(X : list,
                          Y : list = None,
                          gamma : float = 1.67,
                          sinkhorn_alpha : float = 0.05,
                          sinkhorn_err_limit : float = 1e-6,
                          sum_instead_of_sinkhorn : bool = False,
                          use_MMD_instead : bool = False)

    --------------------------------------------------------------------------------------------------
    Inputs:
        X: list of arrays, each with shapes (N,n_mol,dim), preferably normalised long axis 2.
           N: number of frames (can be different in each array).
           n_mol: number of molecules in that frame.
           dim: a constant ( e.g., = 2*n_torsions ).

        Y: (optional) another list of arrays, similar to X, but containing different data.
           [Note: REMatch distance (d) is symmetric, yet comparison between d(x,y) and d(y,x)
            reveals is a small (1e-6) discrepancy. This originates from the stochastic nature of
            the sinkhorn iterations, and therefore is minimised by decreasing sinkhorn_err_limit
            which is set to non-zero default value (1e-6) for reduced computational cost.]  
    
    Parameter:
        gamma: bandwidth parameter for the Gaussian gram matrices [ gamma = 0.5/variance ].

    Optional Parameters:
        alpha: Sinkhorn regularisation parameter. Discussed briefly under the sinkhorn_() function.
               Default value of 0.05 is between 0.1 (good, faster) and 0.01 (more accurate, slower).

        sinkhorn_err_limit: err_limit is discussed under the Sinkhorn_() function.

        sum_instead_of_sinkhorn : bool : (default is False). True is faster but not recommended.

        use_MMD_instead: if True (default False) MMD (less accurate but faster) is used instead of REMatch.

    Outputs:
        D: (n0,n1) shaped distance matrix between frames of trajectories provided in inputs X (Y).
            Elements in range [0,1] iff use_MMD_instead = False.
            n0 = sum([x.shape[0] for x in X]
            n1 = sum([y.shape[0] for y in Y]
    """
    print(info)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def REMatch_distance_(X : list,
                      Y : list = None,
                      gamma : float = 1.67,
                      sinkhorn_alpha : float = 0.05,
                      sinkhorn_err_limit : float = 1e-6,
                      sum_instead_of_sinkhorn : bool = False,
                      use_MMD_instead : bool = False,
                      ):
    """
    More info_REMatch_distance_()

    Summary of what is happening below:
        For the outer terms : xx, yy = f1_(f0_(Gxx)), f1_(f0_(Gyy))
        For the inner terms : xy = f0_(Gxy)
        For output          : distance = f2_(xx, yy, xy)

    Three possible outputs:

        if sum_instead_of_sinkhorn False and use_MMD_instead False: [normalised]
            
            distance = 1.0 - sinkhorn_(Gxy) / sqrt_(sinkhorn_(Gxx)*sinkhorn_(Gyy))

        if sum_instead_of_sinkhorn True and use_MMD_instead False: [normalised]

            distance = 1.0 - sum_(Gxy) / sqrt_(sum_(Gxx)*sum_(Gyy))

        if sum_instead_of_sinkhorn ANY and use_MMD_instead True: [not normalised]

            distance = np.sqrt( mean_(Gxx) + mean_(Gyy) - 2.0*mean_(Gxy) )
    """
    if not use_MMD_instead: # REmatch:
        if sum_instead_of_sinkhorn: f0_ = lambda x, _, __ : x.sum() # -> scalar.
        else:                       f0_ = sinkhorn_ # -> scalar.
        f1_ = lambda x : np.sqrt(x)
        f2_ = lambda xx, yy, xy : 1.0 - xy / (xx*yy) # f1,f1,f0

    else: # MMD:
        f0_ = lambda x, _, __ : x.mean() # -> scalar.
        f1_ = lambda x : x
        f2_ = lambda xx, yy, xy : np.sqrt( xx + yy - 2.0*xy ) # f0,f0,f0

    X = clamp_list_(X)
    if Y is None: Y = X ; Y_is_None = True
    else:
        Y_is_None = False
        Y = clamp_list_(Y)
    
    N0 = len(X) ; ns0 = [] ; siijj0 = []
    
    for i in range(N0):
        n_i = len(X[i]) ; ns0.append(n_i)
        gX_i = np.exp(-gamma*cdist_sqeuclidean_row_(X[i]))
        for j in range(n_i):
            sim_iijj = f1_( f0_(gX_i[j], sinkhorn_alpha, sinkhorn_err_limit) )
            siijj0.append( sim_iijj )
                
    if Y_is_None:
        N1 = N0 ; ns1 = ns0 ; siijj1 = siijj0
    else:
        N1 = len(Y) ; ns1 = [] ; siijj1 = []
        for i in range(N1):
            n_i = len(Y[i]) ; ns1.append(n_i)
            gY_i = np.exp(-gamma*cdist_sqeuclidean_row_(Y[i]))
            for j in range(n_i):
                sim_iijj = f1_( f0_(gY_i[j], sinkhorn_alpha, sinkhorn_err_limit) )
                siijj1.append( sim_iijj ) 
    
    n0 = sum(ns0) ; n1 = sum(ns1)
    
    D = np.zeros([n0,n1])

    if Y_is_None:
        timer = TIMER((n0**2 - n0)//2) ; loading = 0
        for i in range(N0):
            for j in range(N1):
                if i>j:
                    for ii in range(ns0[i]):
                        gXX_ij_ii = np.exp(-gamma*cdist_sqeuclidean_row_(X[i][ii:ii+1], X[j]))
                        for jj in range(ns1[j]):
                            a = sum(ns0[:i])+ii ; b = sum(ns1[:j])+jj
                            
                            sim_ij_ii_jj = f0_(gXX_ij_ii[jj], sinkhorn_alpha, sinkhorn_err_limit) 
                            D[a,b] = f2_(siijj0[a], siijj1[b], sim_ij_ii_jj)

                            timer.check_(loading) ; loading += 1                      
                elif i==j:
                    #for ii in range(ns0[i]):
                    #    Gxy_ij_ii = np.exp(-gamma*cdist_sqeuclidean_row_(X[i][ii], X[j]))
                    for ii in range(ns0[i]):
                        for jj in range(ns1[j]):
                            a = sum(ns0[:i])+ii ; b = sum(ns1[:j])+jj
                            if a > b:
                                gXX_ij_ii_jj = cdist_rbf_(X[i][ii], X[j][jj], gamma = gamma)

                                sim_ij_ii_jj = f0_(gXX_ij_ii_jj, sinkhorn_alpha, sinkhorn_err_limit)
                                D[a,b] = f2_(siijj0[a], siijj1[b], sim_ij_ii_jj)

                                timer.check_(loading) ; loading += 1
                            else: pass
                else: pass
        if use_MMD_instead: D = np.where(np.isnan(D),0.0,D)
        else: pass
        return D + D.T
    
    else:
        timer = TIMER(n0*n1) ; loading = 0
        for i in range(N0):
            for j in range(N1):
                for ii in range(ns0[i]):
                    gXY_ij_ii = np.exp(-gamma*cdist_sqeuclidean_row_(X[i][ii:ii+1], Y[j]))
                    for jj in range(ns1[j]):
                        a = sum(ns0[:i])+ii ; b = sum(ns1[:j])+jj

                        sim_i_j_ii_jj = f0_(gXY_ij_ii[jj], sinkhorn_alpha, sinkhorn_err_limit)
                        D[a,b] = f2_(siijj0[a], siijj1[b], sim_i_j_ii_jj)

                        timer.check_(loading) ; loading += 1
                        
        if use_MMD_instead: D = np.where(np.isnan(D),0.0,D)
        else: pass
        return D

