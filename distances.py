import numpy as np

from utils_jit import cdist_

from utils import sta_array_, psd_mat_power_, clamp_list_, TIMER

##

def hellinger_distance_(X : np.ndarray, Y : np.ndarray = None, normalise_histograms : bool = True):
    '''
    Inputs:
        X : (Nx, n_feat, n_bins) shaped array.
        Y : (Ny, n_feat, n_bins) shaped array or None.
            Nx, Ny = total number of frames in all trajectories.
            n_feat = 1 if histograms are joint (highly recommended).
            n_bins = # bins of flattened normalised histograms.
        normalise_histograms : bool. By default input should already be normalised.
    Outout:
        HD_matrix: (Nx,Ny) shaped array with elements in range [0,1].
    '''
    X = np.array(X)
    if Y is None: Y = X
    else: Y = np.array(Y)
    
    Nx, n_feat, n_bins = X.shape ; Ny = Y.shape[0]
    
    X = X.reshape(Nx, n_feat*n_bins)
    Y = Y.reshape(Ny, n_feat*n_bins)
    
    if normalise_histograms:
        X /= X.sum(1, keepdims=True)
        Y /= Y.sum(1, keepdims=True)
    else: pass

    HD_matrix = cdist_(np.sqrt(X), np.sqrt(Y), metric='euclidean') / 1.4142135623730951
    
    return HD_matrix

""" Both 'marginal_method's are bad.
def hellinger_distance_(X : np.ndarray, Y : np.ndarray = None, marginal_method = 0, ws = None):
    X = np.array(X)
    if Y is None: Y = X
    else:  Y = np.array(Y)
    if marginal_method == 0: # Both similarly bad when joint=False in histograms. Joint density is essential for hellinger distance.
        n_feat = X.shape[1] # always 1 if joint.
        if ws is None:
            ws = np.ones([n_feat,])/n_feat
        else:
            ws = ws.reshape([n_feat,]) # np.exp(alpha*s)
            ws /= ws.sum()
        HD = 0.0
        for q in range(n_feat):
            pxq = X[:,q,:] ; pxq /= pxq.sum(1, keepdims=True) # normalising here just very incase.
            pyq = Y[:,q,:] ; pyq /= pyq.sum(1, keepdims=True) # normalising here just very incase.
            HD += ws[q]*cdist_(np.sqrt(pxq), np.sqrt(pyq), metric='euclidean') / 1.4142135623730951
    else:
        Nx, n_feat, n_bins = X.shape ; Ny = Y.shape[0]
        X = X.reshape(Nx, n_feat*n_bins) ; X /= X.sum(1, keepdims=True)
        Y = Y.reshape(Ny, n_feat*n_bins) ; Y /= Y.sum(1, keepdims=True)
        HD = cdist_(np.sqrt(X), np.sqrt(Y), metric='euclidean') / 1.4142135623730951
    return HD
"""

##

def simple_matrix_distance_(X,Y):
    ''' A basic way to compare two distance matrices (results). '''
    return np.linalg.norm(sta_array_(X)-sta_array_(Y))

def SO_distance_element_(Input : list = None, # [C1,C2]
                         Input_sqrt_trace : list = None, # [[sqrt1,trace1],[sqrt2,trace2]]
                         both_are_distance_matrices : bool = True, 
                         standardise : bool = True,
                         verbose : bool = False):
    '''
    A way to compare two distance matrices (Input != None, both_are_distance_matrices = True) : default,
    or two covariance matrices (Input != None, both_are_distance_matrices = False).

    Input = None but Input_sqrt_trace != None, is used by SO_distance_().
    '''
    if Input_sqrt_trace is None:
        C1, C2 = Input

        if standardise:
            C1 = sta_array_(C1)
            C2 = sta_array_(C2)
        else: pass

        if both_are_distance_matrices:
            C1 = 1.0 - C1 # if C1[0,1] > C1[0,0]: C1 = 1-C1
            C2 = 1.0 - C2 # if C2[0,1] > C2[0,0]: C2 = 1-C2
        else: pass # both are similarity matrices.
        
        C1_sqrt = psd_mat_power_(C1,0.5) ; C1_trace = np.trace(C1)
        C2_sqrt = psd_mat_power_(C2,0.5) ; C2_trace = np.trace(C2)
        
    else:
        C1_sqrt, C1_trace = Input_sqrt_trace[0]
        C2_sqrt, C2_trace = Input_sqrt_trace[1]
            
    difference = C1_sqrt - C2_sqrt
    difference = np.sqrt(np.trace(difference.T.dot(difference)))
    overlap = 1.0 - (difference / np.sqrt(C1_trace + C2_trace))
    
    if verbose: print("subspace overlap is: " + str(overlap * 100.0) + " %")
    else: pass
    
    return 1.0 - overlap

def SO_distance_(CsCt_list_X : list, CsCt_list_Y : list = None):
    X = CsCt_list_X
    Y = CsCt_list_Y

    if Y is None:
        Nx = len(X)
        Output = np.eye(Nx)*0.0
        for i in range(Nx):
            for j in range(Nx):
                if j >=i:
                    d = SO_distance_element_(Input=None,Input_sqrt_trace=[X[i],X[j]])
                    Output[i,j] = Output[j,i] = d
                else: pass
        return Output

    else: 
        Nx, Ny = len(X), len(Y)
        Output = np.zeros([Nx,Ny])
        for i in range(Nx):
            for j in range(Ny):
                d = SO_distance_element_(Input=None, Input_sqrt_trace=[X[i],Y[j]])
                Output[i,j] = d
        return Output

## Depreciated:

'''
def wrapper_(f,args):
    return f(*args)
import torch
einsum_ = lambda test_torch, X : wrapper_(torch.einsum, [test_torch] + [torch.as_tensor(x) for x in X]).numpy()

def landmark_mmd_(X : list, Y : list = None):
    
    #! EXPERIMENTAL
    #Input:
    #    X : = Gs : list [(N,m,m2) shaped array of gram matrices].
    #        Gs = get_gram_features_(data_in_primary_features, landmark_molecules, gamma = gamma)
    #        lands : (n_landmarks, primary_feature_dimension)
    #Output:
    #    D : array

    c = 1.0
    
    X = clamp_list_(X)
    if Y is None: Y = X ; Y_is_None = True
    else:
        Y_is_None = False
        Y = clamp_list_(Y)
    
    N0 = len(X) ; ns0 = [] ; siijj0 = []
    
    for i in range(N0):
        n_i = len(X[i]) ; ns0.append(n_i)
        gX_i = np.sum(einsum_('ijk,ilk->ijl',[X[i],X[i]]), axis=(-2,-1))
        for j in range(n_i):
            siijj0.append( gX_i[j] )
                
    if Y_is_None:
        N1 = N0 ; ns1 = ns0 ; siijj1 = siijj0
    else:
        N1 = len(Y) ; ns1 = [] ; siijj1 = []
        for i in range(N1):
            n_i = len(Y[i]) ; ns1.append(n_i)
            gY_i = np.sum(einsum_('ijk,ilk->ijl',[Y[i],Y[i]]), axis=(-2,-1))
            for j in range(n_i):
                siijj1.append( gY_i[j] ) 
    
    n0 = sum(ns0) ; n1 = sum(ns1)
    
    D = np.zeros([n0,n1])

    if Y_is_None:
        timer = TIMER((n0**2 - n0)//2) ; loading = 0
        for i in range(N0):
            for j in range(N1):
                if i>j:
                    for ii in range(ns0[i]):
                        # X[i][ii:ii+1] ~ (1,m,d)
                        # X[j] ~ (N,m,d)

                        gXX_ij_ii = np.sum(einsum_('ijk,ilk->ijl',[X[i][ii:ii+1],X[j]]), axis=(-2,-1))
                        
                        for jj in range(ns1[j]):
                            a = sum(ns0[:i])+ii ; b = sum(ns1[:j])+jj
                            #dist = (siijj0[a] + siijj1[b]) / gXX_ij_ii[jj] - 2.0
                            #dist = np.sqrt( siijj0[a] + siijj1[b] - 2.0*gXX_ij_ii[jj] )
                            dist = 1.0 - gXX_ij_ii[jj] / np.sqrt(siijj0[a]*siijj1[b])
                            D[a,b] = dist * c
                            
                            timer.check_(loading) ; loading += 1                      
                elif i==j:
                    for ii in range(ns0[i]):
                        for jj in range(ns1[j]):
                            a = sum(ns0[:i])+ii ; b = sum(ns1[:j])+jj
                            if a > b:
                                # X[i][ii] ~ (m,d)
                                # X[j][jj] ~ (n,d)
                                gXX_ij_ii_jj = np.sum(einsum_('jk,lk->jl',[X[i][ii], X[j][jj]]), axis=None)
                                
                                #dist = (siijj0[a] + siijj1[b]) / gXX_ij_ii_jj - 2.0
                                #dist = np.sqrt( siijj0[a] + siijj1[b] - 2.0*gXX_ij_ii_jj)
                                dist = 1.0 - gXX_ij_ii_jj / np.sqrt(siijj0[a]*siijj1[b])
                                D[a,b] = dist * c

                                timer.check_(loading) ; loading += 1
                            else: pass
                else: pass
        D = np.where(np.isnan(D),0.0,D)
        return D + D.T
    
    else:
        timer = TIMER(n0*n1) ; loading = 0
        for i in range(N0):
            for j in range(N1):
                for ii in range(ns0[i]):

                    gXY_ij_ii = np.sum(einsum_('ijk,ilk->ijl',[X[i][ii:ii+1],Y[j]]), axis=(-2,-1))

                    for jj in range(ns1[j]):
                        a = sum(ns0[:i])+ii ; b = sum(ns1[:j])+jj
                        #dist = (siijj0[a] + siijj1[b]) / gXY_ij_ii[jj] - 2.0
                        #dist = np.sqrt( siijj0[a] + siijj1[b] - 2.0*gXY_ij_ii[jj].mean() )
                        dist = 1.0 - gXY_ij_ii[jj].sum() / np.sqrt(siijj0[a]*siijj1[b])
                        D[a,b] = dist * c

                        timer.check_(loading) ; loading += 1
                        
        D = np.where(np.isnan(D),0.0,D)
        return D

H_ = lambda x , axis: - (x*np.log(x)).sum(axis)

def landmark_mi_(X : list,
                 Y : list = None,
                 ):
    # less sensitive to choice of landmarks, but less stable.
    # X = Gs : list [(N,m,m2) shaped array of not-normalised 'histograms' over grid of landmark molecules].

    c = 1.0
    
    X = clamp_list_(X)
    if Y is None: Y = X ; Y_is_None = True
    else:
        Y_is_None = False
        Y = clamp_list_(Y)
    
    N0 = len(X) ; ns0 = [] ; siijj0 = []
    
    for i in range(N0):
        n_i = len(X[i]) ; ns0.append(n_i)
        gX_i = H_(einsum_('ijk,ilk->ijl',[X[i],X[i]]), axis=(-2,-1))
        for j in range(n_i):
            siijj0.append( gX_i[j] )
                
    if Y_is_None:
        N1 = N0 ; ns1 = ns0 ; siijj1 = siijj0
    else:
        N1 = len(Y) ; ns1 = [] ; siijj1 = []
        for i in range(N1):
            n_i = len(Y[i]) ; ns1.append(n_i)
            gY_i = H_(einsum_('ijk,ilk->ijl',[Y[i],Y[i]]), axis=(-2,-1))
            for j in range(n_i):
                siijj1.append( gY_i[j] ) 
    
    n0 = sum(ns0) ; n1 = sum(ns1)
    
    D = np.zeros([n0,n1])

    if Y_is_None:
        timer = TIMER((n0**2 - n0)//2) ; loading = 0
        for i in range(N0):
            for j in range(N1):
                if i>j:
                    for ii in range(ns0[i]):
                        # X[i][ii:ii+1] ~ (1,m,d)
                        # X[j] ~ (N,m,d)

                        gXX_ij_ii = H_(einsum_('ijk,ilk->ijl',[X[i][ii:ii+1],X[j]]), axis=(-2,-1))

                        for jj in range(ns1[j]):
                            a = sum(ns0[:i])+ii ; b = sum(ns1[:j])+jj
                            dist = (siijj0[a] + siijj1[b]) / gXX_ij_ii[jj] - 2.0
                            #dist = np.sqrt( siijj0[a] + siijj1[b] - 2.0*gXX_ij_ii[jj] )
                            D[a,b] = dist * c
                            
                            timer.check_(loading) ; loading += 1                      
                elif i==j:
                    for ii in range(ns0[i]):
                        for jj in range(ns1[j]):
                            a = sum(ns0[:i])+ii ; b = sum(ns1[:j])+jj
                            if a > b:
                                # X[i][ii] ~ (m,d)
                                # X[j][jj] ~ (n,d)
                                
                                gXX_ij_ii_jj = H_(einsum_('jk,lk->jl',[X[i][ii], X[j][jj]]), axis=(0,1))
                                
                                dist = (siijj0[a] + siijj1[b]) / gXX_ij_ii_jj - 2.0
                                #dist = np.sqrt( siijj0[a] + siijj1[b] - 2.0*gXX_ij_ii_jj)
                                D[a,b] = dist * c

                                timer.check_(loading) ; loading += 1
                            else: pass
                else: pass
        D = np.where(np.isnan(D),0.0,D)
        return D + D.T
    
    else:
        timer = TIMER(n0*n1) ; loading = 0
        for i in range(N0):
            for j in range(N1):
                for ii in range(ns0[i]):

                    gXY_ij_ii = H_(einsum_('ijk,ilk->ijl',[X[i][ii:ii+1],Y[j]]), axis=(-2,-1))

                    for jj in range(ns1[j]):
                        a = sum(ns0[:i])+ii ; b = sum(ns1[:j])+jj
                        dist = (siijj0[a] + siijj1[b]) / gXY_ij_ii[jj] - 2.0
                        #dist = np.sqrt( siijj0[a] + siijj1[b] - 2.0*gXY_ij_ii[jj].mean() )
                        D[a,b] = dist * c

                        timer.check_(loading) ; loading += 1
                        
        D = np.where(np.isnan(D),0.0,D)
        return D
'''

