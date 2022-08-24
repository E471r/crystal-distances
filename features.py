import numpy as np

from utils import clamp_list_, psd_mat_power_, sta_array_

from utils_jit import cdist_

from utils_einsum import cdist_sqeuclidean_row_

## Intra-molecular torsions:

def get_torsions_(Rs : np.ndarray or list, torsion_indices : list or np.ndarray):
    
    ''' 
    Inputs:
        Rs : list [(N,m,n,3) shaped array of cartesian coordinates]
            N = # frames
            m = # molecules
            n = # atoms
            3 = # xyz coordinates
        torsion_indices : list or array with shape (n_torsions, 4).
    Outputs:
        As : list [(N,m,n_torsions) shaped array of torsional angles].
    '''
    
    Rs = clamp_list_(Rs)
        
    n_torsions = len(torsion_indices)
    
    As = []
    for j in range(len(Rs)):

        N, m = Rs[j].shape[:2]

        A = np.zeros([N, m, n_torsions])

        for i in range(n_torsions):

            abcd = Rs[j][:,:,torsion_indices[i]]

            v0 = abcd[:,:,1] - abcd[:,:,0]
            v1 = abcd[:,:,2] - abcd[:,:,1]
            v2 = abcd[:,:,3] - abcd[:,:,2]

            u1 = np.cross(v0,v1)
            norm_u1 = np.linalg.norm(u1,axis=2, keepdims=True)
            norm_u1 = np.where(norm_u1<=0.0,1e-10,norm_u1)
            u1 /= norm_u1

            u2 = np.cross(v1,v2)
            norm_u2 = np.linalg.norm(u2,axis=2, keepdims=True)
            norm_u2 = np.where(norm_u2<=0.0,1e-10,norm_u2)
            u2 /= norm_u2

            w = (u1*u2).sum(2) ; w = np.where(w>1.0,1.0,w) ; w = np.where(w<-1.0,-1.0,w) # for any +/-1.00000001 cases.

            A[:,:,i] = np.sign((v0*u2).sum(2)) * np.arccos( w )

        # for 100% stability this is clamped one more time:
        A = np.where(A > np.pi, np.pi, A)   # torsional angles need to be exactly in the [-pi,pi] range.
        A = np.where(A < -np.pi, -np.pi, A) # torsional angles need to be exactly in the [-pi,pi] range.
        
        As.append(A)
        
    return As

def cossin_torsions_(As):
    '''
    Input:
        As : list [(N,m,n_torsions) shaped array of torsional angles].
            N = # frames
            m = # molecules
    Ouput:
        csAs : list [(N,m,2*n_torsions) shaped array of normalised torsional features.]
    '''
    As = clamp_list_(As)
    csAs = [np.concatenate([np.cos(x),np.sin(x)],axis=-1)/np.sqrt(x.shape[-1]) for x in As]
    return csAs

## Intra-molecular distances:

def get_intramolecular_distances_(Rs):
    ''' 
    Input:
        Rs : list [(N,m,n,3) shaped array of cartesian coordiantes]
            N = # frames
            m = # molecules
            n = # atoms
            3 = # xyz coordinates
    Output:
        Ds : list [(N,m,d) shaped array of unique intramolecular distances between atoms].
            d = (n**2-n) / 2
    '''
    Rs = clamp_list_(Rs)
    n_atoms = Rs[0].shape[2]
    d = (n_atoms**2 - n_atoms) // 2
    Ds = []
    for i in range(len(Rs)):
        R = Rs[i]
        n_frames, n_molecules = R.shape[:2]
        Ds_ij = np.zeros([n_frames,n_molecules,d])
        for j in range(n_frames):
            full_Ds_ij = np.sqrt(cdist_sqeuclidean_row_(R[j])) # parallel over molecules in frame.
            for k in range(n_molecules):
                Ds_ij[j,k,:] = full_Ds_ij[k][np.triu_indices(n_atoms, k=1)]
        Ds.append(Ds_ij)
    return Ds

## Conformer landmarks as features:

def recommend_landmark_gamma_(x, c, inds_centroids, inds_dims = None):
    '''
    Inputs:
        x : (N,dim) shaped array of single-molecule conformers.
        c : (N,) shaped array of cluster assignments in {0,...,n}.
        inds_centroids : n+1 shaped array or list of indices for x[i] which are centroids.
        inds_dims : None : all, or only some dimensions can be selected.
    Outputs:
        gamma : recommended gamma vector, for landmarks. [spheres of non-equal sizes.]
    '''
    N, dim = x.shape
    c = np.array(c).reshape([N,])
    
    n_clusters = len(inds_centroids)
    
    if inds_dims is not None:
        x = x[:,inds_dims]
    else: pass
    
    Vars = []
    counts = []
    for i in range(n_clusters):
        ind_centroid = inds_centroids[i]
        
        mu = x[ind_centroid]
        ind_mu = c[ind_centroid]
        
        xi = x[np.where(c==ind_mu)[0]]
        counts.append(xi.shape[0])
        Vars.append(((xi-mu)**2).mean(0).sum()) # total variance or cluster i.

    counts = np.array(counts).astype(float)
    pi = counts/counts.sum()
    
    Vars = np.array(Vars)
    
    gamma = 0.5/(Vars+1e-10)
    
    return gamma, pi

def get_gram_features_(csAs, cs_landmarks, gamma : float or np.ndarray = 1.0):
    ''' 
    ! EXPERIMENTAL

    Inputs:
        csAs : list [(N,m,dim) shaped array of normalised torsional features.]
            N = # frames
            m = # molecules
            dim = 2*n_torsions
        cs_landmarks : (m2,dim) shaped array.
            m2 = # landmark molecules (centroids).
        gamma : float, bandwidth (here just a scalar) for the gram matrices.
            TODO: confirm gamma better as gamma = [0.5/SD[i]**2 for i in n_clusters]
    Output:
        Gs : list [(N,m,m2) shaped array of gram matrices.]
            [Gs has a lot of information if cs_landmarks and gamma are chosen carefully.]
    '''

    if type(gamma) not in [int, float]: gamma = gamma[np.newaxis,np.newaxis,:]
    else: gamma = np.ones([1,1,cs_landmarks.shape[0]])*gamma

    csAs = clamp_list_(csAs)
    cs_landmarks = np.expand_dims(cs_landmarks, axis=0) # (m2,dim) -> (1,m2,dim)
    Gs = []
    for i in range(len(csAs)):

        # (N,m,dim), (1,m2,dim) -> (N,m,m2) # parallel over frames in trajectory.
        x = cdist_sqeuclidean_row_(csAs[i], cs_landmarks)
        G = np.exp(-gamma*x) # (N,m,m2)
        Gs.append(G)

    return Gs

def get_landmark_histograms_(csAs, cs_landmarks, gamma : float = 1.0, force_normalise : bool = True):
    ''' 
    !! EXPERIMENTAL

    Input:
        csAs : list [(N,m,dim) shaped array of normalised torsional features.]
            N = # frames
            m = # molecules
            dim = 2*n_torsions
        cs_landmarks : (m2,dim) shaped array.
            m2 = # landmark molecules (centroids).
        gamma : float or (m2,) shaped vector, bandwidth for the gram matrices.
            if vector : gamma = [alpha * (0.5/Var[i]**2) for i in n_clusters]
        force_normalise : bool.
    Output:
        Hs : (a,1,m2) shaped array of normalised 'histograms' over grid of landmark molecules.
            a = total # frames in all trajectories of input.
            Hs ready for hellinger_distance_():
                + Still fast at very high number of landmarks, compared to get_forced_landmark_CsCt_list_().
                - Delete this function.
    '''
    csAs = clamp_list_(csAs)
    n_frames_total = sum([x.shape[0] for x in csAs])
    m2 = cs_landmarks.shape[0]
    Hs = np.zeros([n_frames_total,1,m2])
    a = 0
    for i in range(len(csAs)):
        G = get_gram_features_(csAs[i], cs_landmarks, gamma = gamma)[0] # (N,m,m2)
        for j in range(G.shape[0]): # N
            W = G[j] # W ~ (m,m2)
            if force_normalise: W /= W.sum(1, keepdims=True)
            else: pass
            Hs[a,0,:] = W.mean(0) # m2 is constant.
            a+=1
    return Hs

def get_landmark_CsCt_list_(csAs, cs_landmarks, gamma : float = 1.0):
    ''' 
    EXPERIMENTAL # Best landmark method. Keep.

    Input:
        csAs : list [(N,m,dim) shaped array of normalised torsional features.]
            N = # frames
            m = # molecules
            dim = 2*n_torsions
        cs_landmarks : (m2,dim) shaped array.
            m2 = # landmark molecules (centroids).
        gamma : float or (m2,) shaped vector, bandwidth for the gram matrices.
            if vector : gamma = [alpha * (0.5/Var[i]**2) for i in n_clusters]
    Output:
        CsCt_list : list [ [C_sqrt, C_trace] for covariance matrices C, one per frame.] of length a.
            a = total # frames in all trajectories of input.
            CsCt_list ready for SO_distance_():
                - Slow at high number of landmarks.
    '''
    csAs = clamp_list_(csAs)
    n_frames_total = sum([x.shape[0] for x in csAs])
    m2 = cs_landmarks.shape[0]
    Hs = np.zeros([n_frames_total,1,m2])
    CsCt_list = []
    for i in range(len(csAs)):
        G = get_gram_features_(csAs[i], cs_landmarks, gamma = gamma)[0] # (N,m,m2)
        N = G.shape[0]
        for j in range(N):
            W = G[j] # W ~ (m,m2)
            C = W.T.dot(W)/N # (m2,m2)
            C_sqrt = psd_mat_power_(C, 0.5) # (m2,m2) # m2 is constant.
            C_trace = np.trace(C)
            CsCt_list.append( [C_sqrt, C_trace] )
    return CsCt_list


'''
from sklearn.kernel_ridge import KernelRidge

softmax_ = lambda x, axis=1, T=0.1 : np.exp(x/T) / np.exp(x/T).sum(axis=axis,keepdims=True)

class LR:
    # Can ridge regreession approximate clustering assigments for (faster) evalutation on each molecule in frames?
    # [h_test,...,h_test] for molecules . mean() -> histogram for frame -> HD.
    # Not very well.

    def __init__(self,
                 x_train,
                 y_train, # cluster assigments to train on.
                 gamma = 1.0,
                ):
        y_train = np.eye(y_train.max()+1)[y_train]
        
        if gamma is not None:
            self.krr = KernelRidge(alpha=gamma)
            self.krr.fit(X=x_train, y=y_train)
        else:
            x_train = self.pad_(x_train)
            self.theta = np.linalg.pinv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
        self.gamma = gamma
        
    def pad_(self, x): # instead of subtracting mean.
        return np.concatenate([x, np.ones([x.shape[0],1])], axis=1)
    
    def f_(self, x_test, T = 0.1):
        if self.gamma is not None:
            y_test = self.krr.predict(x_test)
        else:
            x_test = self.pad_( x_test )
            y_test = x_test.dot(self.theta)
            
        h_test = softmax_(y_test ,axis=1, T=T)
        c_test = np.argmax(h_test, axis=1)
        return c_test
''' 