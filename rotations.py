import numpy as np
from utils import clamp_list_, clamp_range_, TIMER
from histograms import histogram_np_, histogram_kde_

####

def unitvectors_supervised_(R, three_atom_indices):
    """ 
    Inputs:
        R : array with shape (N,m,n,3) : trajectory of a crystal.
            N = # frames
            m = # molecules
            n = # atoms
        three_atom_indices : list of 3 indices for rigid core atoms in each molecule.

    Output:
        U : (N,m,3_vectors,3) array of local orthonormal coordinates.
            [u0,u1,u2] on each molecule.
    """
    R = np.array(R)
    
    A,B,C = three_atom_indices

    # R[:,:,index,:] ~ (N,m,3)
    
    AB = R[:,:,B,:] - R[:,:,A,:] # A->B
    AC = R[:,:,C,:] - R[:,:,A,:] # A->C
    
    u0 = np.cross(AB, AC) ; u0 /= np.linalg.norm(u0, axis=-1, keepdims=True) # oop
    u1 = AB / np.linalg.norm(AB, axis=-1, keepdims=True)                     # op A->B
    u2 = np.cross(u0, u1)                                                    # normal to oop and op.

    return np.stack([u0,u1,u2], axis=-2) # (N,m,3_vectors,3)

class Unitvectors_Unsupervised:
    """ ! Experimental.

    NOT used in practice. TODO: fix the reason.

    The idea is simple of using mass-weighted PCA to define 
    a local orthonormal coordinate system on each conformer 
    which is aligned with the net direction of mass. Most steps 
    here are aimed to clamp the directions of the eigenvectors 
    to fixed sections on the molecule. The 'datapoints' here 
    are the atoms and so the sign (+ or -) in projection of the 
    data (atoms) onto these eigenvectors needs to be fixed.

    This object (Unitvectors_Unsupervised) is initialized on 
    ground state molecule. The principal axes found serve as a 
    reference for what the signs should be in subsequent evaluations.

    !! Warning: This method can be correlated to configurations,
        This can be reduced by changing ws (originally atomic massed) 
        to be higher near the core.
        For example:
            Letting r = R[0,0] be a representative conformer, (n,3) array.
            ws = np.exp(-np.linalg.norm(r-r.mean(0), axis=1)/10)
            ws /= ws.sum()
            ws = ws*np.array(xyz.masses)
            uu = Unitvectors_Unsupervised(r.shape[0], ws)
    """
    def __init__(self,
                 n : int,
                 ws : list or np.ndarray,
                 d3_output : bool = False,
                 store : list = None):
        # n = # atoms.
        # ws = atom masses or custom weights.
        self.atom_set = set(np.arange(n))
        ws = np.array(ws).reshape(n,1)
        ws /= ws.sum()
        self.ws = ws
        self.I = np.eye(n)*ws
        self.n = n
        if store is None: self.store = None
        else:             self.store = store
        if d3_output: self.evaluate_ = self.d3_evaluate_
        else: self.evaluate_ = self.d2_evaluate_

    def pca_(self, r, dim_out : int): # r ~ (n,3)
        r_ = r - (self.ws*r).sum(0)
        C = r_.T.dot(self.I).dot(r_)
        U = np.linalg.svd(C)[0][:,:dim_out]
        p = r_.dot(U)
        return p, U # p ~ (n,2) or (n,3) ; U ~ (3,2) or (3,3)

    def d3_evaluate_(self, R):
        # R ~ (N,m,n,3)
        N,m = R.shape[:2]

        if self.store is not None:
            set_u0, set_u1, set_u2, comp_u0, comp_u1, comp_u2 = self.store
        else: pass
        
        Output = np.zeros([N,m,3,3])
        for i in range(N):
            for j in range(m):
                p, U = self.pca_(R[i,j], 3) # (n,3) -> (n,3)
                
                if self.store is None:
                    set_u0 = set(np.where(p[:,0]>0)[0])
                    set_u1 = set(np.where(p[:,1]>0)[0])
                    set_u2 = set(np.where(p[:,2]>0)[0])
                    comp_u0 = self.atom_set - set_u0
                    comp_u1 = self.atom_set - set_u1
                    comp_u2 = self.atom_set - set_u2
                    self.store = [set_u0, set_u1, set_u2, comp_u0, comp_u1, comp_u2]
                    u0 = U[:,0] ; u1 = U[:,1] ; u2 = U[:,2]
            
                else:
                    set_u0_i = set(np.where(p[:,0]>0)[0])
                    set_u1_i = set(np.where(p[:,1]>0)[0])
                    set_u2_i = set(np.where(p[:,2]>0)[0])
                    if len(comp_u0&set_u0_i) > len(set_u0&set_u0_i): u0 = -U[:,0]
                    else:                                            u0 = U[:,0]
                    if len(comp_u1&set_u1_i) > len(set_u1&set_u1_i): u1 = -U[:,1]
                    else:                                            u1 = U[:,1]
                    if len(comp_u2&set_u2_i) > len(set_u2&set_u2_i): u2 = -U[:,2]
                    else:                                            u2 = U[:,2]                        
                Output[i,j,0] = u0
                Output[i,j,1] = u1
                Output[i,j,2] = u2
        return Output

    def d2_evaluate_(self, R):
        # R ~ (N,m,n,3)
        N,m = R.shape[:2]

        if self.store is not None:
            set_u0, set_u1, comp_u0, comp_u1 = self.store
        else: pass
        
        Output = np.zeros([N,m,2,3])
        for i in range(N):
            for j in range(m):
                p, U = self.pca_(R[i,j], 2) # (n,3) -> (n,2)
                
                if self.store is None:
                    set_u0 = set(np.where(p[:,0]>0)[0])
                    set_u1 = set(np.where(p[:,1]>0)[0])
                    comp_u0 = self.atom_set - set_u0
                    comp_u1 = self.atom_set - set_u1
                    self.store = [set_u0, set_u1, comp_u0, comp_u1]
                    u0 = U[:,0] ; u1 = U[:,1]
                else:
                    set_u0_i = set(np.where(p[:,0]>0)[0])
                    set_u1_i = set(np.where(p[:,1]>0)[0])
                    if len(comp_u0&set_u0_i) > len(set_u0&set_u0_i): u0 = -U[:,0]
                    else:                                            u0 = U[:,0]
                    if len(comp_u1&set_u1_i) > len(set_u1&set_u1_i): u1 = -U[:,1]
                    else:                                            u1 = U[:,1]
                Output[i,j,0] = u0
                Output[i,j,1] = u1
        return Output

def get_rotational_hists_traj_(R,
                               supervised = [3,6,9], 
                               n_bins = 20,
                               use_kde = False,
                               kde_parameter = 100,
                               d3_histograms = False,
                               flatten_output = True):
    """ 
    Inputs:
        R : array with shape (N,m,n,3) : trajectory of a crystal.
            N = # frames
            m = # molecules
            n = # atoms
        supervised : list, or pre-initialised instance of Unitvectors_Unsupervised.
            if list : 3 indices for 3 atoms near the core (e.g., central ring). Faster.
            if object : Unitvectors_Unsupervised(n, ws, d3_output=d3_histograms). Slower.
        n_bins : int : small number is better, because use_kde = False is advised.

    Optional:
        use_kde : bool : False because slow here.
        kde_parameter : higher value approaches np.histogram output (less smooth).
        d3_histograms : False gives better final results.
        flatten_output : True unless only to visualise.

    Output:
        hists : (N,1,n_bins**2) shaoed array.
    """
    
    N,m,n = R.shape[:3]
    
    if type(supervised) is list:
        U = unitvectors_supervised_(R, supervised[:3]) # ~ (N,m,3_vectors,3)
    elif isinstance(supervised, Unitvectors_Unsupervised):
        U = supervised.evaluate_(R) # ~ (N,m,3_vectors,2or3)
    else:
        print("please review the input for the argument called 'supervised'.")

    if d3_histograms:
        u0, u1, u2, d = U[:,:,0], U[:,:,1], U[:,:,2], 3
    else:
        u0, u1, d = U[:,:,0], U[:,:,1], 2
    
    del U
    
    hists = np.zeros([N,1]+[n_bins]*d)
    x_range = [[0.0,3.1415927]]*d # [0,pi]
    
    if use_kde: histogram_ = histogram_kde_
    else:       histogram_ = histogram_np_
    
    inds_triu = np.triu_indices(m, k=1)
    
    # Euler angles method on full (both sides) einsum_('iAjl,iBkl->iABjk',[U,U]) # (N,m,m,3,3), 
    # still depends on kind of rotation that act on full crystal, half of them are bad.
    # If this is fixable, Euler angles has better properties (**) in the 3D histograms.
    #   (**) Random rotations give uniform density.
    # Maybe something to consider, but U not clean enough to warrant this slow approach.
    
    # Further work: hydrogen bonding network with rematch.

    # slower than loop:
    #inds_triu_A, inds_triu_B = inds_triu[0], inds_triu[1]
    #dots = einsum_('iAjl,iBjl->iABj',[U,U])[:,inds_triu_A,inds_triu_B] # (N,K,3) ; K = (m**2-m)//2.
    #dots = np.arccos(clamp_range_(dots,[-1.,1.])) # default because this uses more of available space. (*)
    #for i in range(N):
    #    hists[i,0,...] = histogram_(dots[i,:,:d],...)

    # Loop of original method (diagonal of A,B rotation matrix):

    if d3_histograms:
        for i in range(N):
            u0i = u0[i] # ~ (m,3)
            u1i = u1[i] # ~ (m,3)
            u2i = u2[i] # ~ (m,3)
            dots = np.stack([u0i.dot(u0i.T)[inds_triu],
                             u1i.dot(u1i.T)[inds_triu],
                             u2i.dot(u2i.T)[inds_triu]], axis=1) # (K,3) ; K = (m**2-m)//2.
            dots = np.arccos(clamp_range_(dots,[-1.,1.]))
            
            hists[i,0,...] = histogram_(dots, x_range=x_range, n_bins=n_bins, param=kde_parameter, periodic=False)
    else:
        for i in range(N):
            u0i = u0[i] # ~ (m,3)
            u1i = u1[i] # ~ (m,3)
            dots = np.stack([u0i.dot(u0i.T)[inds_triu],
                             u1i.dot(u1i.T)[inds_triu]], axis=1) # (K,2) ; K = (m**2-m)//2.
            dots = np.arccos(clamp_range_(dots,[-1.,1.]))

            hists[i,0,...] = histogram_(dots, x_range=x_range, n_bins=n_bins, param=kde_parameter, periodic=False)

    if flatten_output: return hists.reshape(N,1,n_bins**d)
    else:              return hists
        

def get_rotational_hists_(Rs : list or np.ndarray,
                          supervised = [3,6,9], 
                          # ^ indices of 3 atoms, or provide the initilased class here.
                          n_bins = 20,
                          use_kde = False,
                          kde_parameter = 100,
                          d3_histograms = False,
                          flatten_output = True,
                          concatenate_output = True,
                          verbose = True):
    """ For same output can do [get_rotational_hists_traj_(R,...) for R in Rs].
    A way to pool histograms from neighbouring frames:
        hists output from here can just be added/averaged along adjacent timesteps/frames. 
        No time-crystals.
    """

    Rs = clamp_list_(Rs)
    if verbose: timer = TIMER(len(Rs))
    else: pass

    hists = []
    for i in range(len(Rs)):
        hists_traj = get_rotational_hists_traj_(Rs[i],
                                                supervised = supervised,
                                                n_bins = n_bins,
                                                use_kde = use_kde,
                                                kde_parameter = kde_parameter,
                                                d3_histograms = d3_histograms,
                                                flatten_output = flatten_output)
        hists.append(hists_traj)
        if verbose: timer.check_(i)
        else: pass
    if concatenate_output: return np.concatenate(hists, axis=0)
    else: return hists


''' Euler angles from rotation matrix R.

def f1_(R):
    
    if R[2,0] not in [-1.0,1.0]:
        theta = np.arctan2(-R[2,0],np.sqrt(R[2,1]**2+R[2,2]**2))
        psi = np.arctan2(R[2,1],R[2,2])
        phi = np.arctan2(R[1,0],R[0,0])
        
    else:
        theta = - R[2,0] * np.pi / 2.0
        psi = np.arctan2(R[0,1]/theta,R[0,2]/theta)
        phi = 0.0
        
    return np.array([theta, psi, phi])

def f2_(R): # full.
    if R[2,0] not in [-1.0,1.0]:

        theta1 = - np.arcsin(R[2,0]) ; cos_theta1 = np.cos(theta1)
        theta2 = np.pi - theta1      ; cos_theta2 = np.cos(theta2)

        psi1 = np.arctan2(R[2,1]/cos_theta1, R[2,2]/cos_theta1)
        psi2 = np.arctan2(R[2,1]/cos_theta2, R[2,2]/cos_theta2)

        phi1 = np.arctan2(R[1,0]/cos_theta1, R[0,0]/cos_theta1)
        phi2 = np.arctan2(R[1,0]/cos_theta2, R[0,0]/cos_theta2)

        vector1 = np.array([theta1, psi1, phi1])
        vector2 = np.array([theta2, psi2, phi2])

    else:
        phi = 0.0

        if R[2,0] == -1.0: theta = np.pi / 2.0
        else:              theta = - np.pi / 2.0

        psi = np.arctan2(R[0,1]/theta,R[0,2]/theta)

        vector1 = np.array([theta, psi, phi])
        vector2 = np.array([theta, psi, phi])    

    return vector1, vector2

def random_rotation_():
    X = np.random.randn(10,3)
    Y = np.random.randn(10,3) 
    U,S,V = np.linalg.svd(X.T.dot(Y))
    return U.dot(V)

'''