import numpy as np

from utils import clamp_list_

try: import tensorflow as tf 
except: print('no tensorflow found. This only effects tsne_() which is not used.')

## PCA:

class cPCA:
    '''
    Two linear methods of reducing dimensionality of torsions in a molecule.
        Works best when all (proper + improper) torsions are given.
    '''
    def __init__(self,
                 X,
                 c = None,
                ):
        '''
        Input:
            X : (N,n_torsions) shaped array of conformers of single molecule.
            c : optional : (N,) shaped array of cluster assigments.
        '''
        X = np.array(X)
        self.f_ = lambda x : np.exp(1j*x) / np.sqrt(x.shape[1])
        X = self.f_(X)
        self.mu = X.mean(0, keepdims=True)
        self.MU = self.mu[np.newaxis,:,:]
        self.f_ = lambda x, mu : np.exp(1j*x) / np.sqrt(x.shape[1]) - mu

        self.K = 1
        if c is None: self.init_pca_(X)
        else: self.init_lda_(X=X, c=c)

    def init_pca_(self, X):
        X -= self.mu
        self.C = np.conjugate(X).T.dot(X) / X.shape[0]
        U,s = np.linalg.svd(self.C)[:2]
        self.U = np.conjugate(U)
        self.s = s
    
    def init_lda_(self, X, c):

        K, d = int(np.max(c)+1), X.shape[1] ; self.K = K
        
        Cb,Cw = (np.eye(d)*0).astype('complex128'), (np.eye(d)*0).astype('complex128')
        for i in range(K):
            selector = (np.where(c == i, 1, -1)*(np.arange(len(c))+1))
            selector = (np.delete(selector, np.where(selector < 0 )[0].tolist(), axis=0)-1)
            x = X[selector]
            mux = x.mean(axis=0)
            Cw += np.conjugate(x-mux).T.dot(x-mux) / len(x)
            Cb += np.conjugate(x-self.mu).T.dot(x-self.mu) / len(x)
        Cw/=K ; Cb/=K
        
        # Cb/Cw : cov between clusters (normalised by cov within clsuters) are maximised by U.
        Uw,Lw = np.linalg.svd(Cw)[:2]
        Ww = Uw.dot((1/np.sqrt(Lw+1e-10))*np.eye(d))
        Cb_white = np.conjugate(Ww).T.dot(Cb).dot(Ww)
        U1,L1 = np.linalg.svd(Cb_white)[:2]
        U = Ww.dot(U1)

        self.U = np.conjugate(U) # 
        self.s = L1
    
    @property
    def feature_ranks(self):
        # informative only when superived (self.U comes from lda).
        n_dims = 3
        return np.array([self.s[i]*np.abs(self.U[:,i])**2 for i in range(n_dims)]).T
    
    def sandwich_(self, x_r, x_i):
        return np.concatenate([np.stack([x_r[...,i],x_i[...,i]],axis=-1) for i in range(x_r.shape[-1])],axis=-1)
    
    def unit_torus_(self, x_r, x_i):
        return np.arctan2(x_r, x_i)
    
    def torus_(self, x_r, x_i):
        r = np.sqrt(x_r**2+x_i**2) # / self.K # scale manually.
        phi = self.unit_torus_(x_r, x_i)
        return self.sandwich_(phi, r)
        
    def embed_torsions_(self, As, n_PCs=2):
        '''
        Input:
            A : list [(N,m,n_torsions) shaped arrays.]
        Output:
            Zs : list [(N,m,2*n_PCs) shaped array. periodic = [True,False]*n_PCs]
                periodic axes on [-pi,pi], non periodic axes on [0,1].
        '''
        As = clamp_list_(As)
        Zs = []
        for i in range(len(As)):
            Zi = self.f_(As[i], self.MU).dot(self.U[:,:n_PCs])
            Zs.append( self.torus_(Zi.real, Zi.imag) )
        return Zs
    
    def embed_torsions_mol_(self, X, n_PCs=2):
        '''
        Input:
            X : (N,n_torsions) shaped array of conformers of single molecule.
        Output:
            z : (N,2*n_PCs) shaped array. periodic = [True,False]*n_PCs
                periodic axes on [-pi,pi], non periodic axes on [0,1].
                z[:,:3] for visualising.
        '''
        z = self.f_(np.array(X), self.mu).dot(self.U[:,:n_PCs])
        z = self.torus_(z.real, z.imag)
        return z

## tSNE:

def tsne_(D,                    # distance matrix.
          lr : float = 0.0001,  # learning rate.
          n_itter : int = 1000, # number of optimisation steps
          dim : int = 3,        # dimension to visualise (2 or 3).
          drop_rate : float = 0.1, # can be 0.0
         ):
    
    learning_rate = lr
    
    D_target = tf.constant(D, dtype=tf.float32)
    
    x = tf.constant(np.random.randn(D_target.shape[0], dim), dtype = tf.float32)
    hist_errs = []
    for it in range(n_itter):
        with tf.GradientTape() as tape:
            tape.watch(x)
            p = tf.reduce_sum(x**2, axis=1, keepdims=True)
            D_moving = tf.einsum('ij->ji',p) + p - 2.0*tf.einsum('ij,kj->ik',x,x)
            err = tf.reduce_sum((D_target-D_moving)**2)
        grad = tape.gradient(err, x)
        
        grad = tf.keras.layers.Dropout(rate=drop_rate)(grad, training=True)
        x -= grad*learning_rate
        hist_errs.append(err.numpy())
        
        if it in [1,100,200,300,400,500,600]:
            if hist_errs[-1] - hist_errs[-2] > 0:
                print('maybe reduce learning rate (lr)')
            else: pass
        else: pass
    return x.numpy(), hist_errs

