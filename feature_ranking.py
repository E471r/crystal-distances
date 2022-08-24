import numpy as np

from histograms import histogram_np_

from utils import TIMER

####

'''
S_ : Shannon entropy. Input: histogram.
I_ : Shannon information between marginals of 2D histogram. Input: 2D histogram. Output: float >= 0.0.
nI_ : normalised I_. Input: 2D histogram. Output: float in [0.0,1.0].
'''
S_ = lambda Px : - ( Px*np.ma.log(Px) ).sum()
I_ = lambda Pxy : ( Pxy*np.ma.log(np.ma.divide( Pxy, np.outer(Pxy.sum(1),Pxy.sum(0)) )) ).sum() # S_(Pxy.sum(1)) + S_(Pxy.sum(0)) - S_(Pxy)
# nI_ = lambda Pxy : I_(Pxy) / (S_(Pxy)+1e-10)
def nI_(Pxy):
    I = I_(Pxy)
    S = S_(Pxy)
    # I/S should be a float in range [0,1], but in MNIST dataset some inaccuracies due to round-off were seen, so:  
    if S <= 0.0:
        if I <= 0.0: return 0.0
        else: return 1.0
    else: return I/S

def PcSi_(c, X,
          i : int,
          S : list = [],
          x_range : list = [-np.pi,np.pi],
          n_bins : int = 20,
          shuffle_i : bool = False):
    ''' nD histogram (i).
    c : (N,) shaped array : categorical (numerical) variable. [axis=0 in output hist.]
    X : (N,d) shaped array with all elements in x_range.
    i : int (index) not found in S. [axis=-1 in output hist.]
    S : list of ints (indices [!=i]) for other columns in X. [if empty, output is 2D array.]
    x_range : list like [a,b]. All elements in X must be in range a <= x <= b.
    n_bins : number of histogram bins for X[:,i] and all X[:,j] for j in S.
    shuffle_i : bool whether to shuffle data in column X[:,i], before histogram. [Not used here.]
        [nTE was tested for this problem and works also instead of mRMR, but having a hyperparameter (beta) is actually better.]

    Output: (len(set(c)),...,n_bins) shaped array, sums to 1 along any axes (histogram).
    '''
    N = c.shape[0]
    dim_X = len(S) + 1
    
    cSi = [c.reshape(N,1)]
    cSi += [X[:,j:j+1] for j in S]
    if shuffle_i: cSi += [X[np.random.choice(N, N,replace=False),i:i+1]]
    else:         cSi += [X[:,i:i+1]]
    cSi = np.concatenate(cSi, axis=1)

    x_range = [[c.min(),c.max()]] + [x_range]*dim_X
    n_bins = [c.max()-c.min()+1] + [n_bins]*dim_X

    return histogram_np_(cSi, x_range=x_range, n_bins=n_bins)

def PS_(X, S,
        x_range : list = [-np.pi,np.pi],
        n_bins : int = 20):
    ''' nD histogram (ii).
    X : (N,d) shaped array, with all elements in x_range.
    S : list of indices referring to columns in X.
    x_range : list like [a,b]. All elements in X must be in range a <= x <= b.
    n_bins : number of histogram bins for any X[:,j], where j in S.
    
    Output : (n_bins,n_bins) shaped array, sums to 1 along any axes (histogram).
    '''
    dim_X = len(S)
    
    S = [X[:,j:j+1] for j in S]
    S = np.concatenate(S, axis=1)
    x_range = [x_range]*dim_X
    n_bins = [n_bins]*dim_X
    
    return histogram_np_(S, x_range=x_range, n_bins=n_bins)

def mRMR_(c, X,
          Iic = None,
          Iij = None,
          dim_max : int = 3,
          x_range : list = [-np.pi,np.pi],
          n_bins : int = 20,
          beta : float = 1.0,
          verbose : bool = True):
    '''
    c : (N,) shaped array : (hard) cluster assignments (categorical variable).
    X : (N,d) shaped array, with all elements in x_range.
    Iic : (d,) shaped array or None (default).
        [Normalised MI matrix between features and c.]
    Iij : (d,d) shaped array or None (default).
        [Normalised MI matrix between pairs of features.]
    dim_max : int <= d.
    x_range : list [a,b] all elements in X must be in range a <= x <= b.
    n_bins : number of histogram bins for any column X[:,j].
    beta : float : hyperparameter for balance between Iic vs. Iij terms.
    verbose : bool : can be handy if d is high when first time running to get Iij (slowest part).
    '''
    dim = X.shape[1]
    set_X = np.arange(dim).tolist()
    
    if Iic is None:
        Iic = np.array([nI_(PcSi_(c=c, X=X, i=i, x_range=x_range, n_bins=n_bins)) for i in set_X])
    else: pass
    set_S = [np.argmax(Iic)]
    log = [Iic.max()] # [[set_X,Iic]]
    
    if Iij is None:
        if verbose: timer = TIMER(dim**2)
        else: pass
        a = 0
        Iij = np.eye(dim)*0.0
        for i in range(dim):
            for j in range(dim):
                if i >= j:
                    Iij[i,j] = Iij[j,i] = nI_(PS_(X, [i,j], x_range=x_range, n_bins=n_bins))
                else: pass
                a+=1
            if verbose: timer.print_(a)
            else: pass
    else: pass
    
    for _ in range(dim_max-1):
        f_list = []
        inds_search = list(set(set_X) - set(set_S))
        for i in inds_search:
            fi = Iic[i] - beta*Iij[i,set_S].mean()
            f_list.append(fi)
        
        set_S += [inds_search[np.argmax(f_list)]]
        log += [max(f_list)] # [[inds_search,f_list]]

    return np.array(set_S), np.array(log), Iic, Iij

# softmax_ = lambda x, axis=1, T=0.1 : np.exp(x/T) / np.exp(x/T).sum(axis=axis,keepdims=True)

cossin_ = lambda x : np.concatenate([np.cos(x),np.sin(x)], axis=1)

def selective_cossin_(X, periodic_mask):
    ''' X ~ (N,d)
        X ~ (N,n) ; n==d iff periodic_mask.sum() = 0.0, else n>d.
    '''
    X_out = []
    indexing_mask = []
    for i in range(X.shape[1]):
        if periodic_mask[i] == 1: 
            X_out.append(cossin_(X[:,i:i+1]))
            indexing_mask += [i,i]
        else: 
            X_out.append(X[:,i:i+1])
            indexing_mask += [i]
    return np.concatenate(X_out, axis=1), np.array(indexing_mask)

def selective_select_(X, set_S, indexing_mask):
    X_out = []
    for index in set_S:
        x = X[:,np.where(indexing_mask==index)[0]]
        X_out.append(x)
    return np.concatenate(X_out, axis=1)

def evaluate_linear_regression_(x_train, y_train, x_test):
    theta = np.linalg.pinv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
    y_test = x_test.dot(theta)
    return y_test

####

class FEATURE_RANKING:
    def __init__(self,
                 c, X,
                 X_periodic : np.ndarray or bool = True,
                 x_range : list = [-np.pi,np.pi],
                 n_bins : int = 20,
                ):
        '''
        c : (N,) shaped array : (hard) cluster assigments (categorical variable).
            [As usual with output of clustering algorithms: c[i] \in {0,..,K} forall i in range(N).]
        X : (N,d) shaped array, with all elements in x_range.
            [The columns are CVs (e.g., torsions) but can be evaluations of any bounded function.]
        X_periodic : bool or (N,) shaped np.ndarray with elements in {0,1}.
            [This is only relevant for the mRMR_autobeta_ function, where X is made euclidean.]
        x_range: list [a,b] all elements in X must be in range a <= x <= b, i.e., bounded.
        n_bins : number of histogram bins for any column X[:,j].
        '''

        self.N, self.dim = X.shape
        c = np.array(c).astype(int).reshape([self.N,])
        self.K = c.max()-c.min()+1

        if type(X_periodic) is bool and X_periodic: self.periodic_mask = np.ones([self.dim])
        elif type(X_periodic) == bool: self.periodic_mask = np.zeros([self.dim])
        else: self.periodic_mask = X_periodic

        self.x_range = x_range
        self.n_bins = n_bins

        self.c = c
        self.c_onehots = np.eye(self.K)[c]
        self.X = X
        self.X_for_auto, self.indexing_mask = selective_cossin_(X, self.periodic_mask)
        self.X_for_auto = np.concatenate([self.X_for_auto, np.ones([self.N,1])], axis=1)

        self.initialised = False

    def mRMR_(self,
              beta : float = 1.0,
              dim_max : int = None,
              verbose : bool = True):
        '''
        beta : float : the only actual hyperparameter. Default is 1.0 but this is unlikely best.
        dim_max : int in range [0,self.dim]
        verbose : The Iij matrix can take a while to make, loading bar displayed.
        
        Outputs:
            set_S : indices refer to columns of X. Indices are ordered in descending order of rank.
            log : some information which can be plotted. Scores.
        '''
        if dim_max is None or dim_max>self.dim: dim_max = self.dim
        else: pass

        if not self.initialised:
            set_S, log, Iic, Iij = mRMR_(self.c, self.X,
                                         Iic = None,
                                         Iij = None,
                                         dim_max = dim_max, x_range = self.x_range, n_bins = self.n_bins, beta = beta, verbose = verbose)
            self.Iic = Iic
            self.Iij = Iij
            self.initialised = True
            return set_S, log
        else:
            if verbose: print('Using stored Iic, Iij arrays, which were made using',self.n_bins,'bins.')
            else: pass
            set_S, log = mRMR_(self.c, self.X,
                               Iic = self.Iic,
                               Iij = self.Iij,
                               dim_max = dim_max, x_range = self.x_range, n_bins = self.n_bins, beta = beta, verbose = False)[:2]
            return set_S, log

    def mRMR_autobeta_(self,
                       grid_beta = np.linspace(0.2,8.0,20),
                       dim_max : int = None,
                       verbose : bool = True):
        '''
        grid_beta : uniform grid of trial floats, for grid search of beta automatically.
        dim_max : int in range [0,self.dim]. Best to set this to around self.dim//2.
        verbose : if True and self.dim very high, this is the loading bar.
        
        Outputs : 
            optimum_set_S_valid : the final answer, for best beta in grid.
            optimum_scoares_valid : coresponding scores for the indices of features in optimum_set_S_valid.
            
            # plotting error landscape: plt.plot(self._grid_beta,self._L)
                # if not convex enough can adjust dim_max, and run this function agian (it is fast at lower dim_max).
        '''
        
        if not self.initialised: self.mRMR_(beta=1.0, dim_max=1, verbose=verbose)
        else: pass

        if verbose: timer = TIMER(len(grid_beta))
        else: pass
        a = 0
        stack_set_S = []
        stack_log = []
        stack_norms = []
        if dim_max is None or dim_max>self.dim: dim_max = self.dim
        else: pass
        grid_dim_max = np.arange(1,dim_max+1)
        for i in range(len(grid_beta)):
            set_S, log = self.mRMR_(beta=grid_beta[i], dim_max=dim_max, verbose=False)[:2]
            stack_set_S.append(set_S)
            stack_log.append(log)

            norms = []
            for j in grid_dim_max:
                x_train = selective_select_(self.X_for_auto, set_S[:j], self.indexing_mask)
                y_hat = evaluate_linear_regression_(x_train = x_train,
                                                    y_train = self.c_onehots,
                                                    x_test = x_train)
                norms.append(np.linalg.norm(self.c_onehots - y_hat))

            stack_norms.append(norms)
            a+=1
            if verbose: timer.print_(a)
            else: pass
        
        L = np.array(stack_norms).mean(1)
        index_beta_grid_optimum = np.argmin(L)
        optimum_beta = np.array(grid_beta)[index_beta_grid_optimum]
        optimum_set_S = np.array(stack_set_S[index_beta_grid_optimum])
        optimum_scoares = np.array(stack_log[index_beta_grid_optimum])
        inds_valid = np.where(optimum_scoares>0.0)[0] # on mnist this thereshold removes spurious result.
        optimum_set_S_valid = optimum_set_S[inds_valid]
        optimum_scoares_valid = optimum_scoares[inds_valid]
            
        if verbose:
            print('optimum beta (from grid_beta provided) is:', optimum_beta)
            print('returning array of indices of features in order of decreasing score.')
            print('returning array of scores for these indices.')
        else: pass
        
        # not returned but just in case stored:
        #
        self._grid_beta = np.array(grid_beta)
        self._L = L
        self._index_beta_grid_optimum = index_beta_grid_optimum
        self._optimum_beta = optimum_beta
        self._optimum_set_S = optimum_set_S
        self._optimum_scoares = optimum_scoares
        self._inds_valid = inds_valid
        self._optimum_set_S_valid = optimum_set_S_valid
        self._optimum_scoares_valid = optimum_scoares_valid
        ##
        self._stack_norms = stack_norms
        self._stack_set_S = stack_set_S
        self._stack_log = stack_log
        
        return optimum_set_S_valid, optimum_scoares_valid
        
def check_clustering_agreements_(list_cs, verbose = False):
    '''
    Input:
        list_cs : list [arrays of clustering assignments from different clusterings]
            [each array should be filled with integers (cluster indices).]
    Output:
        M : (n,n) array, where n is length of the input list (list_cs).
            Pairwiese nI_ values between different clustering results.
            [np.argmax(M.mean(0)) is index of result which agrees best with others.]
    '''    
    n = len(list_cs) ; N = len(list_cs[0])
    n_clusters = max([x.max() for x in list_cs]) + 1 #len(set(list_cs[i]))
    
    list_oh = [np.eye(n_clusters)[list_cs[i]] for i in range(n)]
    M = np.eye(n)
    loading = np.arange(0,n,n//min(n,5))
    for i in range(n):
        if verbose:
            if i in loading: print(i,'/',n)
            else: pass
        else: pass
        for j in range(n):
            if j >= i:
                M[i,j] = M[j,i] = nI_(list_oh[i].T.dot(list_oh[j]) / N)
            else: pass
    #agreements = M.mean(1)
    return M


'''
def nT_(P_x1_x0_y0, P_x1_x0_ys):

    S_x1_x0_y0 = S_(P_x1_x0_y0)

    P_x1_x0    = P_x1_x0_y0.sum(-1)
    S_x1_x0    = S_(P_x1_x0)
    S_x0       = S_(P_x1_x0.sum(0))

    S_x0_y0    = S_(P_x1_x0_y0.sum(0))

    S_x1_x0_ys = S_(P_x1_x0_ys)

    S_x0_ys    = S_(P_x1_x0_ys.sum(0))

    return (S_x1_x0_ys - S_x0_ys - S_x1_x0_y0 + S_x0_y0) / (S_x1_x0 - S_x0)

def Ticj_(c, X,
          x_range,
          n_bins):
    dim = X.shape[1]
    Ticj = np.eye(dim)*0.0
    for i in range(dim):
        print(i)
        for j in range(dim):
            Pcji = PcSi_(c=c, X=X, i=i, X_subset = [j], shuffle_i=False, x_range=x_range, n_bins=n_bins)
            Pcjis = PcSi_(c=c, X=X, i=i, X_subset = [j], shuffle_i=True, x_range=x_range, n_bins=n_bins)
            Ticj[i,j] = nT_(Pcji, Pcjis)
    return Ticj
            
def find_this_(c, X,
                 Ticj = None,
                 dim_max : int = 3,
                 x_range : list = [-np.pi,np.pi],
                 n_bins : int = 20):
    
    dim = X.shape[1]
    log = []
    
    if Ticj is None:
        Ticj = Ticj_(c, X, x_range=x_range, n_bins=n_bins)
    else: pass
    
    set_X = np.arange(dim).tolist()
    init_index = np.argmax(Ticj.sum(1))
    #init_index = np.argmax([nI_(PcSi_(c=c, X=X, i=i, x_range=x_range, n_bins=n_bins)) for i in set_X])
    set_S = [init_index]
    

    for k in range(min(dim_max-1,dim-1)):
        set_S_complement = list(set(set_X) - set(set_S))

        Tic = Ticj[:,set_S].reshape(dim, len(set_S)).sum(1)
        values = Tic[set_S_complement]
        
        set_S += [set_S_complement[np.argmax(values)]]
        log.append(np.max(values))
        
    return set_S, log, Ticj
    

# S,log = find_this_(c=c, X=A_molecules,n_bins=20)
#print(S)
#plt.plot(log)

'''

