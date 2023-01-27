import numpy as np

from utils import sta_array_

from utils_jit import periodic_W_F_, interval_W_F_ # non-linearities for kde.

''' Cant get this flexible import to work correctly yet.
try:
    try:
        import torch
        einsum__ = torch.einsum
        clamp_type_ = lambda x : torch.tensor(x)
    except:
        import tensorflow as tf
        einsum__ = tf.einsum
        clamp_type_ = lambda x : tf.constant(x, dtype = tf.float64)
except:
    einsum__ = np.einsum
    clamp_type_ = lambda x : x
    print('! : kde : could not import pytorch or tensorflow, slower np.einsum is used instead.')   
'''
import torch
einsum__ = torch.einsum
clamp_type_ = lambda x : torch.tensor(x)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def info_KDE_():
    info = ['## KDE here is landmark kernel density estimator. ## ',
            '',
            'Template : ',
            '',
            'kde = KDE(x : np.ndarray, periodic = True, weights = None, x_range = None, n_bins : list or int = 40, param = 200, verbose = True)',
            '',
            'Initialisation inputs : ',
            '',
            '    x : (N,dim) shaped array of data. ',
            '    periodic : bool or list/array (mask) of ones and zeros (ones = True) ',
            '    weights: default None, or (N,1) or (N,) shapes array of weights. ',
            '    x_range: list (e.g., [a,b]) if dim=1, or list of lists if dim>1. ',
            '    n_bins: int or list for # bins for marginals. ',
            '    param: float or list of length dim of bandwidths (gamma) for each variable. ',
            '       [This parameter is carefully tuned heuristically once at the start.]',
            '    verbose: default True, to print warnings. ',
            '',
            'Properties (once initialised) : ',
            '',
            '    kde.histogram: histogram array (sum()=1), shaped based on n_bins parameter. ',
            '    kde.normalised_histogram : scaled histogram by the area of the range. ',
            '    kde.axes: list shaped [(n_bins[k],)]*dim of axes (marginal) of histogram bin centres. ',
            '',
            '    kde.px: (N,1) shaped array of continuous probabilities (not normalised).',
            '    kde.normalised_px : correctly scaled continuous probabilities. ',
            '',
            '    kde.dp_dx: (N,dim) shaped array of continuous derivatives of px. ',
            '    kde.normalised_dp_dx : continuous derivatives of normalised_px. ',
            '',
            'Methods (once initialised) : ',
            '',
            '    kde.evaluate_points_(x, return_gradient = True, normalised = true, verbose = True) ',
            '        returns normalised px (and dp_dx) for any input x within the x_range domain. ',
            '        points outside the domain are ignored and message shown if verbose.',
           ]
    for line in info:
        print(line)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def clamp_shape_(x):
    x = np.array(x) ; N = x.shape[0]
    if len(x.shape) == 1: return x.reshape(N,1), N, 1
    else: return x, N, x.shape[1]

def get_x_ranges_(x): # if x_range not provided.
    return np.array([[a,b] for a,b in zip(x.min(0), x.max(0))]) # (dim,2)

def get_inds_in_range_(x, x_range): # indices of points which are in x_range.
    dim = x.shape[1]
    bools = np.stack([np.where((x[:,k]>=x_range[k,0])&(x[:,k]<=x_range[k,1]), 1, 0) for k in range(dim)],axis=1)
    inds_in_range = np.where(bools.sum(1)==dim)[0]
    return inds_in_range

def wrapper_(f,args):
    return f(*args)

class KDE:
    def __init__(self,
                 x : np.ndarray,
                 periodic : bool or list = True,
                 weights : np.ndarray = None, # TODO: to verify this works correct for >1D.
                 x_range : list = None,
                 n_bins : list or int = 40,
                 param : list or float = 200,
                 verbose : bool = True,
		 longdataset_block_size = None,
                 ):  
        """ info_KDE_()
        """
        # checking input : x : (1/2)
        x, N, dim = clamp_shape_(np.array(x))
        self.N = N
        self.dim = dim

        # checking parameter: n_bins 
        if type(n_bins) is int:
            self.n_bins = [n_bins] * self.dim
        elif type(n_bins) is list:
            if len(n_bins) != self.dim:
                if verbose: print('!! : n_bins incorrect length, defaulting to: n_bins = [n_bins[0]]*dim')
                self.n_bins = [n_bins[0]]*self.dim
            else: self.n_bins = n_bins
        else: print('!! : incorrect type(n_bins)')

        # checking parameter: param
        if type(param) is float or type(param) is int:
            self.param = [float(param)] * self.dim
        else:
            if len(param) != self.dim:
                print('!! : param incorrect length, defaulting to: param = [param[0]]*dim')
                self.param = [param[0]]*self.dim
            else:  self.param = np.array(param).astype(float)

        # checking parameter: x_range
        if self.dim == 1 and x_range is not None:
            x_range = np.array(x_range).reshape(1,2)
        else: pass
        if x_range is None or np.array(x_range).shape != (self.dim,2):
            if verbose: print('!! : defaulting to: default range.')
            self.x_range_0 = get_x_ranges_(x)
        else:
            self.x_range_0 = np.array(x_range)

        # checking input : x : (2/2)
        self.inds_in_range_0 = get_inds_in_range_(x, self.x_range_0)
        x = x[self.inds_in_range_0]
        self.N_effective = len(self.inds_in_range_0)
        
        if self.N_effective/self.N != 1.0:
            if verbose:
                print(self.N - self.N_effective,'of',self.N,'input samples are outside of specified x_range.')
                print('Indices of valid points can be accessed under: self.inds_in_range_0')
            else: pass
        else: pass
        
        self.x_abscisa = [np.linspace(self.x_range_0[k,0] + (self.x_range_0[k,1]-self.x_range_0[k,0])*0.5/self.n_bins[k],
                                      self.x_range_0[k,1] - (self.x_range_0[k,1]-self.x_range_0[k,0])*0.5/self.n_bins[k],
                                      self.n_bins[k]) for k in range(self.dim)] # [(n_bins[k],)]*dim
        
        x, to_01_scaling_factors = self.fit_x_to_01_(x) # x is now ready for histogram.
        self.to_01_scaling_factors = to_01_scaling_factors
        self.px_norm_constant = np.product(to_01_scaling_factors*np.array(self.n_bins))
        self.dp_dx_norm_constants = (to_01_scaling_factors * self.px_norm_constant).reshape(1,self.dim)

        # checking input : weights
        if weights is not None:
            weights = np.array(weights)
            if weights.shape[0] != N:
                print('!! : len(x) != len(weights), defaulting to: uniform weights.')
                self.weights = np.ones([self.N_effective,1])
            else:
                weights = weights.reshape(N,1)[self.inds_in_range_0]
                self.weights = self.N_effective * weights / weights.sum()
        else:
            self.weights = np.ones([self.N_effective,1])
            
        ## summary:
        # self.n_bins : list ~ (dim,)
        # self.param : list ~ (dim,)
        # self.x_range : array ~ (dim,2)
        # x : array ~ (N_effective, dim)
        # weights : array ~ (N_effective, 1)
        #
        # self.to_01_scaling_factors : array ~ (dim,)
        ##
        if type(periodic) is not bool:
            self.mixed_BCs = []
            for value in np.array(periodic).reshape([self.dim,]):
                if value in [1,1.0]:
                    self.mixed_BCs.append(periodic_W_F_)
                else:
                    self.mixed_BCs.append(interval_W_F_)
        else:
            if periodic: self.mixed_BCs = [periodic_W_F_]*self.dim
            else:        self.mixed_BCs = [interval_W_F_]*self.dim
        self.periodic = periodic
            
        self.letters = 'abcdefghvwxyz'
	self.list_letters = ['i' + self.letters[k] + ',' for k in range(self.dim)]
        ##
        # ['ia,ib,ic->abc']
        #string_h = ''.join(['i'+_+',' for _ in self.letters[:self.dim]])[:-1]+'->'+''.join(_ for _ in self.letters[:self.dim])
        self.string_h = [''.join(self.list_letters)[:-1] + '->' + self.letters[:self.dim]]
        
        # ['ia,ib,ic,abc->i']
        #string_px = ''.join(['i'+_+',' for _ in self.letters[:self.dim]])+''.join(_ for _ in self.letters[:self.dim])+'->'+'i'
        self.string_px = [''.join(self.list_letters) + self.letters[:self.dim] + '->i']
        
        # [['ib,ic,abc,ia->i'], ['ia,ic,abc,ib->i'], ['ia,ib,abc,ic->i']]
        self.strings_dp_dx = [[''.join(self.list_letters[:k] + self.list_letters[k+1:]) + self.letters[:self.dim] + ',' + self.list_letters[k][:-1] +'->i'] for k in range(self.dim)]
        ##
	
        if longdataset_block_size is None: self.place_WF_matrices_and_the_histogram_(x)
        else: self.place_histogram_longdataset_(x, block_size = longdataset_block_size) # need to use evalute points to get p(x) when this is used, because the necesary matrices are not stored to prevent momory overload.

    def fit_x_to_01_(self, x):
        x_out = np.zeros([x.shape[0], self.dim])
        to_01_scaling_factors = np.zeros([self.dim,])

        for k in range(self.dim):
            xk, Jk = sta_array_(x[:,k], input_range = self.x_range_0[k].tolist(), target_range = [0.0, 1.0], return_J = True)
            x_out[:,k] = xk
            to_01_scaling_factors[k] = Jk
        return x_out, to_01_scaling_factors 

    def place_WF_matrices_and_the_histogram_(self, x : np.ndarray):

        list_W = []
        self.list_Wt = []
        self.list_Ft = []

        weights = self.weights**(1/self.dim)
        for k in range(self.dim):
            W_k, Wt_k, Ft_k = self.mixed_BCs[k](x[:,k], n_bins=self.n_bins[k], param=self.param[k])
            list_W.append(clamp_type_(W_k * weights))
            self.list_Wt.append(clamp_type_(Wt_k))
            self.list_Ft.append(clamp_type_(Ft_k))

        self._histogram = np.array(wrapper_(einsum__, self.string_h + list_W  )) / self.N_effective
        self._histogram /= self._histogram.sum()
        self._histogram = clamp_type_(self._histogram)
	
    def place_histogram_longdataset_(self, x : np.ndarray, block_size = 2000):
        N = x.shape[0]
        block_size = min(block_size,N)
        n_block = N//block_size 
        n_remainder = N%block_size

        weights = self.weights**(1/self.dim)
        _histogram = 0.0 
        for i in range(n_block):
            a = i*block_size
            b = (i + 1)*block_size
            list_W = []
            for k in range(self.dim):
                W_k = self.mixed_BCs[k](x[a:b,k], n_bins=self.n_bins[k], param=self.param[k], tight=self.tight, test=self.test)[0]
                list_W.append(clamp_type_(W_k * weights[a:b]))
            _histogram += np.array(wrapper_(einsum__, self.string_h + list_W  ))
            
        if n_remainder == 0: pass
        else:
            list_W = []
            for k in range(self.dim):
                W_k = self.mixed_BCs[k](x[-n_remainder:,k], n_bins=self.n_bins[k], param=self.param[k], tight=self.tight, test=self.test)[0]
                list_W.append(clamp_type_(W_k * weights[-n_remainder:]))    
            _histogram += np.array(wrapper_(einsum__, self.string_h + list_W  ))  

        self._histogram = _histogram / self.N_effective
        self._histogram /= self._histogram.sum()
        self._histogram = clamp_type_(self._histogram)
	
    @ property
    def histogram(self):
        return np.array(self._histogram)

    @ property
    def normalised_histogram(self):
        return self.histogram * self.px_norm_constant

    @ property  
    def axes(self):
        return self.x_abscisa

    @ property
    def px(self):
        px = wrapper_(einsum__, self.string_px + self.list_Wt + [self._histogram] )
        return np.array(px)

    @ property
    def normalised_px(self):
        return self.px * self.px_norm_constant

    @ property
    def dp_dx(self):
        dp_dx = np.zeros([self.N_effective,self.dim])
        for k in range(self.dim):
            dp_dx[:,k-self.dim] = np.array(wrapper_(einsum__, self.strings_dp_dx[k] + self.list_Wt[:k] + self.list_Wt[k+1:] + [self._histogram] + [self.list_Ft[k]]))
        return dp_dx

    @ property
    def normalised_dp_dx(self):
        return self.dp_dx * self.dp_dx_norm_constants

    def evaluate_points_(self, x : np.ndarray, return_gradient : bool = True, normalised : bool = True, verbose : bool = True):
        
        if normalised:
            c_px = self.px_norm_constant
            c_dp_dx = self.dp_dx_norm_constants
        else:
            c_px = 1.0
            c_dp_dx = 1.0

        # checking input : x : (1/2)
        x, n, d = clamp_shape_(np.array(x))
        if d != self.dim: print('!! : incompatible dimensionality.')
        else: pass

        # checking input : x : (2/2)
        inds_in_range_0 = get_inds_in_range_(x, self.x_range_0)
        n_effective = len(inds_in_range_0)
        self.inds_x_eval_in_range = inds_in_range_0
        if n_effective < n and verbose:
            print('!! : Warning: Some points fall outside of initialised domain. They were removed.')
            print('...  There are',n - n_effective,'of',n,'such points (',100.*(n - n_effective)/n,'% )')
            print('...  Indices of valid points can be accessed under: self.inds_x_eval_in_range')
        else: pass
        x = x[inds_in_range_0]
        x = self.fit_x_to_01_(x)[0]

        list_Wt = []
        list_Ft = []

        for k in range(self.dim):
            Wt_k, Ft_k = self.mixed_BCs[k](x[:,k], n_bins=self.n_bins[k], param=self.param[k])[1:]
            list_Wt.append(clamp_type_(Wt_k))
            list_Ft.append(clamp_type_(Ft_k))

        px = np.array(wrapper_(einsum__, self.string_px + list_Wt + [self._histogram] ))

        if return_gradient is False:
            return px*c_px
        else:
            dp_dx = np.zeros([n_effective,self.dim])
            for k in range(self.dim):
                dp_dx[:,k-self.dim] = np.array(wrapper_(einsum__, self.strings_dp_dx[k] + list_Wt[:k] + list_Wt[k+1:] + [self._histogram] + [list_Ft[k]]))

            return  px*c_px, dp_dx*c_dp_dx

