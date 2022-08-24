
def wrapper_(f,args):
    return f(*args)

try:
    try:
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

        einsum_ = lambda test_torch, X : wrapper_(torch.einsum, [test_torch] + [torch.as_tensor(x) for x in X]).numpy()
        
    except:
        import tensorflow as tf

        def cdist_sqeuclidean_row_( X : np.ndarray,
                                    Y : np.ndarray = None):

            X = tf.constant(X)

            if Y is None: Y = X
            else: Y = tf.constant(Y)
            
            p1 = tf.reduce_sum(X**2, axis=2, keepdims=True)
            p2 = tf.reduce_sum(Y**2, axis=2, keepdims=True)
            D = p1 + tf.einsum('ijk->ikj', p2) - 2.0*tf.einsum('ijk,ilk->ijl', X, Y)
            D = tf.where(D<0.0,0.0,D)
            return D.numpy()

        einsum_ = lambda text_tf, X : wrapper_(tf.einsum, [text_tf] + [tf.constant(x) for x in X]).numpy()

except:
    import numpy as np
    
    def cdist_sqeuclidean_row_( X : np.ndarray,
                                Y : np.ndarray = None):

        if Y is None: Y = X
        else: pass    
        
        p1 = (X**2).sum(2, keepdims=True)
        p2 = (Y**2).sum(2, keepdims=True)
        D = p1 + np.einsum('ijk->ikj', p2) - 2.0*np.einsum('ijk,ilk->ijl', X, Y)
        D = np.where(D<0.0,0.0,D)
        return D

    einsum_ = lambda text_np, X : wrapper_(np.einsum, [text_np] + X)

