import numpy as np
import matplotlib.pyplot as plt
import py3Dmol
# import nglview as nv

from utils_jit import cdist_, clamp_shape_traj_

##

def plot_mat_(mat,
              clip : list = None,
              show_colorbar : bool = True,
              figsize : tuple = (10,5),
              **kwargs,
             ):
    """ Plot matrix.
        mat : (N,M) shaped array.
    """
    plt.figure(figsize=figsize)
    plt.matshow(mat, fignum=1, **kwargs)
    if clip is not None: plt.clim(clip[0],clip[1])
    if show_colorbar: plt.colorbar()
    else: pass
    plt.show()

def plot_points_(X,
                la=None,
                s=10,cmap='jet',
                show_axes=True,
                figsize=(5, 3),
                autoscale=True,
                show_colorbar=True,
                **kwargs):
    """ Plot 2D or 3D points. 
        X : (N,d) shaped array of N points. 
        la = (N,) shaped array of N labels, for colour. 
    """
    d = X.shape[1]
    
    if la is None: la = np.arange(len(X))
    else: pass 
    
    fig = plt.figure(figsize=figsize)
    if d >= 3: ax = fig.add_subplot(111, projection='3d') ; ax.set_zlabel('z')
    elif d == 2: ax = fig.add_subplot(111)
    else: pass
    
    if autoscale: pass 
    else: ax.autoscale(enable=None, axis='both', tight=False)
        
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if not show_axes: ax.set_axis_off()
    else: pass
    if d >= 3: img = ax.scatter(X[:,0],X[:,1],X[:,2],s=s,c=la,cmap=cmap,**kwargs)
    elif d == 2: img = ax.scatter(X[:,0],X[:,1],s=s,c=la,cmap=cmap,**kwargs)
    else: pass
    if show_colorbar: fig.colorbar(img)
    else: pass
    plt.show()

def plot_torsion_indices_(mol, torsion_indices):
    
    """ Returns a set of 3D (rotatable) plots of the molecule, highlighting the atoms associated with the torsional angles.
        mol : mol rdkit object.
        torsion_indices : (N,4) shaped array or list.
    """

    def drawit_(m, la=None, p=None, confId=-1, thickness=.35):
        
        ## This handy plotting tool was adopted from the internet.
        
        mb = Chem.MolToMolBlock(m,confId=confId)
        if p is None:
            p = py3Dmol.view(width=300,height=300)
        p.removeAllModels()
        p.addModel(mb,'sdf')
        p.setStyle({'sphere':{'scale':thickness}})
        #p.setStyle({'stick':{'radius':thickness}})
        p.setBackgroundColor('0xeeeeee')
        p.zoomTo()
        if la is not None:
            p.setStyle({'serial':[la[1],la[2]]},{'sphere':{'color':'black','scale':thickness}})
            p.setStyle({'serial':[la[0],la[3]]},{'sphere':{'color':'grey','scale':thickness}})
            #p.setStyle({'serial':la},{'stick':{'color':'black','radius':thickness}});
        p.show()
    
    for i in range(len(torsion_indices)):
        print(i,':')
        drawit_(mol,la=torsion_indices[i])

def plot_single_supercell_(x,    # (n_molecules, n_atoms, 3)
          inds_jk = None, # [index_molecule, index_atom]
          with_bonds=True, # can take a long time
         ):
    x = clamp_shape_traj_(x)
    m,n = x.shape[1],  x.shape[2]
    y = x.reshape(m*n,3)
    x = y.reshape(m,n,3)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(y[:,0],y[:,1],y[:,2],s=1)
    
    if with_bonds:
        D = cdist_(y,y,metric='euclidean')
        for i in range(y.shape[0]):
            for j in range(y.shape[0]):
                if i > j:
                    if D[i,j] < 2:
                        show = np.stack([y[i,:],y[j,:]],axis=0)
                        ax.plot(show[:,0],show[:,1],show[:,2], color='black')
                    else: pass
                else: pass
    else: pass
            
    if inds_jk is None: pass
    else:
        j,k = inds_jk
        ax.scatter(x[j,k,0],x[j,k,1],x[j,k,2],s=20,color='red')

