import numpy as np
import matplotlib.pyplot as plt
import py3Dmol
# import nglview as nv

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

