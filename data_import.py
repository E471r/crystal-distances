import numpy as np

from rdkit import Chem
from rdkit.Chem.Lipinski import RotatableBondSmarts

import mdtraj

import os

from utils import TIMER

##

def find_index_by_name_(string_in_filename : str, list_of_paths : list):
    a = 0
    where = []
    for path in list_of_paths:
        if string_in_filename in path:
            where.append(a)
        a+=1
    if len(where) == 1: where = where[0]
    else: pass
    return where

def get_torsion_indices_(mol, # drkit mol object.
                         verbose = True,
                        ):
    """ For unsupervised finding proper torsional angles in single molecule.
        Inputs:
            mol : initialised rdkit object. Used by XYZ.
        Output:
            torsion_indices : (n_torsions, 4) shaped list of lists.
    """
    for i, a in enumerate(mol.GetAtoms()): a.SetAtomMapNum(i)
    rot_atom_pairs = mol.GetSubstructMatches(RotatableBondSmarts)
    rot_atom_pairs = np.array([list(pair) for pair in rot_atom_pairs])
    n_torsions = len(rot_atom_pairs)

    if verbose: print(n_torsions,'torsional angles found in this molecule, between atom pairs:') ; [print(x) for x in rot_atom_pairs]
    else: pass

    am = Chem.rdmolops.GetAdjacencyMatrix(mol) ; node_degree = am.sum(0)
    
    torsion_indices = []
    for i in range(n_torsions):
        b = rot_atom_pairs[i,0]
        c = rot_atom_pairs[i,1]
        a_ = list(set(np.where(am[b]>0)[0]) - {c})
        d_ = list(set(np.where(am[c]>0)[0]) - {b})
        a = a_[np.argmax(node_degree[a_])]
        d = d_[np.argmax(node_degree[d_])]
        torsion_indices.append([a,b,c,d])
    torsion_indices = np.array(torsion_indices).tolist()

    if verbose: print('\nchosen torsion indices:') ; [print(x) for x in torsion_indices]
    else: pass

    return torsion_indices

def get_box_(traj_openmm_boxes_in_frame):
    return np.array([ [traj_openmm_boxes_in_frame[0].x,
                       traj_openmm_boxes_in_frame[0].y,
                       traj_openmm_boxes_in_frame[0].z],

                      [traj_openmm_boxes_in_frame[1].x,
                       traj_openmm_boxes_in_frame[1].y,
                       traj_openmm_boxes_in_frame[1].z],

                      [traj_openmm_boxes_in_frame[2].x,
                       traj_openmm_boxes_in_frame[2].y,
                       traj_openmm_boxes_in_frame[2].z] ]).T

class XYZ:
    def __init__(self,
                 path_sm_top,        # like 'path/single_molecule.pdb'
                 paths_tops : list,  # like ['path/filename1.gro', 'path/filename2.gro', ...]
                 paths_trajs : list, # like ['path/filename1.xtc', 'path/filename2.xtc', ...]
                 units : str = 'A',
                ):
        """ For getting cartesian coordinates of crystal trajectories into python using mdtraj:

        Inputs: * Indexing of atoms in a molecule has to match in all of these:
            path_sm_top : path/file (pdb) of single molecule in vacuum (including hydrogens *).
            paths_tops : list of path/file (e.g., gro or top) of crystals in vacuum.
            paths_trajs : list of path/file (e.g., xtc) of trajectories associated with paths_tops.
            units : str : 'A' or 'nm'. Assuming the inputs above are in nanometres, default is 'A'.

        """
        # checking input 'units':
        
        units = units.lower()
        if units in ['angstrom', 'angstroms', 'a']:
            self.ten_or_one = 10.0
        elif units in ['nanometer', 'nanometers', 'nm']:
            self.ten_or_one = 1.0
        else: print(" !! : XYZ.__init__() : input 'units' was not recognised. ")
        self.units = units
        
        # making use of input 'PDB_single_mol':
        
        mol = Chem.MolFromPDBFile(path_sm_top, removeHs = False)
        for i, a in enumerate(mol.GetAtoms()): a.SetAtomMapNum(i)
        self.atomic_masses = np.array([x.GetMass() for x in mol.GetAtoms()])
        self.n_atoms_all = len(self.atomic_masses)
        self.heavy_atomic_masses = self.atomic_masses[np.where(self.atomic_masses > 1.008)[0]]
        self.n_atoms_heavy = len(self.heavy_atomic_masses)
        self.mol_full = mol
        
        # trajectory_files are stored, extracted by indexing this list in self.return_crystal_arrays_().

        self.path_sm_top = path_sm_top
        self.paths_tops = paths_tops
        self.paths_trajs = paths_trajs
        self.n_trajs = len(paths_trajs)
        if self.n_trajs != len(self.paths_trajs):
            print(' !! : XYZ.__init__() : unequal list lengths : len(paths_tops) != len(paths_trajs). ')
        else: print('Paths for', self.n_trajs, 'trajectories provided.')
    
    def xtc_to_array_(self,
                      path_top : str, # pdb, gro
                      path_traj : str, # xtc, pdb
                      keep_hydrogens : bool = False,
                      apply_gmx_no_jump : bool = False,
                      verbose : bool = False,
                     ):
        
        if apply_gmx_no_jump: # try:
            os.system("echo 0 0 | gmx trjconv -f "+path_traj+" -s "+path_top+" -center -pbc nojump -o "+path_traj[:-4]+"__nj.xtc")
            path_traj = path_traj[:-4]+"__nj.xtc"
        else: pass
        
        traj = mdtraj.load(path_traj, top=path_top)
        heavy_inds = [atom.index for atom in traj.topology.atoms if atom.element.symbol != 'H']
        coordinates = traj.xyz * self.ten_or_one # default is in nm (so 1).
        N, mn = coordinates.shape[:2]
        m = int(mn / self.n_atoms_all) # number of molecules.
        
        if keep_hydrogens: R = coordinates.reshape(N, m, self.n_atoms_all, 3)                   # (N, m, n, 3)
        else:              R = coordinates[:,heavy_inds,:].reshape(N, m, self.n_atoms_heavy, 3) # (N, m, n_h,3)
        
        traj_shape = R.shape
        
        boxes = np.array([get_box_( traj.openmm_boxes(i) ) * self.ten_or_one for i in range(N)])
        # the box vectors in boxes are stacked vertically.
        box_vector_lengths = np.linalg.norm(boxes[0], axis=0)
        min_box_length = np.min(box_vector_lengths)
        
        if verbose:
            print ('Before frame indexing, traj shape:',str(traj_shape)+' %s.' % ('(hydrogens present)' if keep_hydrogens else '(hydrogens removed)'))
        else: pass

        return R, boxes, min_box_length
    
    def import_files_by_index_(self,
                               indices_trajs : list,
                               keep_hydrogens : bool = False,

                               skip_first_frame : bool = True,
                               stride : int = 1,
                               only_n_frames_from_ends : int or bool = False,
                               stack_n_consecutive_frames : int = False,

                               apply_no_jump : bool = False,
                               no_jump_method : str = 'gmx',
			       apply_local_no_jump_to  : str or list= 'all',
			       
                               center_supercells : bool = False,
                               verbose : int = 2,
                               ):
        """ 
        Inputs:
            indices_trajs : list of indices referring to which files (in which order) to import.
                The indices apply to paths_tops initialised earlier.
            keep_hydrogens : bool : False in this case.
            skip_first_frame : bool : True, because that frame seems to be from paths_tops, too symmetric.
            stride : int.
            only_n_frames_from_ends : bool or int : remove frames from the middle of trakectories. keep n frames from ends.
            stack_n_consecutive_frames : bool or int : False or molecules from neighbouring n frames pooled on top of each other.
            
	    apply_no_jump : bool : False here because this was already done in gromacs prior (faster).
                # TODO: test again if True case is still working. # ~yes
            no_jump_method : str : 'gmx' or 'anystring' 
                # TODO: test again if apply_no_jump working. # ~yes
            apply_local_no_jump_to : # in a few cases gmx cant do nojump correctly in certain NPT sim.
                 any str -> applied to all indices_trajs
                 list as subset of  indices_trajs -> applied for only those selected
		
            center_supercells : bool : not implemented.
            verbose : int : 0 = silent, 1 = loading bar, 2 = text with names (not loading bar).
        Outputs:
            None. Everything done here is saved in attributes at the bottom or in self.
                coordinates : list of (N,m,n,3) shaped arrays.
        """

        indices_trajs = list(set(indices_trajs))
        if np.max(indices_trajs) > self.n_trajs - 1:
            print(" !! : XYZ.return_crystal_arrays_() : input 'indices_trajs' has incompatible indices.")
        else: self.inds_trajs = indices_trajs
        
        ##
        self.keep_hydrogens = keep_hydrogens
        mol = Chem.MolFromPDBFile(self.path_sm_top, removeHs = not self.keep_hydrogens)
        for i, a in enumerate(mol.GetAtoms()): a.SetAtomMapNum(i)
        self.mol = mol
        self.rot_atom_pairs = self.mol.GetSubstructMatches(RotatableBondSmarts)
        ##

        if apply_no_jump and no_jump_method == 'gmx': apply_gmx_no_jump = True
        else: apply_gmx_no_jump = False ; self.nj_imported_once = False

        self.Rs = [] ; self.boxes = [] ; self.min_box_lengths = []

        if verbose == 1:
            timer = TIMER(len(indices_trajs))
        else: pass
        n_imported = 0
        for ind in indices_trajs:
            path_top = self.paths_tops[ind]
            path_traj = self.paths_trajs[ind]
            
            if verbose == 2: 
                print('Importing trajectory with index:', ind)
                print('This is trajectory with name:', path_traj.split('/')[-1])
            elif verbose == 1:
                timer.check_(n_imported)
            else: pass
            
            R, boxes, min_box_length = self.xtc_to_array_(path_top = path_top,
                                                          path_traj = path_traj,
                                                          keep_hydrogens = keep_hydrogens,
                                                          apply_gmx_no_jump = apply_gmx_no_jump,
                                                          verbose = bool(min(1,verbose-1)))
            self.min_box_lengths.append(min_box_length)
            
            if skip_first_frame:
                R = R[1:] ; boxes = boxes[1:]
            else: pass            
            
            if stride > 1:
                stride_inds = np.arange(0, R.shape[0], stride)
                R = R[stride_inds] ; boxes = boxes[stride_inds]
            else: pass

            if apply_no_jump and not apply_gmx_no_jump:
                if not self.nj_imported_once: from utils_jit import no_jump_ ; self.nj_imported_once =  True
                else: pass
                if type(apply_local_no_jump_to) == str: R = no_jump_(R, boxes)
                elif ind in apply_local_no_jump_to:     R = no_jump_(R, boxes)
                else: pass 
            else: pass
            
            if center_supercells:
                'R = *implement*(R)'
                print("Option 'center_supercells' is not implemented.")
            else: pass
            
            if only_n_frames_from_ends is not False:
                inds_start = np.arange(only_n_frames_from_ends).tolist()
                inds_end = np.arange(R.shape[0]-only_n_frames_from_ends, R.shape[0]).tolist()
                inds_tails = inds_start + inds_end
            
                boxes = boxes[inds_tails]
                R = R[inds_tails]
			
                #R_start = np.concatenate(R[inds_start])
                #R_end = np.concatenate(R[inds_end])
                #R = np.stack([R_start, R_end])
            else: pass

            if stack_n_consecutive_frames is not False:
                N,m,n = R.shape[:3]
                if N%stack_n_consecutive_frames == 0:
                    n_stack = stack_n_consecutive_frames
                    R = R.reshape(N//n_stack, m*n_stack, n, 3)
                    ''' Reshape above is same as:
                    R_out = np.zeros([N//n_stack,m*n_stack,n,3])
                    for i in range(N//n_stack):
                        R_out[i] = np.concatenate([x for x in r[i*n_stack:i*n_stack+n_stack]], axis=0)
                    '''
                else: print('!! : stack_n_consecutive_frames : could not because of trajectory length. Skipped.')
            else: pass

            n_imported += 1
            if verbose == 2:
                print('After frame indexing, traj shape:', R.shape)
                print('Progress:',n_imported,'of',len(indices_trajs),'trajectories imported.')
                print('')
            else: pass

            self.Rs.append(R)
            self.boxes.append(boxes)
            
    @property
    def names_imported(self):
        return np.array([x.split('/')[-1] for x in self.paths_trajs])[self.inds_trajs]

    @property
    def min_box_length(self):
        return np.min(self.min_box_lengths)
    
    @property
    def coordinates(self):
        return self.Rs
    
    @property
    def masses(self):
        if self.keep_hydrogens: return self.atomic_masses
        else: return self.heavy_atomic_masses
    
    @property
    def torsion_indices(self):
        return get_torsion_indices_(self.mol)

    @property
    def all_torsion_indices(self):
        am = Chem.rdmolops.GetAdjacencyMatrix(self.mol)
        all_torsion_indices = []
        for i in range(am.shape[0]):
            for j in range(am.shape[0]):
                if i < j:
                    set_a = set(np.where(am[i] == 1)[0].tolist()) - set([j])
                    set_b = set(np.where(am[j] == 1)[0].tolist()) - set([i]) 
                    if len(set_a) > 0 and len(set_b) > 0:
                        list_a = list(set_a - set_b)
                        list_b = list(set_b - set_a)
                        if len(list_a) > 0 and len(list_b) > 0:
                            all_torsion_indices.append([list_a[0],i,j,list_b[0]])
                        else: pass
                    else: pass       
                else: pass
        self.am = am
        return all_torsion_indices
    
    
