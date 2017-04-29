import model_utils
from mdtraj import io
from thor import scatter
import numpy as np
import itertools, sys, thor
import mdtraj as md
import cPickle as pickle

""" 
Simulate the diffuse intensities for coordinates in input_pdb, with the disorder
defined by the input covariance matrix (input_V) -- thus this assumes Gaussian 
displacements. Intensities are simulated on the grid of q-vectors defined by the 
maps specifications in system.pickle.

Inputs: input_pdb, coordinates for which to calculate intensities
        system.pickle, dictionary containing map specifications that define q-grid
        input_V, covariance matrix that defines disorder 
        savepath, path at which to save intensity array

Usage: python diffuse_V.py [input_pdb] [system.pickle] [input_V] [savepath]

"""

def compute_qvecs(system):
    """ 
    Compute q vectors and their magnitudes for given hkl indices. Generate grid of
    q vectors, but mask those above a specified resolution.
    """

    # transform hkl to qvecs by deorthogonalization matrix
    A_inv = model_utils.deorth_matrix(system)
    hkl_grid = np.array(list(itertools.product(system['bins']['h'], system['bins']['k'], system['bins']['l'])))
    q_vecs = 2*np.pi*np.inner(A_inv.T, hkl_grid).T 

    s_mags = np.linalg.norm(q_vecs/(2*np.pi), axis=1)
    s_mags[np.where(s_mags==0)] = 0.001 # ensure that origin isn't clipped
    res_valid = np.where(1.0/s_mags > system['d'])[0]

    mask = np.zeros(q_vecs.shape[0])
    mask[res_valid] = 1
    
    return q_vecs[res_valid], mask


if __name__ == '__main__':

    # load input information
    pdb = md.load(sys.argv[1])
    system = pickle.load(open(sys.argv[2], "rb"))
    V = np.load(sys.argv[3])
    savepath = sys.argv[4]
    
    # generate q-grid and compute I_diffuse
    detector, Id_grid = compute_qvecs(system)
    Ib, Id = scatter.simulate_diffuse(pdb, detector, V, ignore_hydrogens=False, device_id=0)
    Id_grid[Id_grid==1] = Id
    np.save(savepath + "_Id.npy", Id_grid)
