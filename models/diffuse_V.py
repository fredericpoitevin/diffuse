import model_utils, Models
from mdtraj import io
from thor import scatter
import numpy as np
import itertools, sys, thor
import mdtraj as md
import cPickle as pickle

""" 
Simulate the diffuse scattering map for coordinates in input_pdb, assuming a model of 
Gaussian disorder defined by the input covariance matrix (or, alternatively, an input 
correlation matrix that is converted to a covariance matrix by atomic displacements).
The dimensions of the output map are determined by system.pickle.

Inputs: input_pdb, coordinates for which to calculate intensities
        system.pickle, dictionary containing map specifications that define q-grid
        savepath, path at which to save intensity array and covariance matrix
        n_asu, number of asus per frame of pdb*
        cov_mat, covariance matrix file of shape (n_atoms, n_atoms)
        OR
        corr_mat, correlation matrix text file of shape (n_atoms, n_atoms)**
        deltas, array of atomic displacements of shape n_atoms**,***

* Some software programs save asus as separate models, in which case n_asu should be set to 1.
** Assuming here that file formats are consistent with FP's NM output.
*** If deltas is 'from_pdb', then use refined B factors.

Usage: python diffuseV.py [input_pdb] [system.pickle] [n_asu] [savepath] [cov_mat]
    or python diffuseV.py [input_pdb] [system.pickle] [n_asu] [savepath] [corr_mat] [deltas]

"""

def split_pdb(uc_pdb, n_asus):
    """ 
    Split asymmetric units in each frame of input_pdb; store in dictionary.
    """
    n_atoms = uc_pdb[0].n_atoms/n_asus
    confs = dict()

    counter = 0
    for frame in range(uc_pdb.n_frames):
        for i in range(n_asus):
            confs[counter] = uc_pdb[frame].atom_slice(range(n_atoms*i, n_atoms*(i+1)))
            counter += 1
            
    return confs

def compute_qvecs(system):
    """ 
    Compute q vectors and their magnitudes for given hkl indices. Generate grid of
    q vectors, but mask those above a specified resolution.
    """

    # transform hkl to qvecs by inverse crystal setting matrix
    A_inv = model_utils.deorth_matrix(system)
    hkl_grid = np.array(list(itertools.product(system['bins']['h'], system['bins']['k'], system['bins']['l'])))
    q_vecs = 2*np.pi*np.inner(A_inv.T, hkl_grid).T 

    s_mags = np.linalg.norm(q_vecs/(2*np.pi), axis=1)
    s_mags[np.where(s_mags==0)] = 0.001 # ensure that origin isn't clipped
    res_valid = np.where(1.0/s_mags > system['d'])[0]

    mask = np.zeros(q_vecs.shape[0])
    mask[res_valid] = 1
    
    return q_vecs[res_valid], mask

def save_maps(rs_map, savepath):
    """
    Save maps as separate keys in .h5 file format so that downstream loading
    isn't problematic.
    """

    for i, item in enumerate(rs_map):
        name = 'n%s' % i
        data = {name : item}
        io.saveh(savepath + ".h5", **data)

    return

if __name__ == '__main__':

    # load input information
    pdb = md.load(sys.argv[1])
    system = pickle.load(open(sys.argv[2], "rb"))
    savepath = sys.argv[3]
    n_asu = int(sys.argv[4]) # reformat pdb if necessary
    confs = split_pdb(pdb, n_asu)

    if len(sys.argv) > 5:
        nm_obj = Models.NormalModes()
        if sys.argv[6] == 'from_pdb':
            deltas = model_utils.retrieve_bfactors(sys.argv[1], as_delta = True)
        else:
            deltas = np.loadtxt(sys.argv[6], usecols = [1])
        corr_mat =nm_obj.read_corrmat(sys.argv[5], pdb.n_atoms)
        V = nm_obj.corr_to_cov(corr_mat, deltas)
        np.save(savepath + "_V.npy", V)
    else:
        V = np.load(sys.argv[5])

    # generate q-grid and storage array
    detector, mask = compute_qvecs(system)
    rs_map = np.zeros((len(confs.keys()), mask.shape[0]))

    # process detector in segments of 1e6 q-vectors to avoid CUDA error
    n_segments = 1
    if detector.shape[0] > 1e6:
        n_segments = int(np.ceil(detector.shape[0] / 1e6))
    nq_per_seg = detector.shape[0]/n_segments
    
    # compute scattering for every frame in ensemble
    for frame in range(len(confs.keys())):
        
        print "processing frame %i" %frame
        Id = np.zeros(detector.shape[0])

        # loop over 'segments' of the detector 
        for seg in range(n_segments):
            start, end = seg*nq_per_seg, seg*nq_per_seg + nq_per_seg
            if seg == n_segments - 1:
                end = detector.shape[0]

            dtc = detector[start:end]
            Ib_partial, Id_partial = scatter.simulate_diffuse(confs[frame], dtc, V, ignore_hydrogens=False, device_id=0)
            Id[start:end] = Id_partial

        rs_map[frame][mask==1] = Id

    save_maps(rs_map, savepath)
        
