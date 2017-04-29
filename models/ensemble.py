import model_utils
from mdtraj import io
from thor import scatter
import numpy as np
import itertools, sys, thor
import mdtraj as md
import cPickle as pickle

""" 
Directly simulate intensities or structure factors for coordinates in input_pdb on
a grid of q-vectors defined by map specifications in system.pickle. Output an array
of shape (n_conformations, n_qvectors). Optional input is n_asu, which corresponds
to the number of asymmetric units per frame in input_pdb. NB: some software programs
automatically write distinct asus to separate models when reconstructing the unit 
cell; in this case MDTraj will interpret each asu as a separate frame, so the n_asu
flag needn't be reset.

Inputs: input_pdb, coordinates for which to calculate intensities
        system.pickle, dictionary containing map specifications that define q-grid
        map_type, 'sf' or 'I' for structure factors and intensities, respectively
        savepath, path at which to save intensity array
        n_asu, number of distinct asus per frame of pdb (optional)

Usage: python ensemble.py [input_pdb] [system.pickle] [map_type] [savepath] [n_asu]

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
        io.saveh(savepath, **data)

    return

if __name__ == '__main__':

    # load input information
    pdb = md.load(sys.argv[1])
    system = pickle.load(open(sys.argv[2], "rb"))
    map_type = sys.argv[3]
    if not ((map_type == 'I') or (map_type == 'sf')):
        raise Exception('Please choose I (intensities) or sf (structure factors for map type.')
    savepath = sys.argv[4]

    # reformat pdb
    n_asu = 1
    if len(sys.argv) > 5:
        n_asu = int(sys.argv[5])
    confs = split_pdb(pdb, n_asu)
    
    # generate q-grid and storage array
    detector, mask = compute_qvecs(system)
    if map_type == 'I':
        rs_map = np.zeros((len(confs.keys()), mask.shape[0]))
    else:
        rs_map = np.zeros((len(confs.keys()), mask.shape[0]), dtype = complex)

    # process detector in segments of 1e6 q-vectors to avoid CUDA error
    n_segments = 1
    if detector.shape[0] > 1e6:
        n_segments = int(np.ceil(detector.shape[0] / 1e6))
    nq_per_seg = detector.shape[0]/n_segments
    
    # compute scattering for every frame in ensemble
    for frame in range(len(confs.keys())):
        print "processing frame %i" %frame

        if map_type == 'I':
            amps = np.zeros(detector.shape[0])
        else:
            amps = np.zeros(detector.shape[0], dtype=complex)

        # loop over 'segments' of the detector 
        for seg in range(n_segments):
            start, end = seg*nq_per_seg, seg*nq_per_seg + nq_per_seg
            if seg == n_segments - 1:
                end = detector.shape[0]

            dtc = detector[start:end]
            amps[start:end] = scatter.simulate_atomic(confs[frame], 1, dtc, finite_photon=False, ignore_hydrogens=True, dont_rotate=True, devices=[0])

        # convert to intensities depending on specified map type
        if map_type == 'I':
            rs_map[frame][mask==1] = np.square(np.abs(amps))
        else:
            rs_map[frame][mask==1] = amps

    save_maps(rs_map, savepath)
        
