import model_utils, map_utils
from mdtraj import io
from thor import scatter
import numpy as np
import itertools, sys, thor, time
import mdtraj as md
import cPickle as pickle

""" 
Simulate diffuse scattering produced by rotational disorder. An ensemble of rotated
coordinates is generated, with angles along each real space direction drawn sampled
from independent normal distributions, each with variance of 1 degree * input scale. 
A diffuse scattering for map with dimensions specified by system.pickle is computed
using Guinier's equation.

Inputs: input_pdb, coordinates for which to calculate intensities
        system.pickle, dictionary containing map specifications that define q-grid
        n_samples, number of rotations to perform
        scale, scale factor that weights amplitudes of rotations
        savepath, path at which to save intensity array

Usage: python guinier.py input_pdb system.pickle n_samples scale savepath

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

    start_time = time.time()

    # load input information
    pdb = md.load(sys.argv[1])
    system = pickle.load(open(sys.argv[2], "rb"))
    n_samples, scale = int(sys.argv[3]), float(sys.argv[4])
    savepath = sys.argv[5]
    
    # generate q-grid and storage array
    detector, mask = compute_qvecs(system)
    fc, fc_square = np.zeros(mask.shape[0], dtype=complex), np.zeros(mask.shape[0])

    # process detector in segments of 1e6 q-vectors to avoid CUDA error
    n_segments = 1
    if detector.shape[0] > 1e6:
        n_segments = int(np.ceil(detector.shape[0] / 1e6))
    nq_per_seg = detector.shape[0]/n_segments

    # generate n_samples rotated instances of the pdb and compute scattering
    rand_angles = np.random.randn(n_samples, 3)
    np.save(savepath + "_angles.npy", rand_angles)
    
    for i in range(n_samples):

        # generate rotated pdb
        pdb_rot = md.load(sys.argv[1])
        Rx = map_utils.rotation_matrix(np.array([1.0, 0.0, 0.0]), scale*np.deg2rad(rand_angles[i,0]))
        Ry = map_utils.rotation_matrix(np.array([0.0, 1.0, 0.0]), scale*np.deg2rad(rand_angles[i,1]))
        Rz = map_utils.rotation_matrix(np.array([0.0, 0.0, 1.0]), scale*np.deg2rad(rand_angles[i,2]))

        Rall = np.dot(Rx, np.dot(Ry, Rz))
        pdb_rot.xyz[0,:,:] = np.dot(np.squeeze(pdb_rot.xyz), Rall)
        pdb = pdb.join(pdb_rot)

        # loop over 'segments' of detector
        amps = np.zeros(detector.shape[0], dtype=complex)
        for seg in range(n_segments):

            start, end = seg*nq_per_seg, seg*nq_per_seg + nq_per_seg
            if seg == n_segments - 1:
                end = detector.shape[0]

            dtc = detector[start:end]
            amps[start:end] = scatter.simulate_atomic(pdb_rot, 1, dtc, finite_photon=False, ignore_hydrogens=True, dont_rotate=True, devices=[0])

        fc[mask==1] += amps
        fc_square[mask==1] += np.square(np.abs(amps))

    rs_map = fc_square/float(n_samples) - np.square(np.abs(fc/float(n_samples)))
    #save_maps(rs_map, savepath)
    np.save(savepath + "_map.npy", rs_map)
    pdb.save(savepath + ".pdb")
    
    print "elapsed time is %.2f" %((time.time() - start_time)/60.0)
