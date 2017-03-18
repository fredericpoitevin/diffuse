import thor, sys, time, os.path
from thor import scatter, parse
import cPickle as pickle
import numpy as np
import mdtraj as md
import map_utils

"""

Script for generating a synthetic dataset by simulating a series of diffraction patterns
on a Pilatus detector across a rotation range specified by the input system file. If the
key 'num_asus' in the system file is greater than 1, then it's assumed we're working with 
a unit cell, and the incoherent sum of the asymmetric units will be simulated.

Usage: python simulate.py [input_pdb] [input_dtc] [system.pickle]

"""

def split_unitcell(uc_pdb, n_asus):
    """ 
    Split asymmetric units of input unit cell and store as separate entries
    in returned dictionary.
    """

    # stack asus if written out as separate models in input pdb
    if uc_pdb.n_frames > 1:
        processed = uc_pdb[0]
        for i in range(1, uc_pdb.n_frames):
            processed = processed.stack(uc_pdb[i])
    else:
        processed = uc_pdb

    asus = dict()
    n_atoms = processed.n_atoms/n_asus

    for i in range(n_asus):
        asus[i] = processed.atom_slice(range(n_atoms*i, n_atoms*(i+1)))

    return asus

def simulate(pdb_file, dtc, system, img_num, traj, rot_mats, sim_dir):
    """
    Simulate the incoherent sum of the asus in input pdb file after rotating xyz coordinates. 
    Simulated intensities are computed for all pixels on a mock Pilatus detecotr (dtc). Keep
    track of rotated pdb coordinates and rotation matrix for debugging purposes.
    """
    
    # check that output file doesn't already exist
    savename = sim_dir + "simulated_" + format(img_num, '05') + ".npy"
    if os.path.isfile(savename) == False:

        # re-load pdb file each time for original xyz
        pdb = md.load(pdb_file)
        
        # rotate oriented pdb file 
        phi = -1.0*np.deg2rad(system['rot_phi'] * (img_num - 1) + system['rot_phi'])
        norm = np.linalg.norm(system['A'].copy(), axis=1)
        norm_A = np.asarray([system['A'].copy()[i]/norm[i] for i in range(3)])

        rot_gonio = map_utils.rotation_matrix(system['rot_axis'].copy(), phi)
        rot_cryst = np.dot(norm_A, rot_gonio)
        pdb.xyz[0,:,:] = np.dot(np.squeeze(pdb.xyz), rot_cryst)

        # save rotation matrix and pdb for downstream verification
        if img_num != 1:
            traj = traj.join(pdb)
            rot_mats = np.concatenate((rot_mats, rot_cryst))
        if img_num == 1:
            traj = pdb
            rot_mats = rot_cryst

        # split unit cell into separate ASUs and sum simulated shot for each.
        asus = split_unitcell(pdb, system['n_asus'])
        inc_sum = np.zeros(system['shape'][0] * system['shape'][1])
        for key in asus.keys():
            inc_sum += np.square(np.abs(scatter.simulate_atomic(asus[key], 1, dtc, finite_photon=False, ignore_hydrogens=True, dont_rotate=True, devices=[0])))
        np.save(savename, inc_sum)

        return traj, rot_mats
    
if __name__ == '__main__':

    # load detector and start trajectory
    dtc = thor.load(sys.argv[2])
    system = pickle.load(open(sys.argv[3], "rb"))
    rot_mats, traj = None, None

    # set up directories for simualted images and checks if don't already exist
    sim_dir, checks_dir = system['dir'] + "simulated/", system['dir'] + "checks/"
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)
    if not os.path.exists(checks_dir):
        os.makedirs(checks_dir)

    # simulate detector images across a broad oscillation range
    for img in range(1, system['n_images'] + 1):
        print "on image %i" %img
        traj, rot_mats = simulate(sys.argv[1], dtc, system, img, traj, rot_mats, sim_dir)

    # save useful debugging information
    traj.save(checks_dir + "sim_traj.pdb")
    rot_mats = rot_mats.reshape(system['n_images'], 3, 3)
    np.save(checks_dir + "rot_mats.npy", rot_mats)
