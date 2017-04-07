import Models, model_utils
import cPickle as pickle
from scipy import signal
import numpy as np
import sys, itertools, time
from mdtraj import io

"""
Scan over values of sigma and gamma to optimize parameters for LLM model.
Usage: python wall_scan_exp.py system.pickle symm_ops transform_prefix experimental_path model save_prefix
"""

def prompt_for_params():
    """
    Prompt for range across which to scan sigma and gamma values.
    """

    sigma_range = raw_input("sigma low, sigma high, sigma interval: ")
    sigma_range = [float(i) for i in sigma_range.split()]
    sigma_range = np.arange(sigma_range[0], sigma_range[1] + sigma_range[2], sigma_range[2])

    gamma_range = raw_input("gamma low, gamma high, gamma interval: ")
    gamma_range = [float(i) for i in gamma_range.split()]
    gamma_range = np.arange(gamma_range[0], gamma_range[1] + gamma_range[2], gamma_range[2])

    return sigma_range, gamma_range

def process_transform(dir_prefix):
    """
    Load molecular transform, assuming that this was saved in two parts:
    intensity for valid q and mask.
    """

    transform = np.load(dir_prefix + "_mask.npy")
    transform[transform > 0] = np.load(dir_prefix + "_I.npy")

    return transform

def process_experimental(exp_path, system, symm_idx):
    """
    Symmetrize experimental map and subtract radial average.
    """

    exp_unsym = io.loadh(exp_path + "/final_maps.h5", "I")
    exp = model_utils.symmetrize(exp_unsym, symm_idx, from_asu = False)
    exp_aniso = model_utils.subtract_radavg(system, exp.copy(), 100)
    
    return exp_aniso


def scan(system, transform, exp_aniso, sigma_range, gamma_range, model):
    """
    Compute CC_aniso between experimental and transforms, across sigma and gamma ranges.
    """

    map_shape = (len(system['bins']['h']), len(system['bins']['k']), len(system['bins']['l']))
    qmags = model_utils.compute_qmags(system)

    cc_aniso = np.zeros(len(sigma_range) * len(gamma_range))
    counter = 0

    wall_obj = Models.Wall()
    for gamma in gamma_range:
        for sigma in sigma_range:

            pred = wall_obj.scale(transform.copy(), qmags, map_shape, sigma, gamma, model)
            pred_aniso = model_utils.subtract_radavg(system, pred, 100)
            cc_aniso[counter] = model_utils.mweighted_cc(pred_aniso.copy(), exp_aniso.copy(), mult = mult)

            print "gamma: %.2f, sigma: %.2f, CC: %.4f" %(gamma, sigma, cc_aniso[counter])
            counter += 1

    cc_aniso = cc_aniso.reshape(len(sigma_range), len(gamma_range))
    return cc_aniso

if __name__ == '__main__':

    start = time.time()
    sigma_range, gamma_range = prompt_for_params()

    # load system and generate symmetry information
    system = pickle.load(open(sys.argv[1] + "/system.pickle", "rb"))
    symm_ops = pickle.load(open(sys.argv[2], "rb"))['p222']
    symm_idx, grid, mult = model_utils.generate_symmates(symm_ops, system['bins'], system['subsampling'])

    # load molecular transform and experimental maps
    transform = process_transform(sys.argv[3])
    exp_aniso = process_experimental(sys.argv[1], system, symm_idx)

    # scan across sigma and gamma ranges; save mesh and cc_aniso to same .h5 file
    cc_aniso = scan(system, transform, exp_aniso, sigma_range, gamma_range, sys.argv[4])
    io.saveh(sys.argv[5] + "/%s.h5" %(sys.argv[4]), sigmas = sigma_range)
    io.saveh(sys.argv[5] + "/%s.h5" %(sys.argv[4]), gammas = gamma_range)
    io.saveh(sys.argv[5] + "/%s.h5" %(sys.argv[4]), cc = cc_aniso)

    print "elapsed time is %.3f" %((time.time() - start)/60.0)
