import Models, model_utils
import cPickle as pickle
from scipy import signal
import numpy as np
import sys, itertools, time
from mdtraj import io

"""
Scan over values of sigma and gamma to optimize parameters for LLM model.
Usage: python wall_scan.py system.pickle point_group transform_map exp_map model save_path
"""

def scan(system, pred_map, exp_map, sigma_range, gamma_range, model, mult):
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

            pred = wall_obj.scale(pred_map.copy(), qmags, map_shape, sigma, gamma, model)
            pred_aniso = model_utils.subtract_radavg(system, pred)
            cc_aniso[counter] = model_utils.mweighted_cc(pred_aniso.copy(), exp_map.copy(), mult = mult)

            print "gamma: %.2f, sigma: %.2f, CC: %.4f" %(gamma, sigma, cc_aniso[counter])
            counter += 1

    cc_aniso = cc_aniso.reshape(len(gamma_range), len(sigma_range))
    return cc_aniso

if __name__ == '__main__':

    start = time.time()
    #sigma_range, gamma_range = np.arange(0.05, 1.55, 0.05), np.arange(3.0, 93.0, 3.0)
    sigma_range, gamma_range = np.arange(0.5, 0.61, 0.01), np.arange(12.0, 21.0)
    
    # load system and generate symmetry information
    system = pickle.load(open(sys.argv[1], "rb"))
    symm_ops = pickle.load(open("reference/symm_ops.pickle", "rb"))[sys.argv[2]]
    symm_idx, grid, mult = model_utils.generate_symmates(symm_ops, system, laue=False)

    # load molecular transform and experimental maps
    transform = np.load(sys.argv[3])
    experimental = np.load(sys.argv[4])

    # scan across sigma and gamma ranges; save mesh and cc_aniso to same .h5 file
    cc_aniso = scan(system, transform, experimental, sigma_range, gamma_range, sys.argv[5], mult)
    io.saveh(sys.argv[6] + "/%s.h5" %(sys.argv[5]), sigmas = sigma_range)
    io.saveh(sys.argv[6] + "/%s.h5" %(sys.argv[5]), gammas = gamma_range)
    io.saveh(sys.argv[6] + "/%s.h5" %(sys.argv[5]), cc = cc_aniso)

    print "elapsed time is %.3f" %((time.time() - start)/60.0)
