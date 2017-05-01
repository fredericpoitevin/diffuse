import model_utils
import itertools, time, sys
import cPickle as pickle
import numpy as np
from mdtraj import io
import scipy.stats

"""
Compute various map statistics across n_bins, including: 
1. CC_Bijovet: correlation coefficient between +(h,k,l) and -(h,k,l)
2. CC_Friedel: correlation coefficient between original and Friedel-symmetrized maps
3. CC_Laue: correlation coefficient between original and Laue-symmetrized maps
for both the full and radial average-subtracted (anisotropic) maps.

Save CC_overall and by shell in a pickle file, and the Laue- and Friedel-symmetrized map
as a numpy array.

Usage: python map_statistics.py [system.pickle] [point_group] [final_maps.h5] [n_bins] [output_path]

"""

def cc_bijovet(system, input_map, n_bins):
    """
    Compute CC between Friedel mates and return Friedel-symmetrized map.
    """

    # retrieve Friedel mates and set weights/multiplicities for CC calculation to 1
    Ipos, Ineg = input_map.copy().flatten(), input_map.copy().flatten()[::-1]
    mult_ones = np.ones(Ipos.shape)

    # compute CC overall and by resolution bin
    valid = np.where((Ipos > 0) & (Ineg > 0))[0]
    cc_overall = np.corrcoef(Ipos[valid], Ineg[valid])[0,1]
    cc_shells = model_utils.cc_by_shell(system, n_bins, Ipos, Ineg, mult_ones)

    # average Friedel pairs to generate Friedel-symmetrized map
    vals = np.vstack((Ipos, Ineg))
    vals[vals==0] = np.nan
    
    fsymm = np.nanmean(vals.T, axis=1)
    fsymm[np.isnan(fsymm)] = 0

    return cc_overall, cc_shells, fsymm


def compute_ccs(system, original_map, symm_map, n_bins):
    """
    Compute (unweighted) CC overall and by shell between original and symmetrized maps.
    """

    cc_overall = model_utils.mweighted_cc(original_map, symm_map)
    mult_ones = np.ones(original_map.flatten().shape)
    cc_shells = model_utils.cc_by_shell(system, n_bins, original_map, symm_map, mult_ones)

    return cc_overall, cc_shells


if __name__ == '__main__':

    start = time.time()
    
    # process command line arguments
    system = pickle.load(open(sys.argv[1], "rb"))
    symm_ops = pickle.load(open("reference/symm_ops.pickle", "rb"))[sys.argv[2]]
    n_bins = int(sys.argv[4])
    output_path = sys.argv[5]

    maps = dict()
    maps['full'] = io.loadh(sys.argv[3])["I"].flatten()
    maps['aniso'] = model_utils.subtract_radavg(system, maps['full'].copy())

    # set up dictionary for storing CC arrays
    cc_stats = dict()
    cc_stats['full'], cc_stats['aniso'] = dict(), dict()

    print "computing map statistics..."
    for key in ['full', 'aniso']:
        print "...for %s map" %key
        overall, by_shell, map_friedel = cc_bijovet(system, maps[key].copy(), n_bins)
        cc_stats[key]['bijovet'] = (overall, by_shell)
        cc_stats[key]['friedel'] = compute_ccs(system, maps[key].copy(), map_friedel, n_bins)
        map_laue, l_mult = model_utils.symmetrize(system, symm_ops, maps[key].copy(), laue = True, from_asu = False)
        cc_stats[key]['laue'] = compute_ccs(system, maps[key].copy(), map_laue, n_bins)
    
    print "saving map statistics"
    with open(output_path + "/symm_statistics.pickle", "wb") as handle:
        pickle.dump(cc_stats, handle)

    # generate Friedel and Laue symmetrized map and its anistropic counterpart
    print "generating Laue- and Friedel-symmetrized maps"
    fl_symm, fl_mult = model_utils.symmetrize(system, symm_ops, maps['full'].copy(), laue = False, from_asu = False)
    fl_symm_aniso = model_utils.subtract_radavg(system, fl_symm.copy())
    maps['symm'], maps['symm_aniso'], maps['mult'] = fl_symm, fl_symm_aniso, fl_mult

    print "saving symmetrized maps"
    for key in maps.keys():
        io.saveh(output_path + "/processed_maps.h5", **maps)

    print "elapsed time is %.3f" %((time.time() - start)/60.0)

