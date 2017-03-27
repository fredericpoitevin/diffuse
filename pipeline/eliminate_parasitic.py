import matplotlib, sys, os.path
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import EliminateParasitic as ep
from collections import OrderedDict
import glob, sys, time, map_utils
import cPickle as pickle
import numpy as np

"""
Wrapper for the EliminateParasitic class. Script has two modes: 1. 'profile', 
which generates radial intensity profiles for the complete dataset, and 2. 
'remove', which generates and applies specified method of background subtraction.

Usage: python eliminate_parasitic.py profile system.pickle [intensity directory] [n_bins]
       python eliminate_parasitic.py remove system.pickle [intensity directory] [rI_mtx]
"""

def plot_strategy(mtx1, mtx2, mtx3):

    meshX, meshY = np.meshgrid(mtx1[0], np.arange(1, mtx1.shape[0]))

    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12,5))

    ax1.pcolormesh(meshX, meshY, mtx1[1:], vmin=0, vmax=50)
    ax2.pcolormesh(meshX, meshY, mtx2[1:], vmin=0, vmax=50)
    ax3.pcolormesh(meshX, meshY, mtx3[1:], vmin=0, vmax=50)

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_xlim(0.01, 0.8)
    for ax in [ax1, ax2, ax3]:
        ax.set_ylim(1, mtx1.shape[0])
    for ax in [ax4, ax5, ax6]:
        ax.set_xlabel("|S| ($\AA$$^{-1}$)")

    for prf in range(1, mtx1.shape[0]):
        ax4.plot(mtx1[0], mtx1[prf])
        ax5.plot(mtx2[0], mtx2[prf])
        ax6.plot(mtx3[0], mtx3[prf])

    ax1.set_ylabel("Image Number")
    ax4.set_ylabel("Intensity")

    f.savefig(system['map_path'] + "checks/eliminate_parasitic.png", dpi=300, bbox_inches='tight')

def remove_background(dir_pI, output_dir, params):

    file_glob = glob.glob(system['map_path'] + dir_pI + "/*.npy")
    filelist = sorted(file_glob, key = lambda name: int(name.split('_')[-1].split('.')[0]))

    ab = ep.ApplyBackground(system, params)
    for i in range(len(filelist)):
        
        print "processing image %i" %i
        num = filelist[i].split('_')[-1].split('.')[0]
        indexed = np.load(system["map_path"] + "indexed/hklI_%s.npy" %num)

        if dir_pI == 'indexed':
            imgI = indexed[:,-1]
        else:
            print "loading from %s" % dir_pI ## for testing purposes
            imgI = np.load(filelist[i])

        s_mags = ab.compute_smags(indexed, int(num))
        for key in params.keys():
            if key == 'water':
                print "removing water profile" ## for testing purposes  
                imgI -= ab.scale_water(s_mags, int(num))
            if key == 'paratone':
                print "removing paratone profile" ## for testing purposes 
                imgI -= ab.scale_paratone(s_mags, int(num))
            if key == 'pca_bgd':
                print "removing pca bgd" ## for testing purposes 
                imgI -= ab.interp_pca(s_mags, int(num))

        np.save(output_dir + "pI_%s.npy" %num, imgI)
        
    return

if __name__ == '__main__':

    start = time.time()
    system = pickle.load(open(sys.argv[2], "rb"))

    if sys.argv[1] == 'profile':

        dir_pI, n_bins = sys.argv[3], int(sys.argv[4])
        rI_mtx = map_utils.mtx_rprofile(system, dir_pI, n_bins, median=True)

        print "Saving intensity profiles to temp/%s_rI.npy" % dir_pI
        np.save(system['map_path'] + 'temp/%s_rI.npy' % dir_pI, rI_mtx)
        
    if sys.argv[1] == 'remove':
        
        # generate background dictionary
        dir_pI, rI_input = sys.argv[3], np.load(sys.argv[4])
        gb = ep.GenerateBackground(system)
        
        opt_par = gb.opt_paratone(rI_input.copy())
        sort_evals, sort_evecs, proj_data = gb.run_pca(opt_par.copy())
        n_eig = 2
        opt_pca = gb.pca_params(opt_par.copy(), n_eig, sort_evals, sort_evecs, proj_data)

        # checks of background strategy
        print "Num. PCs chosen explain %.2f percent of the variance" %(np.sum(sort_evals[:n_eig])/np.sum(sort_evals))
        plot_strategy(rI_input, opt_par, opt_pca)

        # apply background dictionary
        output_dir = system['map_path'] + "processedI/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        remove_background(dir_pI, output_dir, gb.params)
            
    print "elapsed time is %f" %((time.time() - start)/60.0)
