import matplotlib, sys, os.path
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np
import glob, time

"""
As a check of Bragg masking procedure, plot intensity distributions from post-
indexing, -Bragg masking, and -general outlier removal stages. Save as .png.

Usage: python check_masks.py [system.pickle]

"""

def plotI(num, system, smags, I0, I1, I2):
    
    savename = system["map_path"] + "checks/maskedI_%s.png" %num
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))

    ax1.scatter(smags[((1.0/smags > system['d'])*(I0>0))], I0[((1.0/smags > system['d'])*(I0>0))])
    ax2.scatter(smags[((1.0/smags > system['d'])*(I1>0))], I1[((1.0/smags > system['d'])*(I1>0))])
    ax3.scatter(smags[((1.0/smags > system['d'])*(I2>0))], I2[((1.0/smags > system['d'])*(I2>0))])

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0, 0.67)
        ax.set_xlabel(r'|S|($\AA$$^{-1}$)')
    ax1.set_ylabel("Intensity")

    ax1.set_title("Post indexing")
    ax2.set_title("Post Bragg masking")
    ax3.set_title("Post outlier removal")

    f.savefig(savename, bbox_inches='tight', dpi=100)
    plt.clf()

    return

def process_files(system, filelist):

    A_inv = np.linalg.inv(np.diag(system['cell'][:3]))

    for i in range(len(filelist)):
        print "on image %i" %i
        num = filelist[i].split('_')[-1].split('.')[0]

        indexed = np.load(system["map_path"] + "indexed/hklI_%s.npy" %num)
        bmask = np.load(filelist[i])
        gmask = np.load(system["map_path"] + "maskedI/pI_%s.npy" %num)
        smags = np.linalg.norm(np.inner(A_inv, indexed[:,:3]).T, axis=1)

        plotI(num, system, smags, indexed[:,-1], bmask, gmask)

    return

if __name__ == '__main__':

    start = time.time()
    system = pickle.load(open(sys.argv[1], "rb"))

    file_glob = glob.glob(system["map_path"] + "checks/maskedI_*.npy")
    filelist = sorted(file_glob, key = lambda name: int(name.split('_')[-1].split('.')[0]))

    process_files(system, filelist)
    print "elapsed time is %f" %((time.time() - start)/60.0)
