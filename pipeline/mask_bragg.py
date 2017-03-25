import cPickle as pickle
import MaskBragg as mb
import numpy as np
import time, sys, os.path, glob
from mdtraj import io

"""
Wrapper for MaskBragg class. 

Inputs: mode: 'generate', 'compile', or 'apply' which determines what script performs
        system.pickle, file containing system/detector/map information
        sigma_n, parameter that tunes integration region; recommended values are 3-5 (specific to 'generate' mode)
        nbatch, batch number to process, must fall within batch range (specific to 'generate' mode)
        bragg_sigma, determines threshold for excluding pixels spanned by Bragg peaks
        outlier_sigma, determines threshold above radial average for excluding high-intensity pixels
Hardcoded but tunable parameters: length, determines size of bounding box for Bragg Masking. Default is 30 pixels.
                                  n_bins, number of bins for calculating radial intensity profile. Default is 100.

Usage: python mask_bragg.py generate [system.pickle] [sigma_n] [nbatch], to process a single batch
       python mask_bragg.py compile [system.pickle], to compile all batches
       python mask_bragg.py apply [system.pickle] [bragg_sigma] [outlier_sigma], to apply masks and remove outliers

"""

def compile_masks(system):
    """
    Compile per-batch Bragg masks into a composite file.
    """

    n_images = system['batch_size'] * system['n_batch']
    dtc_size = system['shape'][0] * system['shape'][1]
    comb_mask = np.zeros((n_images, dtc_size), dtype=np.uint8)

    # combine all temp files 
    for batch in range(int(system['n_batch'])):
        print "on batch %i" %batch
        for img in range(int(n_images)):
            comb_mask[img] += io.loadh(system['map_path'] + "temp/masks_b%s.h5" %batch, "arr_%i" %img)

    print "saving combined mask"
    for i, item in enumerate(comb_mask):
        name = 'arr_%s' % i
        data = {name : item}
        io.saveh(system['map_path'] + "combined_braggmasks.h5", **data)

    return 

def apply_masks(system, output_dir, bragg_sigma, outlier_sigma, length, n_bins):
    """
    Apply Bragg masks and save the processed intensity arrays.
    """

    # retrieve paths to indexed files
    mask_path = system['map_path'] + "combined_braggmasks.h5"
    file_glob = glob.glob(system['map_path'] + "indexed/*.npy")
    filelist = sorted(file_glob, key = lambda name: int(name.split('_')[-1].split('.')[0]))
    assert len(filelist) == int(system['n_batch']*system['batch_size'])

    # select a random file per batch to additionally save mask file for debugging purposes
    base_batch = np.arange(system['n_batch'])*system['batch_size'] + 1
    rand_batch = np.random.randint(low=0, high=system['batch_size'], size=system['n_batch'])
    check_files = base_batch + rand_batch
    check_files = check_files.astype(int)
    
    for i in range(len(filelist)):
        print "processing image %i" %i

        # set up ApplyMask object
        indexed = np.load(filelist[i])
        img_num = filelist[i].split('_')[-1].split('.')[0]
        bm_obj = mb.ApplyMask(system, indexed, mask_path, int(img_num), length)

        # mask Bragg peaks and remove outliers
        maskedI = bm_obj.ms_threshold(bragg_sigma)
        pI, s_med, n_outliers = bm_obj.remove_outliers(outlier_sigma, n_bins, maskedI.copy())

        npix_removed = np.sum(n_outliers[np.where(1.0/s_med > system['d'])[0]])
        fpix_removed = 100.0*npix_removed/float(system['shape'][0]*system['shape'][1])
        print "%i (%.3f percent) pixels in map resolution range removed as outliers" % (int(npix_removed), fpix_removed)
        
        # save processed image and a few masked images if a 'checks' directory exists in map_path
        np.save(output_dir + "pI_%s.npy" %img_num, pI)

        if int(img_num) in check_files:
            if os.path.exists(system['map_path'] + 'checks/'):
                np.save(system['map_path'] + 'checks/maskedI_%s.npy' %img_num, maskedI)
            
    return

if __name__ == '__main__':
    
    start = time.time()
    system = pickle.load(open(sys.argv[2]))

    if sys.argv[1] == 'generate':
        print "generating bragg masks, processing batch %s" %sys.argv[4]
        sigma_n, nbatch = float(sys.argv[3]), int(sys.argv[4])
        assert (nbatch >= 0) and (nbatch < system['n_batch'])
        bm_obj = mb.PredictBragg(system, nbatch, sys_absences = True)
        bm_obj.pred_pixels(sigma_n)

    if sys.argv[1] == 'compile':
        print "compiling bragg masks"
        compile_masks(system)
        
    if sys.argv[1] == 'apply':
        print "applying bragg masks"

        bragg_sigma, outlier_sigma = float(sys.argv[3]), float(sys.argv[4])
        length, n_bins = 30, 100
        
        output_dir = system['map_path'] + "maskedI/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        apply_masks(system, output_dir, bragg_sigma, outlier_sigma, length, n_bins)
        
    print "elapsed time is %f" %((time.time() - start)/60.0)
