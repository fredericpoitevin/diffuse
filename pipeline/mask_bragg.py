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
        sys_absences, string corresponding to reflection conditions along H,K,L axes. '000' if no screw axes.
        nbatch, batch number to process, must fall within batch range (specific to 'generate' mode)
        bragg_sigma, determines threshold for excluding pixels spanned by Bragg peaks
        outlier_sigma, determines threshold above radial average for excluding high-intensity pixels
        s_mask, optional numpy array for untrusted regions (e.g. cryo-loop)
        img_num_low, img_num_high, image range across which to apply the special mask s_mask
Hardcoded but tunable parameters: length, determines size of bounding box for Bragg Masking. Default is 30 pixels.
                                  n_bins, number of bins for calculating radial intensity profile. Default is 100.

Usage: python mask_bragg.py generate [system.pickle] [sigma_n] [sys_absences] [nbatch], to process a single batch
       python mask_bragg.py compile [system.pickle], to compile all batches
       python mask_bragg.py apply [system.pickle] [bragg_sigma] [outlier_sigma] [s_mask] [img_num_low] [img_num_high], to apply masks and remove outliers

"""

def compile_masks(system):
    """
    Compile per-batch Bragg masks into a composite file.
    """

    #n_images = system['batch_size'] * system['n_batch']
    n_images = len(system['img2batch'])
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

    # select a random file per batch to additionally save mask file for debugging purposes
    keys, vals = system['img2batch'].keys(), system['img2batch'].values()
    batch_ends = np.where(np.array([vals[idx+1] - vals[idx] for idx in range(len(vals) - 1)]) != 0)[0]
    batch_ends = np.array([keys[idx] for idx in batch_ends])
    batch_ends = np.concatenate((np.array([keys[0]]), batch_ends, np.array([keys[-1]])))
    #check_files = [np.random.randint(low=batch_ends[idx], high=batch_ends[idx + 1]) for idx in range(len(batch_ends) - 1)]
    check_files = np.dstack([np.random.randint(low=batch_ends[idx], high=batch_ends[idx + 1], size=3) for idx in range(len(batch_ends) - 1)]).flatten()
    
    for i in range(len(filelist)):
        print "processing image %i" %i

        # set up ApplyMask object
        indexed = np.load(filelist[i])
        img_num = filelist[i].split('_')[-1].split('.')[0]
        bm_obj = mb.ApplyMask(system, indexed, mask_path, int(img_num), length)

        # mask Bragg peaks and remove outliers
        maskedI = bm_obj.ms_threshold(bragg_sigma)
        pI = bm_obj.remove_outliers(outlier_sigma, n_bins, maskedI.copy())
        
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
        print "generating bragg masks, processing batch %s" %sys.argv[5]
        sigma_n, nbatch = float(sys.argv[3]), int(sys.argv[5])
        assert (nbatch >= 0) and (nbatch < system['n_batch'])
        bm_obj = mb.PredictBragg(system, nbatch, sys_absences = sys.argv[4])
        bm_obj.pred_pixels(sigma_n)

    if sys.argv[1] == 'compile':
        print "compiling bragg masks"
        compile_masks(system)
        
    if sys.argv[1] == 'apply':
        print "applying bragg masks"

        bragg_sigma, outlier_sigma = float(sys.argv[3]), float(sys.argv[4])
        if len(sys.argv) > 5:
            system['s_mask'] = np.load(sys.argv[5])
            system['s_mask_img'] = range(int(sys.argv[6]), int(sys.argv[7])+1)
        
        length, n_bins = 30, 100
        
        output_dir = system['map_path'] + "maskedI/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        apply_masks(system, output_dir, bragg_sigma, outlier_sigma, length, n_bins)
        
    print "elapsed time is %f" %((time.time() - start)/60.0)
