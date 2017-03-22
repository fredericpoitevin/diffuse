import cPickle as pickle
import PredictBragg as bm
import numpy as np
import time, sys
from mdtraj import io

"""
Wrapper for PredictBragg class. 

Inputs: system.pickle, file containing system/detector/map information
        sigma_n, parameter that tunes integration region; recommended values are 3-5
        nbatch, batch number to process, must fall within batch range
           or
        'compile', which causes script to compile pre-existing per-batch files

Usage: python generate_braggmasks.py [system.pickle] [sigma_n] [nbatch], to process a single batch
       python generate_braggmasks.py [system.pickle] ['compile'], to compile all batches

"""

def compile(system):
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

if __name__ == '__main__':
    
    start = time.time()
    system = pickle.load(open(sys.argv[1]))

    if sys.argv[2] == 'compile':
        "print in compile mode"
        compile(system)
        
    else:
        "print in batch mode, processing batch %s" %sys.argv[3]
        sigma_n, nbatch = float(sys.argv[2]), int(sys.argv[3])
        assert (nbatch >= 0) and (nbatch < system['n_batch'])
        bm_obj = bm.PredictBragg(system, nbatch, sys_absences = True)
        bm_obj.pred_pixels(sigma_n)

    print "elapsed time is %f" %((time.time() - start)/60.0)
