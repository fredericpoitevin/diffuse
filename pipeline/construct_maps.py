from collections import defaultdict
import MapConstructor, map_utils
import cPickle as pickle
import os.path, time, glob, sys
import numpy as np
from mdtraj import io
import itertools

"""
Wrapper for MapConstructor.py, with two modes: 1. the 'compile' mode, which bins 
pre-indexed pixels into voxels by resolution shell, and 2. the 'reduce' mode,
which compiles the output for different resolution shells into the final map and
computes statistics by resolution shell.

Usage: python construct_maps.py "compile" [map_dir] [num_shells] [n_shell], or
python construct_maps.py "reduce" [map_dir] [num_shells]

Notes about inputs: map_dir is assumed to contain system.pickle and indexed/; 
num_shells refers to number of shells over which to parallelize calculation, and
n_shell refers to the resolution shell to be compiled.

"""

def compile_shell(map_dir, system, num_shells, n_shell):
    """
    Process all indexed images for resolution shell given by n_shell; return result
    as dict[voxel]= [list of intensities].
    """
    
    # get indexed file list
    file_glob = glob.glob(map_dir + "indexed/*.npy")
    filelist = sorted(file_glob, key = lambda name: int(name.split('_')[-1].split('.')[0]))
    assert len(filelist) == int(system['n_batch']*system['batch_size'])

    # initialize map constructor object
    mc_obj = MapConstructor.GenerateMap(system, num_shells)
    combined = defaultdict(list)

    for img in range(len(filelist)):
        print "processing indexed image %i" %img
        indexed = np.load(filelist[img])
        single = mc_obj.process_image(indexed, n_shell)

        if img == 0:
            combined = single
        else:
            combined = mc_obj.merge_dicts(single, combined)

    # reduce voxels from list of intensities to mean intensity and get statistics
    maps, grids = mc_obj.reduce_shell(combined)
    return maps, grids

def reduce_shells(map_dir, system, num_shells):
    """
    Combine shells into final maps for average intensity, <I>/sigma(I), and number of pixels.
    Also compute mean of these parameters per resolution shell; return as shell_stats.
    """
    
    map_shape = (len(system['bins']['h']), len(system['bins']['k']), len(system['bins']['l']))
    map_keys = ["I", "I_sigI", "n_pixels"]

    file_glob = glob.glob(map_dir + "temp/grid_rshell*.h5")
    filelist = sorted(file_glob, key = lambda name: int(name.split('rshell')[-1].split('_')[0]))
    assert len(filelist) == num_shells

    hkl = np.array(list(itertools.product(system['bins']['h'], system['bins']['k'], system['bins']['l'])))
    hkl_res = map_utils.compute_resolution(system['space_group'], system['cell'], hkl)
    hkl_res = hkl_res.reshape(map_shape)
    
    combined_maps, shell_stats = dict(), dict()
    shell_stats['resolution'] = np.zeros(num_shells)
    
    for key in map_keys:
        print "on key %s" %key
        combined_maps[key] = np.zeros(map_shape)
        shell_stats[key] = np.zeros(num_shells)

        for shell in range(len(filelist)):
            data = io.loadh(filelist[shell], key)
            combined_maps[key] += data
            shell_stats[key][shell] = np.mean(data[data>0])

            if key == "I":
                shell_stats['resolution'][shell] = np.median(hkl_res[data>0])
            
    return combined_maps, shell_stats

if __name__ == '__main__':

    start = time.time()
    system = pickle.load(open(sys.argv[2]+"system.pickle"))

    if sys.argv[1] == 'compile':

        # generating temp dir for resolution shell data
        output_dir = sys.argv[2] + "temp/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # processing resolution shell
        num_shells, n_shell = int(sys.argv[3]), int(sys.argv[4])
        shell_maps, shell_grids = compile_shell(sys.argv[2], system, num_shells, n_shell)

        # save resolution shell in grid and dictionary formats
        with open(sys.argv[2] + "temp/dict_rshell%i_t%i.pickle" %(n_shell, num_shells), "wb") as handle:
            pickle.dump(shell_maps, handle)

        io.saveh(sys.argv[2] + "temp/grid_rshell%i_t%i.h5" %(n_shell, num_shells), **shell_grids)
            
    if sys.argv[1] == "reduce":

        # combine resolution shells
        combined_maps, shell_stats = reduce_shells(sys.argv[2], system, int(sys.argv[3]))

        io.saveh(sys.argv[2] + "final_maps.h5", **combined_maps)
        io.saveh(sys.argv[2] + "shell_statistics.h5", **shell_stats)
                
    print "elapsed time is %f" %((time.time() - start)/60.0)
