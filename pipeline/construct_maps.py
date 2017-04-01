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

Usage: python construct_maps.py "compile" [system.pickle] [I_dir] [num_shells] [n_shell], or
python construct_maps.py "reduce" [system.pickle] [num_shells]

Notes about inputs:
I_dir refers to directory name containing intensity profiles, may be different from indexed 
num_shells refers to number of shells over which to parallelize calculation, and
n_shell refers to the resolution shell to be compiled.

"""

def compile_shell(system, I_dir, num_shells, n_shell):
    """
    Process all indexed images for resolution shell given by n_shell; return result
    as dict[voxel]= [list of intensities].
    """
    
    # get intensity file list
    file_glob = glob.glob(system['map_path'] + I_dir + "/*.npy")
    filelist = sorted(file_glob, key = lambda name: int(name.split('_')[-1].split('.')[0]))
    assert len(filelist) == int(system['n_batch']*system['batch_size'])

    # initialize map constructor object
    mc_obj = MapConstructor.GenerateMap(system, num_shells)
    combined = defaultdict(list)

    for img in range(len(filelist)):
        
        print "processing indexed image %i" %img
        num = filelist[img].split('_')[-1].split('.')[0]
        indexed = np.load(system['map_path'] + "indexed/hklI_%s.npy" %num)

        if I_dir != 'indexed':
            assert filelist[img].split('_')[-1].split('.')[0] == num
            indexed[:,-1] = np.load(filelist[img])

        single = mc_obj.process_image(indexed, n_shell)

        if img == 0:
            combined = single
        else:
            combined = mc_obj.merge_dicts(single, combined)

    # reduce voxels from list of intensities to mean intensity and get statistics
    maps, grids = mc_obj.reduce_shell(combined)
    return maps, grids

def reduce_shells(system, num_shells):
    """
    Combine shells into final maps for average intensity, <I>/sigma(I), and number of pixels.
    Also compute mean of these parameters per resolution shell; return as shell_stats.
    """
    
    map_shape = (len(system['bins']['h']), len(system['bins']['k']), len(system['bins']['l']))
    map_keys = ["I", "I_sigI", "n_pixels"]

    file_glob = glob.glob(system['map_path'] + "temp/grid_rshell*.h5")
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
            shell_stats[key][shell] = np.median(data[data>0])

            if key == "I":
                shell_stats['resolution'][shell] = np.median(hkl_res[data>0])
            
    return combined_maps, shell_stats

if __name__ == '__main__':

    start = time.time()
    system = pickle.load(open(sys.argv[2], "rb"))

    if sys.argv[1] == 'compile':

        # generating temp dir for resolution shell data
        output_dir = system['map_path'] + "temp/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # processing resolution shell
        I_dir, num_shells, n_shell = sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
        shell_maps, shell_grids = compile_shell(system, I_dir, num_shells, n_shell)

        # save resolution shell in grid and dictionary formats
        with open(system['map_path'] + "temp/dict_rshell%i_t%i.pickle" %(n_shell, num_shells), "wb") as handle:
            pickle.dump(shell_maps, handle)

        io.saveh(system['map_path'] + "temp/grid_rshell%i_t%i.h5" %(n_shell, num_shells), **shell_grids)
            
    if sys.argv[1] == "reduce":

        # combine resolution shells
        combined_maps, shell_stats = reduce_shells(system, int(sys.argv[3]))

        io.saveh(system['map_path'] + "final_maps.h5", **combined_maps)
        io.saveh(system['map_path'] + "shell_statistics.h5", **shell_stats)
                
    print "elapsed time is %f" %((time.time() - start)/60.0)
