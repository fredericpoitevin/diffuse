import cPickle as pickle
import numpy as np
import math, os.path, sys
import map_utils

""" 
Amend systems.pickle dictionary with information necessary for map constrution,
specifically: global (rather than per batch) unit cell constants and space group.
User is prompted for desired maximum resolution (d) and subsampling between Braggs;
from this information voxel centers along h,k,l are computed and stored in key 'bins'.

Usage: python amend_system.py [systems.pickle]

Required files: systems.pickle, XDS_ASCII.HKL in key 'xds_path' of systems.pickle
Output files: revised systems.pickle

"""

def prompt_for_map_info():
    """
    Prompt for maximum resolution of map (d) and subsampling between Braggs.
    """

    system['d'] = float(raw_input("Maximum resolution of final map: "))
    system['subsampling'] = int(raw_input("Subsampling between Bragg peaks: "))

    return

def extract_cell_info(xds_path):
    """ 
    Extract final refined cell parameters and space group information 
    from XDS_ASCII.HKL.
    """

    system['A_batch'] = system['A'] # move per batch A matrices to separate key

    filename = xds_path + "XDS_ASCII.HKL"
    with open(filename, "r") as f:

        content = f.readlines()
        
        a_axis = np.asarray([s.split()[1:] for s in content if "!UNIT_CELL_A-AXIS=" in s][0], dtype=float)
        b_axis = np.asarray([s.split()[1:] for s in content if "!UNIT_CELL_B-AXIS=" in s][0], dtype=float)
        c_axis = np.asarray([s.split()[1:] for s in content if "!UNIT_CELL_C-AXIS=" in s][0], dtype=float)
        system['A'] = np.vstack((a_axis, b_axis, c_axis)) # final crystal setting matrix, with orientation

        system['cell'] = np.asarray([s.split()[1:] for s in content if "!UNIT_CELL_CONSTANTS=" in s][0], dtype=float)
        system['space_group'] = int([s.strip('\n').split()[-1] for s in content if "!SPACE_GROUP_NUMBER=" in s][0])

    return


if __name__ == '__main__':

    system = pickle.load(open(sys.argv[1], "rb"))
    prompt_for_map_info()
    extract_cell_info(system['xds_path'])
    system['bins'] = map_utils.determine_map_bins(system['cell'], system['space_group'], system['d'], system['subsampling'])

    # save revised systems.pickle file
    with open(sys.argv[1], 'wb') as handle:
        pickle.dump(system, handle)
