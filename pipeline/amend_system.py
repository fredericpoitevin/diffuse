import cPickle as pickle
import numpy as np
import math, os.path, sys

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

def determine_map_bins(cell_constants, space_group, d, subsampling):
    """
    Determine the grid for map construction.
    Inputs: np.array of cell constants in order (a, b, c, alpha, beta, gamma),
            space group (number), d: maximum resolution, subsampling
    Output: bins dictionary containing voxel centers along h,k,l

    Note: checked using http://www.ruppweb.org/new_comp/reciprocal_cell.htm.

    """
    
    a, b, c, alpha, beta, gamma = cell_constants
    
    # valid for orthorhombic, cubic, and tetragonal 
    if ((space_group >= 16) and (space_group <=142)) or ((space_group >= 195) and (space_group <=230)):
        mh, mk, ml = math.ceil(a/d), math.ceil(b/d), math.ceil(c/d)

    # valid for hexagonal (and possibly trigonal? if so change loewr bound to 143)
    elif (space_group >= 168) and (space_group <=194):
        mh, mk = math.ceil(np.sqrt(3)*a/(2*d)), math.ceil(np.sqrt(3)*a/(2*d))
        ml = math.ceil(c/d)

    # valid for monoclinic
    elif (space_group >= 3) and (space_group <=15):
        mh = math.ceil(a*np.sin(np.deg2rad(beta))/d)
        mk = math.ceil(b/d)
        ml = math.ceil(c*np.sin(np.deg2rad(beta))/d)

    else:
        print "This space group is currently unsupported. Please add 'bins' key manually to systems.pickle."

    # generate bins containing voxel center information
    bins = dict()
    try:
        bins['h'] = np.linspace(-1 * mh, mh, 2 * mh * subsampling + 1)
        bins['k'] = np.linspace(-1 * mk, mk, 2 * mk * subsampling + 1)
        bins['l'] = np.linspace(-1 * ml, ml, 2 * ml * subsampling + 1)
        system['bins'] = bins
        
    except NameError:
        pass
    
    return 

if __name__ == '__main__':

    system = pickle.load(open(sys.argv[1], "rb"))
    prompt_for_map_info()
    extract_cell_info(system['xds_path'])
    determine_map_bins(system['cell'], system['space_group'], system['d'], system['subsampling'])

    # save revised systems.pickle file
    with open(sys.argv[1], 'wb') as handle:
        pickle.dump(system, handle)
