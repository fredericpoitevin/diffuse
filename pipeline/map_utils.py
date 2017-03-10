import itertools, math
import numpy as np

""" Miscellanious scripts useful for the construction of diffuse maps. """

def rotation_matrix(axis, omega):
    """ 
    Return 3x3 rotation matrix for counter-clockwise rotation around specified 
    axis by omega radians. 
    """

    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(omega/2.0)
    b, c, d = -axis*math.sin(omega/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    rot_mat = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                        [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                        [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    return rot_mat


def compute_resolution(space_group, cell_constants, s_grid):
    """
    Compute d-spacing / resolution for a set of scattering vectors, 
    where q = 2*pi*s.
    Inputs: np.array of cell constants in order (a, b, c, alpha, beta, gamma),
            space group (number), grid of scattering vectors
    Output: np.array of d-spacings 
    """

    a, b, c, alpha, beta, gamma = cell_constants
    h, k, l = s_grid[:,0], s_grid[:,1], s_grid[:,2]

    # valid for orthorhombic, cubic, and tetragonal
    if ((space_group >= 16) and (space_group <=142)) or ((space_group >= 195) and (space_group <=230)):
        res = 1.0/np.sqrt(np.square(h/a) + np.square(k/b) + np.square(l/c))

    # valid for hexagonal (and possibly trigonal? if so change lower bound to 143)
    elif (space_group >= 168) and (space_group <=194):
        res = 1.0/np.sqrt(4.0*(np.square(h) + h*k + np.square(l))/(3*np.square(a)) + np.square(l/c))

    # valid for monoclinic 
    elif (space_group >= 3) and (space_group <=15):
        res = 1.0/np.sqrt( np.square(h/(a*np.sin(np.deg2rad(beta)))) + np.square(k/b) + np.square(l/c) \
                               + 2*h*l*np.cos(np.deg2rad(beta)) / (a*c*np.square(np.sin(np.deg2rad(beta)))))

    else:
        print "This space group is currently unsupported. Please add 'bins' key manually to systems.pickle."
        res = np.empty(0)

    return res


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

    # valid for hexagonal (and possibly trigonal? if so change lower bound to 143)
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
        
    except NameError:
        pass
    
    return bins


