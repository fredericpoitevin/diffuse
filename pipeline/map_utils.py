import itertools, math, os.path, glob
import cPickle as pickle
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
    Compute d-spacing / resolution for a set of scattering vectors, where q = 2*pi*s.
    Set (0,0,0) to arbitrarily high (but not infinite) resolution.
    Inputs: np.array of cell constants in order (a, b, c, alpha, beta, gamma),
            space group (number), grid of scattering vectors
    Output: np.array of d-spacings, which is empty if space group is unsupported. 
    """

    a, b, c, alpha, beta, gamma = cell_constants
    h, k, l = s_grid[:,0], s_grid[:,1], s_grid[:,2]

    # valid for orthorhombic, cubic, and tetragonal
    if ((space_group >= 16) and (space_group <=142)) or ((space_group >= 195) and (space_group <=230)):
        inv_d = np.sqrt(np.square(h/a) + np.square(k/b) + np.square(l/c))

    # valid for hexagonal (and possibly trigonal? if so change lower bound to 143)
    elif (space_group >= 168) and (space_group <=194):
        inv_d = np.sqrt(4.0*(np.square(h) + h*k + np.square(k))/(3*np.square(a)) + np.square(l/c))

    # valid for monoclinic 
    elif (space_group >= 3) and (space_group <=15):
        beta = np.deg2rad(beta)
        inv_d = np.sqrt( np.square(h/(a*np.sin(beta))) + np.square(k/b) + np.square(l/(c*np.sin(beta)))\
                             + 2*h*l*np.cos(beta) / (a*c*np.square(np.sin(beta))))

    else:
        print "This space group is currently unsupported. Please add 'bins' key manually to systems.pickle."
        return np.empty(0)

    inv_d[inv_d==0] = 1e-5
    res = 1.0 / inv_d

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


def ind_rprofile(system, indexed, n_bins, num = -1, median=False):
    """
    Compute radial intensity profile for an indexed image.
    Inputs: system, dictionary that contains cell dimensions
            indexed, indexed image with columns (h,k,l,I)
            n_bins, number of bins in q-space
            num, image number. if -1, use cell constants for A
            median, if False (default), compute mean intensity
    Outputs: rS, mean or median |S| for each bin
             rI, radial intensity profile
    """

    # compute magnitude of scattering vector
    if num == -1:
        A_inv = np.linalg.inv(np.diag(system['cell'][:3]))
    else:
        #A_inv = np.linalg.inv(system['A_batch'][(num - 1)/system['batch_size']])
        A_inv = np.linalg.inv(system['A_batch'][system['img2batch'][num]])

    s_vecs = np.inner(A_inv, indexed[:,:3]).T
    s_mags = np.linalg.norm(s_vecs, axis=1)

    # bin intensities into shells in scattering space
    x, y = s_mags, indexed[:,-1].copy()
    n_per_shell, shells = np.histogram(x[y > 0], bins = n_bins)
    shells = np.concatenate((shells, [shells[-1] + (shells[-1] - shells[-2])]))

    dx = np.digitize(x, shells)
    bin_sel = np.array([len(y[(dx == i) & (y > 0)]) for i in range(np.min(dx), np.max(dx) + 1)])
    bin_idx = np.where(bin_sel > 2)[0] + np.min(dx) # avoid empty bins

    # compute median or mean mag(S) and intensity
    if median is True:
        rS = np.array([np.median(x[(dx == i) & (y > 0)]) for i in bin_idx])
        rI = np.array([np.median(y[(dx == i) & (y > 0)]) for i in bin_idx])
    else:
        rS = np.array([np.mean(x[(dx == i) & (y > 0)]) for i in bin_idx])
        rI = np.array([np.mean(y[(dx == i) & (y > 0)]) for i in bin_idx])

    return rS, rI


def mtx_rprofile(system, dir_pI, n_bins, median = False):
    """
    Generate a matrix of radial intensity profiles for a dataset. The matrix will
    have shape (n_images + 1, n_bins), where the first row corresponds to the |S|
    at which intensities in each image were evaluated.

    Inputs: system, dictionary that contains cell dimensions
            dir_pI, directory/prefix for intensity files
            n_bins, number of bins in scattering space
            median, if False (default), compute mean intensity
    Output: rSI_mtx, matrix of radial intensity profiles and associated |S|
    """

    # retrieve ordered file lists for indexed images and intensities
    file_glob = glob.glob(system["map_path"] + "indexed/*.npy")
    filelist = sorted(file_glob, key = lambda name: int(name.split('_')[-1].split('.')[0]))

    if dir_pI != 'indexed':
        I_glob = glob.glob(system["map_path"] + dir_pI + "/*.npy")
        Ilist = sorted(I_glob, key = lambda name: int(name.split('_')[-1].split('.')[0]))

    # set up matrices for preliminary storage and output
    prelim, rSI_mtx = np.zeros((len(filelist)*2, n_bins)), np.zeros((len(filelist)+1, n_bins))

    # loop over each file, computing radial intensity profile
    for i in range(len(filelist)):

        indexed = np.load(filelist[i])
        if dir_pI != 'indexed':
            assert filelist[i].split('_')[-1].split('.')[0] == Ilist[i].split('_')[-1].split('.')[0]
            imgI = np.load(Ilist[i])
            indexed[:,-1] = imgI            

        #img_num = i + 1 # 1-indexed
        img_num = int(filelist[i].split('_')[-1].split('.')[0])
        prelim[i], prelim[i+len(filelist)] = ind_rprofile(system, indexed, n_bins, img_num, median)

    # compute mean |S| profile and interpolate all q's onto this S
    rSI_mtx[0] = np.mean(prelim[:len(filelist)].T, axis=1)
    for i in range(len(filelist)):
        rSI_mtx[i+1] = np.interp(rSI_mtx[0], prelim[i], prelim[i+len(filelist)])

    return rSI_mtx


def process_stol():
    """ 
    Convert *.stol files in 'reference' directory from F(sin(theta)/lambda) to I(S).
    Inputs: None
    Output: dict[scattering source] = array(|S|, I)
    """

    stol_sI = dict()
    if os.path.exists("reference/water.stol"):
        stol_sI['water'] = np.loadtxt("reference/water.stol")
    if os.path.exists("reference/Paratone-N.stol"):
        stol_sI['paratone'] = np.loadtxt("reference/Paratone-N.stol")

    for key in stol_sI.keys():
        stol_sI[key][:,0] = 2.0*stol_sI[key][:,0]
        stol_sI[key][:,1] = np.square(stol_sI[key][:,1])

    if len(stol_sI.keys()) != 2:
        print "Warning: one or both of the .stol files not found."
        
    return stol_sI
