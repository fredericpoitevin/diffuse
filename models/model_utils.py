from itertools import islice
import cPickle as pickle
import numpy as np
import itertools, time
import mdtraj as md

""" Miscellaneous functions useful for analyzing diffuse maps. """

def split_unitcell(uc_pdb, n_asus):
    """
    Split asymmetric units of input unit cell and return as separate entries in output
    dictionary.
    """

    # stack asus if written out as separate models in input pdb
    if uc_pdb.n_frames > 1:
        processed = uc_pdb[0]
        for i in range(1, uc_pdb.n_frames):
            processed = processed.stack(uc_pdb[i])
    else:
        processed = uc_pdb

    asus = dict()
    n_atoms = processed.n_atoms/n_asus

    for i in range(n_asus):
        asus[i] = processed.atom_slice(range(n_atoms*i, n_atoms*(i+1)))

    return asus


def generate_symmates(symm_ops, bins, subsampling):
    """
    Return a dictionary of np.arrays such that the nth term of each array is a set of
    symmetry-equivalent indices in raveled format. Also return the corresponding set 
    of hkl vectors and multiplicities.

    Inputs: symm_ops, dictionary containing symmetry operators
            bins, dictionary of voxel centers along h,k,l
            subsampling, degree of oversampling relative to integer hkl
    Outputs: symm_idx, dict of np.arrays of symmetry-equivalent indices
             grid, np.array of hkl vectors
             multiplicities, np.array of multiplicities
    """

    grid = np.array(list(itertools.product(bins['h'], bins['k'], bins['l'])))
    extent = np.max([len(bins['h']), len(bins['k']), len(bins['l'])])
    platform = np.max([np.max(bins['h']), np.max(bins['k']), np.max(bins['l'])])

    # generate symmetry-equivalent indices
    symm_idx = dict()
    for key in symm_ops.keys():
        rot = np.inner(symm_ops[key], grid).T
        rot_bump = subsampling*(rot + platform)
        rot_bump = np.around(rot_bump).astype(int)
        
        idx = np.ravel_multi_index(rot_bump.T, (extent, extent, extent))
        symm_idx[key] = np.argsort(idx)

    # compute multiplicities
    combined = np.zeros((len(symm_idx.keys()), len(symm_idx[0])))
    for key in symm_idx.keys():
        combined[key] = symm_idx[key]
    multiplicities = np.array([len(np.unique(combined.T[i])) for i in range(combined.shape[1])])

    return symm_idx, grid, multiplicities


def symmetrize(input_map, symm_idx, from_asu = False):
    """
    Symmetrize input map according to the symmetry-equivalent indices encoded in symm_idx;
    here it is assumed that the second half of arrays in symm_idx correspond to Friedels.
    If from_asu is True, sum all symmetry-equivalent values; otherwise, assume input_map is
    an unsymmetrized map of the unit cell, and average non-zero symmetry equivalent indices.
    """
    
    unsymm_map = input_map.copy().flatten()
    symm_map = np.zeros(unsymm_map.shape)
    nonzero_counts = len(symm_idx.keys())*np.ones(unsymm_map.shape)

    if from_asu is False:
        for key in symm_idx.keys():
            symm_map += unsymm_map[symm_idx[key]]
            nonzero_counts[np.where(unsymm_map[symm_idx[key]]==0)[0]] -= 1
        symm_map[nonzero_counts!=0] /= nonzero_counts[nonzero_counts!=0]

    else:
        for key in symm_idx.keys()[:len(symm_idx.keys())/2]: 
            symm_map += unsymm_map[symm_idx[key]]

    return symm_map


def generate_mesh(system):
    """
    Generate qvector mesh for use with plt.pcolormesh from A matrix and bins.
    """
    A_inv = np.linalg.inv(np.diag(system['cell'][:3]))

    mesh = dict()
    mesh['projX'] = np.meshgrid(2*np.pi*np.dot(A_inv[2][2], system['bins']['l']), 2*np.pi*np.dot(A_inv[1][1], system['bins']['k']))
    mesh['projY'] = np.meshgrid(2*np.pi*np.dot(A_inv[2][2], system['bins']['l']), 2*np.pi*np.dot(A_inv[0][0], system['bins']['h']))
    mesh['projZ'] = np.meshgrid(2*np.pi*np.dot(A_inv[1][1], system['bins']['k']), 2*np.pi*np.dot(A_inv[0][0], system['bins']['h']))

    return mesh


def compute_qmags(system):
    """
    Compute the magnitudes of the q vectors for the flattened map grid specified 
    in the system.pickle file.
    """
    
    A_inv = np.linalg.inv(np.diag(system['cell'][:3]))
    hkl_grid = np.array(list(itertools.product(system['bins']['h'], system['bins']['k'], system['bins']['l'])))
    
    q_vecs = 2*np.pi*np.inner(A_inv, hkl_grid).T 
    return np.linalg.norm(q_vecs, axis=1)


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


def subtract_radavg(system, input_map, n_bins, medians = False):
    """
    Subtract the radial average from the input_map. Platform such that smallest value of
    non-empty voxel is greater than zero (and empty voxels remain zero).

    Inputs: system, system.pickle file with map and space group information
            input_map, map from which to subtract the radial average
            n_bins, number of bins across which to determine the radial average profile
    Output: aniso_map, input_map with radial average subtracted
    """

    # compute resolution associated with voxels
    hkl = np.array(list(itertools.product(system['bins']['h'], system['bins']['k'], system['bins']['l'])))
    hkl_res = compute_resolution(system['space_group'], system['cell'], hkl)

    # perform binning, linear in 1/d
    x, y = 1.0/hkl_res[input_map.flatten()>0], input_map.copy().flatten()[input_map.flatten()>0]
    n_per_shell, shells = np.histogram(x, bins = n_bins)
    shells = np.concatenate((shells, [shells[-1] + (shells[-1] - shells[-2])]))
    print 'shell spacing: %f, avg, min voxels per shell: %i, %i' \
        %((shells[-1] - shells[-2]), np.mean(n_per_shell), np.min(n_per_shell))
    
    # create resolution vs average radial intensity profile
    dx = np.digitize(x, shells)
    if medians is False:
        xm = np.array([np.mean(x[np.where(dx == i)]) for i in range(np.min(dx), np.max(dx)+1)])
        ym = np.array([np.mean(y[np.where(dx == i)]) for i in range(np.min(dx), np.max(dx)+1)])
    else:
        xm = np.array([np.median(x[np.where(dx == i)]) for i in range(np.min(dx), np.max(dx)+1)])
        ym = np.array([np.median(y[np.where(dx == i)]) for i in range(np.min(dx), np.max(dx)+1)])

    # subtract radial average
    masked_sub = y - np.interp(x, xm, ym)
    aniso_map = np.zeros(hkl_res.shape)
    aniso_map[input_map.flatten()>0] = masked_sub - np.min(masked_sub) + np.min(y)
    
    return aniso_map


def mweighted_cc(map1, map2, mult = None):
    """ 
    Compute correlation coefficient between maps. Valid (positive, nonzero) intensity values are 
    weighted by their multiplicity; it is assumed that input maps exhibit the same symmetry. 
    """

    def w_mean(x, weights):
        return np.sum(x * weights) / np.sum(weights)

    def w_cov(x, y, weights):
        return np.sum(weights * (x - w_mean(x, weights)) * (y - w_mean(y, weights))) / np.sum(weights)

    # process multiplicities array, generating one with equal weights if not given
    if mult is None:
        mult = np.ones(map1.shape).flatten()
    mult = 1.0/mult
    mult /= np.sum(mult)

    # only consider voxels with valid (positive) intensities                                                                                                              
    map1_sel, map2_sel = map1.copy().flatten(), map2.copy().flatten()
    valid_idx = np.where((map1_sel > 0) & (map2_sel > 0))[0]
    map1_sel, map2_sel, mult_sel = map1_sel[valid_idx], map2_sel[valid_idx], mult[valid_idx]

    return w_cov(map1_sel, map2_sel, mult_sel) / np.sqrt(w_cov(map1_sel, map1_sel, mult_sel) * w_cov(map2_sel, map2_sel, mult_sel))


def cc_by_shell(system, n_shells, map1, map2, mult):
    """  
    Compute multiplicity-weighted correlation coefficient across n_shells resolution shells 
    (with spacing even in 1/d^3); return np.arrays of CC and median resolution of shells. 
    """

    hkl_grid = np.array(list(itertools.product(system['bins']['h'], system['bins']['k'], system['bins']['l'])))
    res = compute_resolution(system['space_group'], system['cell'], hkl_grid)

    inv_dcubed = 1.0/(res**3.0)
    nvox_per_shell, shell_bounds = np.histogram(inv_dcubed[res > system['d']], bins = n_shells)

    assert system['d'] > 1.0
    shell_bounds = np.concatenate((shell_bounds, [1.0]))
    vx_dig = np.digitize(inv_dcubed, shell_bounds)
    res_bins = (1.0/shell_bounds[:-1])**(1.0/3)

    cc_shell, res_shell = np.zeros(n_shells), np.zeros(n_shells)
    map1, map2 = map1.flatten(), map2.flatten()
    for i in range(1, n_shells+1):
        idx = np.where(vx_dig==i)[0]
        cc_shell[i-1] = mweighted_cc(map1[idx], map2[idx], mult = mult)
        valid = reduce(np.intersect1d, (np.where(map1>0)[0], np.where(map2>0)[0], idx))
        res_shell[i-1] = np.mean(res[reduce(np.intersect1d, (np.where(map1>0)[0], np.where(map2>0)[0], idx))])

    return res_shell, cc_shell
