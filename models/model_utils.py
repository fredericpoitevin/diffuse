from itertools import islice
import cPickle as pickle
import numpy as np
import itertools, time

""" Miscellaneous functions useful for analyzing diffuse maps. """

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

def generate_mesh(A, bins):
    """
    Generate qvector mesh for use with plt.pcolormesh from A matrix and bins.
    """
    A_inv = np.linagl.inv(A)

    mesh = dict()
    mesh['projX'] = np.meshgrid(2*np.pi*np.dot(A_inv[2][2], bins['l']), 2*np.pi*np.dot(A_inv[1][1], bins['k']))
    mesh['projY'] = np.meshgrid(2*np.pi*np.dot(A_inv[2][2], bins['l']), 2*np.pi*np.dot(A_inv[0][0], bins['h']))
    mesh['projZ'] = np.meshgrid(2*np.pi*np.dot(A_inv[1][1], bins['k']), 2*np.pi*np.dot(A_inv[0][0], bins['h']))

    return mesh

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
