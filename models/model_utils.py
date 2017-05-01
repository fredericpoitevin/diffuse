import itertools, time, scipy.stats
from itertools import islice
import cPickle as pickle
import numpy as np
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


def retrieve_bfactors(pdb_path, as_delta = False):
    """
    Retrieve isotropic B factors from input_pdb; Note that pdb_path corresponds to 
    the pdb text file, _not_ an MDTraj pdb object. If as_delta is True, convert B 
    factors to root mean square atomic displacements, deltas.
    """

    # fetch values from B factor column in PDB file
    pdb_bfactors = list()
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith('ATOM'):
                pdb_bfactors.append(float(line[60:66]))
    pdb_bfactors = np.array(pdb_bfactors)

    # return B factors or deltas
    if as_delta is False:
        return pdb_bfactors
    else:
        return np.sqrt(pdb_bfactors/8/np.square(np.pi))


def symmetrize(system, symm_ops, input_map, laue = True, from_asu = True):
    """
    Symmetrize input_map based on dictionary of symmetry operations specified in symm_ops.
    Boolean flags set whether to 1. symmetrize Laue-equivalent voxels only or both Laue-
    equivalents and Friedel pairs and 2. whether input_map is derived from an ASU or unit 
    cell; in the former case, the incoherent sum of all ASUs is returned, and in the latter,
    symmetry-equivalent voxels are averaged. Map specifications are provided in system input.

    Inputs: system, dictionary with map specifications (space group, cell, bins, d/max res.)
            symm_ops, dictionary of symmetry operations; second half of keys are Friedels
            input_map, map to be symmetrized
            laue, boolean flag. if True, also symmetrize Friedel pairs
            from_asu, boolean flag. if True, sum rather than average equivalent voxels
    Outputs: symm_map, symmetrized map
             multiplicities, multiplicities of each voxel in symm_map

    """
    
    space_group = system['space_group']
    grid = np.array(list(itertools.product(system['bins']['h'], system['bins']['k'], system['bins']['l'])))

    mh, mk, ml = [np.max(system['bins'][key]) for key in ['h', 'k', 'l']]
    lh, lk, ll = [len(system['bins'][key]) for key in ['h', 'k', 'l']]
    if len(input_map.shape) == 1:
        input_map = input_map.reshape(lh, lk, ll)

    # valid for orthorhombic and tetragonal
    if ((space_group >= 16) and (space_group <=142)) or ((space_group >= 195) and (space_group <=230)):
        expand = False
        bins = system['bins'].copy()

    # valid for hexagonal, possibly trigonal
    elif (space_group >= 168) and (space_group <=194):
        expand = True
        bins = dict()
        bins['h'] = np.linspace(-mh - mk, mh + mk, lh + lk - 1)
        bins['k'] = np.linspace(-mh - mk, mh + mk, lh + lk - 1)
        bins['l'] = system['bins']['l']

    else:
        print "This space group is currently not supported"
        return

    # expand input map (for consistency with np.ravel_mult_index output)
    extent = np.max([len(bins[key]) for key in bins.keys()])
    cen = int(extent / 2)
    expand_map = np.zeros((extent, extent, extent))
    expand_map[cen-lh/2:cen+lh/2+1, cen-lk/2:cen+lk/2+1, cen-ll/2:cen+ll/2+1] = input_map
    platform = np.max([np.max(bins[key]) for key in bins.keys()])

    # if laue is True, use only first half of keys (Bijovet positive)
    num_keys = len(symm_ops.keys())
    if laue is True:
        num_keys = len(symm_ops.keys())/2

    # reduce grid to voxels that fall within map resolution
    res = compute_resolution(system['space_group'], system['cell'], grid)
    r_grid = grid[np.where(res > system['d'])[0]]

    # generate symmetry mates for all voxels within map resolution
    symm_idx = np.zeros((num_keys, r_grid.shape[0]))
    for key in range(num_keys):
        print "on key %i" %key

        rot = np.inner(symm_ops[key], r_grid).T
        rot_bump = system['subsampling']*(rot + platform)
        rot_bump = np.around(rot_bump).astype(int)

        idx = np.ravel_multi_index(rot_bump.T, (extent, extent, extent))
        symm_idx[key] = idx

    # retrieve column-aligned symmetry-equivalent values
    symm_idx = symm_idx.astype(int)
    vals = expand_map.flatten()[symm_idx]

    # average values if not from asu; otherwise, sum
    if from_asu is False:
        vals[vals==0] = np.nan
        symm_vals = np.nanmean(vals.T, axis=1)
        symm_vals[np.isnan(symm_vals)] = 0
    else:
        symm_vals = np.sum(vals.T, axis=1)

    symm_map = np.zeros_like(input_map.flatten())
    symm_map[np.where(res > system['d'])[0]] = symm_vals

    # generate multiplicities array; set 0 values to max value to avoid downstream problems
    m_partial = np.array([len(np.unique(symm_idx.T[i])) for i in range(symm_idx.shape[1])])
    multiplicities = np.zeros_like(input_map.flatten())
    multiplicities[np.where(res > system['d'])[0]] = m_partial
    multiplicities[multiplicities==0] = np.max(m_partial)

    return symm_map, multiplicities


def deorth_matrix(system):
    """
    Compute deorthogonalization matrix from cell constants. Equation for this matrix, M, here:
    http://www.ruppweb.org/Xray/tutorial/Coordinate%20system%20transformation.htm.
    """

    a, b, c, alpha, beta, gamma = system['cell']
    alpha, beta, gamma = np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma)

    V = a*b*c*np.sqrt(1.0 - np.square(np.cos(alpha)) - np.square(np.cos(beta)) \
                      - np.square(np.cos(gamma)) + 2.0*np.cos(alpha)*np.cos(beta)*np.cos(gamma))
    M = np.array([[1.0/a, -np.cos(gamma)/(a*np.sin(gamma)), \
                   ((b*c*np.cos(gamma)*(np.cos(alpha) - np.cos(beta)*np.cos(gamma)))/np.sin(gamma) \
                    - b*c*np.cos(beta)*np.sin(gamma))*(1.0/V)],
                  [0, 1.0/(b*np.sin(gamma)), -1.0*a*c*(np.cos(alpha) - np.cos(beta)*np.cos(gamma))/(V*np.sin(gamma))],
                  [0, 0, a*b*np.sin(gamma)/V]])

    return M


def generate_extents(system):
    """
    Generate dictionary of tuples to be used for plotting with plt.imshow.
    """
    
    A_inv = deorth_matrix(system)
    hkl_grid = np.array(list(itertools.product(system['bins']['h'], system['bins']['k'], system['bins']['l'])))
    q_vecs = 2*np.pi*np.inner(A_inv.T, hkl_grid).T

    max_h, max_k, max_l = [np.max(q_vecs[:,i]) for i in range(3)]
    extent = dict()
    extent['0kl'] = (-1.0*max_l, max_l, -1.0*max_k, max_k)
    extent['h0l'] = (-1.0*max_l, max_l, -1.0*max_h, max_h)
    extent['hk0'] = (-1.0*max_k, max_k, -1.0*max_h, max_h)

    return extent


def compute_qmags(system):
    """
    Compute the magnitudes of the q vectors for the flattened map grid specified 
    in the system.pickle file.
    """
    
    A_inv = deorth_matrix(system)
    hkl_grid = np.array(list(itertools.product(system['bins']['h'], system['bins']['k'], system['bins']['l'])))
    
    q_vecs = 2*np.pi*np.inner(A_inv.T, hkl_grid).T 
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


def subtract_radavg(system, input_map, bin_width = None, medians = False):
    """
    Subtract the radial average from the input_map. Platform such that smallest value of
    non-empty voxel is greater than zero (and empty voxels remain zero).

    Inputs: system, system.pickle file with map and space group information
            input_map, map from which to subtract the radial average
            bin_width, in units of inverse Angstrom. Default is None, in which case bin width 
                       is computed using the Freedman-Diaconis rule: 2*IQR(x)/(n^(1/3))
            medians, if True, compute median rather than mean radial profile. Default: False
    Output: aniso_map, input_map with radial average subtracted
    """

    # compute resolution associated with voxels
    hkl = np.array(list(itertools.product(system['bins']['h'], system['bins']['k'], system['bins']['l'])))
    hkl_res = compute_resolution(system['space_group'], system['cell'], hkl)

    # perform binning, linear in 1/d
    x, y = 1.0/hkl_res[input_map.flatten()>0], input_map.copy().flatten()[input_map.flatten()>0]
    if bin_width is None:
        bin_width = 2*scipy.stats.iqr(x)/(float(len(x))**(1.0/3.0))
    bins = np.arange(np.min(x), np.max(x) + bin_width, bin_width)

    n_per_shell, shells = np.histogram(x, bins = bins)
    shells = np.concatenate((shells, [shells[-1] + (shells[-1] - shells[-2])]))

    print 'bin width: %f, avg, min voxels per shell: %i, %i, num shells: %i' \
        %((shells[-1] - shells[-2]), np.mean(n_per_shell), np.min(n_per_shell), len(shells))
    
    # create resolution vs average radial intensity profile
    dx = np.digitize(x, shells)
    if medians is False:
        xm = np.array([np.mean(x[np.where(dx == i)]) for i in range(np.min(dx), np.max(dx)+1)])
        ym = np.array([np.mean(y[np.where(dx == i)]) for i in range(np.min(dx), np.max(dx)+1)])
    else:
        xm = np.array([np.median(x[np.where(dx == i)]) for i in range(np.min(dx), np.max(dx)+1)])
        ym = np.array([np.median(y[np.where(dx == i)]) for i in range(np.min(dx), np.max(dx)+1)])

    # dealing with case where some bins are empty
    xm_real = np.delete(xm, np.where((np.isnan(xm)) | (np.isnan(ym)))[0])
    ym_real = np.delete(ym, np.where((np.isnan(xm)) | (np.isnan(ym)))[0])

    # subtract radial average
    masked_sub = y - np.interp(x, xm_real, ym_real)
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

    # only consider voxels with valid (positive) intensities                                                                                                              
    map1_sel, map2_sel = map1.copy().flatten(), map2.copy().flatten()
    valid_idx = np.where((map1_sel > 0) & (map2_sel > 0))[0]
    map1_sel, map2_sel, mult_sel = map1_sel[valid_idx], map2_sel[valid_idx], mult[valid_idx]
    mult_sel /= float(np.sum(mult_sel))

    return w_cov(map1_sel, map2_sel, mult_sel) / np.sqrt(w_cov(map1_sel, map1_sel, mult_sel) * w_cov(map2_sel, map2_sel, mult_sel))


def cc_by_shell(system, n_shells, map1, map2, mult, hkl_grid = None):
    """  
    Compute multiplicity-weighted correlation coefficient across n_shells resolution shells 
    (with spacing even in 1/d^3); return np.arrays of CC and median resolution of shells. 
    """

    if hkl_grid is None:
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


def cc_friedels(system, input_map, n_shells):
    """
    Compute correlation coefficient between (h,k,l) and (-h,-k,-l), both overall and by
    resolution shell. Currently treating all voxels as independent rather than weigthed
    by multiplicity. Return Friedel-symmetrized map.
    """

    # computing CC, overall and by bin
    Ipos, Ineg = input_map.copy().flatten(), input_map.copy().flatten()[::-1]
    mult = np.ones(Ipos.shape)

    cc_overall = np.corrcoef(Ipos[np.where((Ipos > 0) & (Ineg > 0))[0]], Ineg[np.where((Ipos > 0) & (Ineg > 0))[0]])[0,1]
    res_shells, cc_shells = cc_by_shell(system, n_shells, Ipos, Ineg, mult)

    # symmetrizing Friedel pairs
    vals = np.vstack((Ipos, Ineg))
    vals[vals==0] = np.nan

    fsymm = np.nanmean(vals.T, axis=1)
    fsymm[np.isnan(fsymm)] = 0

    return cc_overall, res_shells, cc_shells, fsymm


