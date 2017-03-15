from collections import defaultdict
import map_utils, itertools, time
import cPickle as pickle
import numpy as np

class GenerateMap:

    def __init__(self, data_path, nshells):
        """
        Initialize class with inputs: data_path, folder in which dictionary containing
        map specifications lives; nshells, number of resolution shells over which to 
        parallelize map generation.
        """

        self.system = pickle.load(open(data_path + "system.pickle", "rb"))
        self.nshells = nshells
        self.set_up()

    def set_up(self):
        """
        Divide the map space into resolution shells of equal numbers of voxels; 
        return a np.array of shape (n_voxels, 3) that specifies the voxel centers
        for the entire map and a np.array of shape (n_voxels) that specifies what 
        resolution shell each voxel belongs to.
        """

        # bins associated with the digitized data, where relevant values are the voxel's upper bounds
        self.edges = dict()
        offset = 1.0/(2.0*self.system['subsampling'])
        for key in self.system['bins'].keys():
            centers = self.system['bins'][key].copy()
            self.edges[key] = np.linspace(centers.min() - offset, centers.max() + offset, len(centers) + 1)

        # hkl grid for voxel centers associated with edges dictionary
        vx_centers = np.array(list(itertools.product(self.edges['h'] - offset, \
                                                         self.edges['k'] - offset, self.edges['l'] - offset)))

        # digitized map shape -- added row along each axis relative to final map to be built
        self.dmap_shape = (len(self.edges['h']), len(self.edges['k']), len(self.edges['l'])) 

        # divide voxels among resolution shells of equal numbers of voxels (so linear in 1/d^3)
        voxel_res = map_utils.compute_resolution(self.system['space_group'], self.system['cell'], vx_centers)
        inv_dcubed = 1.0/(voxel_res**3)

        nvox_per_shell, shell_bounds = np.histogram(inv_dcubed[voxel_res > self.system['d']], bins = self.nshells)
        shell_bounds = np.concatenate((shell_bounds, [1])) # assuming here we're not working with <1 A data
        self.vx_dig = np.digitize(inv_dcubed, shell_bounds)

        return vx_centers

    def process_image(self, indexed, shell):
        """
        Generate a dictionary with keys as 1D voxel identifier and values a list 
        of intensities that fall into that voxel from an indexed image array. Only
        voxels that fall into the resolution bin 'shell' are considered. 
        """
        
        # determine voxel indices in this resolution shell
        if (shell < 1) or (shell > self.nshells):
            print "Warning: not a valid resolution shell for this calculation."

        vx_indices = np.where(self.vx_dig == shell)

        # eliminate invalid datapoints in the indexed image
        data_res = map_utils.compute_resolution(self.system['space_group'], self.system['cell'], indexed)
        data_subset = indexed[np.where(data_res > self.system['d'] - 0.1)[0]]

        # digitize and ravel data such that each datapoint is associated with a 1D identifier
        dig_data = np.array((np.digitize(data_subset[:,0], self.edges['h']), 
                             np.digitize(data_subset[:,1], self.edges['k']), 
                             np.digitize(data_subset[:,2], self.edges['l'])))
        rav_data = np.ravel_multi_index(dig_data, self.dmap_shape)

        # determine the voxels observed in this image and specified shell
        intersection = np.intersect1d(rav_data, vx_indices)
        
        # generate dictionary of voxel ID : list of intensities
        shell_dict = defaultdict(list)
        for voxel in range(len(intersection)):
            shell_dict[intersection[voxel]] = data_subset[np.where(rav_data==intersection[voxel])[0]][:,-1].tolist()

        return shell_dict

    def merge_dicts(self, d1, d2):
        """
        Merge dictionaries by combining key values when keys overlap.
        """

        d_comb = defaultdict(list)
        
        # merge keys into a combined dictionary
        for d in (d1, d2):
            for key, value in d.iteritems():
                d_comb[key].append(value)

        # flatten values of combined dictionary (so no longer stored as list of lists)
        for key, value in d_comb.iteritems():
            d_comb[key] = [item for sublist in value for item in sublist]

        return d_comb
