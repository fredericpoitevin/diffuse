import model_utils
from scipy import signal
import scipy.optimize
import time, itertools, glob
import mdtraj as md
import numpy as np

"""
Distinct disorder models are written into different classes, with functions relevant to
building or fitting that model as class functions.
"""

class Chapman:
    
    """
    'Chapman' model that attributes diffuse features to translational disorder of the
    rigid body contained in the asymmetric unit. 

    I_diffuse = m*(1 - exp(-q^2*sigma^2))*|F(q)|^2 + b

    Free parameters are m, a multiplicative scale factor
                        b, a constant platform
                        sigma, parameter that tunes the disorder

    Optionally can optimize over log sigma values; empirically it was found to help
    to scan over sigma in a linear rather than log scale.

    """

    def scale(self, transform, qmags, m, b, sigma, log = False):
        """
        Scale the molecular transform map to the diffuse map.
        """

        if log is True:
            wilson_factor = np.exp(-1*np.square(qmags*10**(sigma)))
        else:
            wilson_factor = np.exp(-1*np.square(qmags*sigma))

        return m*transform*(1.0 - wilson_factor) + b

    def optimize(self, transform, target, qmags, sigma_guess, log = False):
        """
        Solve for best fit parameters (m, b, sigma) that scale transform to target.
        """

        def error(args):
            m, b, sigma = args
            scaled = self.scale(transform_sel, qmags_sel, m, b, sigma, log)
            residuals = scaled - target_sel
            return residuals

        valid_transform, valid_tar = np.where(transform > 0)[0], np.where(target > 0)[0]
        valid_idx = np.intersect1d(valid_transform, valid_tar)
        transform_sel, target_sel, qmags_sel = transform.copy()[valid_idx], \
            target.copy()[valid_idx], qmags.copy()[valid_idx]

        x0 = (np.mean(transform_sel)/np.mean(target_sel), 0.0, sigma_guess) # initial condition
        res = scipy.optimize.leastsq(error, x0, ftol=1e-15)
        m_f, b_f, sigma_f = res[0] # least squares result
        
        scaled_transform = self.scale(transform, qmags, m_f, b_f, sigma_f, log)
        scaled_transform[transform==0] = 0

        return scaled_transform, m_f, b_f, sigma_f


class Wall:
    """
    Wall/liquid-like motions model. Recommendation is to fit parameters by grid search since 
    achieving convergence is difficult. This was the approach taken by Wall, ME et al. 1997.
    """

    def scale(self, transform, qmags, map_shape, sigma, gamma, model):
        """
        Scale the input molecular transform using the Wall model. Tunable disorder parameters 
        are sigma and gamma, which correspond to the size of the global atomic displacements 
        and the correlation length, respectively. Code courtesy TJL.
        """

        transform, qmags = transform.reshape(map_shape), qmags.reshape(map_shape)

        # generate kernel
        kernel = np.zeros_like(qmags)
        if model == 'exponential':
            kernel = 8.0 * np.pi * (gamma**3) / np.square(1 + np.square(gamma * qmags))
        elif model == 'stretched':
            kernel = 4.0 * np.pi * (gamma**3) / (1 + np.square(gamma * qmags))
        else:
            raise ValueError('model must be {stretched, exponential}')

        # convolve kernel and map
        wall_map = signal.fftconvolve(transform, kernel, mode='same')

        # appy scaling
        q2s2 = np.square(qmags * sigma)
        wall_map *= np.exp(-q2s2) * q2s2
        wall_map[transform==0] = 0

        return wall_map


class NormalModes:
    """
    Various utilities useful for assessing normal modes models.
    """

    def read_corrmat(self, filepath, n_atoms):
        """
        Return atomic correlation matrix text file as a .npy array.
        """
        
        from itertools import islice
        corr_mat = np.zeros((n_atoms, n_atoms))
        counter, N = 0, 200
        
        with open(filepath, "r") as infile:
            while counter < n_atoms/N + 1:
                gen = islice(infile, N)
                arr = np.genfromtxt(gen, dtype=None)
                start, end = counter*N, counter*N + N
                if end > n_atoms:
                    end = n_atoms
                corr_mat[start:end] = arr
                if arr.shape[0] < N:
                    break
                counter += 1

        return corr_mat

    def corr_to_cov(self, corr_mat, deltas):
        """
        Compute covariance matrix, V, from correlation matrix (corr_mat) and root 
        mean square displacements (deltas). 
        """

        V = np.zeros_like(corr_mat)
        for i in range(corr_mat.shape[0]):
            for j in range(corr_mat.shape[1]):
                V[i][j] = np.sqrt(deltas[i]*deltas[j])*corr_mat[i][j]

        return V


class Ensemble:
    """
    Compute diffuse intensities from an ensemble of structure factors using
    Guinier's equation: I(q) = <|F_n(q)|^2> - |<F_n(q)^2|>.
    """

    def compute(self, system, ensemble, probs, symm_idx = None, sigma = None):
        """
        Calculate I(q) using Guinier's equation.
        Inputs: system, dictionary specifying map; used for calculating qmags
                ensemble, dictionary of structure factors for each state
                probs, probabilities of each state in same order as ensemble's keys
                symm_idx, if not None, symmetrize according to symm_idx dictionary
                sigma, if not None, scale with a global isotropic B factor
        """
        
        assert np.sum(probs) == 1.0
        assert len(probs) == len(ensemble.keys())

        sum_fc_sq, sum_fc = np.zeros_like(ensemble[ensemble.keys()[0]]), np.zeros_like(ensemble[ensemble.keys()[0]])
        for i in range(len(ensemble.keys())):
            sum_fc_sq += probs[i]*np.square(np.abs(ensemble[ensemble.keys()[i]]))
            sum_fc += probs[i]*ensemble[ensemble.keys()[i]]

        I_diffuse = (sum_fc_sq - np.square(np.abs(sum_fc))).real # NB: complex part is zero, but results in warning

        if symm_idx is not None:
            I_diffuse = model_utils.symmetrize(I_diffuse, symm_idx, from_asu = True)

        if sigma is not None:
            qmags = model_utils.compute_qmags(system)
            I_diffuse *= np.exp(-1*np.square(qmags*sigma))

        return I_diffuse


class Autocorrelation: 

    """
    Various methods for computing the autocorrelation function of a 3d map
    in reciprocal space. 
    """

    def acf_extents(self, system, htrim, ktrim, ltrim, hstride, kstride, lstride):
        """
        Generate a dictionary that defines dimensions of ACF slices in real space. The trim and 
        stride inputs refer to the number of rows trimmed from the intensity map and number of 
        spectra taken along each dimension; these will be 0 and 1, respectively if the ACF is 
        computed for a map of shape specified by the system dictionary. NB: extent corresponds
        to real space shape estimate (so factor of two has already been taken into account.
        """
        
        A_inv = np.linalg.inv(np.diag(system['cell'][:3]))
        trim, stride = dict(), dict()
        trim['h'], trim['k'], trim['l'] = htrim, ktrim, ltrim
        
        # trim bins as needed
        bins = dict()
        for key in trim.keys():
            if trim[key] != 0:
                bins[key] = system['bins'][key].copy()[trim[key]:-trim[key]]
            else:
                bins[key] = system['bins'][key].copy()

        hkl_grid = np.array(list(itertools.product(bins['h'], bins['k'], bins['l'])))
        s_vecs = np.inner(A_inv, hkl_grid).T

        max_h, max_k, max_l = [np.max(s_vecs[:,i]) for i in range(3)]
        max_x = len(bins['h'])/(2.0*max_h)/2.0 
        max_y = len(bins['k'])/(2.0*max_k)/2.0
        max_z = len(bins['l'])/(2.0*max_l)/2.0

        # center the extent tuple around the origin
        acf_extent = dict()
        acf_extent['projX'] = (-1.0*max_z/2.0, max_z/2.0, -1.0*max_y/2.0, max_y/2.0)
        acf_extent['projY'] = (-1.0*max_z/2.0, max_z/2.0, -1.0*max_x/2.0, max_x/2.0)
        acf_extent['projZ'] = (-1.0*max_y/2.0, max_y/2.0, -1.0*max_x/2.0, max_x/2.0)

        print "real space voxel dimensions are: "
        print "%.3f angstrom along X" %(max_x/(len(bins['h'])/hstride))
        print "%.3f angstrom along Y" %(max_y/(len(bins['k'])/kstride))
        print "%.3f angstrom along Z" %(max_z/(len(bins['l'])/lstride))

        return acf_extent

    def standard_acf(self, input_map):
        """
        Standard ACF calculation using np.fft; assumes input_map has correct map shape.
        """
        
        map_acf = np.abs(np.fft.ifftn(input_map))
        return np.fft.fftshift(map_acf.real)

    def acf_with_mask(self, input_map, mask=None):
        """
        Compute the Nd autocorrelation from a diffraction pattern with optional masked 
        values (uses the FFT). Code courtesy TJL.
        
        Parameters
        ----------
        input_map : np.ndarray
            The image to autocorrelate (can be any dimension)

        mask : np.ndarray, bool
            Optional mask. If None, no mask applied. If an array, must a boolean
            array the same shape as input_map.

        Returns
        -------
        map_acf : np.ndarray
            The autocorrelation function.
        """

        if mask is not None:
            norm = np.abs(np.fft.ifftn(1 - mask))
            input_map[mask] = 0.0
        else:
            norm = 1.0

        map_acf = np.abs(np.fft.ifftn(input_map))
        map_acf /= norm

        return np.fft.fftshift(map_acf.real)
        
    def acf_spectogram(self, input_map, xcuts, ycuts, exclude_center = False, exclude_edges = False, method = None):
        """
        Compute ACF using spectrogram approach. xcuts and ycuts describe number of slices along 
        the x and y dimensions of the image, respectively. Can optionally exclude the center and 
        edge ACF spectra, and compute the average or median of the ACF spectra. If no method is 
        given, then all ACF spectra are returned.
        """

        # generate mask based on inputs
        mask = np.zeros((xcuts, ycuts), dtype=int)
        if exclude_center is True:
            mask[xcuts/2, ycuts/2] = 1
        if exclude_edges is True:
            mask[0], mask[-1], mask[:,0], mask[:,-1] = 1, 1, 1, 1

        xstride, ystride = input_map.shape[0]/xcuts, input_map.shape[1]/ycuts
        acf_compiled = np.zeros((xcuts*ycuts, xstride, ystride))

        counter = 0
        for i in range(xcuts):
            for j in range(ycuts):

                # take acf of slice if at least 90 percent of pixels are present
                partial = input_map[xstride*(i):xstride*(i+1), ystride*(j):ystride*(j+1)]
                if len(np.where(partial.flatten() == 0)[0])/float(partial.shape[0] * partial.shape[1]) < 0.1:
                    acf_partial = np.abs(np.fft.fftshift(np.fft.ifftn(partial)))
                    
                    # add to average acf if not supposed to be masked
                    if mask[i, j] == 0:
                        acf_compiled[i*xcuts+j] = acf_partial
                        counter += 1

        if method is None:
            acf_processed = acf_compiled

        else:
            acf_processed = np.zeros((xstride, ystride))
            acf_compiled[acf_compiled==0] = np.nan
            for i in range(xstride):
                for j in range(ystride):
                    if method == 'average':
                        acf_processed[i,j] = np.nanmean(acf_compiled[:,i,j])
                    if method == 'median':
                        acf_processed[i,j] = np.nanmedian(acf_compiled[:,i,j])

        return acf_processed
