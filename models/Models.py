import model_utils, map_utils
from scipy import signal
import scipy.optimize
import time, itertools, glob
import mdtraj as md
import numpy as np

"""
Distinct disorder models are written into different classes, generally with a 'scale'
and an 'optimize' feature.
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
        
        scaled_sel = self.scale(transform_sel, qmags_sel, m_f, b_f, sigma_f, log)
        scaled_transform = np.zeros(len(transform))
        scaled_transform[valid_idx] = scaled_sel

        return scaled_transform, m_f, b_f, sigma_f


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
