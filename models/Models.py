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
