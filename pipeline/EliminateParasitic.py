import scipy.optimize, scipy.spatial
import cPickle as pickle
import numpy as np
import map_utils, os.path
import glob, sys, time, itertools

class GenerateBackground:

    def __init__(self, system, dir_pI, n_bins, rI_input = None):
        self.stols = map_utils.process_stol()
        self.system = system
        self.profiles, self.params = dict(), dict()
        if rI_input is None:
            self.profiles['input'] = map_utils.mtx_rprofile(system, dir_pI, n_bins, median=True)
        else:
            self.profiles['input'] = rI_input

    def opt_ind_paratone(self, S, profile):
        """ 
        Eliminate signature oil peak at |S| = 0.2 1/A by minimizing curvature (specifically, fit to a 
        second order polynomial) of the background-subtracted curve between |S| = 0.1 and |S| = 0.3 1/A. 
        Fit parameters are a multiplicative scale factor and an offset in |S|. Return parameters and the
        background-subtracted profile.
        """

        def error(args):
            offset, scale = args
            background = scale*np.interp(S, self.stols['paratone'][:,0] - offset, self.stols['paratone'][:,1])
            f_pfit = np.poly1d(np.polyfit(S[idx], profile[idx] - background[idx], 2))
            residual = profile[idx] - background[idx] - f_pfit(S[idx])
            return residual

        idx = np.where((S > 0.1) & (S < 0.3))[0]
        x0 = (0.0, 0.1) # guess initial condition

        res = scipy.optimize.leastsq(error, x0)
        f_offset, f_scale = res[0]
        bgd_profile = f_scale*np.interp(S, self.stols['paratone'][:,0] - f_offset, self.stols['paratone'][:,1])
        
        return f_offset, f_scale, bgd_profile

    def opt_paratone(self, rI_input):
        """ 
        Remove paratone signature from every profile in input intensity matrix. Note that input
        intensity matrix, rI_mtx, should have shape (n_images + 1, n_bins), where the first row
        corresponds to the magnitude of the scattering vector, |S|.
        """

        # fit scale and offset parameters for each image
        param_popt = np.zeros((rI_input.shape[0] - 1, 2))
        rI_output = rI_input.copy()
        
        for num in range(1, rI_input.shape[0]):
            fo, fs, bgd_profile = self.opt_ind_paratone(rI_input[0].copy(), rI_input[num].copy())
            param_popt[num - 1] = np.array([fo, fs])
            rI_output[num] -= bgd_profile

        self.params['paratone'] = param_popt

        return rI_output

    def opt_pairwise(self, key, rI_input, init, lbound, ubound):
        """ 
        Optimize all profiles simultaneously with subtraction of a modified scattering profile given;
        the only free parameter is a scale factor (bounded by lbound and ubound) for each profile. The
        input key determines whether paratone or water is subtracted, and the first column of rI_input
        is assumed to correspond to |S| at which subsequent rows of I were computed.
        """
        
        def error(args):
            ps_scales = np.array(args)
            b_mtx = ps_interp*ps_scales[:,np.newaxis]
            residual = np.sum(np.abs(np.triu(scipy.spatial.distance.cdist(rI_input[1:] - b_mtx, rI_input[1:] - b_mtx))))
            return residual

        x0 = tuple(init)
        ps_interp = np.interp(rI_input[0].copy(), self.stols[key][:,0], self.stols[key][:,1])
        bounds = list(itertools.repeat((lbound, ubound), rI_input.shape[0] - 1))

        res = scipy.optimize.minimize(error, x0, method ='L-BFGS-B', bounds=tuple(bounds))
        if res.success is False:
            print "optimization failed (likely due to non-convergence)"

        if self.params.has_key(key):
            print "Warning: overwriting pre-existing key %s" %key
        self.params[key] = res.x

        rI_output = rI_input.copy()
        rI_output[1:] -= ps_interp*res.x[:,np.newaxis]
        
        return rI_output

    def run_pca(self, rI_input):
        """ 
        Perform principal component analysis. Input: matrix of image profiles aligned in q;
        outputs: sorted eigenvalues, sorted eigenvectors, and projection of the eigenvectors on
        the mean-subtracted data (which can be used to generate the background matrix). 
        """

        trA = rI_input[1:].T.copy()
        msA = trA - trA.mean(axis=1, keepdims=True)
        covA = np.cov(msA)

        evals, evecs = np.linalg.eig(covA)
        sort_idx = np.argsort(evals)[::-1]
        sort_evals, sort_evecs = evals[sort_idx], evecs[:,sort_idx]

        proj_data = np.dot(np.real(sort_evecs.transpose()), msA)

        return sort_evals, sort_evecs, proj_data

    def pca_params(self, rI_input, neig, evals, evecs, proj_data):
        """ 
        Generate a background matrix from first neig eigenvectors of PCA results. For params
        dict, first row is q and subsequent nth row is background to substract from (n-1)th image.
        """

        pca_bgd = np.zeros(rI_input.shape)
        
        for i in range(neig):
            nscale = np.array(evals/evals[0])[i] # scaling by eigenvalue
            nsproj = nscale*proj_data[i]

            for profile in range(rI_input.shape[0] - 1):
                pca_bgd[profile + 1] += evecs[:,i]*nsproj[profile]

        pca_bgd[0] = rI_input[0]
        self.params['pca_bgd'] = pca_bgd

        rI_output = rI_input.copy()
        rI_output[1:] -= pca_bgd[1:]
        
        return rI_output
        
