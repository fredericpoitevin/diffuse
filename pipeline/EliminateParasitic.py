from collections import OrderedDict
import scipy.optimize, scipy.spatial
import cPickle as pickle
import numpy as np
import map_utils, os.path
import glob, sys, time, itertools

class GenerateBackground:

    """
    Class with functions useful for generating an approach for eliminating parasitic scattering,
    particularly from paratone or water. Sub-strategies include principal component analysis and
    subtraction of fitted water or paratone scattering profiles, with the goal of minimizing the 
    difference between different images' radial intensity profiles. The stored params dictionary
    contains parameters that can be used with the ApplyBackground class for processing images.
    """
    
    def __init__(self, system):
        self.stols = map_utils.process_stol()
        self.system = system
        self.params = OrderedDict()

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
        

class ApplyBackground:

    """
    Class for applying the background subtraction strategy devised/specified by GenerateBackground.
    Supported functions are subtraction of scaled water and paratone scattering profiles, in addition
    to subtraction of the background profile calculated from principal component analysis. The order
    specified by the GenerateBackground output dictionary is preserved in processing images.
    """

    def __init__(self, system, params):

        self.stols = map_utils.process_stol()
        self.system = system
        self.params = params

    def compute_smags(self, indexed, num):
        """
        Compute the magnitude of the scattering/hkl vector for every datapoint in indexed.
        Input num corresponds to 1-indexed image number; if -1, then 'cell' parameters are
        used to compute crystal setting matrix.
        """
        
        if num == -1:
            A_inv = np.linalg.inv(np.diag(self.system['cell'][:3]))
        else:
            A_inv = np.linalg.inv(self.system['A_batch'][(num - 1)/self.system['batch_size']])

        s_vecs = np.inner(A_inv, indexed[:,:3]).T

        return np.linalg.norm(s_vecs, axis=1)

    def scale_water(self, s_mags, num):
        """
        Return scaled water profile to be subtracted. 
        """

        assert self.params.has_key('water')

        w_scale = self.params['water'][num - 1]
        return w_scale*np.interp(s_mags, self.stols['water'][:,0], self.stols['water'][:,1])

    def scale_paratone(self, s_mags, num):
        """
        Return scaled paratone profile to be subtracted. If paratone parameter array is 2D,
        assume first and second column correspond to q-offset and scale, respectively; if
        1D, assume that array corresponds to scale factors only.
        """

        assert self.params.has_key('paratone')
        img_params = self.params['paratone'][num - 1]

        if img_params.shape[0] == 2:
            p_offset, p_scale = img_params[0], img_params[1]
        else:
            p_offset, p_scale = 0, img_params

        return p_scale*np.interp(s_mags, self.stols['paratone'][:,0] - p_offset, self.stols['paratone'][:,1])

    def interp_pca(self, s_mags, num):
        """
        Interpolate the background profile computed from PCA.
        """
        
        assert self.params.has_key('pca_bgd')

        pca_S, pca_Ibgd = self.params['pca_bgd'][0], self.params['pca_bgd'][num]
        return np.interp(s_mags, pca_S, pca_Ibgd)
