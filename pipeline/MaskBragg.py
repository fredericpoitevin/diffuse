import itertools, math, time, os.path
import cPickle as pickle
import numpy as np
from mdtraj import io
import map_utils

class PredictBragg:

    """ 
    Class that predicts centers and shapes of Bragg reflections. This calculation is performed per batch, 
    i.e. shapes are predcited for all of the Miller indices whose centers are predicted to fall within 
    that batch, but the reflections can extend beyond batch bounds, so the final output per batch is an 
    .h5 file with one key of shape (flattened detector) per image in the oscillation range. Pixels predicted 
    to be spanned by Bragg reflections are marked with a value of 1; Bragg centers are marked by a value of 10.    
    """

    def __init__(self, system, nbatch, sys_absences):
        """
        Initialize class. Required inputs are system.pickle, XDS batch number, and a 3-digit 
        string corresponding to reflection conditions along the H,K,L axes, e.g. '222' for 
        P212121, '002' for P6322, and '000' for no screw axes.
        """
        self.system = system
        self.nbatch = nbatch
        self.sys_absences = sys_absences 
        self.predicted = self.predict_s1()

    def _filter_hkl(self, hkl):
        """
        Minimally remove the origin Miller index (0,0,0) and Miller indices that are of higher
        resolution than the map resolution - 0.2 from the grid of hkl vectors. Optionally remove
        Miller indices that will be extinguished due to screw axes along each unit cell direction
        if self.sys_absences is not '000'.
        """

        # dictionary for storing invalid indices
        invalid = dict()
        
        # remove systematic absences resulting from 21 screw axes along each axis
        if int(self.sys_absences[0]) != 0:
            invalid['h_idx'] = np.where((hkl[:,0] % int(self.sys_absences[0]) != 0) & (hkl[:,1] == 0) & (hkl[:,2] == 0))[0]
        if int(self.sys_absences[1]) != 0:
            invalid['k_idx'] = np.where((hkl[:,0] == 0) & (hkl[:,1] % int(self.sys_absences[1]) != 0) & (hkl[:,2] == 0))[0]
        if int(self.sys_absences[2]) != 0:
            invalid['l_idx'] = np.where((hkl[:,0] == 0) & (hkl[:,1] == 0) & (hkl[:,2] % int(self.sys_absences[2]) != 0))[0]
            
        # remove origin (0,0,0) since we never see this
        invalid['center'] = np.where((hkl[:,0]==0) & (hkl[:,1]==0) & (hkl[:,2]==0))[0]

        # remove Millers beyond resolution cut-off
        A_inv = np.linalg.inv(self.system['A_batch'][self.nbatch])
        qvecs = np.inner(A_inv, hkl).T
        qmags = np.linalg.norm(qvecs, axis=1)
        qmags[qmags==0] = 1e-5 # set origin miller arbitrarily small to avoid errors
        invalid['res'] = np.where(1.0/qmags < (self.system['d'] - 0.2))[0]

        invalid_idx = np.concatenate([invalid[key] for key in invalid.keys()])        
        return np.delete(hkl, invalid_idx, axis=0)

    def _predict_phi(self, bins):
        """ 
        Compute phi angles at which Miller indices will be observed, based on derivation in
        Kabsch, W. Acta Cryst, D66. 2010. Similar to DIALS implementation:
        https://github.com/dials/dials/blob/2390ee95b7949dea224c11af83d82e06a325e349/\
        algorithms/spot_prediction/rotation_angles.h
        """

        hkl = np.array(list(itertools.product(bins['h'], bins['k'], bins['l'])))
        hkl = self._filter_hkl(hkl)
        hkl_phi = None

        for batch in range(self.system['n_batch']):
            A_inv = np.linalg.inv(self.system['A_batch'][batch])
            s0 = self.system['beam'][batch]
            mask = np.zeros(len(hkl), dtype = bool)

            # compute goniometer basis set
            m2 = self.system['rot_axis']
            m1 = np.cross(m2, s0)/np.linalg.norm(np.cross(m2, s0))
            m3 = np.cross(m1, m2)/np.linalg.norm(np.cross(m1, m2))

            s0_d_m2 = np.dot(s0, m2)
            s0_d_m3 = np.dot(s0, m3)

            # compute unrotated S vectors, pstar0
            pstar0 = np.inner(A_inv, hkl).T
            pstar0_len_sq = (np.linalg.norm(pstar0, axis=1))**2
            blind = np.where(pstar0_len_sq > 4*(np.linalg.norm(s0))**2)[0]
            mask[blind] = True

            # compute various dot products
            pstar0_d_m1 = np.dot(pstar0, m1)
            pstar0_d_m2 = np.dot(pstar0, m2)
            pstar0_d_m3 = np.dot(pstar0, m3)

            pstar_d_m3 = (-(0.5*pstar0_len_sq) - np.dot(pstar0_d_m2, s0_d_m2))/s0_d_m3
            rho_sq = pstar0_len_sq - pstar0_d_m2**2
            blind = np.where(rho_sq < pstar_d_m3**2)[0]
            mask[blind] = True
            pstar_d_m1 = np.sqrt(rho_sq - pstar_d_m3**2)

            # compute oscillation angles at which hkl indices will be observed
            cosphi1 = pstar_d_m1*pstar0_d_m1 + pstar_d_m3*pstar0_d_m3
            cosphi2 = -1*pstar_d_m1*pstar0_d_m1 + pstar_d_m3*pstar0_d_m3
            sinphi1 = pstar_d_m1*pstar0_d_m3 - pstar_d_m3*pstar0_d_m1
            sinphi2 = -1*pstar_d_m1*pstar0_d_m3 - pstar_d_m3*pstar0_d_m1

            phi1 = np.rad2deg(np.arctan2(sinphi1, cosphi1))
            phi2 = np.rad2deg(np.arctan2(sinphi2, cosphi2))
            phi1[mask] = 1000 # arbritrary number beyond what will be covered by experiment
            phi2[mask] = 1000

            # wrap phi beyond 180, whose Bragg may be partially observed on final images
            wrap_idx1 = np.where(phi1<-170)[0]
            wrap_idx2 = np.where(phi2<-170)[0]
            phi1[wrap_idx1] += 360
            phi2[wrap_idx2] += 360

            # determine which Miller indices will be observed in this batch
            idx = np.where(np.array(self.system['img2batch'].values()) == batch)[0]
            bps, l_idx, u_idx = len(idx)*self.system['rot_phi'], idx[0] + 1, idx[-1] + 1
            lower = l_idx*self.system['rot_phi'] - 0.5*self.system['rot_phi'] + self.system['start_phi']
            upper = u_idx*self.system['rot_phi'] + 0.5*self.system['rot_phi'] + self.system['start_phi']
            
            if batch == 0: 
                lower -= bps
            if batch == self.system['n_batch'] - 1:
                upper += bps

            idx1 = np.where((phi1 >= lower) & (phi1 < upper))[0]
            idx2 = np.where((phi2 >= lower) & (phi2 < upper))[0]
            hkl_phi1 = np.column_stack((hkl[idx1], phi1[idx1]))
            hkl_phi2 = np.column_stack((hkl[idx2], phi2[idx2]))

            if hkl_phi is None:
                hkl_phi = np.row_stack((hkl_phi1, hkl_phi2))
            else:
                hkl_phi = np.row_stack((hkl_phi, hkl_phi1, hkl_phi2))

        return hkl_phi

    def predict_s1(self):
        """ Compute scattering vectors for predicted Millers. Returns array in shape (n_millers, 7);
        columns are h, k, l, phi (degrees), s1_x, s1_y, s1_z. """

        bins = map_utils.determine_map_bins(self.system['cell'], self.system['space_group'], self.system['d'] - 0.2, 1.0)
        hkl_phi = self._predict_phi(bins)
        hkl_s1_phi = np.zeros((hkl_phi.shape[0], 7))

        # determine Millers whose centroids fall in batch bounds
        idx = np.where(np.array(self.system['img2batch'].values()) == self.nbatch)[0]
        bps, l_idx, u_idx = len(idx)*self.system['rot_phi'], idx[0] + 1, idx[-1] + 1
        lower = l_idx*self.system['rot_phi'] - 0.5*self.system['rot_phi'] + self.system['start_phi']
        upper = u_idx*self.system['rot_phi'] + 0.5*self.system['rot_phi'] + self.system['start_phi']
        
        if self.nbatch == 0:
            lower -= bps
        if self.nbatch == self.system['n_batch'] - 1:
            upper += bps

        phi_idx = np.where((hkl_phi[:,3] >= lower) & (hkl_phi[:,3] < upper))[0]
        hkl_phi_batch = hkl_phi[phi_idx]
        hkl_s1_phi = np.zeros((hkl_phi_batch.shape[0], 7))
        hkl_s1_phi[:,:4] = hkl_phi_batch

        # compute scattering vectors for miller indices predicted for batch
        A_inv = np.linalg.inv(self.system['A_batch'][self.nbatch])
        pstar0_batch = np.inner(A_inv, hkl_phi_batch[:,:3]).T

        for idx in range(hkl_phi_batch.shape[0]):
            rot_mat = map_utils.rotation_matrix(self.system['rot_axis'], np.deg2rad(hkl_phi_batch[idx, 3]))
            s1 = np.dot(rot_mat, pstar0_batch[idx]) + self.system['beam'][self.nbatch]
            hkl_s1_phi[idx, 4:] = s1

        return hkl_s1_phi

    def _px_to_sdash(self):
        """ 
        Returns wavelength-normalized scattering vectors corresponding to each pixel. 
        """
        # convert pixel coordinates into meters in laboratory frame
        mg = np.mgrid[0:self.system['shape'][0]-1:1j*self.system['shape'][0],
                      0:self.system['shape'][1]-1:1j*self.system['shape'][1]]
        xyz = np.outer(mg[0].flatten(), self.system['s']) + np.outer(mg[1].flatten(), self.system['f'])
        xyz += self.system['p'][self.nbatch]

        # calculate scattering vectors and normalize by wavelength
        norms = np.linalg.norm(xyz, axis=1)
        s1_dtc = np.divide(xyz.T, norms).T
        sdash_batch = s1_dtc/self.system['wavelength']

        return sdash_batch

    def _save_masks(self, rs_mask):
        """ 
        Save mask in h5 format, with each key corresponding to a separate image. Currently
        stored in a temporary directory since arrays for the same image from different batches
        must still be compiled.
        """

        output_dir = self.system['map_path'] + "temp/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, item in enumerate(rs_mask):
            name = 'arr_%s' % i
            data = {name : item}
            io.saveh(output_dir + "masks_b%s.h5" %self.nbatch, **data)

        return
    
    def pred_pixels(self, sigma_n):
        """ 
        Predict extent/shape of Bragg peaks, basd on approach described by Kabsch, W. Acta Cryst,
        D66. 2010. Returns an array with shape of (frames, flattened detector), where 1 and 10 indicate
        pixels predicted to be covered by a reflection and a reflection center, respectively. 
        """

        # set tolerances; sigma_d and sigma_m from SYSTEM
        delta_d, delta_m = self.system['sigma_d']*sigma_n, self.system['sigma_m']*sigma_n
        s1_batch = self.predicted[:, -3:]
        s1_batch_norm = np.linalg.norm(s1_batch, axis=1)

        # compute basis vectors [e1, e2, e3] for batch scattering vectors
        e1 = np.cross(s1_batch, self.system['beam'][self.nbatch])
        e1 = np.divide(e1.T, np.linalg.norm(e1, axis=1)).T
        e2 = np.cross(s1_batch, e1)
        e2 = np.divide(e2.T, np.linalg.norm(e2, axis=1)).T
        e3 = s1_batch + self.system['beam'][self.nbatch]
        e3 = np.divide(e3.T, np.linalg.norm(e3, axis=1)).T

        # detector-relevant items: setting up masks and calculating associated s1_vecs
        num_images = len(self.system['img2batch'])
        dtc_size = self.system['shape'][0]*self.system['shape'][1]
        rs_mask = np.zeros((num_images, dtc_size), dtype=np.uint8)
        sdash = self._px_to_sdash()

        # compute image span (eps3 in Kabsch nomenclature) across which Millers will be observed
        zeta = np.tile(np.inner(self.system['rot_axis'], e1), (num_images, 1)).T
        end_phi = num_images*self.system['rot_phi'] + self.system['rot_phi'] + self.system['start_phi']
        dpsi = np.tile(np.arange(self.system['rot_phi'] + self.system['start_phi'], end_phi, self.system['rot_phi']), \
                           (s1_batch.shape[0], 1)) - np.tile(self.predicted[:,3], (num_images, 1)).T
        frames = np.where(np.abs(zeta * dpsi) < delta_m)

        # predict detector coordinates that each miller will span
        counter = 0
        stime = time.time()
        for idx in np.unique(frames[0]):
            
            eps1 = 180*np.inner(e1[idx], sdash - s1_batch[idx])/(np.pi*s1_batch_norm[idx]) 
            eps2 = 180*np.inner(e2[idx], sdash - s1_batch[idx])/(np.pi*s1_batch_norm[idx])
            xy_obs = np.where((np.abs(eps1) < delta_d) & (np.abs(eps2) < delta_d))[0]

            if len(xy_obs) > 0:
                images = np.array(frames[1][np.where(frames[0]==idx)[0]])
                rs_mask[images[0]:images[-1]+1, xy_obs] = 1

                xcen, ycen = int(np.median(xy_obs/self.system['shape'][1])), int(np.median(xy_obs % self.system['shape'][1]))
                flattened = xcen*self.system['shape'][1] + ycen
                rs_mask[images[0]:images[-1]+1, flattened] = 10 

                if counter < 200:
                    print idx, ' '.join(map(str, self.predicted[idx, :4])), xcen, ycen, len(images), len(xy_obs)
                    counter += 1
                else:
                    print "elapsed time is %f for 200 of %i reflections" %((time.time() - stime)/60.0, len(np.unique(frames[0])))

        self._save_masks(rs_mask)
                
        return

    
class ApplyMask:

    """
    Class for masking Bragg peaks; the masked pixel value is set to -1. Pixels predicted to be spanned by
    a Bragg peak are excluded if intensity exceeds a certain threshold above the intensity of nearby pixels
    not predicted to be spanned by the peak. Currently supported thresholds are based on percentile or sigma
    standard deviations above the mean.
    """

    def __init__(self, system, indexed, mask_path, num, length):
        self.num = num # 1-indexed, consistent with image nomenclature
        self.extent = length/2 # length is number of pixels per block side
        self.img_mask = io.loadh(mask_path, "arr_%i" % (num - system['start_image']))
        self.system = system
        self.indexed = indexed

    def _set_up(self):
        """
        Gather square blocks of pixels whose centers coincide with the predicted Bragg peak center
        and whose dimensions span (-length/2, +length/2) pixels along the X and Y dimensions of the 
        detector image. Return all pixels within this block and the selection of pixels in each
        block not predicted to be spanned by a Bragg peak.
        """

        # determine index of upper left hand corner of each pixel block centered at a reflection
        centers = np.array(np.where(self.img_mask.reshape(self.system['shape'])>=10))
        corners = centers - self.extent
        fcorners = corners[0]*self.system['shape'][1] + corners[1]
        
        # compute indices (for equivalent flattened array) for these pixel windows
        n_centers = len(np.where(self.img_mask >= 10)[0])
        dtc_idx = np.zeros((n_centers, np.square(2*self.extent)), dtype=int)
        for idx in range(n_centers):
            dtc_idx[idx] = np.array([range(fcorners[idx]+self.system['shape'][1]*i,
                                           fcorners[idx]+self.system['shape'][1]*i+self.extent*2) for i in range(self.extent*2)]).flatten()

        # retrieve intensity and mask values for pixels in windows
        fimg = self.indexed[:,3].copy().flatten()
        window_I = fimg[dtc_idx]
        window_mask = self.img_mask[dtc_idx]

        # retrieve array for valid (non-Bragg, positive) pixels in each block
        array_mask = np.zeros((dtc_idx.shape), dtype=bool)
        array_mask[np.where((window_I <= 0) | (window_mask > 0))] = True
        masked_I = np.ma.array(window_I, mask = array_mask)

        return masked_I, window_I, window_mask, dtc_idx

    def ms_threshold(self, sigma):
        """ 
        Mask pixels in indexed image, specifically those 1. predicted to be spanned by a Bragg peak
        and 2. whose intensities exceed the mean + sigma * std. dev. of neighboring pixels. Assign
        masked pixels a value of -1 in returned intensity array.
        """

        masked_I, window_I, window_mask, dtc_idx = self._set_up()
        means = np.ma.mean(masked_I, axis=1).data
        stdev = np.ma.std(masked_I, axis=1).data
        
        threshold = np.repeat(means + sigma*stdev, np.square(2*self.extent)).reshape(masked_I.shape)
        window_I[(window_I > threshold) & (window_mask > 0)] = -1

        masked_img = self.indexed[:,3].copy().flatten()
        masked_img[dtc_idx] = window_I
        
        return masked_img

    def p_threshold(self, percentile):
        """
        Mask pixels in indexed image, specifically those 1. predicted to be spanned by a Bragg peak
        and 2. whose intensities fall in the input intensity percentile of neighboring pixels. Assign
        masked pixels a value of -1 in returned intensity array.
        """

        masked_I, window_I, window_mask, dtc_idx = self._set_up()
        r = np.nanpercentile(masked_I.filled(np.nan), percentile, axis=1)

        threshold = np.repeat(r, np.square(2*self.extent)).reshape(masked_I.shape)
        window_I[(window_I > threshold) & (window_mask > 0)] = -1

        if len(np.where(np.isnan(r))[0])>0: # deal with blocks for which all pixels are masked
            window_I[np.where(np.isnan(r))] = -1
        
        masked_img = self.indexed[:,3].copy().flatten()
        masked_img[dtc_idx] = window_I

        return masked_img

    def remove_outliers(self, sigma, n_bins, imgI):
        """
        Remove pixels whose intensities exceed the (median + sigma * median absolute deviation) 
        intensity of its resolution shell. Number of shells and sigma are determined by inputs. 
        These pixels are assigned a value of -1. 
        """

        # bin intensities into n_bins shells of equal spacing in inverse Angstrom
        rot_svecs = np.inner(np.linalg.inv(self.system['A_batch'][self.system['img2batch'][self.num]]), self.indexed[:,:3]).T
        s_mags = np.linalg.norm(rot_svecs, axis=1)

        x, y = s_mags, imgI.copy()
        if ('s_mask' in self.system.keys()) & (self.num in self.system['s_mask_img']):
            print "applying special mask"
            y[~self.system['s_mask'].flatten()] = 0

        n_per_shell, shells = np.histogram(x[y > 0], bins = n_bins)
        shells = np.concatenate((shells, [shells[-1] + (shells[-1] - shells[-2])]))

        dx = np.digitize(x, shells)
        bin_sel = np.array([len(y[(dx == i) & (y > 0)]) for i in range(np.min(dx), np.max(dx) + 1)])
        bin_idx = np.where(bin_sel > 2)[0] + np.min(dx) # avoid empty bins

        # compute median and median absolute deviation intensity per shell; avoid sensitive mean
        ymedians = np.array([np.median(y[(dx == i) & (y > 0)]) for i in bin_idx])
        ymad = np.array([1.4826*np.median(np.abs(y[(dx == bin_idx[i]) & (y > 0)] - ymedians[i])) for i in range(len(bin_idx))])
        threshold = ymedians + sigma*ymad

        # interpolate threshold and remove pixels that fall above
        s_medians = np.array([np.median(s_mags[dx == bin_idx[i]]) for i in range(len(bin_idx))])
        threshold_int = np.interp(s_mags, s_medians, threshold)
        imgI[np.where(imgI > threshold_int)] = -1

        # for pixels in the 'special mask' area, set a lower threshold and remove as necessary
        if ('s_mask' in self.system.keys()) & (self.num in self.system['s_mask_img']):
            lthreshold = ymedians + 2.5*ymad
            lthreshold_int = np.interp(s_mags, s_medians, lthreshold)
            imgI[np.where((imgI > lthreshold_int) & (self.system['s_mask'].flatten() == False))] = -1

        return imgI
