import itertools, math, time, os.path
import numpy as np
from mdtraj import io
import map_utils

class PredictBragg:

    """ Class that predicts centers and shapes of Bragg reflections. Generates a mask per batch in shape 
    of (frames, flattened detector) that indicates pixels predicted to be spanned by Bragg peaks with a 
    value of 1. """

    def __init__(self, system, nbatch, sys_absences = False):
        """
        Initialize class. Required inputs are system.pickle and XDS batch number.
        If optional input sys_absences is True, then it's assumed that the space
        group has 21 screw axes along each unit cell axis.

        """
        self.system = system
        self.nbatch = nbatch
        self.sys_absences = sys_absences # currently only 212121 systematic absences are supported
        self.predicted = self.predict_s1()

    def _filter_hkl(self, hkl):
        """
        Minimally remove the origin Miller index (0,0,0) and Miller indices that are of higher
        resolution than the map resolution - 0.2 from the grid of hkl vectors. Optionally remove
        Miller indices that will be extinguished due to 21 screw axes along each unit cell direction
        if self.sys_absences is True.
        """

        # remove systematic absences resulting from 21 screw axes along each axis
        if self.sys_absences is True:
            h_idx = np.where((hkl[:,0]%2!=0) & (hkl[:,1]==0) & (hkl[:,2]==0))[0]
            k_idx = np.where((hkl[:,0]==0) & (hkl[:,1]%2!=0) & (hkl[:,2]==0))[0]
            l_idx = np.where((hkl[:,0]==0) & (hkl[:,1]==0) & (hkl[:,2]%2!=0))[0]

        # remove origin (0,0,0) since we never see this
        center = np.where((hkl[:,0]==0) & (hkl[:,1]==0) & (hkl[:,2]==0))[0]

        # remove Millers beyond resolution cut-off
        A_inv = np.linalg.inv(self.system['A_batch'][self.nbatch])
        qvecs = np.inner(A_inv, hkl).T
        qmags = np.linalg.norm(qvecs, axis=1)
        qmags[qmags==0] = 1e-5 # set origin miller arbitrarily small to avoid errors
        res_invalid = np.where(1.0/qmags < (self.system['d'] - 0.2))[0]

        if self.sys_absences is True:
            invalid_idx = np.concatenate((h_idx, k_idx, l_idx, center, res_invalid))
        else:
            invalid_idx = np.concatenate((center, res_invalid))

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

            # determine which millers will be observed in that batch
            bps = self.system['batch_size']*self.system['rot_phi'] # batch phi span
            lower = bps*batch + self.system['rot_phi']/2.0
            upper = bps*batch + bps + self.system['rot_phi']/2.0
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
        bps = self.system['batch_size']*self.system['rot_phi'] # batch phi span
        lower = bps*self.nbatch + self.system['rot_phi']/2.0
        upper = bps*self.nbatch + bps + self.system['rot_phi']/2.0
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
        num_images = self.system['batch_size']*self.system['n_batch']
        dtc_size = self.system['shape'][0]*self.system['shape'][1]
        rs_mask = np.zeros((num_images, dtc_size), dtype=np.uint8)
        sdash = self._px_to_sdash()

        # compute image span (eps3 in Kabsch nomenclature) across which Millers will be observed
        zeta = np.tile(np.inner(self.system['rot_axis'], e1), (360, 1)).T
        end_phi = self.system['batch_size']*self.system['n_batch']*self.system['rot_phi'] + self.system['rot_phi']
        dpsi = np.tile(np.arange(self.system['rot_phi'], end_phi, self.system['rot_phi']), \
                           (s1_batch.shape[0], 1)) - np.tile(self.predicted[:,3], (num_images, 1)).T
        frames = np.where(np.abs(zeta * dpsi) < delta_m)

        # predict detector coordinates that each miller will span
        for idx in np.unique(frames[0])[:10]:
            
            eps1 = 180*np.inner(e1[idx], sdash - s1_batch[idx])/(np.pi*s1_batch_norm[idx]) 
            eps2 = 180*np.inner(e2[idx], sdash - s1_batch[idx])/(np.pi*s1_batch_norm[idx])
            xy_obs = np.where((np.abs(eps1) < delta_d) & (np.abs(eps2) < delta_d))[0]

            if len(xy_obs) > 0:
                images = np.array(frames[1][np.where(frames[0]==idx)[0]])
                rs_mask[images[0]:images[-1]+1, xy_obs] = 1

                xcen, ycen = int(np.median(xy_obs/self.system['shape'][1])), int(np.median(xy_obs % self.system['shape'][1]))
                flattened = xcen*self.system['shape'][1] + ycen
                rs_mask[images[0]:images[-1]+1, flattened] = 10 

                print idx, ' '.join(map(str, self.predicted[idx, :4])), len(images), len(xy_obs)

        #self._save_masks(rs_mask)
                
        return
