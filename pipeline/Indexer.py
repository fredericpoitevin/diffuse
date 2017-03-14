import numpy as np
import math, map_utils

class Indexer:

    """ 
    Class for indexing diffraction images, with geometrical corrections to intensities
    optionally performed. Diffraction images (in .npy format) are flattened per Python's 
    convention, and each pixel is indexed based on the parameters in the input system 
    dictionary. Per image output is a np.array of shape (n_pixels, 4), where the four 
    columns are (h,k,l,I).

    Correct indexing is optionally checked against the pixel coordinates/Bragg positions
    in an XDS INTEGRATE.HKL file. The following geometric corrections to intensities can
    be performed: parallax correction, solid angle normalization, polarization correction.

    The coordinate system is defined to mirror the conventions of Thor and XDS, with the 
    X and Y axes respectively defining the fast and slow scan directions of the detector. 
    For consistency with Python's row-major ordering, pixel coordinates are (slow, fast).
    The positive Z direction points from the crystal towards the detector.

    """

    def __init__(self, system):
        """ 
        Initialize class. Required input is dictionary defining system geometry. 

        """
    
        self.system = system
        if 'xds_path' in system.keys(): # store reference hkl if available
            self.ref_hkl = np.loadtxt(system['xds_path'] + 'INTEGRATE.HKL', \
                                          comments = '!', usecols = [0, 1, 2, 5, 6, 7])
        self.hklI = np.zeros((system['shape'][0]*system['shape'][1], 4))

    def _parallax_correction(self, image_num, s1):
        """ 
        Compute the parallax correction, which accounts for the shift in the position 
        between where photons are detected and recorded due to non-negligible detector
        thickness. Below code follows the framework outlined by cxtbx/dxtbx:
        https://github.com/cctbx/cctbx_project/blob/\
        b0460954eac07a3a3639dbe7efd942c21e12ebc1/dxtbx/model/parallax_correction.h
        
        """

        u_fast = self.system['f']/np.linalg.norm(self.system['f'])
        u_slow = self.system['s']/np.linalg.norm(self.system['s'])

        normal = np.cross(u_fast, u_slow)
        dist = np.dot(self.system['p'][(image_num - 1)/self.system['batch_size']], normal)
        if dist < 0:
            normal = -1*normal

        cos_t = np.inner(s1.copy(), normal)
        attenuation = 1.0/self.system['mu'] - (self.system['t0']/cos_t + 1.0/self.system['mu'])\
            *np.exp(-self.system['mu']*self.system['t0']/cos_t)

        xcorr = attenuation*np.inner(s1.copy(), u_fast)
        ycorr = attenuation*np.inner(s1.copy(), u_slow)

        corrections = np.zeros(s1.shape)
        corrections[:,0], corrections[:,1] = xcorr, ycorr

        return corrections

    def _validate(self, hkl, image_num):
        """ 
        Check accuracy of indexing against the subset of Bragg peaks predicted 
        to be observed in this image by INTEGRATE.HKL. Recall that slicing is
        (y, x) in this coordinate system convention.

        """

        millers_reshape = hkl.reshape(self.system['shape'][0], self.system['shape'][1], 3)
        millers_xds, xyzcal = self.ref_hkl.T[:3], self.ref_hkl.T[3:5]
        inds = np.where(self.ref_hkl.T[5] == image_num)[0]

        centroids = np.around(xyzcal.T[inds]).astype(int)
        deltas = millers_reshape[centroids.T[1], centroids.T[0]] - millers_xds.T[inds]
        avg_delta = np.mean(np.abs(deltas.T), axis = 1)

        print "Average difference in h: %f" %(avg_delta[0])
        print "Average difference in k: %f" %(avg_delta[1])
        print "Average difference in l: %f" %(avg_delta[2])

        return avg_delta

    def index(self, image_num):
        """ 
        Index each detector pixel, from detector position/rotation angle (X,Y,phi) to (h,k,l). 
        Input: image number, note that these are integral and  1-indexed for a rotation series.
        Output: flattened np.array whose columns are (h,k,l).

        """

        # convert pixel coordinates into meters in laboratory frame
        mg = np.mgrid[0:self.system['shape'][0]-1:1j*self.system['shape'][0],
                      0:self.system['shape'][1]-1:1j*self.system['shape'][1]]
        xyz = np.outer(mg[0].flatten(), self.system['s']) + np.outer(mg[1].flatten(), self.system['f'])
        xyz += self.system['p'][(image_num - 1)/self.system['batch_size']]

        norms = np.linalg.norm(xyz, axis=1)
        s1 = np.divide(xyz.T, norms).T

        # perform parallax correction if specified
        if 'parallax' in self.system['corrections']:
            if 'parallax' not in self.system.keys():    
                self.system['parallax'] = self._parallax_correction(image_num, s1.copy())
            xyz -= self.system['parallax']

        # calculate scattering vectors, S, related to the q vector by: q = 2*pi*S
        beam = self.system['beam'][(image_num - 1)/self.system['batch_size']]\
            /np.linalg.norm(self.system['beam'][(image_num - 1)/self.system['batch_size']])
        S = (1.0/self.system['wavelength'])*(s1 - beam)

        # rotate orientation matrix and compute hkl
        rot_mat = map_utils.rotation_matrix(self.system['rot_axis'], \
                                                -1*np.deg2rad(self.system['rot_phi']*(image_num-1)+self.system['rot_phi']))
        rot_cryst = np.dot(self.system['A_batch'][(image_num - 1)/self.system['batch_size']], rot_mat)
        hkl = np.inner(rot_cryst, S).T
        delta = self._validate(hkl.copy(), image_num)

        # compute polarization and solid angle corrections if specified
        if 'polarization' in self.system['corrections']:
            if "polarization" not in self.system.keys():
                self.system['polarization'] = self._polarization_correction(s1, S, beam)

        if 'solid angle' in self.system['corrections']:
            if "solid_angle" not in self.system.keys():
                self.system['solid_angle'] = self._solid_angle_correction(S)

        self.hklI[:,0:3] = hkl
        return delta

    def _polarization_correction(self, s1, S, beam):
        """ 
        Compute array of polarization correction factors with shape (n_pixels). Based on Thor implementation:
        https://github.com/tjlane/thor/blob/9c8ccfff06756ef2f2438574b6df4edc1a9f1816/src/python/xray.py,
        with equation from Hura et al. J Chem Phys. 113, 9140 (2000):
        P_in-plane [ 1 - (sin(phi)*sin(theta))**2 ] + P_out-of-plane [ 1 - (cos(phi)sin(theta))**2 ],
        where theta is the diffraction angle and phi is an angle in the plane of the detector.
        """

        S_mags = np.linalg.norm(S.copy(), axis=1)
        thetas = np.arcsin(0.5 * self.system['wavelength'] * S_mags)
        sin_thetas = np.sin(2.0 * thetas)

        phis = np.arctan2(s1[:,0] - beam[0], s1[:,1] - beam[1]) # differs from Thor: phi=0 aligned with y
        phis[phis < 0.0] += 2 * np.pi

        pol_correction = (1.0 - self.system['pf'])*(1.0 - np.square(sin_thetas*np.cos(phis))) + \
                     self.system['pf']*(1.0 - np.square(sin_thetas*np.sin(phis)))

        return pol_correction

    def _solid_angle_correction(self, S):
        """ 
        Compute a solid angle correction factor, as measured are proportional to the solid angle 
        subtended by each pixel: solid_angle = (cos(theta))**3. Since our interest is in relative 
        intensities, we do not compute the absolute solid angle: (A/d**2 * solid_angle), where A
        and d are pixel area and minimum distance to detector, respectively. Absolute solid angle
        is computed by Thor and LUNUS (Wall, Methods in Mol Bio, 544: 269-279. 2009.), though the 
        latter subsequently scales intensities by a constant factor.

        """

        S_mags = np.linalg.norm(S.copy(), axis=1)
        thetas = np.arcsin(0.5 * self.system['wavelength'] * S_mags)
        solid_angle = np.cos(2.0*thetas)*np.cos(2.0*thetas)*np.cos(2.0*thetas)

        return solid_angle

    def process_intensities(self, intensities, image_num):
        """ 
        Perform following corrections if specified: masking of untrusted detector region, 
        applying per image scale factor, polarization correction, solid angle normalization. 
        Return corrected, flattened intensity array of shape (n_pixels). 
        """

        # apply mask and corrections factors
        if 'mask' in self.system.keys():
            intensities[ ~self.system['mask'] ] = -1
        intensities = intensities.flatten().astype(float)
        if 'scales' in self.system.keys():
            intensities /= self.system['scales'][image_num - 1]
        if 'solid_angle' in self.system.keys():
            intensities /= self.system['solid_angle']
        if 'polarization' in self.system.keys():
            intensities /= self.system['polarization']

        # append intensity array to hkl
        self.hklI[:,3] = intensities
        return intensities

    def clear_hklI(self):
        """ 
        Clear self.hklI of indexing and intensity data. 
        """

        self.hklI = np.zeros(self.hklI.shape)
        assert np.all(self.hklI == 0)
        
        print "Data cleared"
        return 

    def clear_corrections(self):
        """ 
        Clear self.system of polarization, parallax, and solid angle correction arrays, 
        as these depend on refined parameters and so are calculated per batch. 
        """

        self.system.pop('polarization', None)
        self.system.pop('solid_angle', None)
        self.system.pop('parallax', None)

        print "Corrections cleared"
        return
