import cPickle as pickle
import matplotlib, sys, os.path
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
from matplotlib import gridspec
import numpy as np


class TestIndexer:

    """ 
    Tests of the correction factors calculated by the Indexer class. Note that tests
    of accurate indexing are performed by the Indexer class itself.

    """

    def setup(self):

        import Indexer

        self.system = pickle.load(open("reference/system.pickle", "rb"))
        self.image_num, self.batch = 1, 0
        indexer_obj = Indexer.Indexer(self.system)
        indexer_obj.index(1)
        self.system = indexer_obj.system

        # computing scattering vectors used for polarization and parallax tests
        mg = np.mgrid[0:self.system['shape'][0]-1:1j*self.system['shape'][0],
                      0:self.system['shape'][1]-1:1j*self.system['shape'][1]]
        xyz = np.outer(mg[0].flatten(), self.system['s']) + np.outer(mg[1].flatten(), self.system['f'])
        xyz += self.system['p'][self.batch]

        norms = np.linalg.norm(xyz, axis=1)
        self.s1 = np.divide(xyz.T, norms).T


    def test_solidangle(self):

        """
        Comparing implementation of Thor and Indexer for the relative (not absolute)
        solid angle subtended by detector pixels.

        """
        print 'testing solid angle normalization...'

        import thor
        from thor.misc_ext import SolidAngle

        # reference calculation uses Thor's rigorous approach
        dtc = thor.load("reference/11-1_pilatus6M.dtc")
        constant = np.linalg.norm(self.system['f'])*np.linalg.norm(self.system['s'])/\
            np.square(self.system['p'][self.batch][-1])
        beam = self.system['beam'][self.batch]/np.linalg.norm(self.system['beam'][self.batch])

        dtc.beam_vector, dtc._basis_grid._ps = beam, [np.array(self.system['p'][self.batch])]
        solidangle_obj = SolidAngle(dtc)
        solidangle_ref = solidangle_obj.__call__(np.ones(dtc.num_pixels))/constant

        # test calculation from Indexer class
        solidangle_tst = self.system['solid_angle']

        # compare the results: expectation is <7% difference
        frac_diff = np.abs(solidangle_ref - solidangle_tst)/solidangle_tst
        print "Avg. fractional difference between Indexer and Thor calculations: %f " %(np.mean(frac_diff))
        print "Max. fractional difference between Indexer and Thor calculations: %f " %(np.max(frac_diff))

        # plot reference and test
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        
        ax1.hist((solidangle_ref - solidangle_tst)/solidangle_tst)
        ax2.imshow(solidangle_ref.reshape(self.system['shape']), vmax = 1)
        im = ax3.imshow(solidangle_tst.reshape(self.system['shape']), vmax = 1)
        cbar_ax = f.add_axes([0.93, 0.18, 0.02, 0.65])
        f.colorbar(im, cax=cbar_ax)

        ax1.set_title("Fractional difference")
        ax2.set_title("Ref. implementation")
        ax3.set_title("Test implementation")

        f.savefig("./checks/solidangle.png", dpi=300, bbox_inches='tight')

        return

    
    def test_polarization(self):

        """ 
        Comparing implementation of Indexer class (based on Thor), DIALS, and Holton for
        calculation of the polarization correction factor. DIALS approach is coded here:
        https://github.com/dials/dials/blob/2390ee95b7949dea224c11af83d82e06a325e349\
        /algorithms/integration/corrections.h; Thor reference is: Hura et al. J Chem 
        Phys. 113, 9140 (2000); Holton reference is: Holton and Frankel. Acta Cryst D. 
        66, 393 (2010).

        """
        print "testing polarization correction..."

        # reference calculation modeled from DIALS approach 
        pol_normal = np.array([0.0, 1.0, 0.0]) # polarization plane normal

        P1 = np.inner(pol_normal, self.s1.copy())
        P2 = (1.0-2.0*self.system['pf'])*(1.0-np.square(P1))
        P3 = np.dot(self.s1.copy(), self.system['beam'][self.batch])/np.linalg.norm(self.system['beam'][self.batch])
        P4 = self.system['pf']*(1.0+np.square(P3))
        polarization_ref0 = P2+P4

        # second reference calculation based on Holton et al.
        beam = self.system['beam'][self.batch]/np.linalg.norm(self.system['beam'][self.batch])
        S = (1.0/self.system['wavelength'])*(self.s1.copy() - beam)
        S_mags = np.linalg.norm(S.copy(), axis=1)
        thetas = np.arcsin(0.5 * self.system['wavelength'] * S_mags)
        alpha = np.arctan2(self.s1.copy()[:,1] - beam[1], self.s1.copy()[:,0] - beam[0]) # 0 deg. corresponds to 3 o'clock
        
        polarization_ref1 = 0.5*(1.0 + np.square(np.cos(2.0*thetas)) - self.system['pf']*np.cos(2.0*alpha)*np.square(np.sin(2.0*thetas)))

        # test calculation from Indexer class
        polarization_tst = self.system['polarization']
        
        # compare the results        
        frac_diff = (polarization_ref0 - polarization_tst)/polarization_tst
        print "Max. fractional difference to DIALS reference: %f" %(np.max(np.abs(frac_diff)))
        print "Avg. fractional difference to Holton reference: %f" %(np.mean(np.abs(frac_diff)))

        # plot reference and test
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))

        ax1.hist((polarization_ref0-polarization_tst)/polarization_tst)
        ax2.imshow(polarization_ref0.reshape(self.system['shape']), vmin=0.4, vmax=1)
        ax3.imshow(polarization_ref1.reshape(self.system['shape']), vmin=0.4, vmax=1)
        im = ax4.imshow(polarization_tst.reshape(self.system['shape']), vmin=0.4, vmax=1)
        cbar_ax = f.add_axes([0.92, 0.18, 0.015, 0.65])
        f.colorbar(im, cax=cbar_ax)

        ax1.set_title("Fractional Diff. (DIALS, Test)")
        ax2.set_title("DIALS ref. implementation")
        ax3.set_title("Holton ref. implementation")
        ax4.set_title("Test implementation")

        f.savefig("./checks/polarization.png", dpi=300, bbox_inches='tight')
        
        return
