import cPickle as pickle
import matplotlib, sys, os.path
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
from matplotlib import gridspec
from matplotlib.patches import Ellipse
import numpy as np

""" 
Generate a systems dictionary from detector and refined geometry information 
extracted from output of XDS. Input arguments should be absolute paths.

Usage: python generate_system.py [xds directory] [map directory] [image type]

Required files in input path: INTEGRATE.LP, INIT.LP, CORRECT.LP, XYCORR.LP 
Currently supported image types: npy, h5, ccd
Output files: system.pickle, checks/mask.png

"""

def prompt_for_corrections():
    """ Prompt user for whether to include polarization, parallax, and solid
    angle corrections. If any is yes, store in 'corrections' key of system. """
    
    corrections = list()
    if 'y' in raw_input('Include polarization correction? (y/n) : '):
        corrections.append('polarization')
    if 'y' in raw_input('Include solid angle correction? (y/n) : '):
        corrections.append('solid angle')
    if 'y' in raw_input('Include parallax correction? (y/n) : '):
        corrections.append('parallax')

    system['corrections'] = corrections

    return

def extract_geometry(xds_path):
    """ Extract detector and refined geometry parameters from INTEGRATE.LP. """

    filename = xds_path + "INTEGRATE.LP"
    with open(filename, "r") as f:
        
        content = f.readlines()

        # extract pixel size and detector shape
        dtc_size = [s for s in content if "NX" in s][0].split()
        system['shape'] = (int(dtc_size[3]), int(dtc_size[1]))
        system['pixel_size'] = 0.001*np.array([float(dtc_size[7]), float(dtc_size[5])]) # units: m
        
        # extract information about batch size
        system['batch_size'] = int([s for s in content if "NUMBER_OF_IMAGES_IN_CACHE" in s][0].split()[-1]) - 1
        img_range = [s for s in content if "DATA_RANGE" in s][0].split()
        system['start_image'] = int(img_range[-2])
        system['start_phi'] = float([s for s in content if "STARTING_ANGLE" in s][0].split()[1])
        #system['n_batch'] = int(img_range[-1]) / system['batch_size']

        # add image2batch mapping to circumvent problem of missing images
        bounds = [(int(s.split()[-3]), int(s.split()[-1])+1) for s in content if "PROCESSING OF IMAGES" in s]
        system['img2batch'] = dict()
        for nbatch in range(len(bounds)):
            for image in range(bounds[nbatch][0], bounds[nbatch][1]):
                system['img2batch'][image] = nbatch

        # extract per image scale factor
        idx = [i for i,s in enumerate(content) if 'IMAGE IER' in s]
        system['n_batch'] = len(np.array(idx))
        scales = list()
        for start in idx:
            start += 1
            while content[start]!='\n':
                scales.append(float(content[start].split()[2]))
                start += 1
        system['scales'] = np.array(scales)
                
        # extract orientation (A) matrices and wavelength
        A_matrices = [s.strip('\n').split()[5:] for s in content if "COORDINATES OF UNIT CELL" in s]
        A_matrices = np.asarray(A_matrices, dtype=float)
        system['A_batch'] = A_matrices.reshape(system['n_batch'], 3, 3)
        system['wavelength'] = float([s.split()[1] for s in content if "X-RAY_WAVELENGTH" in s][0])

        # extract p, s, f, and origin vectors, with units in meters
        beam = [s.strip('\n').split()[-3:] for s in content if "DIRECT BEAM COORDINATES" in s]
        system['beam'] = np.asarray(beam, dtype=float)

        p_xy = np.asarray([s.strip('\n').strip('\n').split()[-2:] for s in content if "DETECTOR ORIGIN (PIXELS) AT" in s], dtype=float)
        p_z = 0.001*np.asarray([s.strip('\n').split()[-1] for s in content if "CRYSTAL TO DETECTOR DISTANCE (mm)" in s], dtype=float)
        p_xy *= np.tile(-1*system['pixel_size'], system['n_batch']).reshape(p_xy.shape)
        system['p'] = np.vstack((p_xy.T, p_z)).T

        system['f'] = np.array([system['pixel_size'][0], 0, 0])
        system['s'] = np.array([0, system['pixel_size'][1], 0])

        # extract rotation axis -- should use XDS_ASCII.HKL values intead?
        system['rot_axis'] = np.asarray([s.split()[1:] for s in content if "ROTATION_AXIS" in s][0], dtype=float)
        system['rot_phi'] = float([s.split()[1] for s in content if "OSCILLATION_RANGE" in s][0])

        # extract estimated beam divergence and reflecting range e.s.d's (latter is mosaicity)
        idx = [i for i,s in enumerate(content) if 'SUGGESTED VALUES FOR INPUT PARAMETERS' in s][0]
        assert "BEAM_DIVERGENCE_E.S.D." in content[idx + 1]
        system['sigma_d'] = float(content[idx + 1].split()[-1])
        assert "REFLECTING_RANGE_E.S.D." in content[idx + 2] # conventional mosaicity parameter
        system['sigma_m'] = float(content[idx + 2].split()[-1])

    return

def extract_corrections(xds_path):
    """ Extract details for geometrical corrections and a mask of untrusted detector regions. 
    Save a .png file of the latter; assumes folder ./checks exists. """

    # generate boolean mask in shape of detector from INIT.LP, with False indicating untrusted pixel
    filename = xds_path + "INIT.LP"
    mask = np.ones(system['shape'], dtype = bool)

    with open(filename, "r") as f:
        
        content = f.readlines()
        
        # mask untrusted rows and columns of detector
        corners = np.asarray([s.strip('\n').split()[-6:-2] for s in content if "UNTRUSTED_RECTANGLE" in s], dtype=float).astype(int)
        for corner in corners:
            if corner[0] == 0:
                for i in range(corner[2], corner[3]):
                    mask[i] = False
            if corner[2] == 0:
                for i in range(corner[0], corner[1]):
                    mask[:,i] = False

        # mask shadow of beamstop arm if it's marked as an untrusted region by XDS
        if any("UNTRUSTED_QUADRILATERAL" in s for s in content):
            vertices = np.asarray([s.strip('\n').split()[-10:-2] for s in content if "UNTRUSTED_QUADRILATERAL" in s][0], dtype=int)
            bbPath = mplPath.Path(np.array([[vertices[0], vertices[1]],
                                            [vertices[2], vertices[3]],
                                            [vertices[4], vertices[5]],
                                            [vertices[6], vertices[7]],]))
            for x in range(mask.shape[1]):
                for y in range(mask.shape[0]):
                    if bbPath.contains_point((x, y)):
                        mask[y, x] = False
        
        # mask beamstop if marked as an untrusted region by XDS
        ebounds = [s.strip('\n').split()[1:5] for s in content if "UNTRUSTED_ELLIPSE" in s][0]
        ebounds = np.array([float(eb) for eb in ebounds])

        xwidth, ywidth = ebounds[1] - ebounds[0], ebounds[3] - ebounds[2]
        center = np.array([int(xwidth/2.0 + ebounds[0]), int(ywidth/2.0 + ebounds[2])])
        e = Ellipse(xy=center, width=xwidth, height=ywidth)
        for xpoint in np.arange(ebounds[0]-5, ebounds[1]+5):
            for ypoint in np.arange(ebounds[2]-5, ebounds[3]+5):
                if e.contains_point((xpoint, ypoint)):
                    mask[int(ypoint), int(xpoint)] = False

        system['mask'] = mask
        
    # extract parameters for parallax correction
    if 'parallax' in system['corrections']:
        
        filename = xds_path + "XYCORR.LP"
        with open(filename, "r") as f:
            
            content = f.readlines()
            
            # attentuation coefficient, mu, in units of m^{-1}; thickness, t0, in units of meters
            system['mu'] = 1000*float([s.strip('\n').split()[-1] for s in content if "SILICON" in s][0]) 
            system['t0'] = 0.001*float([s.strip('\n').split()[-1] for s in content if "SENSOR_THICKNESS" in s][0]) 

    # extract polarization fraction and check that plane normal is aligned along y
    if 'polarization' in system['corrections']:
        
        filename = xds_path + "CORRECT.LP"
        with open(filename, "r") as f:

            content = f.readlines()
            system['pf'] = float([s.strip('\n').split()[-1] for s in content if "FRACTION_OF_POLARIZATION" in s][0]) 
            pol_axis = np.asarray([s.strip('\n').split()[1:] for s in content if "POLARIZATION_PLANE_NORMAL" in s][0], dtype=float)

            try:
                assert pol_axis[0]==0 and pol_axis[1]==1 and pol_axis[2]==0
            except:
                "Polarization axis unexpected and flipped relative to Indexer.py"

    return

def plot_mask_and_scales(data_path):
    """ Plot mask and scales from system dictionary. """

    # generate a checks directory if doesn't already exist
    output_dir = data_path + "checks/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    f = plt.figure(figsize=(8,6))

    # following line was edited by Frederic Poitevin on June 19 2018 
    gs = gridspec.GridSpec(2, 2, height_ratios=[4,1], width_ratios=[1,1]) 
    #gs = gridspec.GridSpec(2, 1, height_ratios=[4,1], width_ratios=[1,1])
    ax0 = plt.subplot(gs[0])
    ax0.imshow(system['mask'], cmap='Blues')

    ax1 = plt.subplot(gs[1])
    ax1.plot(system['scales'])
    ax1.set_xlim(0, len(system['scales']))

    for tick in ax1.yaxis.get_ticklabels()[1::2]:
        tick.set_visible(False)

    ax0.set_title("Mask of untrusted regions")
    ax1.set_title("Image scale factors")
        
    f.savefig(data_path + "checks/mask_and_scales.png", dpi=100, bbox_inches='tight')

    return


if __name__ == '__main__':
    # build system dictionary

    system = dict()
    system['xds_path'] = sys.argv[1]
    system['map_path'] = sys.argv[2]
    system['image_type'] = sys.argv[3]

    prompt_for_corrections()
    print "Now extracting information from XDS files..."
    extract_geometry(sys.argv[1])
    extract_corrections(sys.argv[1])

    # plot mask & scales; save in pickle format
    plot_mask_and_scales(sys.argv[2])
    with open(sys.argv[2] + 'system.pickle', 'wb') as handle:
        pickle.dump(system, handle)
    print "Files saved to base directory: %s" %(sys.argv[2])
