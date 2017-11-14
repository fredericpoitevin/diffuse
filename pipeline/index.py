import tables, h5py, bitshuffle.h5, fabio
import matplotlib, sys, time, glob
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np
import Indexer, os.path

"""
Wrapper for Indexer class, which can accept .npy or .h5 image files. Files in 
.cbf format must be converted to .npy format first.

Usage: python index.py [map directory] [image directory]

"""

def load_h5(file_name):
    """
    Load compressed h5 file and convert to np.array format using bitshuffle.
    """

    try:
        f = tables.File(file_name, filters=32008) # filter number from bitshuffle website
        data = f.get_node('/entry/data').data.read()
    
        mask = np.zeros(data[0].shape)
        mask[np.where(data[0]==data[0].max())] = 1
        data_filtered = data[0].copy()
        data_filtered[mask==1] = 0
    
        return data_filtered

    except:
        return
    
def index_data(system, image_dir, output_dir):
    """ 
    Index images with Indexer class.
    Input parameters: system dictionary, image directory, directory for indexed .npy files
    Outputs: indexed .npy files, plot of average differences between Indexer results and INTEGRATE.HKL
    """
    
    # assume that file numbers are 1-indexed and prefaced by final underscore in file name
    file_glob = glob.glob(image_dir + "*")

    if system['image_type'] == 'h5':
        idx = [i for i,s in enumerate(file_glob) if 'master' in s]
        if len(idx) != 0:
            file_glob.pop(idx) # eliminate the master.h5 file if present

    if system['image_type'] == 'ccd':
        filelist = sorted(file_glob, key = lambda name: int(name.split('.')[-1]))
    else:
        filelist = sorted(file_glob, key = lambda name: int(name.split('_')[-1].split('.')[0]))
    num_images = len(filelist)
    deltas = np.zeros((num_images, 3))
        
    # set up instance of Indexer class and retrieve batch bounds
    indexer_obj = Indexer.Indexer(system)
    keys, vals = system['img2batch'].keys(), system['img2batch'].values()
    batch_ends = np.where(np.array([vals[v+1] - vals[v] for v in range(len(vals) - 1)]) != 0)[0]
    batch_ends = np.array([keys[v] for v in batch_ends])
    
    for img in range(num_images):

        if system['image_type'] == 'ccd':
            img_num = filelist[img].split('.')[-1]
        else:
            img_num = filelist[img].split('_')[-1].split('.')[0]
        print "Indexing image %s" %img_num

        # index image and process intensities; save as hklI
        deltas[img] = indexer_obj.index(int(img_num))

        if system['image_type'] == 'h5':
            imgI = load_h5(filelist[img])
        elif system['image_type'] == 'ccd':
            imgI = fabio.open(filelist[img]).data
        elif system['image_type'] == 'npy':
            imgI = np.load(filelist[img])
        else:
            print "Cannot open image files"

        if imgI is not None:
            indexer_obj.process_intensities(imgI, int(img_num))
            np.save(output_dir + "hklI_%s.npy" %img_num, indexer_obj.hklI)
        else:
            print "Warning: image %s could not be processed" %img_num 

        # reset hklI per image and corrections per batch
        indexer_obj.clear_hklI()
        if int(img_num) in batch_ends:
            indexer_obj.clear_corrections()
    
    return deltas

def plot_deltas(deltas, map_dir):
    """
    Plot per image average residual in h,k,l between results of Indexer class 
    and INTEGRATE.HKL.
    """

    f, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,8))

    ax1.plot(deltas[:,0])
    ax2.plot(deltas[:,1])
    ax3.plot(deltas[:,2])

    for ax in [ax1, ax2, ax3]:
        ax.set_ylim(0, 0.05)
        ax.set_xlim(0, deltas.shape[0])

    ax1.set_ylabel(r'$\Delta$ h')
    ax2.set_ylabel(r'$\Delta$ k')
    ax3.set_ylabel(r'$\Delta$ l')

    ax1.set_title("Comparison to INTEGRATE.HKL")
    ax3.set_xlabel("Image Number")

    f.savefig(map_dir + "checks/index.png", dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    
    start = time.time()
    system = pickle.load(open(sys.argv[1]+"system.pickle"))
    
    output_dir = sys.argv[1]+"indexed/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    deltas = index_data(system, sys.argv[2], output_dir)
    if 'xds_path' in system.keys():
        plot_deltas(deltas, sys.argv[1])

    print "elapsed time is %f" %((time.time() - start)/60.0)
