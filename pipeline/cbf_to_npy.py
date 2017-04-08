import numpy, sys, pycbf

""" This script reads a cbf file and writes the observed intensities as a 2d numpy array 
with the dimensions of the detector. This is a modified version of image.py from DIALS, 
which is under the BSD license and writes the cbf file as a scitbx flex array: 
https://github.com/dials/dials/blob/a852845612701cfc18bce4844e43fae42ad843a0/util/image.py.

Usage: dials.python cbf_to_npy.py image.cbf savename.npy."""

class reader:
    """A class to read the CBF files used in DIALS"""
    def __init__(self):
        pass

    def read_file(self, filename):
        """Read the CBF file"""
        self.cbf_handle = pycbf.cbf_handle_struct()
        self.cbf_handle.read_file(filename, pycbf.MSG_DIGEST)
        self.cbf_handle.rewind_datablock()

    def extract_data(self):
        """Get the gain array from the file"""

        # Select the first datablock and rewind all the categories
        self.cbf_handle.select_datablock(0)
        self.cbf_handle.select_category(0)
        # cols 0 and 1 contain detector/expt information; col 2 contains data
        self.cbf_handle.select_column(2)
        self.cbf_handle.select_row(0)

        # Read the image data into an array
        image_string = self.cbf_handle.get_integerarray_as_string()
        image = numpy.fromstring(image_string, numpy.int32)

        # Resize image based on detector dimensions
        parameters = self.cbf_handle.get_integerarrayparameters_wdims()
        image_size = (parameters[10], parameters[9])
        image = image.reshape(image_size[0], image_size[1])

        # Return the image
        return image

if __name__ == '__main__':
    handle = reader()
    handle.read_file(sys.argv[1])
    image = handle.extract_data()
    numpy.save(sys.argv[2], image)
