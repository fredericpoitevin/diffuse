import cPickle as pickle
import numpy as np

"""
Store symmetry operations in .h5 format, with key denoting point group operations.
For consistency with other functions, symmetry operations for each point group are 
stored as a dictionary, with the second half of the keys corresponding to Friedel
pairs.
"""

symm_ops = dict()

symm_ops['p222'] = dict()
symm_ops['p222'][0] = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]], dtype=float)
symm_ops['p222'][1] = np.array([[-1, 0, 0],[0, -1, 0],[0, 0, 1]], dtype=float)
symm_ops['p222'][2] = np.array([[-1, 0, 0],[0, 1, 0],[0, 0, -1]], dtype=float)
symm_ops['p222'][3] = np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]], dtype=float)
for i in range(4):
    symm_ops['p222'][i+4] = -1*symm_ops['p222'][i]

symm_ops['p622'] = dict()
symm_ops['p622'][0] = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]], dtype=float)
symm_ops['p622'][1] = np.array([[0, -1, 0],[1, 1, 0],[0, 0, 1]], dtype=float)
symm_ops['p622'][2] = np.array([[-1, -1, 0],[1, 0, 0],[0, 0, 1]], dtype=float)
symm_ops['p622'][3] = np.array([[-1, 0, 0],[0, -1, 0],[0, 0, 1]], dtype=float)
symm_ops['p622'][4] = np.array([[0, 1, 0],[-1, -1, 0],[0, 0, 1]], dtype=float)
symm_ops['p622'][5] = np.array([[1, 1, 0],[-1, 0, 0],[0, 0, 1]], dtype=float)
symm_ops['p622'][6] = np.array([[0, -1, 0],[-1, 0, 0],[0, 0, -1]], dtype=float)
symm_ops['p622'][7] = np.array([[1, 0, 0],[-1, -1, 0],[0, 0, -1]], dtype=float)
symm_ops['p622'][8] = np.array([[1, 1, 0],[0, -1, 0],[0, 0, -1]], dtype=float)
symm_ops['p622'][9] = np.array([[0, 1, 0],[1, 0, 0],[0, 0, -1]], dtype=float)
symm_ops['p622'][10] = np.array([[-1, 0, 0],[1, 1, 0],[0, 0, -1]], dtype=float)
symm_ops['p622'][11] = np.array([[-1, -1, 0],[0, 1, 0],[0, 0, -1]], dtype=float)
for i in range(12):
    symm_ops['p622'][i+12] = -1*symm_ops['p622'][i]

symm_ops['p422'] = dict()
symm_ops['p422'][0] = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]], dtype=float)
symm_ops['p422'][1] = np.array([[0, -1, 0],[1, 0, 0],[0, 0, 1]], dtype=float)
symm_ops['p422'][2] = np.array([[-1, 0, 0],[0, -1, 0],[0, 0, 1]], dtype=float)
symm_ops['p422'][3] = np.array([[0, 1, 0],[-1, 0, 0],[0, 0, 1]], dtype=float)
symm_ops['p422'][4] = np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]], dtype=float)
symm_ops['p422'][5] = np.array([[0, 1, 0],[1, 0, 0],[0, 0, -1]], dtype=float)
symm_ops['p422'][6] = np.array([[-1, 0, 0],[0, 1, 0],[0, 0, -1]], dtype=float)
symm_ops['p422'][7] = np.array([[0, -1, 0],[-1, 0, 0],[0, 0, -1]], dtype=float)
for i in range(8):
    symm_ops['p422'][i+8] = -1*symm_ops['p422'][i]

with open("reference/symm_ops.pickle", "wb") as handle:
    pickle.dump(symm_ops, handle)
