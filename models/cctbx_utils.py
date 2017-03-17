import mmtbx.f_model, cctbx
from iotbx import pdb
import numpy as np

""" Collection of scripts that rely on cctbx.python. """

def asu_to_p1(pdb_name, savename):
    """ 
    Expands an input pdb to p1 based on symmetry defined by CRYST1 record,
    based on: http://cctbxwiki.bravais.net/CCTBX_Wiki#Generating_a_Unit_Cell.
    Alternative methods (which use a GUI interface) are Mercury and Chimera.

    Parameters
    ----------
    pdb_name : string
        path to pdb file of asymmetric unit, must have intact CRYST1 records

    savename : string
        path to output pdb of unit cell

    """

    pdb_inp = pdb.input(file_name = pdb_name)
    
    # retrieve symmetry information for input model
    symm = pdb_inp.crystal_symmetry()
    cell = symm.unit_cell()
    sg = symm.space_group()
    hierarchy = pdb_inp.construct_hierarchy()
    model = hierarchy.models()[0]

    # set up new hierarchy object
    hierarchy_new = pdb.hierarchy.root()
    model_new = pdb.hierarchy.model()
    hierarchy_new.append_model(model_new)

    chain_ids = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    for i, op in enumerate(sg.all_ops()):
        rotn = op.r().as_double()
        # orthogonalizing translational compoment to operate on atoms
        tln = cell.orthogonalize(op.t().as_double())
        for chain in model.chains():
            c = chain.detached_copy()
            atoms = c.atoms()
            xyz = atoms.extract_xyz()
            atoms.set_xyz((rotn*xyz)+tln)
            # set c.id = 'A' for all chains to share chain ID
            c.id = chain_ids.next()
            model_new.append_chain(c)

    hierarchy_new.write_pdb_file(file_name = savename)

    return


def compute_sf(pdb_name, resolution, savename, p1=False, intensities=False, set_b=None):
    """ 
    Compute structure factors from an input pdb file; save as .npy file 

    Parameters
    ----------
    pdb_name : string
        path to pdb file
    
    resolution : float
        highest resolution to which SFs will be computed

    savename : string
        path to output .npy file

    p1 : bool, optional
        If True, expand Miller array to p1. Default: False.

    intensities : bool, optional
        If True, return intensities rather than amplitudes. Default: False.

    set_b : float, optional
        Value to uniformly set atomic B factors to if not None. Default: None.

    Returns
    -------
    miller : cctbx miller array
        Miller array (indices and data) of I(hkl) or SF(hkl).

    """

    pdb_name = pdb.input(file_name = pdb_name)
    pdb_target = pdb_name.xray_structure_simple()
    if set_b is not None:
        pdb_target.set_b_iso(set_b)

    miller = pdb_target.structure_factors(d_min = resolution).f_calc()    
    if p1 is True:
        miller = miller.expand_to_p1()
    if intensities is True:
        miller = miller.as_intensity_array()

    hkl = np.asarray(list(miller.indices()))
    vals = np.asarray(list(miller.data()))
    stacked = np.vstack((hkl.T, vals)).T

    np.save(savename, stacked)
    return stacked
