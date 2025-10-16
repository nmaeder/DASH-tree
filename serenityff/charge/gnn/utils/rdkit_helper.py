from typing import List, Optional, Sequence

import numpy as np
import torch as pt
from rdkit import Chem

from serenityff.charge.gnn.utils import CustomData, MolGraphConvFeaturizer
from serenityff.charge.utils import Molecule


def mols_from_sdf(sdf_file: str, removeHs: Optional[bool] = False) -> Sequence[Molecule]:
    """
    Returns a Sequence of rdkit molecules read in from a .sdf file.

    Args:
        sdf_file (str): path to .sdf file.
        removeHs (Optional[bool], optional): Wheter to remove Hydrogens. Defaults to False.

    Returns:
        Sequence[Molecule]: rdkit mols.
    """
    return Chem.SDMolSupplier(sdf_file, removeHs=removeHs)


def get_mol_prop_as_array(prop_name: Optional[str], mol: Chem.Mol) -> np.ndarray:
    """Get atomic properties from an RDKit molecule object as an array.

    The property is expected to be a string of '|' separated numerical
    values, one for each atom in the molecule.

    Parameters
    ----------
    prop_name
        The name of the property to retrieve from the molecule.
    mol
        The RDKit molecule object.

    Returns
    -------
    np.ndarray
        The atomic properties converted to a NumPy array.

    Raises
    ------
    ValueError
        If `prop_name` is None or if the property is not found in the molecule.
    TypeError
        If any of the parsed property values are NaN or not convertable to float.
    """
    if prop_name is None:
        raise ValueError("Property name can not be None when no_y == False.")
    if not mol.HasProp(prop_name):
        raise ValueError(f"Property {prop_name} not found in molecule.")  # noqa E713
    array = np.fromstring(mol.GetProp(prop_name), sep="|", dtype=float)
    if np.isnan(array).any():
        raise TypeError(f"Nan found in {prop_name}.")
    return array


def get_mol_prop_as_tensor(prop_name: Optional[str], mol: Chem.Mol) -> pt.Tensor:
    """Get atomic properties from an RDKit molecule object as a tensor.

    The property is expected to be a string of '|' separated numerical
    values, one for each atom in the molecule.

    Parameters
    ----------
    prop_name
        The name of the property to retrieve from the molecule.
    mol
        The RDKit molecule object.

    Returns
    -------
    pt.Tensor
        The atomic properties converted to a PyTorch tensor.

    Raises
    ------
    ValueError
        If `prop_name` is None or if the property is not found in the molecule.
    TypeError
        If any of the parsed property values are NaN or not convertable to float.
    """
    return pt.from_numpy(get_mol_prop_as_array(prop_name=prop_name, mol=mol))


def get_graph_from_mol(
    mol: Molecule,
    index: int,
    sdf_property_name: Optional[str] = "MBIScharge",
    allowable_set: Optional[List[str]] = [
        "C",
        "N",
        "O",
        "F",
        "P",
        "S",
        "Cl",
        "Br",
        "I",
        "H",
    ],
    no_y: Optional[bool] = False,
) -> Optional[CustomData]:
    """
    Creates an pytorch_geometric Graph from an rdkit molecule.

    Returns None if the property is not found or contains NaN.
    The graph contains following features:
        > Node Features:
            > Atom Type (as specified in allowable set)
            > formal_charge
            > hybridization
            > H acceptor_donor
            > aromaticity
            > degree
        > Edge Features:
            > Bond type
            > is in ring
            > is conjugated
            > stereo
    Args:
        mol (Molecule): rdkit molecule
        sdf_property_name (Optional[str]): Name of the property in the sdf file to be used for training.
        allowable_set (Optional[List[str]], optional): List of atoms to be \
            included in the feature vector. Defaults to \
                [ "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "H", ].

    Returns:
        CustomData: pytorch geometric Data with .smiles as an extra attribute.
    """

    grapher = MolGraphConvFeaturizer(use_edges=True)
    graph = grapher._featurize(mol, allowable_set).to_pyg_graph()
    if no_y:
        graph.y = pt.tensor(
            [0 for _ in mol.GetAtoms()],
            dtype=pt.float,
        )
    else:
        try:
            graph.y = get_mol_prop_as_tensor(sdf_property_name, mol)
        except TypeError as exc:
            print(exc)
            return None

    graph.batch = pt.tensor([0 for _ in mol.GetAtoms()], dtype=int)
    graph.molecule_charge = Chem.GetFormalCharge(mol)
    graph.smiles = Chem.MolToSmiles(mol, canonical=True)
    graph.sdf_idx = index
    return graph
