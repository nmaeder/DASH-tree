from serenityff.charge.tree.dash_tree import DASHTree
from serenityff.charge.tree.atom_features import AtomFeatures
from serenityff.torsion.data import default_dash_torsion_tree_path
from serenityff.charge.utils.rdkit_typing import Molecule
from serenityff.torsion.tree.tree_utils import get_canon_torsion_feature


class DASHTorsionTree(DASHTree):
    def __init__(
        self,
        tree_folder_path: str = default_dash_torsion_tree_path,
        preload: bool = True,
        verbose: bool = True,
        num_processes: int = 1,
    ):
        super().__init__(tree_folder_path, preload, verbose, num_processes)

    def _get_init_layer(self, mol: Molecule, atom: int, max_depth: int):
        if len(atom) != 4:
            raise ValueError(f"A list of 4 atom indices is required to define a torsion angle. Got {atom} instead.")
        af1, af2, af3, af4 = [AtomFeatures.atom_features_from_molecule(mol, atom_i) for atom_i in atom]
        canon_init_torsion_feature = get_canon_torsion_feature(af1, af2, af3, af4)
        matched_node_path = [canon_init_torsion_feature, 0]
        max_depth = max(max_depth - 3, 0)
        return canon_init_torsion_feature, matched_node_path, atom, max_depth

    def match_new_torsion(
        self,
        atoms_in_torsion: [int],
        mol: Molecule,
        max_depth: int = 16,
        attention_threshold: float = 10,
        attention_increment_threshold: float = 0,
        return_atom_indices: bool = False,
        neighbor_dict=None,
    ):
        if len(atoms_in_torsion) != 4:
            raise ValueError(
                f"A list of 4 atom indices is required to define a torsion angle. Got {atoms_in_torsion} instead."
            )
        # Shh, don't tell anyone, but we match torsions like single atoms, just with a overwriten _get_init_layer method
        return super().match_new_atom(
            atoms_in_torsion,
            mol,
            max_depth,
            attention_threshold,
            attention_increment_threshold,
            return_atom_indices,
            neighbor_dict,
        )
