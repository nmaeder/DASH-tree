from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

from serenityff.charge.tree.node import node
from serenityff.charge.tree.atom_features import atom_features


from serenityff.charge.tree.tree_utils import get_possible_atom_features


class tree:
    def __init__(self):
        self.root = node(level=0)
        self.tree_lengths = defaultdict(int)

    ###############################################################################
    # General helper functions
    ###############################################################################

    def from_file(self, file_path: str):
        self.root = self.from_file(file_path)

    def update_tree_length(self):
        self.tree_lengths.clear()
        self.tree_lengths = self.root.get_tree_length(self.tree_lengths)

    def get_sum_nodes(self):
        self.update_tree_length()
        return sum(self.tree_lengths.values())

    ###############################################################################
    # Tree assignment functions
    ###############################################################################

    def match_new_atom(self, atom, mol):
        current_correct_node = tree.root
        node_path = [self.root]
        connected_atoms = []
        for i in range(20):
            try:
                if i == 0:
                    possible_new_atom_features = [atom_features(mol, atom)]
                    possible_new_atom_idxs = [atom]
                else:
                    possible_new_atom_features, possible_new_atom_idxs = get_possible_atom_features(
                        mol, [int(x) for x in connected_atoms]
                    )
                found_match = False
                for current_node in current_correct_node.children:
                    if found_match:
                        break
                    for possible_atom_feature, possible_atom_idx in zip(
                        possible_new_atom_features, possible_new_atom_idxs
                    ):
                        if possible_atom_feature in current_node.atoms:
                            current_correct_node = current_node
                            connected_atoms.append(possible_atom_idx)
                            node_path.append(current_node)
                            found_match = True
                            break
            except Exception as e:
                print(e)
                break
        return (current_correct_node.result, node_path)

    def match_molecules_atoms(self, mol, mol_idx):
        return_list = []
        for atom in mol.GetAtoms():
            return_dict = {}
            atom_idx = atom.GetIdx()
            return_dict["mol_idx"] = int(mol_idx)
            return_dict["atom_idx"] = int(atom_idx)
            return_dict["atomtype"] = atom.GetSymbol()
            return_dict["truth"] = float(atom.GetProp("molFileAlias"))
            try:
                result, node_path = self.match_new_atom(atom_idx, mol)
                return_dict["tree"] = float(result)
                return_dict["tree_std"] = node_path[-1].stdDeviation
            except Exception as e:
                print(e)
                return_dict["tree"] = np.NAN
                return_dict["tree_std"] = np.NAN
            return_list.append(return_dict)

        # normalize_charge symmetric
        tot_charge_truth = np.round(np.sum([float(x.GetProp("molFileAlias")) for x in mol.GetAtoms()]))
        tot_charge_tree = np.sum([x["tree"] for x in return_list])
        for x in return_list:
            x["tree_norm1"] = x["tree"] - ((tot_charge_tree - tot_charge_truth) / mol.GetNumAtoms())

        # normalize_charge std weighted
        tot_std = np.sum([x["tree_std"] for x in return_list])
        for x in return_list:
            x["tree_norm2"] = x["tree"] + (tot_charge_truth - tot_charge_tree) * (x["tree_std"] / tot_std)

        return return_list

    def match_dataset(self, mol_sup, stop=1000000):
        i = 0
        tot_list = []
        for mol in tqdm(mol_sup):
            if i >= stop:
                break
            tot_list.extend(self.match_molecules_atoms(mol, i))
            i += 1
        return pd.DataFrame(tot_list)
