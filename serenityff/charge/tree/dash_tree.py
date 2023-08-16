import os
import pickle
import gzip as compression

# import lz4.frame as compression
# import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# from multiprocessing import Pool
# from multiprocessing import Process, Manager

from serenityff.charge.tree.atom_features import AtomFeatures
from serenityff.charge.tree.tree_utils import get_possible_atom_features
from serenityff.charge.data import default_dash_tree_path
from serenityff.charge.utils.rdkit_typing import Molecule


class DASHTree:
    def __init__(self, tree_folder_path=default_dash_tree_path, preload=True, verbose=True, num_processes=4):
        self.tree_folder_path = tree_folder_path
        self.verbose = verbose
        self.num_processes = num_processes
        self.tree_storage = defaultdict(lambda: None)
        self.data_storage = defaultdict(lambda: None)
        if preload:
            self.load_all_trees_and_data()

    ########################################
    #   Tree import/export functions
    ########################################

    # tree file format:
    # int(id_counter), int(atom_type), int(con_atom), int(con_type), float(oldNode.attention), []children

    def load_all_trees_and_data(self):
        if self.verbose:
            print("Loading DASH tree data")
        # import all files
        for i in range(AtomFeatures.get_number_of_features()):
            tree_path = os.path.join(self.tree_folder_path, f"{i}.gz")
            df_path = os.path.join(self.tree_folder_path, f"{i}.h5")
            self.load_tree_and_data(tree_path, df_path)

    def load_tree_and_data(self, tree_path, df_path, hdf_key="df"):
        branch_idx = int(os.path.basename(tree_path).split(".")[0])
        with compression.open(tree_path, "rb") as f:
            tree = pickle.load(f)
        df = pd.read_hdf(df_path, key=hdf_key, mode="r")
        self.tree_storage[branch_idx] = tree
        self.data_storage[branch_idx] = df

    def save_all_trees_and_data(self):
        if self.verbose:
            print(f"Saving DASH tree data to {len(self.tree_storage)} files in {self.tree_folder_path}")
        for branch_idx in tqdm(self.tree_storage):
            self.save_tree_and_data(branch_idx)

    def save_tree_and_data(self, branch_idx):
        tree_path = os.path.join(self.tree_folder_path, f"{branch_idx}.lzma")
        df_path = os.path.join(self.tree_folder_path, f"{branch_idx}.h5")
        self._save_tree_and_data(branch_idx, tree_path, df_path)

    def _save_tree_and_data(self, branch_idx, tree_path, df_path):
        with compression.open(tree_path, "wb") as f:
            pickle.dump(self.tree_storage[branch_idx], f)
        self.data_storage[branch_idx].to_hdf(df_path, key="df", mode="w")

    def get_node_data(self, init_af, line_number):
        branch_idx = self.af_to_branch_idx[init_af]
        return self.data_storage[branch_idx].iloc[line_number]

    ########################################
    #   Tree assignment functions
    ########################################

    def _pick_subgraph_expansion_node(
        self, current_node: int, branch_idx: int, possible_new_atom_features: list, possible_new_atom_idxs: list
    ):
        current_node_children = self.tree_storage[branch_idx][current_node][5]
        for child in current_node_children:
            child_tree_node = self.tree_storage[branch_idx][child]
            child_af = [child_tree_node[1], child_tree_node[2], child_tree_node[3]]
            for possible_atom_feature, possible_atom_idx in zip(possible_new_atom_features, possible_new_atom_idxs):
                if possible_atom_feature == child_af:
                    return (child, possible_atom_idx)
        return (None, None)

    def match_new_atom(
        self,
        atom: int,
        mol: Molecule,
        max_depth: int = 16,
        attention_threshold: float = 10,
        attention_incremet_threshold: float = 10,
    ):
        init_atom_feature = AtomFeatures.atom_features_from_molecule_w_connection_info(mol, atom)
        branch_idx = init_atom_feature[0]  # branch_idx is the key of the AtomFeature without connection info
        matched_node_path = [branch_idx, 0]
        cummulative_attention = 0
        # Special case for H -> only connect to heavy atom and ignore H
        if mol.GetAtomWithIdx(atom).GetSymbol() == "H":
            h_connected_heavy_atom = mol.GetAtomWithIdx(atom).GetNeighbors()[0].GetIdx()
            init_atom_feature = AtomFeatures.atom_features_from_molecule_w_connection_info(mol, h_connected_heavy_atom)
            child, _ = self._pick_subgraph_expansion_node(0, branch_idx, [init_atom_feature], [h_connected_heavy_atom])
            matched_node_path.append(child)
            atom_indices_in_subgraph = [h_connected_heavy_atom]  # skip Hs as they are only treated implicitly
            max_depth -= 1  # reduce max_depth by 1 as we already added one node
        else:
            atom_indices_in_subgraph = [atom]
        if max_depth <= 1:
            return matched_node_path
        else:
            for _ in range(1, max_depth):
                possible_new_atom_features, possible_new_atom_idxs = get_possible_atom_features(
                    mol, atom_indices_in_subgraph
                )
                child, atom = self._pick_subgraph_expansion_node(
                    matched_node_path[-1], branch_idx, possible_new_atom_features, possible_new_atom_idxs
                )
                if child is None:
                    return matched_node_path
                matched_node_path.append(child)
                atom_indices_in_subgraph.append(atom)
                node_attention = self.tree_storage[branch_idx][child][4]
                cummulative_attention += node_attention
                if cummulative_attention > attention_threshold:
                    return matched_node_path
                if node_attention > attention_incremet_threshold:
                    return matched_node_path
            return matched_node_path

    def get_atom_properties(self, matched_node_path: list):
        branch_idx = matched_node_path[0]
        atom = matched_node_path[-1]
        df = self.data_storage[branch_idx]
        return df.iloc[atom]
