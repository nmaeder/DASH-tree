from __future__ import annotations

from torch import Tensor
from torch.nn import Module
from torch_geometric.explain.algorithm.gnn_explainer import (
    GNNExplainer_ as GNNExplainer,
)

from serenityff.charge.gnn.utils import CustomData


class Explainer:
    """
    Class holding a pytorch_geometric GNNExplainer object
    and some settings.
    """

    def __init__(
        self,
        model: Module,
        epochs: int = 2000,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            model (Module): model to be
            epochs (int, optional): _description_. Defaults to 2000.
            verbose (bool, optional): _description_. Defaults to False.
        """
        self.gnn_explainer = GNNExplainer(
            model=model,
            epochs=epochs,
            log=verbose,
            return_type="regression",
            feat_mask_type="scalar",
        )
        self.gnnverbose = verbose

    @property
    def gnn_explainer(self) -> GNNExplainer:
        return self._gnn_explainer

    @gnn_explainer.setter
    def gnn_explainer(self, value) -> None:
        if not isinstance(value, GNNExplainer):
            raise TypeError("explainer needs to be of type GNNExplainer")
        else:
            self._gnn_explainer = value

    @property
    def gnnverbose(self) -> bool:
        return self._gnnverbose

    @gnnverbose.setter
    def gnnverbose(self, value: bool) -> None:
        self._gnnverbose = value
        self._gnn_explainer.log = value

    def _explain(self, *args, **kwargs) -> tuple[Tensor, Tensor]:
        """
        parser to the GNNExplainers' explain node Function.
        Give all args also needed in the models forward function.

        Returns:
            Tuple[Tensor, Tensor]: Node and Edge Attentions for node in graph.
        """
        return self.gnn_explainer.explain_node(*args, **kwargs)

    def _explain_atom(
        self,
        node_idx: int,
        graph: CustomData,
    ) -> tuple[Tensor, Tensor]:
        """
        Explains a given node in an Graph object.

        Args:
            node_idx (int): Idx of node to be explained.
            data (CustomData): Graph containing the node to be explained.

        Returns:
            Tuple[Tensor, Tensor]: Node and Edge Attentions for node in graph.
        """
        node, edge = self._explain(
            node_idx,
            graph.x,
            graph.edge_index,
            edge_attr=graph.edge_attr,
            batch=graph.batch,
            molecule_charge=graph.molecule_charge,
        )
        return node, edge

    def explain_molecule(
        self,
        graph: CustomData,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """
        Explains all Atom Predictions in a Molecule

        Args:
            data (CustomData): Graph representing the molecule.

        Returns:
            Tuple[List[Tensor]]: Lists of Node and Edge Attentions.
        """
        nodes: list[Tensor] = []
        edges: list[Tensor] = []
        for i in range(graph.num_nodes):
            node, edge = self._explain_atom(node_idx=i, graph=graph)
            nodes.append(node)
            edges.append(edge)
        return nodes, edges
