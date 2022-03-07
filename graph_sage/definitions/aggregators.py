import numpy as np
import torch
import torch.nn as nn


class Aggregator(nn.Module):

    def __init__(self, input_dim, output_dim, device):
        """
        Parameters
        ----------
        input_dim : int or None.
            Dimension of input node features. Used for defining fully
            connected layer in pooling aggregators. Default: None.
        output_dim : int or None
            Dimension of output node features. Used for defining fully
            connected layer in pooling aggregators. Currently only works when
            input_dim = output_dim. Default: None.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

    def forward(self, features, nodes, mapping, rows, num_neighbors=5):
        """
        Parameters
        ----------
        features : torch.Tensor
            An (n' x input_dim) tensor of input node features.
        nodes : numpy array
            nodes is a numpy array of nodes in the current layer of the computation graph.
        mapping : dict
            mapping is a dictionary mapping node v (labelled 0 to |V|-1) to
            its position in the layer of nodes in the computationn graph
            before nodes. For example, if the layer before nodes is [2,5],
            then mapping[2] = 0 and mapping[5] = 1.
        rows : numpy array
            rows[i] is an array of neighbors of node i which is present in nodes.
        num_samples : int
            Number of neighbors to sample while aggregating. Default: 25.

        Returns
        -------
        out : torch.Tensor
            An (len(nodes) x output_dim) tensor of output node features.
            Currently only works when output_dim = input_dim.
        """

        mapped_rows = [np.array([mapping[v] for v in row], dtype=np.int64) for row in rows]
        if num_neighbors == -1:
            sampled_rows = mapped_rows
        else:
            sampled_rows = [np.random.choice(row, min(len(row), num_neighbors), len(row) < num_neighbors) for row in mapped_rows]

        n = len(nodes)

        out = torch.zeros(n, self.output_dim).to(self.device)
        for i in range(n):
            if len(sampled_rows[i]) != 0:
                out[i, :] = self._aggregate(features[sampled_rows[i], :])

        return out

    def _aggregate(self, features):
        """
        Parameters
        ----------

        Returns
        -------
        """
        raise NotImplementedError


class MeanAggregator(Aggregator):

    def _aggregate(self, features):
        """
        Parameters
        ----------
        features : torch.Tensor
            Input features.

        Returns
        -------
        Aggregated feature.
        """
        return torch.mean(features, dim=0)
