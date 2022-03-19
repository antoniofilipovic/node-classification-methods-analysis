import numpy as np
import torch
import torch.nn as nn


class Aggregator(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, device: torch.device):
        """
        Parameters
        ----------
        input_dim : int or None.
            Dimension of input node features. Used for defining fully
            connected layer in pooling aggregators. Default: None.
        output_dim : int or None
            Dimension of output node features. Used for defining fully
            connected layer in pooling aggregators.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

    def forward(self, features, nodes, mapping, rows, num_neighbors):
        raise NotImplementedError

    def _aggregate(self, features):
        """
        Parameters
        ----------

        Returns
        -------
        """
        raise NotImplementedError


class GCNAggregator(Aggregator):

    def __init__(self, input_dim: int, output_dim: int, device: torch.device):
        super().__init__(input_dim, output_dim, device)

    def forward(self, features, nodes, mapping, rows, num_neighbors):

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
        curr_mapped_nodes = np.array([mapping[node] for node in nodes], dtype=np.int64)
        if num_neighbors == -1:
            sampled_rows = mapped_rows
        else:
            sampled_rows = [np.random.choice(row, min(len(row), num_neighbors), replace=False) for row in
                            mapped_rows]
        # here we concatenate sampled neighbors and current node
        sampled_rows = [np.append(row, curr_mapped_nodes[i]) for i, row in enumerate(sampled_rows)]

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
        features : torch.Tensor
            Input features.

        Returns
        -------
        Aggregated feature.
        """
        return torch.mean(features, dim=0)  # mean over rows


class MeanAggregator(Aggregator):

    def __init__(self, input_dim: int, output_dim: int, device: torch.device):
        super().__init__(input_dim, output_dim, device)


    def forward(self, features, nodes, mapping, rows, num_neighbors):

        mapped_rows = [np.array([mapping[v] for v in row], dtype=np.int64) for row in rows]
        if num_neighbors == -1:
            sampled_rows = mapped_rows
        else:
            sampled_rows = [np.random.choice(row, min(len(row), num_neighbors), replace=False) for row in
                            mapped_rows]

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
        features : torch.Tensor
            Input features.

        Returns
        -------
        Aggregated feature.
        """

        return torch.mean(features, 0)  # mean over rows



class MeanPoolAggregator(Aggregator):

    def __init__(self, input_dim: int, output_dim: int, device: torch.device):
        super().__init__(input_dim, output_dim, device)
        self.linear = nn.Linear(input_dim, output_dim)
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.relu = nn.ReLU()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, features, nodes, mapping, rows, num_neighbors):

        mapped_rows = [np.array([mapping[v] for v in row], dtype=np.int64) for row in rows]
        if num_neighbors == -1:
            sampled_rows = mapped_rows
        else:
            sampled_rows = [np.random.choice(row, min(len(row), num_neighbors), replace=False) for row in
                            mapped_rows]

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
        features : torch.Tensor
            Input features.

        Returns
        -------
        Aggregated feature.
        """
        projected_features = self.linear(features)
        projected_features_relu = self.relu(projected_features)
        return torch.mean(projected_features_relu, 0)  # mean over rows


class MaxPoolAggregator(Aggregator):

    def __init__(self, input_dim: int, output_dim: int, device: torch.device):
        super().__init__(input_dim, output_dim, device)
        self.linear = nn.Linear(input_dim, output_dim)
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.relu = nn.ReLU()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, features, nodes, mapping, rows, num_neighbors):

        mapped_rows = [np.array([mapping[v] for v in row], dtype=np.int64) for row in rows]
        if num_neighbors == -1:
            sampled_rows = mapped_rows
        else:
            sampled_rows = [np.random.choice(row, min(len(row), num_neighbors), replace=False) for row in
                            mapped_rows]

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
        features : torch.Tensor
            Input features.

        Returns
        -------
        Aggregated feature.
        """
        projected_features = self.linear(features)
        projected_features_relu = self.relu(projected_features)
        return torch.max(projected_features_relu, 0)[0]  # max over rows
