from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from graph_sage.constants import GraphSAGELayerType, GraphSAGEAggregatorType
from graph_sage.definitions.aggregators import Aggregator, MeanAggregator, MaxPoolAggregator, MeanPoolAggregator


class GraphSAGE(nn.Module):

    def __init__(self, num_of_layers: int, device: torch.device, num_features_per_layer: List[int], num_neighbors: int,
                 aggregator_type: GraphSAGEAggregatorType,
                 dropout=0.6, layer_type=GraphSAGELayerType.IMP1):
        super().__init__()
        assert num_of_layers == len(num_features_per_layer) - 1, f'Enter valid params'
        self.num_of_layers = num_of_layers
        self.device = device
        self.num_neighbors = num_neighbors

        graph_sage_layers = []  # collect GraphSAGE layers
        GraphSAGELayer = get_layer_type(layer_type=layer_type)
        for i in range(num_of_layers):
            layer = GraphSAGELayer(
                layer_num=i + 1,
                max_layer_num=num_of_layers,
                num_in_features=num_features_per_layer[i],
                num_out_features=num_features_per_layer[i + 1],
                dropout_prob=dropout,
                aggregator_type=aggregator_type,
                device=self.device,
                num_neighbors=self.num_neighbors
            )
            graph_sage_layers.append(layer)

        self.graph_sage_net = nn.Sequential(*graph_sage_layers)

    def _form_computation_graph(self, neighbors, idx):
        node_layers = [np.array(idx)]
        for _ in range(self.num_of_layers):
            prev = node_layers[-1]
            arr = [node for node in prev]
            arr.extend([v for node in arr for v in neighbors[node]])
            arr = np.array(list(set(arr)), dtype=np.int64)
            node_layers.append(arr)
        node_layers.reverse()

        mappings = [{j: i for (i, j) in enumerate(arr)} for arr in node_layers]

        return node_layers, mappings

    def forward(self, data):
        node_features, adj_matrix = data
        node_layers, mappings = self._form_computation_graph(adj_matrix.rows, np.arange(0, node_features.shape[0]))
        return self.graph_sage_net((node_features, node_layers, mappings, adj_matrix.rows))


class GraphSAGELayer(nn.Module):
    def __init__(self, layer_num: int, max_layer_num: int, num_in_features: int, num_out_features: int,
                 dropout_prob: float, aggregator_type: GraphSAGEAggregatorType, device: torch.device,
                 num_neighbors: int):
        super().__init__()
        self.layer_num = layer_num
        self.max_layer_num = max_layer_num
        self.num_in_features = num_in_features
        self.num_out_features = num_out_features
        self.dropout_prob = dropout_prob
        self.aggregator_type = aggregator_type

        Aggregator = get_aggregator_type(aggregator_type)
        self.aggregator = Aggregator(input_dim=self.num_in_features, output_dim=self.num_in_features,
                                     device=device)

        if aggregator_type == GraphSAGEAggregatorType.Mean:
            self.linear = nn.Linear(in_features=self.num_in_features, out_features=self.num_out_features)  # no concat
        else:
            self.linear = nn.Linear(in_features=self.num_in_features * 2, out_features=self.num_out_features)
        self.num_neighbors = num_neighbors
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def init_params(self):
        nn.init.xavier_uniform_(self.linear.weight)


class GraphSAGELayerImp1(GraphSAGELayer):
    def __init__(self, layer_num: int, max_layer_num: int, num_in_features: int, num_out_features: int,
                 dropout_prob: float, aggregator_type: GraphSAGEAggregatorType, device: torch.device,
                 num_neighbors: int):
        super().__init__(layer_num, max_layer_num, num_in_features, num_out_features, dropout_prob, aggregator_type,
                         device, num_neighbors)

    def forward(self, data: Tuple[torch.Tensor, List[np.array], List[Dict[int, int]],
                                  List[List[int]]]):
        features, node_layers, mappings, global_neighbors = data
        out = features

        current_layer_index = self.layer_num - 1
        mapping = mappings[current_layer_index]
        nodes = node_layers[current_layer_index]

        init_mapped_nodes = np.array([mappings[0][v] for v in nodes], dtype=np.int64)
        cur_rows = global_neighbors[init_mapped_nodes]
        aggregate = self.aggregator(out, nodes, mapping, cur_rows, self.num_neighbors)

        # unless we don't use MEAN aggregator, we need to concat features from previous layer, but in MEAN
        # we do that in aggregation part already
        if self.aggregator_type != GraphSAGEAggregatorType.Mean:
            cur_mapped_nodes = np.array([mapping[v] for v in nodes], dtype=np.int64)
            out = torch.cat((out[cur_mapped_nodes, :], aggregate), dim=1)

        out = self.linear(out)
        if self.layer_num < self.max_layer_num:
            out = self.relu(out)
            out = self.dropout(out)

        return out, node_layers, mappings, global_neighbors


def get_layer_type(layer_type: GraphSAGELayerType):
    assert isinstance(layer_type, GraphSAGELayerType), f'Expected {GraphSAGELayerType} got {type(layer_type)}.'

    if layer_type == GraphSAGELayerType.IMP1:
        return GraphSAGELayerImp1
    else:
        raise Exception(f'Layer type {layer_type} not yet supported.')


def get_aggregator_type(aggregator_type: GraphSAGEAggregatorType):
    assert isinstance(aggregator_type,
                      GraphSAGEAggregatorType), f'Expected {GraphSAGEAggregatorType} got {type(aggregator_type)}.'

    if aggregator_type == GraphSAGEAggregatorType.Mean:
        return MeanAggregator
    elif aggregator_type == GraphSAGEAggregatorType.MeanPool:
        return MeanPoolAggregator
    elif aggregator_type == GraphSAGEAggregatorType.MaxPool:
        return MaxPoolAggregator
    else:
        raise Exception(f'Aggregator type {aggregator_type} not yet supported.')
