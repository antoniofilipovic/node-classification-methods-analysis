from typing import List

import torch
import torch.nn as nn

from gat.constants import GATLayerType


class GAT(nn.Module):

    def __init__(self, num_of_layers: int, num_features_per_layer: List[int], num_heads_per_layer: List[int],
                 add_skip_connection: False, bias=True, dropout=0.6, layer_type=GATLayerType.IMP1, ):
        super().__init__()

        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid params'
        num_heads_per_layer = [1] + num_heads_per_layer
        gat_layers = []  # collect GCN layers
        GATLayer = get_layer_type(layer_type=layer_type)
        for i in range(num_of_layers):
            layer = GATLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],
                num_out_features=num_features_per_layer[i + 1],
                activation=nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                num_heads=num_heads_per_layer[i+1],
                concat=True if i < num_of_layers - 1 else False

            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(*gat_layers)

    def forward(self, data):
        return self.gat_net(data)


class GATLayer(nn.Module):
    """
    Base class for all implementations
    """

    def __init__(self, num_in_features: int, num_out_features: int, activation, dropout_prob: float,
                 add_skip_connection: bool, bias: bool, num_heads: int, concat: bool):
        super().__init__()
        self.num_out_features = num_out_features
        self.num_in_features = num_in_features
        self.num_heads = num_heads
        self.concat = concat
        self.add_skip_connection = add_skip_connection

        self.head_dim = 1

        self.linear = nn.Linear(self.num_in_features, self.num_heads * self.num_out_features)

        self.attn_fn_source = nn.Parameter(torch.Tensor(1, self.num_heads, self.num_out_features))
        self.attn_fn_target = nn.Parameter(torch.Tensor(1, self.num_heads, self.num_out_features))



        # if bias:
        #    self.bias = nn.Parameter(torch.FloatTensor(num_out_features))

        # self.add_skip_connection = add_skip_connection

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = activation

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear.weight)

        nn.init.xavier_uniform_(self.attn_fn_source)
        nn.init.xavier_uniform_(self.attn_fn_target)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.reshape(-1, self.num_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


class GATLayerImp1(GATLayer):
    """
    Base class for all implementations
    """

    def __init__(self, num_in_features: int, num_out_features: int, activation, dropout_prob: float,
                 add_skip_connection: bool, bias: bool, num_heads: int, concat: bool):
        super().__init__(num_in_features, num_out_features, activation, dropout_prob, add_skip_connection, bias,
                         num_heads, concat)

    def forward(self, data):
        in_node_features, connectivity_mask = data

        in_node_features = self.dropout(in_node_features)

        # shape = (N, num_in) * (num_input, num_heads(NH) *  num_out) -> (N, NH * num_out)

        proj_features = self.linear(in_node_features)
        # shape = (N, NH, num_out)
        proj_features = proj_features.reshape(-1, self.num_heads, self.num_out_features)

        proj_features = self.dropout(proj_features)

        # shape = (1, NH, FOUT) * (N, NH, FOUT) -> sum((N, NH, FOUT)) -> (N, NH, 1)
        target_scores = torch.sum(self.attn_fn_target * proj_features, dim=-1, keepdim=True)
        source_scores = torch.sum(self.attn_fn_source * proj_features, dim=-1, keepdim=True)

        # shape = src (NH, N, 1), target (NH, 1, N)
        source_scores = source_scores.permute(1, 0, 2)
        target_scores = target_scores.permute(1, 2, 0)

        # automatic broadcasting -> shape = (NH, N, N)
        all_scores = self.leaky_relu(source_scores + target_scores)

        all_attn_coefs = self.softmax(all_scores + connectivity_mask)

        # shape = (NH, N, N) * (NH, N, FOUT) -> (NH, N, FOUT)
        out_node_features = torch.bmm(all_attn_coefs, proj_features.transpose(0, 1))

        # shape = (N, NH, FOUT)
        out_node_features = out_node_features.permute(1, 0, 2)

        out_node_features = self.concat_bias(all_attn_coefs, in_node_features, out_node_features)
        return out_node_features, connectivity_mask


#
# Helper functions
#
def get_layer_type(layer_type):
    assert isinstance(layer_type, GATLayerType), f'Expected {GATLayerType} got {type(layer_type)}.'

    if layer_type == GATLayerType.IMP1:
        return GATLayerImp1
    else:
        raise Exception(f'Layer type {layer_type} not yet supported.')
