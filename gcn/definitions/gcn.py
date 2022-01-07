from typing import List

import torch.nn

from gcn.constants import GCNLayerType


class GCN(torch.nn.Module):
    def __init__(self, num_of_layers: int, num_features_per_layer: List[int], add_skip_connection: False, bias=True,
                 dropout=0.6, layer_type = GCNLayerType.IMP1):
        super().__init__()

        assert num_of_layers == len(num_features_per_layer) - 1, f'Enter valid params'

        gcn_layers = []  # collect GCN layers
        GCNLayer = get_layer_type(layer_type=layer_type)
        for i in range(num_of_layers):
            layer = GCNLayer(
                num_in_features=num_features_per_layer[i],
                num_out_features=num_features_per_layer[i + 1],
                activation=torch.nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,

            )
            gcn_layers.append(layer)

        self.gcn_net = torch.nn.Sequential(*gcn_layers)

    def forward(self, data):
        return self.gcn_net(data)


class GCNLayer(torch.nn.Module):
    """
    Base class for all implementations
    """

    def __init__(self, num_in_features: int, num_out_features: int, activation, dropout_prob: float,
                 add_skip_connection: bool, bias:bool, layer_type):
        super().__init__()
        self.num_out_features = num_out_features
        self.num_in_features = num_in_features

        # projection matrix to lower dimension
        self.proj_matrix = torch.nn.Parameter(torch.FloatTensor(self.num_in_features, self.num_out_features))

        self.dropout = torch.nn.Dropout(dropout_prob)
        self.activation = activation

        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(num_out_features))

        self.add_skip_connection = add_skip_connection

        self.init_params()

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.

        """
        torch.nn.init.xavier_uniform_(self.proj_matrix)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class GCNLayerImp1(GCNLayer):
    """
        Implementation #2 was inspired by the official GAT implementation: https://github.com/PetarV-/GAT

        It's conceptually simpler than implementation #3 but computationally much less efficient.

        Note: this is the naive implementation not the sparse one and it's only suitable for a transductive setting.
        It would be fairly easy to make it work in the inductive setting as well but the purpose of this layer
        is more educational since it's way less efficient than implementation 3.

    """

    def __init__(self, num_in_features, num_out_features, activation,
                 dropout_prob=0.6, add_skip_connection=True, bias=True):
        super().__init__(num_in_features, num_out_features, activation,
                         dropout_prob,
                         add_skip_connection, bias, GCNLayerType.IMP1)

    def forward(self, data):
        in_nodes_features, preprocessed_adj_matrix = data  # unpack data
        # shape = (N, FIN) * (FIN, FOUT) = (N,FOUT), project FIN features to smaller dimension
        projected_features = torch.mm(in_nodes_features, self.proj_matrix)
        # shape = (N,N) * (N, FOUT) = (N, FOUT)
        output = torch.mm(preprocessed_adj_matrix, projected_features)

        return (output, preprocessed_adj_matrix)


#
# Helper functions
#
def get_layer_type(layer_type):
    assert isinstance(layer_type, GCNLayerType), f'Expected {GCNLayerType} got {type(layer_type)}.'

    if layer_type == GCNLayerType.IMP1:
        return GCNLayerImp1
    elif layer_type == GCNLayerType.IMP2:
        return GCNLayerImp1
    else:
        raise Exception(f'Layer type {layer_type} not yet supported.')
