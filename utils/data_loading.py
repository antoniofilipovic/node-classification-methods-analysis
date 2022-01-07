import torch
import networkx as nx
import numpy as np
import scipy.sparse as sp
from typing import List, Tuple

from constants import ModelType
from data_parser import cora


def create_adj_matrix(directed_edges: List[Tuple[int, int]], nodes: List[int]):
    adjacency_list_dict = {node: [] for node in nodes}
    for directed_edge in directed_edges:
        adjacency_list_dict[directed_edge[0]].append(directed_edge[1])

    return nx.adjacency_matrix(nx.from_dict_of_lists(adjacency_list_dict)).todense().astype(np.float)


def normalize_features_sparse(node_features_sparse):
    assert sp.issparse(node_features_sparse), f'Expected a sparse matrix, got {node_features_sparse}.'

    # Instead of dividing (like in normalize_features_dense()) we do multiplication with inverse sum of features.
    # Modern hardware (GPUs, TPUs, ASICs) is optimized for fast matrix multiplications! ^^ (* >> /)
    # shape = (N, FIN) -> (N, 1), where N number of nodes and FIN number of input features
    node_features_sum = np.array(node_features_sparse.sum(-1),
                                 dtype=float)  # sum features for every node feature vector

    # Make an inverse (remember * by 1/x is better (faster) then / by x)
    # shape = (N, 1) -> (N)
    node_features_inv_sum = np.power(node_features_sum, -1).squeeze()

    # Again certain sums will be 0 so 1/0 will give us inf so we replace those by 1 which is a neutral element for mul
    node_features_inv_sum[np.isinf(node_features_inv_sum)] = 1.

    # Create a diagonal matrix whose values on the diagonal come from node_features_inv_sum
    diagonal_inv_features_sum_matrix = sp.diags(node_features_inv_sum)

    # We return the normalized features.
    return diagonal_inv_features_sum_matrix.dot(node_features_sparse)


def load_graph_data(training_config, device=None):
    dataset_name = training_config['dataset_name'].lower()
    model_type = training_config['model_type']

    if dataset_name == "cora":
        # directed edges shape = (2,E), where E is number of edges
        # nodes_correct_label shape = (N,2), where N is number of nodes, and 2 for node and it's label
        # node features shape = (N, FIN), where FIN je number of input features
        adj_matrix, node_labels, node_features = cora.get_data_train_balanced()

        # adjacency matrix shape = (N,N)
        adj_matrix = np.array(adj_matrix, dtype=float)
        adj_matrix[adj_matrix > 0] = 1  # multiple edges not allowed

        # not good design pattern
        if model_type == ModelType.NODE2VEC:
            return node_features, node_labels, adj_matrix

        if model_type == ModelType.GCN:
            adj_matrix += np.identity(adj_matrix.shape[0])  # add self connections

            rowsum = np.array(adj_matrix.sum(1))
            r_inv = np.power(rowsum, -0.5).flatten()
            r_inv[np.isinf(r_inv)] = 1.
            r_mat_inv = np.zeros((2708, 2708), float)  # if you put int here, it will be greatest mistake ever

            np.fill_diagonal(r_mat_inv, r_inv)

            adj_matrix = np.matmul(adj_matrix, r_mat_inv)
            adj_matrix = np.matmul(adj_matrix, r_mat_inv)

        # normalizing node features
        nodes_features = sp.csr_matrix(node_features)
        nodes_features = normalize_features_sparse(nodes_features)

        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float, device=device)
        node_labels = torch.tensor(node_labels, dtype=torch.long, device=device)  # Cross entropy expects a long int
        node_features = torch.tensor(nodes_features.todense(), dtype=torch.float, device=device)

        return node_features, node_labels, adj_matrix
