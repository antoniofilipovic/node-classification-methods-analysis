"""


"""

import collections
import enum
import os
import pickle

from typing import List, Tuple

import numpy as np
import scipy.sparse as sp

from utils.util import pickle_save, convert_adj_dict_to_adj_matrix

CORA_NUMBER_OF_LABELS_PER_CLASS = 20

WORK_DIRECTORY = os.getcwd()
DATA_DIR = f"{WORK_DIRECTORY}/data"
CORA_DATA_DIR = f"{DATA_DIR}/CORA/"
PREPROCESSED_DATA_DIR = f"{CORA_DATA_DIR}/preprocessed_data/"

EDGES_DATASET = "cora.cites"
NODES_DATASET = "cora.content"

ADJ_PREPROCESSED = "adj_matrix.dict"
FEATURES_PREPROCESSED = "features.npy"
LABELS_PREPROCESSED = "labels.npy"


# This two functions are used for normal processing of dataset tkipf used in GCN :)
# Just read data and we get adjacency dictionary, features numpy array and lables numpy

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)

    return labels_onehot


def load_data(path=os.path.join(CORA_DATA_DIR), dataset="cora"):
    # this method will read content part of data and save it as numpy array type of string
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))

    # features represent all data between first and last index in each row
    # shape = (N, FIN), where N is number of nodes and FIN is number of input features
    features = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)

    # labels encoded as number -> 0, 1, 2, 3, 4, 5, 6
    # shape = (1,N) where N is number of nodes
    labels = np.where(encode_onehot(idx_features_labels[:, -1]))[1]

    # edges from file cora.cites
    # if you check file you will see that message that if we have paper1 paper2
    # that actually means that paper2 is citing paper1
    # so this order is wrong, but we will change it few lines below
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)

    # get ids of nodes
    nodes_ids = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # map nodes so we have nodes from 0 to N
    nodes_ids_map = {j: i for i, j in enumerate(nodes_ids)}

    edges = np.array(list(map(nodes_ids_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    # here we change order
    edges[:, [1, 0]] = edges[:, [0, 1]]

    # here we crate sparse adjacency matrix
    adj = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # but I want to save it as dict, same as Gordic Aleksa used in his pytorchGAT implementation
    adj = adj.todense()
    adj_dict = {i: np.nonzero(row)[1].tolist() for i, row in enumerate(adj)}

    return adj_dict, features, labels


# All Cora data is stored as pickle
def pickle_read(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)

    return data


# After preprocessing, you can just use this function to load data
def get_data_train_unbalanced():
    # shape = (N, FIN), where N is the number of nodes and FIN is the number of input features
    node_features_npy = pickle_read(os.path.join(PREPROCESSED_DATA_DIR, FEATURES_PREPROCESSED))

    # shape = (N, 1)
    node_labels_npy = pickle_read(os.path.join(PREPROCESSED_DATA_DIR, LABELS_PREPROCESSED))

    # shape = (N, number of neighboring nodes) <- this is a dictionary not a matrix!
    adjacency_list_dict = pickle_read(os.path.join(PREPROCESSED_DATA_DIR, ADJ_PREPROCESSED))

    adjacency_matrix = convert_adj_dict_to_adj_matrix(adjacency_list_dict)

    return adjacency_matrix, node_labels_npy, node_features_npy


# IMPORTANT:
# From here to all way down below functions are used to make training dataset balanced.
# This means we want to have 20 labels of each class on top so that we can train model
# with balanced dataset. Later it got me that same would happen if only I used indices for
# training differently :(

class CoraCategories(enum.Enum):
    Case_Based = 0
    Genetic_Algorithms = 1
    Neural_Networks = 2
    Probabilistic_Methods = 3
    Reinforcement_Learning = 4
    Rule_Learning = 5
    Theory = 6


class CoraDatasetNode:
    def __init__(self, label: int, feature_vector: List[int], node_id: int):
        self.label = label
        self.feature_vector = feature_vector
        self.node_id = node_id
        self.neighbors = []


def get_graph() -> (List[Tuple[int, int]], List[CoraDatasetNode]):
    with open(os.path.join(CORA_DATA_DIR, EDGES_DATASET)) as edges_io:
        edges_lines = map(str.split, edges_io.readlines())

    with open(os.path.join(CORA_DATA_DIR, NODES_DATASET)) as nodes_io:
        node_content = nodes_io.readlines()

    # shape (E,2), where E is number of directed edges and 2 is for source and target node
    # one entry, for example 1,2 corresponds to 1->2
    directed_edges: List[Tuple[int, int]] = []

    for edge in edges_lines:
        # edge of type string, formatting: number number
        # in cora dataset entry 1 2 represents 2->1:
        # If a line is represented by "paper1 paper2" then the link is "paper2->paper1".
        src, target = int(edge[1]), int(edge[0])
        directed_edges.append((src, target))

    # here we will save all nodes
    nodes = []
    for node_info in node_content:
        # node info is represented like this
        # node_id 0 0 0 ... 0 0 1 ... 0 0 Genetic_Algorithms
        # we need to isolate node_id, node label and feature vector
        # So first we parse node info
        node_info_parsed = node_info.split("\t")

        # Take node id from 0 index
        node_id = int(node_info_parsed[0])

        # cora category represents for example Genetic_Algorithms
        cora_category = node_info_parsed[len(node_info_parsed) - 1].strip()

        # here we take enum value
        label = int(CoraCategories[cora_category].value)

        # Take feature vector
        feature_vector = node_info_parsed[1:-1]
        feature_vector = [int(v) for v in feature_vector]

        nodes.append(CoraDatasetNode(label=label, node_id=node_id, feature_vector=feature_vector))

    nodes = preprocess_dataset(nodes)
    return directed_edges, nodes


def preprocess_dataset(nodes: List[CoraDatasetNode]) -> List[CoraDatasetNode]:
    # I am using this function in order to have in list in top 140 places balanced data of 20 nodes per class
    # That means I want to have 20 labels of class

    nodes = sorted(nodes, key=lambda n: n.label)

    train_nodes_list = []
    rest_nodes_list = []

    label_current = 0
    cnt_current_label = 0

    indices = [0] * len(nodes)
    for i in range(len(nodes)):
        node = nodes[i]

        if node.label != label_current:
            label_current += 1
            cnt_current_label = 0

        if node.label == label_current and cnt_current_label < CORA_NUMBER_OF_LABELS_PER_CLASS:
            train_nodes_list.append(node)
            indices[i] = 1
            cnt_current_label += 1
            continue

        rest_nodes_list.append(node)

    train_nodes_list.extend(rest_nodes_list)
    return train_nodes_list


def get_data_train_balanced():
    directed_edges, nodes = get_graph()
    adjacency_list_dict = collections.OrderedDict()

    # This got maybe too complicated, but I will try to explain
    # Now we have 20 labels of each class one after the other in list of nodes
    # 0,0,0 ... 0, 0, 1, 1, 1, ..,1,1 , 2,2,2, ..., 2,2, 3,3,3, ...,3,3 ... , 6,6,...,6,6

    # here we will just map nodes from random ids to nodes with ids from 0 to N-1
    nodes_new_ids = {nodes[i].node_id: i for i in range(len(nodes))}

    # this way in adjacency list we will have node and its neighbors, but first we will have node 0, then for node 1
    # 0: 23, 34, 405
    # 1: 143, 43, 2000,
    # ...
    # 2708: 214, 25 ,44

    for node in nodes:
        adjacency_list_dict[nodes_new_ids[node.node_id]] = []

    for src, target in directed_edges:
        src, target = nodes_new_ids[src], nodes_new_ids[target]
        if src not in adjacency_list_dict:
            adjacency_list_dict[src] = []
        adjacency_list_dict[src].append(target)

    # sort by key, this is not necessary, but just to be sure
    adjacency_list_dict = collections.OrderedDict(sorted(adjacency_list_dict.items()))

    # here we will have labels, first it will be label of node 0, then label of node 1 and so on
    labels = [node.label for node in nodes]

    # shape (N, FIN), where N is number of nodes and FIN is feature number
    # here are node features, first of node 0, then of node 1 and so on...
    nodes_features = [node.feature_vector for node in nodes]

    adj_matrix = []
    for i in range(2708):
        adj_matrix.append([])
        for j in range(2708):
            adj_matrix[i].append(0)

    for node, node_neighbors in adjacency_list_dict.items():
        for n in node_neighbors:
            adj_matrix[node][n] = 1
            adj_matrix[n][node] = 1

    return adj_matrix, labels, nodes_features


if __name__ == "__main__":
    adj_dict, features_numpy, labels_numpy = load_data()

    # this is used to save adj matrix to file so we don't need to preprocess it every time
    adj_dict_path = os.path.join(PREPROCESSED_DATA_DIR, ADJ_PREPROCESSED)
    pickle_save(adj_dict_path, adj_dict)

    # same applies to features which we will save as numpy
    features_dict_path = os.path.join(PREPROCESSED_DATA_DIR, FEATURES_PREPROCESSED)
    pickle_save(features_dict_path, features_numpy)

    # same applies to features which we will save as numpy
    labels_numpy_path = os.path.join(PREPROCESSED_DATA_DIR, LABELS_PREPROCESSED)
    pickle_save(labels_numpy_path, labels_numpy)
