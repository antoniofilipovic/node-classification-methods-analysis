import os
import numpy as np
import scipy.sparse as sp
import pickle

WORK_DIRECTORY = os.getcwd()
DATA_DIR = f"{WORK_DIRECTORY}/data"
CITESEER_DATA_DIR = f"{DATA_DIR}/citeseer/"
PREPROCESSED_DATA_DIR = f"{CITESEER_DATA_DIR}/preprocessed_data/"

EDGES_DATASET = "citeseer.cites"
NODES_DATASET = "citeseer.content"

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


def load_data(path=os.path.join(CITESEER_DATA_DIR), dataset="citeseer"):
    # this method will read content part of data and save it as numpy array type of string
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))



    # features represent all data between first and last index in each row
    # shape = (N, FIN), where N is number of nodes and FIN is number of input features
    features = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)

    num_of_features = features.shape[1]

    # labels encoded as number -> 0, 1, 2, 3, 4, 5, 6
    # shape = (1,N) where N is number of nodes
    labels = np.where(encode_onehot(idx_features_labels[:, -1]))[1]

    # edges from file cora.cites
    # if you check file you will see that message that if we have paper1 paper2
    # that actually means that paper2 is citing paper1
    # so this order is wrong, but we will change it few lines below
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=str)

    # get ids of nodes, type string as some have string id
    nodes_ids = np.array(idx_features_labels[:, 0], dtype=str)
    # map nodes so we have nodes from 0 to N
    nodes_ids_map = {j: i for i, j in enumerate(nodes_ids)}
    max_value = len(nodes_ids_map)
    edges=[]
    for edge in edges_unordered:
        src, target = nodes_ids_map.get(edge[0]), nodes_ids_map.get(edge[1])
        if src is None:
            nodes_ids_map[edge[0]]=max_value
            max_value+=1
            features = np.vstack((features, np.zeros((1, num_of_features))))
            labels = np.append(labels, -1)
        if target is None:
            nodes_ids_map[edge[1]] = max_value
            max_value += 1
            features = np.vstack((features, np.zeros((1, num_of_features))))
            labels = np.append(labels,-1)

        edges.append([nodes_ids_map.get(edge[0]), nodes_ids_map.get(edge[1])])

    edges = np.array(edges, dtype=int)
    edges.reshape(edges_unordered.shape)


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


def convert_adj_dict_to_adj_matrix(adj_dict):
    """
    Convert adj dict to adj matrix
    """
    assert isinstance(adj_dict, dict), f'Expected Dict type got {type(adj_dict)}.'

    N = 2708
    adjacency_matrix = np.zeros((N, N), dtype=int)
    for src, src_neighbors in adj_dict.items():
        for target in src_neighbors:
            adjacency_matrix[src][target] = 1

    return adjacency_matrix  # shape (N,N)


def get_data_train_unbalanced():
    # shape = (N, FIN), where N is the number of nodes and FIN is the number of input features
    node_features_npy = pickle_read(os.path.join(PREPROCESSED_DATA_DIR, FEATURES_PREPROCESSED))

    # shape = (N, 1)
    node_labels_npy = pickle_read(os.path.join(PREPROCESSED_DATA_DIR, LABELS_PREPROCESSED))

    # shape = (N, number of neighboring nodes) <- this is a dictionary not a matrix!
    adjacency_list_dict = pickle_read(os.path.join(PREPROCESSED_DATA_DIR, ADJ_PREPROCESSED))

    adjacency_matrix = convert_adj_dict_to_adj_matrix(adjacency_list_dict)

    return adjacency_matrix, node_labels_npy, node_features_npy


if __name__ == "__main__":
    adj_dict, features_numpy, labels_numpy = load_data()

    # this is used to save adj matrix to file so we don't need to preprocess it every time
    adj_dict_path = os.path.join(PREPROCESSED_DATA_DIR, ADJ_PREPROCESSED)
    with open(adj_dict_path, 'wb') as handle:
        pickle.dump(adj_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # same applies to features which we will save as numpy
    features_dict_path = os.path.join(PREPROCESSED_DATA_DIR, FEATURES_PREPROCESSED)
    with open(features_dict_path, 'wb') as handle:
        pickle.dump(features_numpy, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # same applies to features which we will save as numpy
    labels_numpy_path = os.path.join(PREPROCESSED_DATA_DIR, LABELS_PREPROCESSED)
    with open(labels_numpy_path, 'wb') as handle:
        pickle.dump(labels_numpy, handle, protocol=pickle.HIGHEST_PROTOCOL)


