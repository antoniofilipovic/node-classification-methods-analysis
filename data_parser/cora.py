import collections
import enum
import os

from typing import List, Tuple

CORA_NUMBER_OF_LABELS_PER_CLASS = 20

WORK_DIRECTORY = os.getcwd()
DATA_DIR = f"{WORK_DIRECTORY}/data"
CORA_DATA_DIR = f"{DATA_DIR}/CORA"

EDGES_DATASET = "cora.cites"
NODES_DATASET = "cora.content"


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

    # we will also keep track of all node ids in cora dataset
    all_nodes = set()

    for edge in edges_lines:
        # edge of type string, formatting: number number
        # in cora dataset entry 1 2 represents 2->1:
        # If a line is represented by "paper1 paper2" then the link is "paper2->paper1".
        src, target = int(edge[1]), int(edge[0])
        directed_edges.append((src, target))
        all_nodes.add(src)
        all_nodes.add(target)

    all_nodes = list(sorted(all_nodes))

    # here we will just map nodes from random ids to nodes with ids from 0 to N-1
    nodes_new_ids = {all_nodes[i]: i for i in range(len(all_nodes))}

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

        # take new node id
        node_id = nodes_new_ids[node_id]

        # cora category represents for example Genetic_Algorithms
        cora_category = node_info_parsed[len(node_info_parsed) - 1].strip()

        # here we take enum value
        label = int(CoraCategories[cora_category].value)

        # Take feature vector
        feature_vector = node_info_parsed[1:-1]
        feature_vector = [int(v) for v in feature_vector]

        nodes.append(CoraDatasetNode(label=label, node_id=node_id, feature_vector=feature_vector))

    # Map node ids to new ids, so for example the smallest id in dataset is 35, and we will map it to 0
    directed_edges = [(nodes_new_ids[src], nodes_new_ids[target]) for src, target in directed_edges]

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


def save_data_to_files():
    directed_edges, nodes = get_graph()
    adjacency_list_dict = collections.OrderedDict()

    # here we will just map nodes from random ids to nodes with ids from 0 to N-1
    nodes_new_ids = {nodes[i].node_id: i for i in range(len(nodes))}

    for node in nodes:
        adjacency_list_dict[nodes_new_ids[node.node_id]] = []

    for src, target in directed_edges:
        src, target = nodes_new_ids[src], nodes_new_ids[target]
        if src not in adjacency_list_dict:
            adjacency_list_dict[src] = []
        adjacency_list_dict[src].append(target)

    # sort by key
    adjacency_list_dict = collections.OrderedDict(sorted(adjacency_list_dict.items()))

    # shape (N, 2), where N is number of nodes and 2 is for node and label
    # ordered dict because we want to have correct correspondence between this dictionary and features dict
    # we use collections OrderedDict in order not to lose order of data when adding values from list to dict
    labels = [node.label for node in nodes]

    # shape (N, FIN), where N is number of nodes and FIN is feature number
    nodes_features = [node.feature_vector for node in nodes]

    return adjacency_list_dict, labels, nodes_features


if __name__ == "__main__":
    get_graph()
