import argparse
import collections
import os
import time
from typing import Tuple, Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from constants import ModelType, DatasetType, CORA_NUMBER_OF_TRAIN_EXAMPLES_PER_CLASS, \
    CITESEER_NUMBER_OF_TRAIN_EXAMPLES_PER_CLASS
from node2vec.constants import NODE2VEC_BINARIES_PATH
from node2vec.util.graph import GraphHolder
from utils import util
from utils.data_loading import load_graph_data
from node2vec.definitions import node2vec
from utils.util import convert_adj_to_edge_index, get_node2vec_training_state, pickle_save, get_balanced_train_indices
from utils.visualization import visualize_embeddings


def get_num_training_examples_per_classes(dataset_name):
    if dataset_name.lower() == DatasetType.CORA.name.lower():
        return CORA_NUMBER_OF_TRAIN_EXAMPLES_PER_CLASS
    if dataset_name.lower() == DatasetType.CITESEER.name.lower():
        return CITESEER_NUMBER_OF_TRAIN_EXAMPLES_PER_CLASS


def calculate_embeddings(config, adj_matrix):
    edges = convert_adj_to_edge_index(adj_matrix)

    edges_weights: Dict[Tuple[int, int], int] = {(src, target): 1 for src, target in edges}
    graph = GraphHolder(edges_weights=edges_weights, is_directed=False)
    time_start = time.time()
    print(f'Started training of node2vec')
    embeddings = node2vec.calculate_node_embeddings(graph=graph,
                                                    p=config["p"],
                                                    q=config["q"],
                                                    num_walks=config["num_walks"],
                                                    walk_length=config["walk_length"],
                                                    vector_size=config["vector_size"],
                                                    alpha=config["alpha"],
                                                    window=config["window"],
                                                    min_count=config["min_count"],
                                                    seed=config["seed"],
                                                    workers=config["workers"],
                                                    min_alpha=config["min_alpha"],
                                                    sg=config["sg"],
                                                    hs=config["hs"],
                                                    negative=config["negative"],
                                                    epochs=config["epochs"],
                                                    )
    print(
        f'NODE2VEC training: time elapsed= {(time.time() - time_start):.2f} [s] | epochs={config["epochs"] + 1}')
    return embeddings


def train_classifier(node_labels: Dict[int, int], embeddings: Dict[int, List[float]], dataset_name="cora") -> (
        float, LogisticRegression, List[List[float]]):
    BEST_TEST_ACC = 0.
    BEST_CLF: LogisticRegression
    BEST_EMBEDDINGS: List[List[float]]

    node_labels = collections.OrderedDict(sorted(node_labels.items()))
    embeddings = collections.OrderedDict(sorted((embeddings.items())))

    NUM_TRAIN_LABELS_PER_CLASS = [get_num_training_examples_per_classes(dataset_name=dataset_name), 100, 150]

    time_start = time.time()
    for num_train_nodes in NUM_TRAIN_LABELS_PER_CLASS:
        train_indices, val_indices, test_indices = get_balanced_train_indices(np.array(list(node_labels.values())),
                                                                              num_training_examples_per_class=num_train_nodes)
        train_node_ids = train_indices
        test_node_ids = np.concatenate((val_indices, test_indices))

        train_node_ids = np.arange(0, num_train_nodes*7)
        test_node_ids = np.arange(num_train_nodes * 7, len(node_labels))

        train_embeddings = [embeddings[node_id] for node_id in train_node_ids]
        train_labels = [node_labels[node_id] for node_id in train_node_ids]

        test_embeddings = [embeddings[node_id] for node_id in test_node_ids]
        test_labels = [node_labels[node_id] for node_id in test_node_ids]

        scaler = StandardScaler()
        train_embeddings = scaler.fit_transform(train_embeddings)
        test_embeddings = scaler.transform(test_embeddings)

        clf = LogisticRegression(random_state=0, multi_class='ovr', max_iter=1000)
        clf.fit(train_embeddings, train_labels)
        predictions = clf.predict(test_embeddings)

        train_acc = np.sum(np.array(clf.predict(train_embeddings)) == np.array(train_labels)) / len(train_embeddings)

        accuracy = np.sum(np.array(predictions) == np.array(test_labels)) / len(test_node_ids)

        if accuracy > BEST_TEST_ACC:
            BEST_TEST_ACC = accuracy
            BEST_CLF = clf
            best_embeddings = []
            best_embeddings.extend(train_embeddings)
            best_embeddings.extend(test_embeddings)
            BEST_EMBEDDINGS = best_embeddings
        print(
            f'NODE2VEC embeddings_fitting: time elapsed= {(time.time() - time_start):.2f} [s] | num_train_nodes={num_train_nodes} | train acc={train_acc} | test acc={accuracy} ')

    return BEST_TEST_ACC, BEST_CLF, BEST_EMBEDDINGS


def get_args():
    parser = argparse.ArgumentParser()

    # Training related
    parser.add_argument("--p", type=float, help="Return parameter p", default=2.0)
    parser.add_argument("--q", type=float, help="Inout parameter q", default=0.5)
    parser.add_argument("--num_walks", type=int, help="number of walks for certain node", default=4)
    parser.add_argument("--walk_length", type=int, help="Max walk length", default=5)

    parser.add_argument("--vector_size", type=int, default=64, help='Size of embeddings for each vector')
    parser.add_argument("--alpha", type=float, default=0.025, help='The initial learning rate')
    parser.add_argument("--window", type=int, default=5, help='Maximum distance between the current and predicted '
                                                              'word within a sentence.')
    parser.add_argument("--min_count", type=int, default=1,
                        help='Maximum distance between the current and predicted word within a sentence.')
    parser.add_argument("--seed", type=int, default=1,
                        help='Maximum distance between the current and predicted word within a sentence.')
    parser.add_argument("--workers", type=float, default=1, help='Use these many worker threads to train the model ('
                                                                 '=faster training with multicore machines).')
    parser.add_argument("--min_alpha", type=float, default=0.0001, help='The minimal learning rate')
    parser.add_argument("--sg", type=int, default=1, help='Training algorithm: 1 for skip-gram; otherwise CBOW')
    parser.add_argument("--hs", type=int, default=0, help='The initial learning rate')
    parser.add_argument("--negative", type=int, default=5, help=' If > 0, negative sampling will be used, '
                                                                'the int for negative specifies how many "noise '
                                                                'words" should be drawn (usually between 5-20).')
    parser.add_argument("--epochs", type=int, default=10, help='Number of iterations (epochs) over the corpus. ('
                                                               'Formerly: `iter`)')

    # Dataset related
    parser.add_argument("--dataset_name", choices=[el.name for el in DatasetType], help='dataset to use for training',
                        default=DatasetType.CORA.name)
    parser.add_argument("--should_visualize", action='store_true', help='should visualize the dataset? (no by default)')

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", action='store_true', help="enable tensorboard logging (no by default)")
    parser.add_argument("--console_log_freq", type=int, help="log to output console (epoch) freq (None for no logging)",
                        default=100)
    parser.add_argument("--checkpoint_freq", type=int,
                        help="checkpoint model saving (epoch) freq (None for no logging)", default=1000)
    args = parser.parse_args()

    node2vec_config = {
        "p": 2,
        "q": 0.5,
        "num_walks": 80,
        "walk_length": 10,
        "vector_size": 64,
        "alpha": 0.025,
        "window": 5,
        "min_count": 1,
        "seed": 1,
        "workers": 4,
        "min_alpha": 0.0001,
        "sg": 1,
        "hs": 0,
        "negative": 5,
        "epochs": 10,
        "model_type": ModelType.NODE2VEC,

    }

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    # Add additional config information
    training_config = {**node2vec_config, **training_config}

    print(training_config)

    return training_config


def main():
    config = get_args()
    node_features, node_labels, adj_matrix = load_graph_data(
        {"dataset_name": config["dataset_name"], "model_type": config["model_type"]})
    embeddings = calculate_embeddings(config, adj_matrix)
    best_acc, best_clf, best_embeddings = train_classifier({node: label for node, label in enumerate(node_labels)},
                                                           embeddings)

    config['test_perf'] = best_acc
    node2vec_training_state = get_node2vec_training_state(config, best_clf, best_embeddings)

    pickle_save(path=os.path.join(NODE2VEC_BINARIES_PATH,
                                  util.get_available_binary_name(model_name=ModelType.NODE2VEC.name,
                                                                 binary_name=NODE2VEC_BINARIES_PATH,
                                                                 dataset_name=config['dataset_name'])),
                data=node2vec_training_state)

    all_nodes_unnormalized_scores = np.array(best_clf.predict_proba(best_embeddings))

    visualize_embeddings(all_nodes_unnormalized_scores, np.array(node_labels))
    visualize_embeddings(np.array(best_embeddings), np.array(node_labels))


if __name__ == "__main__":
    main()
