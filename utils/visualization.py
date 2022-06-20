import os

import numpy as np
import igraph as ig
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from constants import DatasetType, cora_label_to_color_map, VisualizationType, ModelType
from gat.constants import GATLayerType, GAT_BINARIES_PATH
from gat.definitions.gat import GAT
from gcn.constants import GCNLayerType, GCN_BINARIES_PATH
from gcn.definitions.gcn import GCN
from graph_sage.constants import GraphSAGELayerType, GRAPH_SAGE_BINARIES_PATH
from graph_sage.definitions.graph_sage import GraphSAGE
from utils.data_loading import load_graph_data
from utils.util import convert_adj_to_edge_index, print_model_metadata

import graph_sage.utils.util as graph_sage_util
import gat.utils.util as gat_util
import gcn.utils.util as gcn_util


def tsne_visualize_embeddings(all_nodes_unnormalized_scores, node_labels):
    assert isinstance(all_nodes_unnormalized_scores,
                      np.ndarray), f'Expected NumPy array got {type(all_nodes_unnormalized_scores)}. '

    num_classes = len(set(node_labels))

    # Feel free to experiment with perplexity it's arguable the most important parameter of t-SNE and it basically
    # controls the standard deviation of Gaussians i.e. the size of the neighborhoods in high dim (original) space.
    # Simply put the goal of t-SNE is to minimize the KL-divergence between joint Gaussian distribution fit over
    # high dim points and between the t-Student distribution fit over low dimension points (the ones we're plotting)
    # Intuitively, by doing this, we preserve the similarities (relationships) between the high and low dim points.
    # This (probably) won't make much sense if you're not already familiar with t-SNE, God knows I've tried. :P
    t_sne_embeddings = TSNE(n_components=2, perplexity=10, method='barnes_hut').fit_transform(
        all_nodes_unnormalized_scores)

    for class_id in range(num_classes):
        # We extract the points whose true label equals class_id and we color them in the same way, hopefully
        # they'll be clustered together on the 2D chart - that would mean that GAT has learned good representations!
        plt.scatter(t_sne_embeddings[node_labels == class_id, 0], t_sne_embeddings[node_labels == class_id, 1],
                    s=20, color=cora_label_to_color_map[class_id], edgecolors='black', linewidths=0.2)
    plt.show()


def visualize_gcn(binary_name: str, dataset_name: str,
                  visualization_type=VisualizationType.EMBEDDINGS):
    """
    Notes on t-SNE:
    Check out this one for more intuition on how to tune t-SNE: https://distill.pub/2016/misread-tsne/

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

    config = {
        'dataset_name': dataset_name,
        'model_type': ModelType.GCN,
        'layer_type': GCNLayerType.IMP1,
    }

    # Step 1: Prepare the data
    if dataset_name == DatasetType.CORA.name or dataset_name == DatasetType.CITESEER.name:
        node_features, node_labels, topology = load_graph_data(config, device)

    # Step 2: Prepare the model
    model_path = os.path.join(GCN_BINARIES_PATH, binary_name)
    model_state = torch.load(model_path, map_location=device)

    gcn = GCN(
        num_of_layers=model_state['num_of_layers'],
        num_features_per_layer=model_state['num_features_per_layer'],
        add_skip_connection=model_state['add_skip_connection'],
        bias=model_state['bias'],
        dropout=model_state['dropout'],
        layer_type=gcn_util.name_to_layer_type(model_state['layer_type']),
    ).to(device)

    print_model_metadata(model_state)
    assert model_state['dataset_name'].lower() == dataset_name.lower(), \
        f"The model was trained on {model_state['dataset_name']} but you're calling it on {dataset_name}."
    gcn.load_state_dict(model_state["state_dict"], strict=True)
    gcn.eval()  # some layers like nn.Dropout behave differently in train vs eval mode so this part is important

    # Step 3: Calculate the things we'll need for different visualization types (attention, scores, edge_index)

    # This context manager is important (and you'll often see it), otherwise PyTorch will eat much more memory.
    # It would be saving activations for backprop but we are not going to do any model training just the prediction.
    with torch.no_grad():
        # Step 3: Run predictions and collect the high dimensional data
        all_nodes_unnormalized_scores = gcn((node_features, topology))[0]  # shape = (N, num of classes)
        all_nodes_unnormalized_scores = all_nodes_unnormalized_scores.cpu().numpy()

    # edge_index = convert_adj_to_edge_index(topology)
    if visualization_type == VisualizationType.EMBEDDINGS:
        # visualize embeddings (using t-SNE)
        node_labels = node_labels.cpu().numpy()
        tsne_visualize_embeddings(all_nodes_unnormalized_scores, node_labels)


def visualize_graph_sage(binary_name: str, dataset_name: str,
                         visualization_type=VisualizationType.EMBEDDINGS):
    """
    Notes on t-SNE:
    Check out this one for more intuition on how to tune t-SNE: https://distill.pub/2016/misread-tsne/

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

    config = {
        'dataset_name': dataset_name,
        'model_type': ModelType.GraphSAGE,
        'layer_type': GraphSAGELayerType.IMP1,
    }

    # Step 1: Prepare the data
    if dataset_name == DatasetType.CORA.name or dataset_name == DatasetType.CITESEER.name:
        node_features, node_labels, topology = load_graph_data(config, device)

    # Step 2: Prepare the model
    model_path = os.path.join(GRAPH_SAGE_BINARIES_PATH, binary_name)
    model_state = torch.load(model_path, map_location=device)

    graph_sage = GraphSAGE(
        num_of_layers=model_state['num_of_layers'],
        num_features_per_layer=model_state['num_features_per_layer'],
        dropout=model_state['dropout'],
        layer_type=graph_sage_util.name_to_layer_type(model_state['layer_type']),
        num_neighbors=model_state["num_neighbors"],
        device=device,
        aggregator_type=graph_sage_util.name_to_agg_type(model_state['aggregator_type'])
    ).to(device)

    print_model_metadata(model_state)
    assert model_state['dataset_name'].lower() == dataset_name.lower(), \
        f"The model was trained on {model_state['dataset_name']} but you're calling it on {dataset_name}."
    graph_sage.load_state_dict(model_state["state_dict"], strict=True)
    graph_sage.eval()  # some layers like nn.Dropout behave differently in train vs eval mode so this part is important

    # Step 3: Calculate the things we'll need for different visualization types (attention, scores, edge_index)

    # This context manager is important (and you'll often see it), otherwise PyTorch will eat much more memory.
    # It would be saving activations for backprop but we are not going to do any model training just the prediction.
    with torch.no_grad():
        # Step 3: Run predictions and collect the high dimensional data
        all_nodes_unnormalized_scores = graph_sage((node_features, topology))[0]  # shape = (N, num of classes)
        all_nodes_unnormalized_scores = all_nodes_unnormalized_scores.cpu().numpy()

    if visualization_type == VisualizationType.EMBEDDINGS:
        # visualize embeddings (using t-SNE)
        node_labels = node_labels.cpu().numpy()
        tsne_visualize_embeddings(all_nodes_unnormalized_scores, node_labels)


def visualize_gat(binary_name: str, dataset_name: str,
                         visualization_type=VisualizationType.EMBEDDINGS):
    """
    Notes on t-SNE:
    Check out this one for more intuition on how to tune t-SNE: https://distill.pub/2016/misread-tsne/

    """

    device = torch.device("cpu")  # checking whether you have a GPU, I hope so!

    config = {
        'dataset_name': dataset_name,
        'model_type': ModelType.GAT,
        'layer_type': GATLayerType.IMP1,
    }

    # Step 1: Prepare the data
    if dataset_name == DatasetType.CORA.name or dataset_name == DatasetType.CITESEER.name:
        node_features, node_labels, topology = load_graph_data(config, device)

    # Step 2: Prepare the model
    model_path = os.path.join(GAT_BINARIES_PATH, binary_name)
    model_state = torch.load(model_path, map_location=device)

    gat = GAT(
        num_of_layers=model_state['num_of_layers'],
        num_features_per_layer=model_state['num_features_per_layer'],
        dropout=model_state['dropout'],
        layer_type=gat_util.name_to_layer_type(model_state['layer_type']),
        num_heads_per_layer=model_state['num_heads_per_layer'],
        add_skip_connection=model_state['add_skip_connection'],
        bias=model_state['bias']
    ).to(device)

    print_model_metadata(model_state)
    assert model_state['dataset_name'].lower() == dataset_name.lower(), \
        f"The model was trained on {model_state['dataset_name']} but you're calling it on {dataset_name}."
    gat.load_state_dict(model_state["state_dict"], strict=True)
    gat.eval()  # some layers like nn.Dropout behave differently in train vs eval mode so this part is important

    # Step 3: Calculate the things we'll need for different visualization types (attention, scores, edge_index)

    # This context manager is important (and you'll often see it), otherwise PyTorch will eat much more memory.
    # It would be saving activations for backprop but we are not going to do any model training just the prediction.
    with torch.no_grad():
        # Step 3: Run predictions and collect the high dimensional data
        all_nodes_unnormalized_scores = gat((node_features, topology))[0]  # shape = (N, num of classes)
        all_nodes_unnormalized_scores = all_nodes_unnormalized_scores.cpu().numpy()

    if visualization_type == VisualizationType.EMBEDDINGS:
        # visualize embeddings (using t-SNE)
        node_labels = node_labels.cpu().numpy()
        tsne_visualize_embeddings(all_nodes_unnormalized_scores, node_labels)