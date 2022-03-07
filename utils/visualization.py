import os

import numpy as np
import igraph as ig
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from constants import DatasetType, cora_label_to_color_map, VisualizationType, ModelType
from gcn.constants import GCNLayerType, GCN_BINARIES_PATH
from gcn.definitions.gcn import GCN
from graph_sage.constants import GraphSAGELayerType
from graph_sage.definitions.graph_sage import GraphSAGE
from utils.data_loading import load_graph_data
from utils.util import convert_adj_to_edge_index, print_model_metadata, name_to_layer_type


def visualize_graph(edge_index, node_labels, dataset_name):
    """
    blog for available graph visualization tools:
        https://towardsdatascience.com/large-graph-visualization-tools-and-approaches-2b8758a1cd59

    Basically depending on how big your graph is there may be better drawing tools than igraph.

    Note:
    There are also some nice browser-based tools to visualize graphs like this one:
        http://networkrepository.com/graphvis.php?d=./data/gsm50/labeled/cora.edges

    Nonetheless tools like igraph can be useful for quick visualization directly from Python

    """
    assert isinstance(edge_index, np.ndarray), f'Expected NumPy array got {type(edge_index)}.'
    if edge_index.shape[0] == edge_index.shape[1]:
        edge_index = convert_adj_to_edge_index(edge_index)
        edge_index = edge_index.transpose()  # 2,N shape

    num_of_nodes = len(node_labels)
    edge_index_tuples = list(zip(edge_index[0, :], edge_index[1, :]))

    # Construct the igraph graph
    ig_graph = ig.Graph()
    ig_graph.add_vertices(num_of_nodes)
    ig_graph.add_edges(edge_index_tuples)

    # Prepare the visualization settings dictionary
    visual_style = {}

    # Defines the size of the plot and margins
    visual_style["bbox"] = (3000, 3000)
    visual_style["margin"] = 35

    # I've chosen the edge thickness such that it's proportional to the number of shortest paths (geodesics)
    # that go through a certain edge in our graph (edge_betweenness function, a simple ad hoc heuristic)

    # line1: I use log otherwise some edges will be too thick and others not visible at all
    # edge_betweeness returns < 1.0 for certain edges that's why I use clip as log would be negative for those edges
    # line2: Normalize so that the thickest edge is 1 otherwise edges appear too thick on the chart
    # line3: The idea here is to make the strongest edge stay stronger than others, 6 just worked, don't dwell on it

    edge_weights_raw = np.clip(np.log(np.asarray(ig_graph.edge_betweenness()) + 1e-16), a_min=0, a_max=None)
    edge_weights_raw_normalized = edge_weights_raw / np.max(edge_weights_raw)
    edge_weights = [w ** 6 for w in edge_weights_raw_normalized]
    visual_style["edge_width"] = edge_weights

    # A simple heuristic for vertex size. Size ~ (degree / 2) (it gave nice results I tried log and sqrt as well)
    visual_style["vertex_size"] = [deg / 2 for deg in ig_graph.degree()]

    # This is the only part that's Cora specific as Cora has 7 labels
    if dataset_name.lower() == DatasetType.CORA.name.lower():
        visual_style["vertex_color"] = [cora_label_to_color_map[label] for label in node_labels]
    else:
        print('Feel free to add custom color scheme for your specific dataset. Using igraph default coloring.')

    # Set the layout - the way the graph is presented on a 2D chart. Graph drawing is a subfield for itself!
    # I used "Kamada Kawai" a force-directed method, this family of methods are based on physical system simulation.
    # (layout_drl also gave nice results for Cora)
    visual_style["layout"] = ig_graph.layout_kamada_kawai()

    print('Plotting results ... (it may take couple of seconds).')
    ig.plot(ig_graph, **visual_style)


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


def visualize_gcn(binary_name:str, dataset_name:str,
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
        'should_visualize': False,  # don't visualize the dataset
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
        layer_type=name_to_layer_type(model_state['layer_type']),
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
        all_nodes_unnormalized_scores, _ = gcn((node_features, topology))  # shape = (N, num of classes)
        all_nodes_unnormalized_scores = all_nodes_unnormalized_scores.cpu().numpy()

    # edge_index = convert_adj_to_edge_index(topology)
    if visualization_type == VisualizationType.EMBEDDINGS:
        # visualize embeddings (using t-SNE)
        node_labels = node_labels.cpu().numpy()
        tsne_visualize_embeddings(all_nodes_unnormalized_scores, node_labels)



def visualize_graph_sage(binary_name:str, dataset_name:str,
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
        'should_visualize': False,  # don't visualize the dataset
    }

    # Step 1: Prepare the data
    if dataset_name == DatasetType.CORA.name or dataset_name == DatasetType.CITESEER.name:
        node_features, node_labels, topology = load_graph_data(config, device)

    # Step 2: Prepare the model
    model_path = os.path.join(GCN_BINARIES_PATH, binary_name)
    model_state = torch.load(model_path, map_location=device)

    graph_sage = GraphSAGE(
        num_of_layers=model_state['num_of_layers'],
        num_features_per_layer=model_state['num_features_per_layer'],
        add_skip_connection=model_state['add_skip_connection'],
        bias=model_state['bias'],
        dropout=model_state['dropout'],
        layer_type=name_to_layer_type(model_state['layer_type']),
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
        all_nodes_unnormalized_scores, _ = graph_sage((node_features, topology))  # shape = (N, num of classes)
        all_nodes_unnormalized_scores = all_nodes_unnormalized_scores.cpu().numpy()

    # edge_index = convert_adj_to_edge_index(topology)
    if visualization_type == VisualizationType.EMBEDDINGS:
        # visualize embeddings (using t-SNE)
        node_labels = node_labels.cpu().numpy()
        tsne_visualize_embeddings(all_nodes_unnormalized_scores, node_labels)

