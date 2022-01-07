import numpy as np
import igraph as ig

from constants import DatasetType, cora_label_to_color_map
from utils.util import convert_adj_to_edge_index


def visualize_graph(edge_index, node_labels, dataset_name):
    """
    Check out this blog for available graph visualization tools:
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
    edge_weights = [w**6 for w in edge_weights_raw_normalized]
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
