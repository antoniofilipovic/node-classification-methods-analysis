from typing import Tuple, List, Dict

from data_parser import blog_catalog
from node2vec import graph, node2vec

edges, node_labels = blog_catalog.get_graph()
edges_weights: Dict[Tuple[int,int], int] = {edge:1 for edge in edges}


graph = graph.GraphHolder(edges_weights=edges_weights, is_directed=False)



embeddings = node2vec.calculate_node_embeddings(graph=graph,
                                   p=2.0,
                                   q=0.5,
                                   num_walks=4,
                                   walk_length=5,
                                   vector_size=100,
                                   alpha=0.025,
                                   window=5,
                                   min_count=1,
                                   seed=1,
                                   workers=1,
                                   min_alpha=0.0001,
                                   sg=1,
                                   hs=0,
                                   negative=5,
                                   epochs=5,
                                   )

for node, embedding in embeddings:
    print(node, embedding)
    break