import os
from typing import List, Tuple, Dict

WORK_DIRECTORY = os.getcwd()
print(WORK_DIRECTORY)
DATA_DIR = f"{WORK_DIRECTORY}/data"
BLOG_DATA_DIR = f"{DATA_DIR}/blog-catalog"

EDGES_DATASET = "edges.csv"
GROUPS_DATASET = "group-edges.csv"
NODES_DATASET = "nodes.csv"


def get_graph() -> (List[Tuple[int, int]], Dict[int, int]):
    with open(os.path.join(BLOG_DATA_DIR, EDGES_DATASET)) as edges_io:
        edges_csv = edges_io.readlines()

    with open(os.path.join(BLOG_DATA_DIR, GROUPS_DATASET)) as edges_io:
        node_group_csv = edges_io.readlines()

    edges: List[Tuple[int, int]] = []
    for edge in edges_csv:
        from_id, to_id = edge.split(",")
        edges.append((int(from_id), int(to_id)))

    node_group_list = [node_group.split(",") for node_group in node_group_csv]
    nodes_correct_label = {int(node_group[0]): int(node_group[1]) for node_group in node_group_list}

    return edges, nodes_correct_label


if __name__ == "__main__":
    print(get_graph())
