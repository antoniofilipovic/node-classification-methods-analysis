import enum
import os


class GraphSAGELayerType(enum.Enum):
    IMP1 = 0


GRAPH_SAGE_BINARIES_PATH = os.path.join(os.path.dirname(__file__), 'binaries')
GRAPH_SAGE_CHECKPOINTS_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints')