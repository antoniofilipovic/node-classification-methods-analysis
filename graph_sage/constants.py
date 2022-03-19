import enum
import os


class GraphSAGELayerType(enum.Enum):
    IMP1 = 0


class GraphSAGEAggregatorType(enum.Enum):
    Mean = "mean"
    MaxPool = "max_pool"
    MeanPool = "mean_pool"


GRAPH_SAGE_BINARIES_PATH = os.path.join(os.path.dirname(__file__), 'binaries')
GRAPH_SAGE_CHECKPOINTS_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints')
