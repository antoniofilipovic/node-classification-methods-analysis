import enum
import os


class GATLayerType(enum.Enum):
    IMP1 = 0


GAT_BINARIES_PATH = os.path.join(os.path.dirname(__file__), 'binaries')
GAT_CHECKPOINTS_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints')