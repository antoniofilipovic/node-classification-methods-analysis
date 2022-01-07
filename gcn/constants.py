import enum
import os


class GCNLayerType(enum.Enum):
    IMP1 = 0
    IMP2 = 1


GCN_BINARIES_PATH = os.path.join(os.path.dirname(__file__), 'binaries')
GCN_CHECKPOINTS_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints')
GCN_DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'data')


