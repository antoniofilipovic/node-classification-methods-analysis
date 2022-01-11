import enum


class DatasetType(enum.Enum):
    CORA = 0
    CITESEER = 1


class VisualizationType(enum.Enum):
    EMBEDDINGS = 0


class ModelType(enum.Enum):
    GCN = 0
    NODE2VEC = 1
    NONE = 3


CORA_NUM_INPUT_FEATURES = 1433
CORA_NUM_CLASSES = 7

# Thomas Kipf et al. first used this split in GCN paper and later Petar Veličković et al. in GAT paper
CORA_NUMBER_OF_TRAIN_EXAMPLES_PER_CLASS = 20
CORA_TOTAL_TRAIN_EXAMPLES = CORA_NUMBER_OF_TRAIN_EXAMPLES_PER_CLASS * CORA_NUM_CLASSES

CORA_TRAIN_RANGE = [0, CORA_TOTAL_TRAIN_EXAMPLES]
CORA_VAL_RANGE = [CORA_TOTAL_TRAIN_EXAMPLES, CORA_TOTAL_TRAIN_EXAMPLES + 500]
CORA_TEST_RANGE = [1708, 1708 + 1000]

CITESEER_NUM_INPUT_FEATURES = 3703
CITESEER_NUM_CLASSES = 6
CITESEER_NUMBER_OF_TRAIN_EXAMPLES_PER_CLASS = 20


# 3 different model training/eval phases used in train.py
class LoopPhase(enum.Enum):
    TRAIN = 0,
    VAL = 1,
    TEST = 2


# Used whenever we need to plot points from different class (like t-SNE in playground.py and CORA visualization)
cora_label_to_color_map = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "yellow", 5: "pink", 6: "gray"}
