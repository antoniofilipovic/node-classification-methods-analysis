from constants import DatasetType, CORA_NUM_INPUT_FEATURES, CITESEER_NUM_INPUT_FEATURES, CORA_NUM_CLASSES, \
    CITESEER_NUM_CLASSES, CORA_NUMBER_OF_TRAIN_EXAMPLES_PER_CLASS, CITESEER_NUMBER_OF_TRAIN_EXAMPLES_PER_CLASS, \
    CORA_TRAIN_RANGE, CORA_VAL_RANGE, CORA_TEST_RANGE, CITESEER_TRAIN_RANGE, CITESEER_VAL_RANGE, CITESEER_TEST_RANGE


def get_num_input_features(dataset_name: str):
    if dataset_name.lower() == DatasetType.CORA.name.lower():
        return CORA_NUM_INPUT_FEATURES
    if dataset_name.lower() == DatasetType.CITESEER.name.lower():
        return CITESEER_NUM_INPUT_FEATURES


def get_num_classes(dataset_name: str):
    if dataset_name.lower() == DatasetType.CORA.name.lower():
        return CORA_NUM_CLASSES
    if dataset_name.lower() == DatasetType.CITESEER.name.lower():
        return CITESEER_NUM_CLASSES


def get_num_training_examples_per_classes(dataset_name: str):
    if dataset_name.lower() == DatasetType.CORA.name.lower():
        return CORA_NUMBER_OF_TRAIN_EXAMPLES_PER_CLASS
    if dataset_name.lower() == DatasetType.CITESEER.name.lower():
        return CITESEER_NUMBER_OF_TRAIN_EXAMPLES_PER_CLASS


def get_train_test_val_ranges(dataset_name: str):
    if dataset_name.lower() == DatasetType.CORA.name.lower():
        return CORA_TRAIN_RANGE[0], CORA_TRAIN_RANGE[1], CORA_VAL_RANGE[0], CORA_VAL_RANGE[1], CORA_TEST_RANGE[0], \
               CORA_TEST_RANGE[1]
    elif dataset_name.lower() == DatasetType.CITESEER.name.lower():
        return CITESEER_TRAIN_RANGE[0], CITESEER_TRAIN_RANGE[1], CITESEER_VAL_RANGE[0], CITESEER_VAL_RANGE[1], \
               CITESEER_TEST_RANGE[0], CITESEER_TEST_RANGE[1]

    else:
        raise Exception(f"Expected {DatasetType.CORA.name} or {DatasetType.CITESEER.name} but got {dataset_name}")
