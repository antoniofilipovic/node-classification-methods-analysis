from gat.constants import GATLayerType


def name_to_layer_type(name: str) -> GATLayerType:
    if name == GATLayerType.IMP1.name:
        return GATLayerType.IMP1
    else:
        raise Exception(f'Name {name} not supported.')
