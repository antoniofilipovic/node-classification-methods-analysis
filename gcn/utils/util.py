from gcn.constants import GCNLayerType


def name_to_layer_type(name:str)->GCNLayerType:
    if name == GCNLayerType.IMP1.name:
        return GCNLayerType.IMP1
    elif name == GCNLayerType.IMP2.name:
        return GCNLayerType.IMP2
    else:
        raise Exception(f'Name {name} not supported.')