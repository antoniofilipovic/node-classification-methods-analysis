from graph_sage.constants import GraphSAGELayerType


def name_to_layer_type(name: str) -> GraphSAGELayerType:
    if name == GraphSAGELayerType.IMP1.name:
        return GraphSAGELayerType.IMP1
    else:
        raise Exception(f'Name {name} not supported.')
