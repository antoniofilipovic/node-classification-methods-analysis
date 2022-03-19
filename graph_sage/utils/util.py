from graph_sage.constants import GraphSAGELayerType, GraphSAGEAggregatorType


def name_to_layer_type(name: str) -> GraphSAGELayerType:
    if name == GraphSAGELayerType.IMP1.name:
        return GraphSAGELayerType.IMP1
    else:
        raise Exception(f'Name {name} not supported.')


def name_to_agg_type(name: str) -> GraphSAGEAggregatorType:
    if name == GraphSAGEAggregatorType.GCN.name:
        return GraphSAGEAggregatorType.GCN
    elif name == GraphSAGEAggregatorType.Mean.name:
        return GraphSAGEAggregatorType.Mean
    elif name == GraphSAGEAggregatorType.MeanPool.name:
        return GraphSAGEAggregatorType.MeanPool
    elif name == GraphSAGEAggregatorType.MaxPool.name:
        return GraphSAGEAggregatorType.MaxPool
    else:
        raise Exception(f'Name {name} not supported.')
