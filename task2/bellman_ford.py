from generate_graph import gnp_random_connected_graph
import numpy as np
import networkx as nx
def bellman_ford_algo(graph: nx.Graph, start_vertex=0) -> dict:
    """
    Bellman-Ford algorithm
    """
    nodes = graph.nodes
    edges_tup = graph.edges(data=True)
    edges = {}
    for first, second, weight in edges_tup:
        edges[(first, second)]=weight['weight']
    path_leng = {vrtx: np.inf for vrtx in nodes}
    path_leng[start_vertex] = 0
    path = {vrtx: [] for vrtx in nodes}
    path[start_vertex]=[start_vertex]
    _ = len(nodes) #iterator
    while _ >=1:
        for (first_v, second_v), weitgh in edges.items():
            if path_leng[second_v] > path_leng[first_v] + weitgh:
                path_leng[second_v] = path_leng[first_v] + weitgh
                path[second_v] = path[first_v] + [second_v]
        _-=1
    for (first_v, second_v), weitgh in edges.items():
        if path_leng[second_v] > path_leng[first_v] + weitgh:
            return None #becouse graph have negative cyrcle
    return path_leng, path

graph = gnp_random_connected_graph(10, 0.5, True, False)
print(bellman_ford_algo(graph))
