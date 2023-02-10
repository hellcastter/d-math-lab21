"""This module was created to study the operation of Prim's algorithm"""
from generate_graph import gnp_random_connected_graph
import networkx as nx

def prim_algo(graph: nx.Graph, start: int):
    """
    This function applies Prim's algorithm to the given graph.
    The initial vertex is set as the start.
    Returns the result of Prim's algorithm to
    the given graph and returns it as a list of edges.
    """
    vertices, res_graph = [start], []
    for _ in range(len(graph.todense())-1):
        min_weight, vertex_1, vertex_2 = 0, None, None
        for first_vertex in vertices:
            for second_vertex, edge_weight in enumerate(graph.todense()[first_vertex]):
                if (min_weight == 0 or edge_weight < min_weight) and\
                    edge_weight != 0 and second_vertex not in vertices\
                    and (first_vertex, second_vertex) not in res_graph:
                    min_weight, vertex_1, vertex_2 = edge_weight, first_vertex, second_vertex
        vertices.append(vertex_2)
        res_graph.append((vertex_1, vertex_2))
    return res_graph

graph = gnp_random_connected_graph(10, 0.5, False, False)
graph = nx.adjacency_matrix(graph, nodelist=None, weight='weight')
print(prim_algo(graph, 0))
