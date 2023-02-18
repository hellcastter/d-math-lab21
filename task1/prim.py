"""This module was created to study the operation of Prim's algorithm"""
from generate_graph import gnp_random_connected_graph
import networkx as nx

def prim_algo(graph: nx.Graph, start = 0) -> nx.Graph:
    """
    This function applies Prim's algorithm to the given graph.
    The initial vertex is set as the start.
    Returns the result of Prim's algorithm to
    the given graph and returns it as a list of edges.
    """
    vertices = set([start])
    res_graph = nx.Graph()

    adjacency_matrix = nx.adjacency_matrix(graph, nodelist=None, weight='weight')
    adjacency_matrix = adjacency_matrix.todense()

    for _ in range(graph.number_of_nodes()):
        min_weight, vertex_1, vertex_2 = 0, None, None

        for first_vertex in vertices:
            for second_vertex, edge_weight in enumerate(adjacency_matrix[first_vertex]):
                if (min_weight == 0 or edge_weight < min_weight) and second_vertex not in vertices:
                    min_weight, vertex_1, vertex_2 = edge_weight, first_vertex, second_vertex

        if vertex_2:
            vertices.add(vertex_2)
            res_graph.add_edge(vertex_1, vertex_2, weight=min_weight)

    return res_graph

if __name__ == "__main__":
    graph = gnp_random_connected_graph(10, 0.5, False, False)
    print(prim_algo(graph))
