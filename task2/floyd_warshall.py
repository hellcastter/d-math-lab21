""" Floyd-Warshall algorithm """
from math import inf
import networkx as nx
from generate_graph import gnp_random_connected_graph

def floyd_warshall(graph: nx.Graph) -> list[list[int | float]] | None:
    """Floyd-Warshall algorithm

    Args:
        graph (nx.Graph): original graph

    Returns:
        list[list[int | float]] | None: distances
    """
    # create adjacency matrix
    nodes_count = len(graph.nodes)
    # init matrix with default value inf
    matrix = [[inf] * nodes_count for _ in range(nodes_count)]

    # set all values
    for edge in graph.edges(data=True):
        matrix[edge[0]][edge[1]] = edge[2]['weight']

    # set main diagonal to 0
    for i in range(nodes_count):
        matrix[i][i] = 0

    for k in range(nodes_count):
        for i in range(nodes_count):
            for j in range(nodes_count):
                matrix[i][j] = min(matrix[i][k] + matrix[k][j], matrix[i][j])

            if matrix[i][i] < 0:
                print("Negative cycle detected")
                return None

    return matrix

if __name__ == '__main__':
    graph = gnp_random_connected_graph(10, 0.8, True)
    print( floyd_warshall(graph) )
