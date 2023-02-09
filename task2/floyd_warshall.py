""" Floyd–Warshall algorithm """
from pprint import pprint
from math import inf
from copy import deepcopy
import networkx as nx
from generate_graph import gnp_random_connected_graph


def floyd_warshall(graph: nx.Graph):
    # create adjactency matrix
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
        for i, row in enumerate(matrix):
            for j, weight in enumerate(row):
                matrix[i][j] = min(matrix[i][k] + matrix[k][j], matrix[i][j])

            if matrix[i][i] < 0:
                print("Negative cycle detected")
                return

    return matrix


    # pprint(matrix)

G = nx.DiGraph()
G.add_nodes_from(range(3))
G.add_edge(0, 1, weight=-2)
G.add_edge(0, 2, weight=3)
G.add_edge(0, 3, weight=-3)
G.add_edge(1, 2, weight=2)
G.add_edge(2, 3, weight=-3)
G.add_edge(3, 2, weight=5)
G.add_edge(3, 1, weight=5)
G.add_edge(3, 0, weight=4)

# a = gnp_random_connected_graph(10, 1, True, False)
print(floyd_warshall(G))
