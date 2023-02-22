from generate_graph import gnp_random_connected_graph
import networkx as nx
from math import inf

def bellman_ford_algo(
        graph: nx.Graph,
        start_vertex=0
    ) -> tuple[dict[int, list[int]], dict[int, int]]:
    """Bellman-Ford algorithm. Find the shortest way from selected node to
    every else

    Args:
        graph (nx.Graph): original graph
        start_vertex (int, optional): selected node. Defaults to 0.

    Returns:
        tuple[dict[int, list[int]], dict[int, int]]: 1st dict is the shortest 
            path to current node. 2nd element is lengths of shortest paths
    """
    nodes = graph.nodes
    edges = graph.edges(data=True)

    # initialize lengths and set start to 0
    lengths = {vertex: inf for vertex in nodes}
    lengths[start_vertex] = 0

    # initialize path
    path = {vertex: [] for vertex in nodes}
    path[start_vertex] = [start_vertex]

    # do n - 1 iterations
    for _ in range(len(nodes) - 1):
        # for every edge
        for node1, node2, weight in edges:
            weight = weight['weight']

            # compare current path and path to node1 + path from node1 to node2
            if lengths[node2] > (new_length := lengths[node1] + weight):
                lengths[node2] = new_length
                path[node2] = path[node1] + [node2]

    # control iteration. If smth changes -- there's negative cycle
    for node1, node2, weight in edges:
        weight = weight['weight']

        if lengths[node2] > lengths[node1] + weight:
            return None # because graph have negative cycle

    return path, lengths

if __name__ == "__main__":
    graph = gnp_random_connected_graph(10, 0.5, True, False)
    print( bellman_ford_algo(graph) )