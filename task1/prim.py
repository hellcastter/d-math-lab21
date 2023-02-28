""" This module was created to study the operation of Prim's algorithm """
from heapq import heappush, heappop
import networkx as nx
from generate_graph import gnp_random_connected_graph

def prim_algo(graph: nx.Graph, start=0) -> nx.Graph:
    """This function applies Prim's algorithm to the given graph.

    Args:
        graph (nx.Graph): original graph
        start (int, optional): node where to start. Defaults to 0.

    Returns:
        nx.Graph: minimum spanning edges
    """
    visited = {start}
    tree = nx.Graph()

    # heapq -- data structure that always looks like SORTED list
    # and the first item is always the least
    heapq = []

    # initialize. add data to heapq
    for node, weight in graph.adj[start].items():
        weight = weight['weight']
        heappush(heapq, (weight, start, node))

    # while there's edges in heapq or not all nodes are in tree
    while heapq and len(visited) != graph.number_of_nodes():
        # get least edge
        weight, node1, node2 = heappop(heapq)

        # this node is already in tree
        if node2 in visited:
            continue

        tree.add_edge(node1, node2, weight=weight)
        visited.add(node2)

        # add all edges from node to N to heapq
        for node, weight in graph.adj[node2].items():
            if node in visited:
                continue

            weight = weight['weight']
            heappush(heapq, (weight, start, node))

    return tree

if __name__ == "__main__":
    random_graph = gnp_random_connected_graph(10, 0.8, False, False)
    print( prim_algo(random_graph) )
