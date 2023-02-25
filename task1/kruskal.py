""" Kruskal's algo """
import networkx as nx
from generate_graph import gnp_random_connected_graph


def kruskal_algo(graph: nx.Graph) -> nx.Graph:
    """make minimum spanning edges by Kruskal's algorithm

    Args:
        graph (nx.Graph): original graph

    Returns:
        nx.Graph: minimum spanning edges
    """
    trees = [set([i]) for i in graph.nodes()]

    # graph represented in (v1, v2, {'weight': w}) and sorted by weight
    graph = sorted(graph.edges(data=True), key=lambda x: x[2]['weight'])

    result = nx.Graph()

    while graph and len(trees) > 1:
        node1, node2, weight = graph.pop(0)
        weight = weight['weight']

        first_tree, second_tree = None, None

        # find in which trees the first and second nodes are
        for tree in trees:
            if node1 in tree:
                first_tree = tree

            if node2 in tree:
                second_tree = tree

            # found 1 and 2 trees
            if first_tree and second_tree:
                break

        # they are in the same trees,
        # so they would do a cycle if we connect them
        if first_tree == second_tree:
            continue

        # add to result
        result.add_edge(node1, node2, weight=weight)

        # extend first tree with the second by reference
        # (it will change anywhere)
        # and delete second tree
        first_tree.update(second_tree)
        del trees[trees.index(second_tree)]

    return result


if __name__ == "__main__":
    random_graph = gnp_random_connected_graph(10, 0.8, False, False)
    print( kruskal_algo(random_graph) )
