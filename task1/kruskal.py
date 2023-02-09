""" Kruskal's algo """
import networkx as nx


def kruskal_algo(graph: nx.Graph) -> nx.Graph:
    """make minimal cut by Kruskal's algorithm

    Args:
        graph (nx.Graph): original graph

    Returns:
        nx.Graph: minimal cut
    """
    trees = [set([i]) for i in graph.nodes()]

    # graph represented in (v1, v2, {'weight': w}) and sorted by weigth
    graph = sorted(graph.edges(data=True), key=lambda x: x[2]['weight'])

    result = nx.Graph()

    while graph and len(trees) > 1:
        edge = graph.pop(0)

        first_tree, second_tree = None, None

        # find in which trees the first and second nodes are
        for tree in trees:
            if edge[0] in tree:
                first_tree = tree

            if edge[1] in tree:
                second_tree = tree

            # found 1 and 2 trees
            if first_tree and second_tree:
                break

        # they are in the same trees,
        # so they would do a cycle if we connect them
        if first_tree == second_tree:
            continue

        # add to result
        result.add_edge(edge[0], edge[1], weight=edge[2]['weight'])

        # extend first tree with the second by reference
        # (it will change anywhere)
        # and delete second tree
        first_tree.update(second_tree)
        del trees[trees.index(second_tree)]

    return result
