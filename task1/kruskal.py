import networkx as nx
from generate_graph import gnp_random_connected_graph

def kruskal_algo(graph):
    """_summary_

    Args:
        graph (_type_): _description_

    Returns:
        _type_: _description_
    """
    nodesCount = len(graph.nodes())
    trees = [set([i]) for i in graph.nodes()]

    # graph represented in (v1, v2, {'weight': w}) and sorted by weigth
    graph = sorted(graph.edges(data=True), key=lambda x: x[2]['weight'])
    
    result = nx.Graph()
    
    for edge in graph:
        firstTree, secondTree = None, None
        
        # find in which trees the first and second nodes are
        for tree in trees:
            if edge[0] in tree:
                firstTree = tree
            
            if edge[1] in tree:
                secondTree = tree
            
            # found 1 and 2 trees
            if firstTree and secondTree:
                break
        
        # they are in the same trees, so they would do a cycle if we connect them
        if firstTree == secondTree:
            continue
        
        # add to result
        result.add_edge(edge[0], edge[1], weight=edge[2]['weight'])

        # extend first tree with the second by reference (it will change anywhere)
        # and delete second tree
        firstTree.update(secondTree)
        del trees[trees.index(secondTree)]

        # all edges are already in one tree
        if len(trees) == 1:
            break
        
    return result
