"""Ford–Warshall algorithm"""
from math import inf
import networkx as nx

def ford_warshall(graph: nx.Graph) -> list[list[int | float]] | None:
    