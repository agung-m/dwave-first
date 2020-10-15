# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 17:04:18 2020

@author: agung
"""

import networkx as nx
import random
import dwave_networkx as dnx
from dwave.system import LeapHybridSampler

problem_node_count = 300
G = nx.random_geometric_graph(problem_node_count, radius=0.0005*problem_node_count)
G.add_edges_from([(u, v, {'sign': 2*random.randint(0, 1)-1}) for u, v in G.edges])

print(G)

sampler = LeapHybridSampler()

imbalance, bicoloring = dnx.structural_imbalance(G, sampler)

set1 = int(sum(list(bicoloring.values())))
print("One set has {} nodes; the other has {} nodes.".format(set1, problem_node_count-set1))  # doctest: +SKIP
print("The network has {} frustrated relationships.".format(len(list(imbalance.keys()))))    # doctest: +SKIP