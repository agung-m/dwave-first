import dimod
import networkx as nx
import random
import hybrid
#import dwave.inspector

#graph = nx.barabasi_albert_graph(100, 3, seed=1)  # Build a quasi-random graph
graph = nx.barabasi_albert_graph(300, 9, seed=1)
# Set node and edge values for the problem
h = {v: 0.0 for v in graph.nodes}
J = {edge: random.choice([-1, 1]) for edge in graph.edges}
bqm = dimod.BQM(h, J, offset=0, vartype=dimod.SPIN)

# define a qbsolv-like workflow
def merge_substates(_, substates):
    a, b = substates
    return a.updated(subsamples=hybrid.hstack_samplesets(a.subsamples, b.subsamples))

# Redefine the workflow: a rolling decomposition window
#subproblem = hybrid.EnergyImpactDecomposer(size=50, rolling_history=0.15)
subproblem = hybrid.Unwind(hybrid.EnergyImpactDecomposer(size=100, rolling_history=0.15, traversal='bfs'))
# QPU
#subsampler = hybrid.QPUSubproblemAutoEmbeddingSampler() | hybrid.SplatComposer()
qpu_sampler = hybrid.Map(
   hybrid.QPUSubproblemAutoEmbeddingSampler()
) | hybrid.Reduce(
   hybrid.Lambda(merge_substates)
) | hybrid.SplatComposer()

rand_sampler = hybrid.Map(
   hybrid.RandomSubproblemSampler()
) | hybrid.Reduce(
   hybrid.Lambda(merge_substates)
) | hybrid.SplatComposer()

iteration = hybrid.RacingBranches(
   hybrid.InterruptableTabuSampler(),
   subproblem | qpu_sampler
) | hybrid.ArgMin() | hybrid.TrackMin(output=True)

workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=3)

# Convert to dimod sampler and run workflow
result = hybrid.HybridSampler(workflow).sample(bqm)

# show execution profile
#hybrid.profiling.print_counters(workflow)

print("Solution: sample={}".format(result.first)) # doctest: +SKIP
#dwave.inspector.show(result)