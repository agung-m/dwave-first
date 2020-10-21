import dimod
import hybrid
from time import perf_counter

# define a qbsolv-like workflow
def merge_substates(_, substates):
    a, b = substates
    return a.updated(subsamples=hybrid.hstack_samplesets(a.subsamples, b.subsamples))


# Construct a problem
#bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': -1, 'ca': 1}, 0, dimod.SPIN)

with open('bqp100_01.qubo') as problem:
   bqm = dimod.BinaryQuadraticModel.from_coo(problem)

subproblem_size = 25

print("BQM size: {}, subproblem size: {}".format(len(bqm), subproblem_size))
# Classical solvers
#subproblem = hybrid.EnergyImpactDecomposer(size=50, rolling_history=0.15)
#subproblem = hybrid.EnergyImpactDecomposer(size=1024, rolling_history=0.15, traversal="bfs")
# Parallel subproblem
subproblem = hybrid.Unwind(
    hybrid.EnergyImpactDecomposer(size=subproblem_size, rolling_history=0.15, traversal="bfs")
)

# QPU
#subsampler = hybrid.QPUSubproblemAutoEmbeddingSampler() | hybrid.SplatComposer()

subsampler = hybrid.Map(
    hybrid.QPUSubproblemAutoEmbeddingSampler()
) | hybrid.Reduce(
    hybrid.Lambda(merge_substates)
) | hybrid.SplatComposer()

# Define the workflow
# iteration = hybrid.RacingBranches(
#     hybrid.InterruptableTabuSampler(),
#     hybrid.EnergyImpactDecomposer(size=2)
#     | hybrid.QPUSubproblemAutoEmbeddingSampler()
#     | hybrid.SplatComposer()
# ) | hybrid.ArgMin()

#iteration = hybrid.RacingBranches(
iteration = hybrid.Race(
    #hybrid.InterruptableTabuSampler(),
    #hybrid.SimulatedAnnealingProblemSampler(),
     subproblem | subsampler
 ) | hybrid.ArgMin()

# iteration = hybrid.Race(
#     hybrid.SimulatedAnnealingProblemSampler(),
#     subproblem | subsampler
# ) | hybrid.ArgMin()

#workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=3)
#workflow = hybrid.Loop(iteration, max_iter=5, convergence=3)
workflow = hybrid.Loop(iteration, max_iter=1)


start_t = perf_counter()
# Solve the problem
init_state = hybrid.State.from_problem(bqm)
final_state = workflow.run(init_state).result()

elapsed_t = perf_counter() - start_t
# Print results
print("Solution: sample={.samples.first}".format(final_state))
print("Elapsed time: {}".format(elapsed_t))
hybrid.profiling.print_counters(workflow)