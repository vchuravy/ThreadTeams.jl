"""
    ThreadTeams

A pool of persistent ("parked") worker tasks that runs the chunks of parallel loops with
low per-loop overhead, and the scheduling of chunks on it. One sticky worker task runs per
thread of the `:default` thread pool. After a loop, idle workers spin for a short while
(see [`ThreadTeams.set_spin_time!`](@ref)) before they sleep, so back-to-back loops pay
for no task creation, scheduling or wake-up.

The package exports nothing. Parallel loops:

- [`ThreadTeams.foreach_chunk`](@ref): a loop over chunk indices, for callers that split
  the work themselves;
- [`ThreadTeams.foreach_range`](@ref): a loop over ranges of indices with the placement of
  `Threads.@threads :static`, optionally with work stealing;
- [`ThreadTeams.run_chunks`](@ref) and [`ThreadTeams.run_fold`](@ref) with a
  [`ThreadTeams.StaticSchedule`](@ref) or [`ThreadTeams.GreedySchedule`](@ref): the
  building blocks of the `TeamScheduler` and `GreedyTeamScheduler` of OhMyThreads.jl,
  which runs `tforeach`, `tmapreduce`, `tmap`, `@tasks`, ... on the pool.

Pool controls: [`ThreadTeams.nworkers`](@ref), [`ThreadTeams.set_spin_time!`](@ref),
[`ThreadTeams.spin_time`](@ref), [`ThreadTeams.parallel_phase`](@ref),
[`ThreadTeams.shutdown!`](@ref).

With Enzyme.jl loaded, parallel operations on the team can be differentiated in forward
and reverse mode (a package extension; see the README).
"""
module ThreadTeams

import ScopedValues

@static if VERSION >= v"1.11"
    eval(
        Meta.parse(
            "public nworkers, foreach_chunk, foreach_range, run_chunks, run_fold, " *
                "StaticSchedule, GreedySchedule, runs_serially, set_spin_time!, " *
                "spin_time, parallel_phase, shutdown!, ChunkFailedException"
        )
    )
end

include("topology.jl")
include("pool.jl")
include("ranges.jl")
include("precompile.jl")

end # module
