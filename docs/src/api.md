# API

The package exports nothing. The schedulers that run OhMyThreads.jl's `tforeach`,
`tmapreduce`, `tmap`, `@tasks`, ... on the team, `TeamScheduler` and
`GreedyTeamScheduler`, are part of OhMyThreads.jl; see the
[OhMyThreads.jl documentation](https://juliafolds2.github.io/OhMyThreads.jl/stable/).

## Parallel loops

```@docs
ThreadTeams.foreach_chunk
ThreadTeams.foreach_range
ThreadTeams.run_chunks
ThreadTeams.run_fold
ThreadTeams.StaticSchedule
ThreadTeams.GreedySchedule
ThreadTeams.runs_serially
```

## The team

```@docs
ThreadTeams
ThreadTeams.nworkers
ThreadTeams.set_spin_time!
ThreadTeams.spin_time
ThreadTeams.parallel_phase
ThreadTeams.shutdown!
ThreadTeams.ChunkFailedException
```
