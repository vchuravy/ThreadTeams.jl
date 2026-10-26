# API

Everything that OhMyThreads.jl exports (`tforeach`, `tmapreduce`, `treduce`, `tmap`,
`tmap!`, `tcollect`, `@tasks`, `@set`, `@local`, `@task_index`, `@allow_boxed_captures`,
`chunks`, `index_chunks`, the other schedulers, ...) is re-exported; see the
[OhMyThreads.jl documentation](https://juliafolds2.github.io/OhMyThreads.jl/stable/).

## Schedulers

```@docs
TeamScheduler
GreedyTeamScheduler
```

## The team

```@docs
ThreadTeams
ThreadTeams.nworkers
ThreadTeams.nchunks
ThreadTeams.foreach_chunk
ThreadTeams.foreach_range
ThreadTeams.set_spin_time!
ThreadTeams.spin_time
ThreadTeams.parallel_phase
ThreadTeams.shutdown!
ThreadTeams.ChunkFailedException
```
