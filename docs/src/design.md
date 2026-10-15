# Design

How a parallel operation runs on the team, and what that means for the code inside it.

- **Chunking.** With `TeamScheduler`, `itr` is split into `min(nthreads(:default), length(itr))`
  contiguous chunks of nearly equal length (with ChunkSplitters.jl, as OhMyThreads does).
  Chunk 1 runs on the calling task, chunk `c` on a worker. `GreedyTeamScheduler` splits
  into `nchunks` chunks (default `4 * nworkers()`) or chunks of `chunksize` elements; the
  caller and the workers take them from a shared counter until none are left.
- **Stealing.** When the calling task has finished chunk 1, it runs every chunk that its
  worker has not picked up yet (the worker is asleep, or the OS has not scheduled its
  thread). So a busy core or a sleeping worker does not stall the loop. Chunks that run at
  the same time always run on different threads, so `threadid()`-indexed scratch buffers
  are safe as long as the body does not yield. Do not assume that chunk `c` always runs on
  the same thread. Prefer buffers indexed by the task index (`@task_index`,
  `OhMyThreads.WithTaskIndex`), which is the chunk index. With `GreedyTeamScheduler`, the
  shared counter balances the load already: the calling task takes back the workers that
  have not started when its own share is done, since no chunk is left for them. Workers
  taken back are not woken, so after a serial gap a burst of short greedy loops brings the
  sleeping workers in over a few loops rather than all at once.
  `ThreadTeams.stolen_chunks()` counts the chunks taken over (or back) this way.
- **Serial fallback.** The loop runs serially when it would get a single chunk: one thread,
  zero or one element, during precompilation, and inside another parallel loop on the team
  (nested loops). OhMyThreads' own schedulers nested in a team loop are not affected.
- **Thread safety.** Only one parallel region runs on the team at a time. A loop started
  from another task while the team is busy does **not** wait for the team (no lock, no
  deadlock). It runs its chunks on `Threads.@spawn`ed tasks instead (chunk 1 on the calling
  task), with the same exception semantics, like `Threads.@threads :dynamic` would. Loops
  nested in those chunks run serially while the team is busy; once the other task has
  released it, the first nested loop to arrive takes the team. Loops may be started from
  any thread, including the interactive thread pool. The worker on the caller's own thread
  is then left out, since the caller occupies that thread.
- **Dynamic scope and world age.** Chunks run in the dynamic scope of the caller
  (`ScopedValue`s and the logger set with `with_logger`) and in its world age, as the tasks
  of `Threads.@spawn` would. `current_task()` and `task_local_storage()` are those of the
  worker (or of the calling task). Task-local values (`@local`, `TaskLocalValue`) live in
  the worker tasks, which persist: they are initialized once per worker and reused across
  loops.
- **Reductions.** Each chunk is reduced with `mapreduce`, and the chunk results are combined
  in chunk order, also with the greedy scheduler. So `op` must be associative, but not
  commutative. The result is deterministic for a given number of chunks, but floating-point
  results can differ in the last digits from a serial loop. `init`, if given, is used once
  per chunk, so it must be a neutral element of `op`.
- **Exceptions.** If chunks throw, the exception that a serial loop would have thrown (that of
  the first failing chunk) is rethrown after *all* running chunks have finished. The workers
  stay usable. Chunks that already started run to completion. If chunk 1 throws, chunks that
  no worker has started yet are skipped. Everything a serial loop would have done before the
  error is always done. A worker's backtrace is kept on the exception
  stack (`current_exceptions()`, shown as "caused by").
- **Lazy start, shutdown.** The workers start with the first parallel loop.
  `ThreadTeams.shutdown!()` stops them, and the next loop starts them again.
- **Waking up.** Sleeping workers are woken in a binary tree: whoever takes chunk `c` wakes
  the workers of chunks `2c` and `2c + 1`. The caller wakes at most two workers itself, so
  waking all workers takes O(log n) steps on its critical path instead of n.
- **Allocates once.** The first run of a parallel loop allocates once (the slot that hands
  the loop body to the workers, and the buffer for the partial results of a reduction).
  Later runs of the same loop allocate nothing. After a loop, the team holds no reference to
  the loop body, the arrays it captured, the partial results or the exceptions of the
  chunks.
- **Barriers.** `@barrier` (`OhMyThreads.Experimental`) needs a fixed number of tasks that
  run at the same time. The team does not promise that (see "Stealing"), so the team
  schedulers reject `ntasks`, which `@barrier` requires. Use `DynamicScheduler` for such loops.
