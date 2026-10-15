# Design

How a parallel operation runs on the team, and what that means for the code inside it.

- **Chunking.** `TeamScheduler` splits `itr` into `min(nthreads(:default), length(itr))`
  contiguous chunks of nearly equal length (with ChunkSplitters.jl, as OhMyThreads does).
  Chunk 1 runs on the calling task, chunk `c` on a worker. `GreedyTeamScheduler` splits
  into `nchunks` chunks (default `4 * nworkers()`) or chunks of `chunksize` elements; the
  caller and the workers take them from a shared counter until none are left.
- **Stealing.** After chunk 1, the calling task runs every chunk its worker has not picked
  up yet (the worker is asleep, or the OS has not scheduled its thread), so a busy core or
  a sleeping worker does not stall the loop. Concurrent chunks always run on different
  threads, so `threadid()`-indexed scratch buffers are safe as long as the body does not
  yield. Do not assume chunk `c` always runs on the same thread; prefer buffers indexed by
  the task index (`@task_index`, `OhMyThreads.WithTaskIndex`), which is the chunk index.
  With `GreedyTeamScheduler` the shared counter already balances the load: when its own
  share is done, the calling task takes back the workers that have not started, since no
  chunk is left for them. Workers taken back are not woken, so after a serial gap a burst
  of short greedy loops brings the sleeping workers in over a few loops rather than all at
  once.
  `ThreadTeams.stolen_chunks()` counts the chunks taken over (or back) this way.
- **Serial fallback.** The loop runs serially when it would get a single chunk: one thread,
  zero or one element, during precompilation, and nested inside another parallel loop on
  the team. OhMyThreads' own schedulers nested in a team loop are not affected.
- **Thread safety.** Only one parallel region runs on the team at a time. A loop started
  from another task while the team is busy does **not** wait (no lock, no deadlock): it
  runs on `Threads.@spawn`ed tasks instead (chunk 1 on the calling task), with the same
  exception semantics, like `Threads.@threads :dynamic`. Loops nested in those chunks run
  serially while the team is busy; once it is released, the first nested loop to arrive
  takes it. Loops may start from any thread, including the interactive pool; the worker on
  the caller's own thread is then left out.
- **Dynamic scope and world age.** Chunks run in the caller's dynamic scope
  (`ScopedValue`s and the logger set with `with_logger`) and world age, as `Threads.@spawn`
  tasks would. `current_task()` and `task_local_storage()` are those of the worker (or of
  the calling task). Task-local values (`@local`, `TaskLocalValue`) live in the persistent
  worker tasks: they are initialized once per worker and reused across loops.
- **Reductions.** Each chunk is reduced with `mapreduce`, and the chunk results are combined
  in chunk order, also with the greedy scheduler. `op` must be associative, not
  commutative. The result is deterministic for a given number of chunks, but floating-point
  results can differ from a serial loop in the last digits. `init`, if given, is used once
  per chunk, so it must be a neutral element of `op`.
- **Exceptions.** If chunks throw, the exception of the first failing chunk (what a serial
  loop would have thrown) is rethrown after *all* running chunks have finished; started
  chunks run to completion, and if chunk 1 throws, chunks no worker has started are
  skipped. Everything a serial loop would have done before the error is done. The workers
  stay usable. A worker's backtrace stays on the exception stack (`current_exceptions()`,
  "caused by").
- **Lazy start, shutdown.** The workers start with the first parallel loop.
  `ThreadTeams.shutdown!()` stops them; the next loop starts them again.
- **Waking up.** Sleeping workers are woken in a binary tree: whoever takes chunk `c` wakes
  the workers of chunks `2c` and `2c + 1`. The caller wakes at most two workers itself, so
  waking all workers takes O(log n) steps on its critical path instead of n.
- **Allocates once.** The first run of a parallel loop allocates once (the slots that hand
  the loop body to the workers, and the buffer for the partial results of a reduction);
  later runs of the same loop allocate nothing. After a loop, the team holds no reference
  to the loop body, the arrays it captured, the partial results or the chunks' exceptions.
- **Barriers.** `@barrier` (`OhMyThreads.Experimental`) needs a fixed number of concurrent
  tasks. The team does not promise that (see "Stealing"), so the team schedulers reject
  `ntasks`, which `@barrier` requires. Use `DynamicScheduler` for such loops.

## Related work

The mechanism combines known pieces:

- Persistent workers that read the job from a per-thread slot the master writes: GCC's
  libgomp (`thr->fn`, `thr->data` in
  [`team.c`](https://gnu.googlesource.com/gcc/+/trunk/libgomp/team.c)); LLVM's libomp
  keeps its teams alive the same way ("hot teams").
- The binary wake-up tree with each worker spinning on a flag its parent writes: the
  tree barrier of Mellor-Crummey and Scott,
  [*Algorithms for Scalable Synchronization on Shared-Memory Multiprocessors*](https://www.cs.rochester.edu/~scott/papers/1991_TOCS_synch.pdf),
  ACM TOCS 1991.
- Running the chunks of workers that have not started: libomp's `static_steal` schedule
  ([D22134](https://reviews.llvm.org/D22134)), restricted here to one thief.
- Spinning for a bounded time before sleeping: Karlin, Li, Manasse, Owicki,
  [*Empirical Studies of Competitive Spinning for a Shared-Memory Multiprocessor*](https://www.cs.princeton.edu/research/techreps/24),
  1991; `KMP_BLOCKTIME` and `GOMP_SPINCOUNT` in the OpenMP runtimes.
- The sleep handshake (announce, re-check, then block; the waker stores, fences, then
  reads the announcement): as documented for
  [rayon's sleep module](https://github.com/rayon-rs/rayon/blob/main/rayon-core/src/sleep/README.md).
- One cache line per message, polled by its reader: Barrelfish's URPC, Baumann et al.,
  [*The Multikernel*](https://pdos.csail.mit.edu/6.828/2010/readings/barrelfish_sosp09.pdf),
  SOSP 2009.
- Per-thread buffers and spin-then-sleep workers in Julia:
  [ThreadingUtilities.jl](https://github.com/JuliaSIMD/ThreadingUtilities.jl) (Polyester),
  which hands a function pointer and raw argument bytes to each thread.
