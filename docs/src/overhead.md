# Overhead

Per-loop time for a loop with 64 trivial iterations (`x[i] += 1.0`), `benchmark/overhead/overhead.jl`,
pinned to 8 cores (`taskset -c 8-15`) of a Ryzen 9 5950X, Julia 1.12.7 and 1.13.1, `OPENBLAS_NUM_THREADS=1`. Polyester.jl 0.7.18 was
loaded from a separate environment for comparison (\* on 1.13 from an earlier run, with
0.7.19); OhMyThreads.jl 0.8.7 and KernelAbstractions.jl 0.10.0 (`foreach_index` on the
`CPU` backend, followed by `synchronize`) come from the benchmark environment. Values are
min / median in µs.

**These numbers come from one shared machine, with other jobs running on the same cores.
Minima are more reliable than medians.**

| mechanism | 1.12 `-t 8` (8+1 interactive) | 1.12 `-t 8,0` | 1.13 `-t 8` | 1.13 `-t 8,0` | allocations per loop |
|---|---|---|---|---|---|
| `tforeach`, `scheduler = TeamScheduler()` | 0.26 / 0.31 | 0.32 / 0.38 | 0.30 / 0.34 | 0.27 / 0.34 | 0 |
| `tmapreduce`, `scheduler = TeamScheduler()` | 0.30 / 0.35 | 0.33 / 0.38 | 0.31 / 0.37 | 0.30 / 0.37 | 0 |
| `@tasks` with `@set scheduler = TeamScheduler()` | 0.29 / 0.33 | 0.30 / 0.37 | 0.29 / 0.35 | 0.28 / 0.34 | 0 |
| `tforeach`, `scheduler = GreedyTeamScheduler()` (32 chunks) | 0.90 / 1.33 | 0.89 / 1.45 | 0.76 / 0.91 | 0.73 / 0.91 | 0 |
| Polyester `@batch` (v0.7.18) | 0.23 / 0.37 | 0.26 / 0.38 | 0.23 / 0.35* | 0.24 / 0.37* | 0 |
| `Threads.@threads :static` | 11.57 / **113** | 1.86 / 2.77 | 10.92 / **114** | 1.65 / 2.67 | 42 (2.9 KB) |
| `Threads.@threads :dynamic` | 2.46 / 3.94 | 2.52 / 3.88 | 2.36 / 3.80 | 2.27 / 3.64 | 42 (2.9 KB) |
| OhMyThreads `tforeach` (v0.8.7, `DynamicScheduler`) | 4.22 / 5.35 | 4.31 / 5.27 | 3.58 / 4.62 | 3.53 / 4.45 | 66–68 (3.6 KB) |
| OhMyThreads `tforeach`, `scheduler = :static` | 17.87 / **118** | 4.63 / 5.30 | 11.09 / **115** | 4.23 / 4.64 | 67–69 (3.6 KB) |
| OhMyThreads `tmapreduce(+)` | 3.98 / 5.04 | 4.10 / 5.03 | 3.52 / 4.39 | 3.34 / 4.26 | 64–66 (3.6 KB) |
| KernelAbstractions `foreach_index` (v0.10.0) | 3.24 / 4.61 | 3.17 / 4.48 | 3.13 / 4.20 | 3.08 / 4.38 | 5 (96 B) |
| serial loop | 0.02 / 0.02 | 0.02 / 0.02 | 0.02 / 0.02 | 0.02 / 0.02 | 0 |

- Pass the scheduler as an object. A symbol (`scheduler = :team`, `@set scheduler = :team`)
  is resolved by OhMyThreads with a dynamic dispatch, which costs about 0.3 µs and 64 bytes
  per loop.
- `GreedyTeamScheduler()` splits the 64 iterations into 32 chunks (4 per thread) that are
  handed out one by one, hence the higher cost; with `chunksize` or `nchunks` set to match
  the loop, it costs about as much as the static schedule.
- `-t 8` is the default since Julia 1.12: 8 default threads **plus one interactive thread**,
  so 9 threads share 8 cores. `Threads.@threads :static` then pays about the scheduler's
  sleep threshold (100 µs) per loop. Its sticky tasks must wait until the OS deschedules a
  thread that is spinning in the scheduler. `:dynamic` is not affected, and neither are the
  team and Polyester. The worker tasks already run, and the caller (on the interactive
  thread) runs chunk 1 while one default thread stays unused.
- OhMyThreads' `:static` scheduler also uses sticky tasks and has the same problem with `-t 8`.
- KernelAbstractions 0.10 runs `CPU` kernels through POCL (`CPU()` is a `POCLBackend()`), so
  `foreach_index` compiles the body as a kernel. A launch costs about as much as a `@threads`
  loop, but it allocates much less.
- With `-t 8,0` (no interactive thread) `@threads :static` costs about 2–3 µs.
- Waking up workers that have gone to sleep costs about 10–15 µs, for the team and for
  `@threads` alike. Raise the spin time when loops are separated by longer serial phases.

## Where the time goes

Hardware counters per loop on the calling thread (`benchmark/overhead/counters.jl`, LinuxPerf.jl,
same loop and machine as above, Julia 1.12, `-t 8`):

| | cycles | instructions | L1d misses | last-level cache misses |
|---|---|---|---|---|
| `tforeach`, `TeamScheduler()` | 1460–1590 | 1690–1710 | 20–24 | 11 |
| `tmapreduce`, `TeamScheduler()` | 1520–1570 | 1750–1760 | 23–29 | 14–15 |
| Polyester `@batch` | 1620–1670 | 500 | 35–36 | 16 |
| serial loop | 16 | 70 | 0 | 0 |

(Ranges over three runs; the spread comes from other jobs on the machine.)

The cache misses are the hand-off: every worker's command field is a cache line that moves
between the calling thread's core and the worker's core, twice per loop (the worker's claim
and its completion). Timestamps at the stages of a loop (8 threads, 64 iterations, median
nanoseconds; the timers themselves add about 20 ns per stage) show where the calling
thread's time goes:

| stage of the calling thread | ns |
|---|---|
| OhMyThreads' entry, `run_fold`, fold of the partials, release | 150 |
| publishing the region and storing the seven commands | 90 |
| chunk 1 (`mapreduce` over 8 elements) | 30 |
| checking which chunks the workers have not taken (seven loads) | 140 |
| waiting for the last worker | 20–100 |
| clearing the slot, the scope and the exceptions | 20 |

A worker claims its chunk within about 20 ns of the stores and is done about 100–150 ns
later, of which the chunk itself is about 60 ns. The rest of its time is the wake-up check
for its two children in the binary tree and the call of the chunk closure (see `job_closure` in `src/pool.jl`).

Four details keep the hand-off at that level:

- The command stores are release stores followed by one fence rather than sequentially
  consistent stores: an `xchg` per store would wait for each cache line in turn, where
  plain stores overlap in the store buffer.
- The check for untaken chunks loads before it compare-and-swaps: a failed
  compare-and-swap would still take the line away from the worker, which then has to take
  it back.
- Each worker's `sleeping` flag sits on its own cache line, apart from its command. The
  flag is written only when the worker goes to sleep or wakes up, so the two flag loads of
  every wake-up check hit a line that stays shared between all cores, instead of the line
  the worker is claiming its chunk on. Before this, the wake-up checks cost the calling
  thread 70 ns and every worker 50–120 ns per loop.
- The slot that hands the chunk function to the workers and the buffer for the partial
  results are constants embedded by `@generated` functions, one per type, instead of
  entries of an `IdDict`: the two lookups cost about 50 ns per loop.

The remaining difference to Polyester is instruction count on the calling thread
(OhMyThreads' entry with its keyword arguments and scheduler object, the job struct,
`mapreduce` over the first chunk): about three times Polyester's at a similar cycle count.
The exception handlers were also reduced to one per loop on the calling thread and one for
the lifetime of each worker; entering the caller's dynamic scope installs a handler only
when the caller has a scope at all (a `ScopedValue` or logger in effect).
