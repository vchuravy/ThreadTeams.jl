# Benchmark sweep

`run.sh` measures, on one machine and for a range of thread counts, the numbers reported in
[#46](https://github.com/vchuravy/ThreadTeams.jl/issues/46): the per-loop overhead of the
team against `Threads.@threads`, OhMyThreads, Polyester and KernelAbstractions; the EPCC
syncbench overheads of the same mechanisms next to those of C OpenMP (libgomp and libomp);
the wake-up latency after a serial gap; hardware counters per loop (LinuxPerf.jl); and the
cost of stealing. `summarize.jl` turns the results of any number of machines into tables.

```sh
git clone https://github.com/vchuravy/ThreadTeams.jl && cd ThreadTeams.jl
benchmark/sweep/run.sh        # 1–2 minutes per thread count and Julia version
julia benchmark/sweep/summarize.jl benchmark/sweep/results/*/results.csv
```

A first look takes `QUICK=1` (shorter runs) and `THREADS="2 4"`. The settings are
listed at the top of `run.sh`; the most useful:

| variable | meaning | default |
|---|---|---|
| `JULIA_CHANNELS` | juliaup channels to compare, e.g. `"1.12 1.13"` | the `julia` on the `PATH` |
| `JULIA` | Julia command, e.g. `/opt/julia-1.12/bin/julia` | `julia` |
| `THREADS` | thread counts | 2, 4, 8, … and the number of cores |
| `BENCHES` | subset of `overhead syncbench epcc wakeup counters steal` | all |
| `REPEAT` | rounds over the whole sweep, to see the spread between processes | 1 |
| `CPUS` | CPUs in the order to use them | one per physical core |
| `INSTALL_JULIA=1` | install juliaup if there is no `julia` | |

## What it does

- **Placement.** Each run uses one hardware thread per physical core: the first `n` CPUs of
  `CPUS` (from `lscpu`, restricted to the CPUs the shell may use), with `taskset`, and every
  Julia thread pinned to its own CPU (`pin.jl`, loaded with `julia -L`). OpenMP gets the
  same CPUs through `OMP_PLACES` and `OMP_PROC_BIND=close`. Julia runs with `-t n,0`. On
  macOS nothing is pinned.
- **Isolation.** Each benchmark runs in a process of its own (`wakeup` and `steal` one per
  mechanism), under a lock file shared by all users of the script on the host (`LOCK`).
  The one-minute load average before each run is recorded with the results: on a shared
  machine, look at it before the numbers.
- **Code alignment.** Julia compiles with `-align-all-nofallthru-blocks=6`
  (`CODE_ALIGN=0` turns it off), so that where the hot loops fall relative to the
  64-byte fetch blocks does not differ between builds and processes. Package images
  leave LLVM options out of their cache key, so the aligned runs use a depot of their own
  (`.cache/<host>/depot-aligned`), compiled once per host and Julia version (a few
  minutes); juliaup still finds its Julia installations in the usual depot.
- **Environment.** `benchmark/sweep/Project.toml` is the environment of
  `benchmark/overhead` plus Polyester; each Julia version gets its own
  `Manifest-vX.Y.toml`.

## The benchmarks

| name | script | measures |
|---|---|---|
| `overhead` | `../overhead/overhead.jl` | min / median time of a loop of 64 trivial iterations, all mechanisms |
| `syncbench` | `syncbench.jl` | EPCC's PARALLEL FOR and REDUCTION overheads (0.1 µs per iteration, one iteration per thread), ported to Julia |
| `epcc` | EPCC syncbench v4.0 in C | the same, and BARRIER, PARALLEL etc., for each OpenMP runtime, by default and with `OMP_WAIT_POLICY=active` |
| `wakeup` | `wakeup.jl` | the loop after a busy gap of 0–10 ms, and for the team with `set_spin_time!(0)` |
| `counters` | `../overhead/counters.jl` | hardware and software counters per loop (cycles, instructions, cache misses, context switches), on the calling thread and over all threads |
| `steal` | `../overhead/steal.jl` | time per loop and share of stolen chunks for chunks of 0–10 µs |

EPCC is fetched at a fixed commit and built with each of `EPCC_CC` (default `gcc clang`)
that compiles OpenMP, with EPCC's own flags; the runtime name in the results is the one
the binary links. `counters` needs Linux with `kernel.perf_event_paranoid` at most 2;
`COUNTERS_EVENTS` selects other events (LinuxPerf's syntax: generic `perf list` names,
grouped in parentheses).

## Results

`benchmark/sweep/results/<host>-<date>/` holds `machine.txt` (CPU, topology, OS, compilers,
Julia `versioninfo`, governor, load), `logs/` (the output of every run) and
`results.csv` with the columns `host, julia, threads, config, bench, mechanism, metric,
value` (`config` is the round). `summarize.jl` prints one Markdown table per benchmark and
metric, a row per host, Julia version and thread count, and a column per mechanism; cells
with several values (rounds, files) show their range:

```sh
julia benchmark/sweep/summarize.jl results/*/results.csv                    # default tables
julia benchmark/sweep/summarize.jl --bench overhead --metric min_us --mechanism team results/*/results.csv
```

## Several machines

`remote.sh` copies this checkout (with uncommitted changes and untracked, not ignored files) to
`~/threadteams-sweep` on each ssh host, starts `run.sh` there in the background, and
collects the results. The settings above are passed on from the local shell. The hosts'
login shell must be POSIX (`sh`, `bash`, `zsh`).

```sh
JULIA_CHANNELS="1.12 1.13" INSTALL_JULIA=1 benchmark/sweep/remote.sh start host1 host2 host3
benchmark/sweep/remote.sh status host1 host2 host3
benchmark/sweep/remote.sh fetch  host1 host2 host3
julia benchmark/sweep/summarize.jl benchmark/sweep/results/*/results.csv
```

juliaup provides Julia for x86_64 and aarch64 Linux and macOS; on other machines
(ppc64le, riscv64) set `JULIA` to a Julia built or unpacked there. On shared
machines, run `REPEAT=3` or more, and compare the recorded load with the spread.
