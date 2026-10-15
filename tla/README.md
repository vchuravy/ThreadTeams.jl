# The hand-off protocol in TLA+

`HandOff.tla` models one parallel region of the team (src/pool.jl: `dispatch!`,
`wait_for_command`, `worker_loop`, `wake_children`, `wait_idle`) under a weak memory model,
with the ordering of every atomic access as a parameter. `check.sh` runs TLC on one
configuration. The model served to review the orderings and weaken those that are not
load-bearing.

## The model

**Threads.** The dispatcher (starts the region, runs chunk 1, steals unclaimed chunks) and
`N` workers; worker `k` gets chunk `k + 1`. Workers start spinning or asleep, in every
combination, and may go to sleep whenever they have read a non-positive command (the real
worker does after the spin time).

**Memory.** A view-based operational model of release/acquire consistency with sc fences
(the promise-free fragment of the promising semantics, Kang et al. 2017). Every location
keeps its list of writes; every thread keeps a view, the newest write of each location it
knows. A load returns any write at or after the view, so stale values are read unless
something orders the access. A release store attaches the writer's view to the write, an
acquire load joins it into the reader's view, a read-modify-write reads the newest write
and carries its view (release sequences), and an sc fence joins the thread's view with a
global one and stores the result back.

Sequentially consistent *accesses* lie between two encodings (constant `SCENC`):

- `fence`: sc store = release store + sc fence; sc load = acquire load; sc RMW = fence +
  acq_rel RMW + fence. How x86 implements them (`xchg`, `mov`, `lock cmpxchg`); a correct
  implementation of C11/Julia sc accesses, so it has at most their behaviours.
- `weak`: sc accesses are plain release/acquire accesses: more behaviours than C11.

A property that holds under `weak` holds under C11 on any hardware; one that fails under
`fence` fails on x86. The explicit `Threads.atomic_fence()` is an sc fence in both.

**Abstractions.** The region data (the closure `ws.job`, the workers' slots, the scope; the
number of chunks and the skipped worker travel in the command word) is one location written
before the commands; a worker's results (`errs[k]`, the partials buffer) are one location
written before its final `cmd = 0`. The worker's `Event` is a flag that carries the
notifier's view (its lock is a release/acquire pair). Loops that re-read a location get a
fairness assumption: a store becomes visible eventually.

**Properties.**

| property | meaning |
|---|---|
| `ExactlyOnce`, `AllRun` | every chunk runs once: the worker's claim (`cmd: chunk => RUNNING`) and the dispatcher's steal (`cmd: chunk => 0`) exclude each other |
| `FreshRegion` | a worker that runs a chunk sees the region data of this region (not of an earlier one) |
| `ResultsVisible` | after `wait_idle`, the dispatcher sees what the worker wrote while running the chunk |
| `NoWorkerLeftBehind` | when the region ends, no worker is still running a chunk of it |
| `WakeupDelivered` | the sleep handshake: a worker that goes to sleep after its command was stored either sees the command and does not wait, or its waker sees that it sleeps and notifies it. Checked without the re-notify of `wait_idle` (`BACKUP = FALSE`) |
| `Terminates` | the region ends (liveness) |

## Results

`N = 1`: the dispatcher hands the chunk to the worker and wakes it. `N = 2` with
`TREE = "chain"`: worker 1 wakes worker 2, the case of the binary tree in which a worker
wakes another (`chain` needs two workers for that instead of three; the binary tree with
two workers is the `N = 1` case twice). Only the woken worker sleeps (`SLEEPERS`), at most
once (`MAXSLEEP`): 10^3 to 10^5 states, against 10^8 with two sleeping workers.

"current" is the orderings before this review, "weakened" the ones the code has now:
relaxed loads in the worker's spin loop and the dispatcher's steal loop, a relaxed steal
compare-and-swap and claim failure ordering, a relaxed `sleeping = false` and re-notify
load in `wait_idle`.

| configuration | N = 1 | N = 2, chain |
|---|---|---|
| current, `fence` | all hold | all hold |
| current, `weak` | `WakeupDelivered` fails (pure release/acquire has no Dekker), rest holds | same |
| weakened, `fence` | all hold | all hold |
| weakened, `weak`, with the re-notify | all hold | all hold |
| command stores relaxed | `FreshRegion` fails | `FreshRegion` fails |
| `cmd = 0` relaxed, or the load of `wait_idle` relaxed | `ResultsVisible` fails | `ResultsVisible` fails |
| claim relaxed | `FreshRegion` fails | `WakeupDelivered` fails |
| claim acquire | all hold | `WakeupDelivered` fails: the parent's loads of the child's `sleeping` need the full barrier of the claim |
| no fence after the command stores | `WakeupDelivered` fails | `WakeupDelivered` fails |
| `sleeping = true` release instead of sc | `WakeupDelivered` fails | `WakeupDelivered` fails |
| the load of `cmd` after `sleeping = true` relaxed | all hold (x86: the fence after the store orders it) | all hold |
| the loads of `wake_chunk` relaxed | all hold (x86) | all hold (x86) |

`Terminates` holds in every configuration, lost wake-ups included: the dispatcher runs
every chunk its worker has not claimed, so a worker that sleeps through its command only
slows the region.

The last two rows hold under the x86 encoding but this model cannot certify them for C11,
whose sc rules lie between `weak` and `fence`. Under C11 a relaxed load after an sc fence
sees the last sc store before the fence (so the dispatcher's wake loads could be relaxed),
but a worker waking a child has no fence, only its sc claim, and there C11 allows a lost
wake-up even with sc loads. The code keeps these accesses sc: free on x86 (every load is a
plain `mov`), one `ldar` instead of `ldr` on AArch64, where the RCsc ordering of
`stlr`/`ldar` and `casal` makes the handshake work in practice.

## The orderings of src/pool.jl

| access | ordering | why |
|---|---|---|
| `ws.job`, the workers' slots, `pool.scope` stores and loads | monotonic | published by the release store of the command, acquired by the worker's claim |
| `ws.cmd = chunk` (`dispatch!`) | release | publishes the region data; seven relaxed-ordered release stores overlap in the store buffer, where sc stores would each wait |
| `Threads.atomic_fence()` after them | sc | the dispatcher's side of the sleep handshake |
| `sleeping`, `cmd` loads in `wake_chunk` | sc | the waker's side of the handshake (see above) |
| `ws.cmd` load in the spin loop | monotonic | the claim acquires |
| `cmd: chunk => RUNNING` (claim) | sc, failure monotonic | acquires the region data; its full barrier orders the parent's wake loads |
| `sleeping = true`; load of `cmd` | sc; sc | the worker's side of the handshake |
| `sleeping = false` | monotonic | a stale `true` costs one spurious notification |
| `ws.cmd = 0` after a chunk | release | publishes the results and exceptions |
| `ws.cmd` load in `wait_idle` | acquire | acquires them |
| `sleeping` load in `wait_idle` (re-notify) | monotonic | fallback only |
| `ws.cmd` load and `cmd: chunk => 0` in the steal loop | monotonic | nothing of the worker's is read after a steal |
| `POOL.busy` compare-and-swap / store | acquire (failure monotonic) / release | a lock |
| `POOL.owner` | monotonic | only compared with the current task |
| `POOL.pool` | acquire / release | publishes the new pool |
| `counter.next` (greedy) | monotonic | only claims indices; results go through `cmd = 0` |

On x86 every load is a plain `mov` and every compare-and-swap a `lock cmpxchg` whatever
the ordering, release and monotonic stores are plain stores, an sc store is an `xchg`: the
weakening changes no instruction there. On AArch64 it turns `ldar` into `ldr` in the two
spin loops and `casal` into `cas` in the steal loop.

## Running

Needs Java 11 or newer and `tla2tools.jar` from
https://github.com/tlaplus/tlaplus/releases (put it in this directory or set
`TLA2TOOLS`):

```sh
tla/check.sh current N=1 BACKUP=FALSE
tla/check.sh chain N=2 'SLEEPERS={2}' BACKUP=FALSE
tla/check.sh relaxed-claim N=1 BACKUP=FALSE OrdClaim=rlx     # fails FreshRegion
```

`check.sh NAME KEY=VALUE...` writes `out/NAME/HandOff.cfg` with the orderings of the code
and the given overrides (`N`, `TREE`, `SLEEPERS`, `MAXSLEEP`, `BACKUP`, `SCENC`,
`LIVE=0` to skip the liveness property, and `OrdCmd`, `OrdClaim`, ... for the accesses),
runs TLC and prints the verdict; `out/NAME/tlc.log` has the counterexample of a failure.
