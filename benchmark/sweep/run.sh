#!/usr/bin/env bash
# Benchmark sweep of ThreadTeams against the alternatives, for one machine: the numbers of
# https://github.com/vchuravy/ThreadTeams.jl/issues/46 for a range of thread counts and
# Julia versions, with EPCC's OpenMP syncbench (libgomp and libomp) as the baseline.
#
#     benchmark/sweep/run.sh                    # everything, with the `julia` on the PATH
#     JULIA_CHANNELS="1.12 1.13" QUICK=1 benchmark/sweep/run.sh
#     julia benchmark/sweep/summarize.jl benchmark/sweep/results/*/results.csv
#
# Each run uses one hardware thread per physical core (`taskset`, and each Julia thread is
# pinned to its own CPU, see pin.jl), `-t N,0`, and the first N cores in CPU order. The
# results go to `benchmark/sweep/results/<host>-<date>/`: `results.csv` (one row per
# number), `machine.txt` (CPU, OS, compilers, load) and the output of every run in `logs/`.
#
# Settings (environment variables):
#   JULIA           Julia command (default `julia`); with `JULIA_CHANNELS`, a juliaup launcher
#   JULIA_CHANNELS  juliaup channels to sweep, e.g. "1.12 1.13" (runs `$JULIA +<channel>`)
#   INSTALL_JULIA=1 install juliaup (and the channels) if there is no `julia`
#   THREADS         thread counts, e.g. "2 4 8" (default: 2, 4, 8, ... and the number of cores)
#   CPUS            CPUs to use in order, e.g. "0,2,4,6" (default: one per physical core)
#   BENCHES         subset of "overhead syncbench epcc wakeup counters steal" (default: all)
#   REPEAT          rounds over the whole sweep, interleaved (default 1)
#   QUICK=1         shorter runs, for a first look
#   OUT             results directory (default `benchmark/sweep/results/<host>-<date>`)
#   LOCK            lock file shared by the benchmarks on this host (default
#                   /tmp/threadteams-bench.lock, or the directory ${LOCK}dir without flock;
#                   empty for none)
#   EPCC_CC         C compilers for EPCC (default "gcc clang": libgomp and libomp)
#   COUNTERS_EVENTS LinuxPerf event groups for `counters` (see benchmark/overhead/counters.jl)
#   CODE_ALIGN      0 to compile with LLVM's default code alignment (default 1, see below)
#   CODE_ALIGN_ARGS the LLVM options for it (default -align-all-nofallthru-blocks=6)
set -euo pipefail

SWEEP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SWEEP/../.." && pwd)
HOST=$(hostname -s 2>/dev/null || hostname)
OS=$(uname -s)
ARCH=$(uname -m)

JULIA=${JULIA:-julia}
JULIA_CHANNELS=${JULIA_CHANNELS:-}
BENCHES=${BENCHES:-overhead syncbench epcc wakeup counters steal}
REPEAT=${REPEAT:-1}
QUICK=${QUICK:-0}
OUT=${OUT:-$SWEEP/results/$HOST-$(date +%Y%m%d-%H%M%S)}
LOCK=${LOCK-/tmp/threadteams-bench.lock}
EPCC_CC=${EPCC_CC:-gcc clang}
EPCC_COMMIT=94628c25ef2b37b2e9bcc3ed7a6f7cd5cbac459a
CACHE=$SWEEP/.cache/$HOST-$ARCH

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
has() { [[ " $BENCHES " == *" $1 "* ]]; }

mkdir -p "$OUT/logs" "$CACHE"
CSV=$OUT/results.csv

if ((QUICK)); then
    export OVERHEAD_SECONDS=1 SYNCBENCH_OUTER_REPS=10 WAKEUP_SAMPLES=200 COUNTERS_SCALE=0.2
    export STEAL_N=2000
    EPCC_REPS=10
else
    EPCC_REPS=20
fi
export OPENBLAS_NUM_THREADS=1 BENCH_HOST=$HOST

# --- Julia --------------------------------------------------------------------------------

# juliaup's launcher is not on the PATH of a non-interactive ssh session
if ! command -v "$JULIA" >/dev/null 2>&1 && [[ $JULIA == julia && -x $HOME/.juliaup/bin/julia ]]; then
    JULIA=$HOME/.juliaup/bin/julia
fi
if ! command -v "$JULIA" >/dev/null 2>&1; then
    if [[ ${INSTALL_JULIA:-0} == 1 ]]; then
        log "installing juliaup"
        curl -fsSL https://install.julialang.org | sh -s -- --yes
        JULIA=$HOME/.juliaup/bin/julia
    else
        echo "no \`$JULIA\`: set JULIA=/path/to/julia, or INSTALL_JULIA=1 to install juliaup" >&2
        exit 1
    fi
fi
JULIAS=()
if [[ -n $JULIA_CHANNELS ]]; then
    for ch in $JULIA_CHANNELS; do
        if command -v juliaup >/dev/null 2>&1 || [[ -x $HOME/.juliaup/bin/juliaup ]]; then
            "$(command -v juliaup || echo "$HOME/.juliaup/bin/juliaup")" add "$ch" >/dev/null 2>&1 || true
        fi
        JULIAS+=("$JULIA +$ch")
    done
else
    JULIAS=("$JULIA")
fi

# --- Code alignment -----------------------------------------------------------------------

# Where the compiler places the hot loops (the workers' spin loop, the dispatcher's wait)
# relative to the 64-byte fetch blocks changes the time per loop by a few percent between
# builds and processes. Aligning every block that is not reached by fall-through to 64
# bytes takes that out of the comparison. The option reaches only code compiled with it:
# the cache key of package images leaves out LLVM options, so an image compiled without
# it would be loaded as valid. The aligned runs therefore use a depot of their own, in
# which every package is compiled with it (once per host and Julia version), with juliaup
# still finding its Julia installations in the usual depot. The system image (Base) keeps
# its own alignment.
CODE_ALIGN=${CODE_ALIGN:-1}
CODE_ALIGN_ARGS=${CODE_ALIGN_ARGS:--align-all-nofallthru-blocks=6}
if ((CODE_ALIGN)); then
    first_depot=${JULIA_DEPOT_PATH:-$HOME/.julia}
    export JULIAUP_DEPOT_PATH=${JULIAUP_DEPOT_PATH:-${first_depot%%:*}}
    export JULIA_DEPOT_PATH=$CACHE/depot-aligned
    export JULIA_LLVM_ARGS="${JULIA_LLVM_ARGS:+$JULIA_LLVM_ARGS }$CODE_ALIGN_ARGS"
fi

# --- Topology -----------------------------------------------------------------------------

# Expands a CPU list such as "0-3,8,10-11" into one CPU per line
expand_cpus() {
    tr ',' '\n' <<<"$1" | awk -F- 'NF == 2 { for (i = $1; i <= $2; i++) print i; next } NF { print }'
}

# One CPU per physical core among the CPUs this process may use: the first CPU of each
# `thread_siblings_list` (the CPUs that share a core; `lscpu`'s core ids repeat per
# cluster on some ARM chips). The cores of the highest `cpu_capacity` come first (the big
# cores of a big.LITTLE chip, so that a small thread count does not run on the small
# ones), each class in CPU order; capacities within 64 of a multiple of 128 count as one
# class (a chip's big cores differ by a few percent). Without `cpu_capacity` (x86), CPU
# order, which keeps the first cores on one chiplet.
physical_cpus() {
    local allowed d n
    allowed=" $(expand_cpus "$(awk '/^Cpus_allowed_list/ { print $2 }' /proc/self/status)" | paste -sd' ' -) "
    for d in /sys/devices/system/cpu/cpu[0-9]*; do
        n=${d##*cpu}
        [[ $allowed == *" $n "* ]] || continue
        echo "$n $(cat "$d/topology/thread_siblings_list" 2>/dev/null || echo "$n") $(cat "$d/cpu_capacity" 2>/dev/null || echo 1024)"
    done | sort -k1,1n | awk '!seen[$2]++ { print int($3 / 128 + 0.5), $1 }' |
        sort -s -k1,1nr | awk '{ print $2 }' | paste -sd, -
}

if [[ -z ${CPUS:-} && $OS == Linux ]]; then
    CPUS=$(physical_cpus)
fi
CPUS=${CPUS:-}
if [[ -n $CPUS ]]; then
    IFS=, read -r -a CPU_LIST <<<"$CPUS"
    NCORES=${#CPU_LIST[@]}
elif [[ $OS == Darwin ]]; then
    NCORES=$(sysctl -n hw.physicalcpu)
else
    NCORES=$(getconf _NPROCESSORS_ONLN)
fi
if [[ -z ${THREADS:-} ]]; then
    THREADS=""
    for ((t = 2; t < NCORES; t *= 2)); do THREADS+="$t "; done
    THREADS+=$NCORES
fi

# The first `n` CPUs of `CPUS`, comma-separated
cpus_for() { if [[ -n $CPUS ]]; then printf '%s\n' "${CPU_LIST[@]:0:$1}" | paste -sd, -; fi; }
pinned() { # n cmd...: runs `cmd` restricted to the first `n` CPUs (if known)
    local n=$1
    shift
    if [[ -n $CPUS ]] && command -v taskset >/dev/null 2>&1; then
        taskset -c "$(cpus_for "$n")" "$@"
    else
        "$@"
    fi
}

load1() {
    if [[ -r /proc/loadavg ]]; then cut -d' ' -f1 /proc/loadavg; else sysctl -n vm.loadavg | awk '{ print $2 }'; fi
}
csv_row() { # julia threads bench mechanism metric value
    [[ -s $CSV ]] || echo "host,julia,threads,config,bench,mechanism,metric,value" >"$CSV"
    printf '%s,%s,%s,%s,%s,"%s",%s,%s\n' "$HOST" "$1" "$2" "$BENCH_CONFIG" "$3" "$4" "$5" "$6" >>"$CSV"
}

# --- Machine description ------------------------------------------------------------------

{
    echo "host: $HOST"
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "uname: $(uname -a)"
    [[ -r /etc/os-release ]] && grep -E '^PRETTY_NAME=' /etc/os-release
    echo "ThreadTeams: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || cat "$ROOT/.sweep-commit" 2>/dev/null || echo unknown)"
    echo "CPUs used, in order: ${CPUS:-unknown (no pinning)}"
    echo "thread counts: $THREADS"
    echo "code alignment: $( ((CODE_ALIGN)) && echo "$CODE_ALIGN_ARGS (depot $JULIA_DEPOT_PATH)" || echo "LLVM default")"
    echo "load: $(cat /proc/loadavg 2>/dev/null || uptime)"
    if [[ -r /proc/sys/kernel/perf_event_paranoid ]]; then
        echo "perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid)"
    fi
    gov=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
    [[ -r $gov ]] && echo "cpufreq governor (cpu0): $(cat $gov)"
    for cc in $EPCC_CC; do command -v "$cc" >/dev/null 2>&1 && echo "$cc: $("$cc" --version | head -1)"; done
    echo
    if command -v lscpu >/dev/null 2>&1; then
        lscpu
    elif [[ $OS == Darwin ]]; then
        sysctl -n machdep.cpu.brand_string
        sysctl hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu 2>/dev/null || true
    fi
    for j in "${JULIAS[@]}"; do
        echo
        $j -e 'using InteractiveUtils; versioninfo()' 2>&1 || true
    done
} >"$OUT/machine.txt"
log "results in $OUT"
log "threads: $THREADS; CPUs: ${CPUS:-unpinned}; Julia: ${JULIAS[*]}"

# --- Lock ---------------------------------------------------------------------------------

if [[ -n $LOCK ]] && command -v flock >/dev/null 2>&1; then
    exec 9>>"$LOCK"
    if ! flock -n 9; then
        log "waiting for $LOCK (another benchmark runs on this host)"
        flock 9
    fi
elif [[ -n $LOCK ]]; then
    # No flock (macOS): a directory as the lock, created atomically by `mkdir`
    if ! mkdir "${LOCK}dir" 2>/dev/null; then
        log "waiting for ${LOCK}dir (another benchmark runs on this host)"
        until mkdir "${LOCK}dir" 2>/dev/null; do sleep 2; done
    fi
    trap 'rmdir "${LOCK}dir" 2>/dev/null' EXIT
fi

# --- Setup: Julia environments and EPCC ---------------------------------------------------

for j in "${JULIAS[@]}"; do
    log "instantiating benchmark/sweep for $j"
    # One manifest per Julia version: Pkg uses `Manifest-vX.Y.toml` only if it exists. A
    # new one in every sweep: `resolve` keeps the packages of an existing manifest even
    # where `[sources]` in Project.toml now names another (e.g. a branch with a fix), so a
    # sweep would silently measure the code of the one before
    $j --startup-file=no -e '
        manifest = joinpath(ARGS[1], "Manifest-v$(VERSION.major).$(VERSION.minor).toml")
        rm(manifest; force = true)
        touch(manifest)
        using Pkg
        # `resolve` neither installs the General registry in a new depot, as `add` does,
        # nor updates a stale one (which may not know the versions of the compat bounds)
        if isempty(Pkg.Registry.reachable_registries())
            Pkg.Registry.add("General")
        else
            Pkg.Registry.update()
        end
        Pkg.activate(ARGS[1]); Pkg.resolve(); Pkg.instantiate(); Pkg.precompile()' \
        "$SWEEP" >"$OUT/logs/instantiate-$($j --version | awk '{ print $3 }').log" 2>&1 ||
        { log "instantiating failed, see $OUT/logs"; exit 1; }
done

EPCC_BINS=()
# The sources' tarball is shared by the hosts (`remote.sh` copies it along, so that a host
# need not reach GitHub); the build is per host. Without it, the sweep runs without EPCC.
EPCC_TAR=$SWEEP/.cache/epcc-$EPCC_COMMIT.tar.gz
if has epcc && [[ ! -s $EPCC_TAR ]]; then
    log "fetching EPCC OpenMP micro-benchmarks"
    for attempt in 1 2 3; do
        curl -fsSL -o "$EPCC_TAR.part" \
            "https://github.com/EPCCed/epcc-openmp-microbenchmarks/archive/$EPCC_COMMIT.tar.gz" &&
            mv "$EPCC_TAR.part" "$EPCC_TAR" && break
        rm -f "$EPCC_TAR.part"
        sleep 5
    done
    [[ -s $EPCC_TAR ]] || log "fetching EPCC failed, running without it"
fi
src=$CACHE/epcc-$EPCC_COMMIT
# Unpacked unless its sources are there (an interrupted run may have left the directory)
if has epcc && [[ -s $EPCC_TAR && ! -f $src/openmpbench_C_v31/syncbench.c ]]; then
    rm -rf "$src" "$src.part"
    mkdir -p "$src.part" && tar -xzf "$EPCC_TAR" -C "$src.part" --strip-components=1 && mv "$src.part" "$src" ||
        { rm -rf "$src.part"; log "unpacking EPCC failed, running without it"; }
fi
if has epcc && [[ -f $src/openmpbench_C_v31/syncbench.c ]]; then
    for cc in $EPCC_CC; do
        command -v "$cc" >/dev/null 2>&1 || { log "no $cc, skipping its OpenMP"; continue; }
        bin=$CACHE/syncbench-$cc
        # v4.0 needs OpenMP 4.5 lock hints, which some runtimes do not export (libgomp 16):
        # then without the hints, which only the LOCK tests use; v3.1 for older compilers.
        # EPCC's own flags (no -O), so that the delay loop is not optimised away
        for attempt in "v40" "v40 -Domp_init_lock_with_hint(l,h)=omp_init_lock(l)" "v31"; do
            [[ -x $bin ]] && break
            read -r v defs <<<"$attempt"
            (cd "$src/openmpbench_C_$v" &&
                $cc -fopenmp ${defs:+"$defs"} ${EPCC_CFLAGS:-} -o "$bin" syncbench.c common.c -lm) \
                >>"$OUT/logs/build-epcc-$cc.log" 2>&1 || rm -f "$bin"
        done
        if [[ -x $bin ]]; then
            EPCC_BINS+=("$bin")
        else
            log "building EPCC with $cc failed, see $OUT/logs/build-epcc-$cc.log"
        fi
    done
fi

# Name of the OpenMP runtime that a binary loads
omp_runtime() {
    local libs
    if [[ $OS == Darwin ]]; then libs=$(otool -L "$1"); else libs=$(ldd "$1" 2>/dev/null || true); fi
    case $libs in
        *libgomp*) echo libgomp ;;
        *libiomp*) echo libiomp ;;
        *libomp*) echo libomp ;;
        *) echo "omp-$(basename "$1")" ;;
    esac
}

PERF_OK=0
if [[ $OS == Linux && -r /proc/sys/kernel/perf_event_paranoid ]] &&
    (($(cat /proc/sys/kernel/perf_event_paranoid) <= 2)); then
    PERF_OK=1
elif has counters; then
    log "perf_event_paranoid > 2 (or not Linux): skipping counters"
fi

# --- Runs ---------------------------------------------------------------------------------

run_julia() { # julia n script name [VAR=value...]
    local j=$1 n=$2 script=$3 name=$4
    shift 4
    local jv logf
    jv=$($j --version | awk '{ print $3 }')
    logf=$OUT/logs/$name-julia$jv-t$n-r$BENCH_CONFIG.log
    log "$name, Julia $jv, $n threads"
    csv_row "$jv" "$n" system "load before $name" load1 "$(load1)"
    if ! pinned "$n" env "$@" BENCH_CSV="$CSV" BENCH_CPUS="$(cpus_for "$n")" \
        $j --startup-file=no -t "$n,0" --project="$SWEEP" -L "$SWEEP/pin.jl" \
        "$script" >"$logf" 2>&1; then
        log "  failed, see $logf"
    fi
}

run_epcc() { # bin n
    local bin=$1 n=$2 rt places="" logf
    rt=$(omp_runtime "$bin")
    [[ -n $CPUS ]] && places=$(printf '{%s},' "${CPU_LIST[@]:0:$n}") && places=${places%,}
    for policy in default active; do
        logf=$OUT/logs/epcc-$rt-$policy-t$n-r$BENCH_CONFIG.log
        log "EPCC syncbench, $rt, wait policy $policy, $n threads"
        csv_row - "$n" system "load before epcc" load1 "$(load1)"
        local envs=(OMP_NUM_THREADS="$n" OMP_PROC_BIND=close)
        [[ -n $places ]] && envs+=(OMP_PLACES="$places")
        [[ $policy == active ]] && envs+=(OMP_WAIT_POLICY=active)
        if ! pinned "$n" env "${envs[@]}" "$bin" --outer-repetitions "$EPCC_REPS" >"$logf" 2>&1; then
            log "  failed, see $logf"
            continue
        fi
        # "PARALLEL FOR overhead     = 0.5 microseconds +/- 0.1" and the median line
        sed -nE 's/^(.*[^ ]) +overhead += *([-0-9.e+]+) microseconds \+\/- *([-0-9.e+]+).*/\1|\2|\3/p' "$logf" |
            while IFS='|' read -r test ovh ci; do
                csv_row - "$n" epcc "$test $rt/$policy" overhead_mean_us "$ovh"
                csv_row - "$n" epcc "$test $rt/$policy" overhead_ci95_us "$ci"
            done
        sed -nE 's/^(.*[^ ]) +median_ovrhd += *([-0-9.e+]+) microseconds.*/\1|\2/p' "$logf" |
            while IFS='|' read -r test med; do
                csv_row - "$n" epcc "$test $rt/$policy" overhead_median_us "$med"
            done
    done
}

for ((round = 1; round <= REPEAT; round++)); do
    export BENCH_CONFIG=$round
    for n in $THREADS; do
        for b in $BENCHES; do
            case $b in
                epcc)
                    # (`${a[@]+...}`: bash 3.2, macOS's, rejects an empty array under `set -u`)
                    for bin in ${EPCC_BINS[@]+"${EPCC_BINS[@]}"}; do run_epcc "$bin" "$n"; done
                    ;;
                overhead | syncbench | counters | wakeup | steal)
                    [[ $b == counters ]] && ((!PERF_OK)) && continue
                    for j in "${JULIAS[@]}"; do
                        case $b in
                            overhead) run_julia "$j" "$n" "$ROOT/benchmark/overhead/overhead.jl" overhead ;;
                            counters) run_julia "$j" "$n" "$ROOT/benchmark/overhead/counters.jl" counters ;;
                            syncbench) run_julia "$j" "$n" "$SWEEP/syncbench.jl" syncbench ;;
                            wakeup)
                                for s in team threads polyester; do
                                    run_julia "$j" "$n" "$SWEEP/wakeup.jl" "wakeup-$s" WAKEUP_SCHED=$s
                                done
                                ;;
                            steal)
                                for s in team greedy polyester; do
                                    run_julia "$j" "$n" "$ROOT/benchmark/overhead/steal.jl" "steal-$s" STEAL_SCHED=$s
                                done
                                ;;
                        esac
                    done
                    ;;
                *) log "unknown benchmark $b" ;;
            esac
        done
    done
done

echo "load after: $(cat /proc/loadavg 2>/dev/null || uptime)" >>"$OUT/machine.txt"
log "done: $CSV"
log "summary: julia $SWEEP/summarize.jl $CSV"
