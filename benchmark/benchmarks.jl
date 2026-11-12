# Benchmark suite for AirspeedVelocity.jl (`benchpkg`, see .github/workflows/Benchmark.yml):
# the per-loop cost of a loop with 64 trivial iterations on the team, through each of the
# package's parallel loops, against `Threads.@threads`. The absolute numbers depend
# strongly on the machine and on how many cores the Julia threads get (see the overhead
# page of the documentation); the suite is for comparing revisions on the same machine.
# `benchmark/overhead/overhead.jl` is the script behind the numbers in the documentation;
# it also measures OhMyThreads' schedulers, the team ones included.
using BenchmarkTools
using ThreadTeams
using ThreadTeams: run_chunks, run_fold, foreach_range, foreach_chunk, StaticSchedule, GreedySchedule
using Enzyme
using FixedSizeArrays: FixedSizeArray
include("aligned.jl")   # `aligned`: copies of the arrays at cache-line-aligned addresses

const SUITE = BenchmarkGroup()

const N = 64
const NW = ThreadTeams.nworkers()
# The greedy schedule takes any number of chunks: four per worker, to give it some to balance
const NGREEDY = min(N, 4 * NW)

# The iterations of chunk `c` of `m` of a loop of `n`
chunk_range(c, n, m) = ((c - 1) * n ÷ m + 1):(c * n ÷ m)

function f_run_chunks(x)
    run_chunks(NW) do c
        for i in chunk_range(c, length(x), NW)
            @inbounds x[i] += 1.0
        end
    end
    return
end
function f_run_fold(x)
    return run_fold(+, NW) do c
        s = 0.0
        for i in chunk_range(c, length(x), NW)
            @inbounds s += x[i]
        end
        s
    end
end
function f_greedy(x)
    run_chunks(NGREEDY, GreedySchedule()) do c
        for i in chunk_range(c, length(x), NGREEDY)
            @inbounds x[i] += 1.0
        end
    end
    return
end
function f_foreach_range(x)
    foreach_range(length(x)) do lo, hi
        for i in lo:hi
            @inbounds x[i] += 1.0
        end
    end
    return
end
function f_foreach_range_steal(x)
    foreach_range(length(x); schedule = :steal) do lo, hi
        for i in lo:hi
            @inbounds x[i] += 1.0
        end
    end
    return
end
function f_foreach_chunk(x)
    foreach_chunk(NW) do c
        for i in chunk_range(c, length(x), NW)
            @inbounds x[i] += 1.0
        end
    end
    return
end
function f_threads_dynamic(x)
    Threads.@threads :dynamic for i in eachindex(x)
        @inbounds x[i] += 1.0
    end
    return
end
function f_serial(x)
    for i in eachindex(x)
        @inbounds x[i] += 1.0
    end
    return
end

# One sample per loop (`evals = 1`): the cost of one loop is what matters, and the workers'
# spin time (100 µs) covers the gap between samples
for (name, f) in (
        ("run_chunks static", f_run_chunks), ("run_fold static", f_run_fold),
        ("run_chunks greedy", f_greedy), ("foreach_range", f_foreach_range),
        ("foreach_range steal", f_foreach_range_steal), ("foreach_chunk", f_foreach_chunk),
        ("Threads.@threads :dynamic", f_threads_dynamic), ("serial loop", f_serial),
    )
    SUITE["overhead"][name] = @benchmarkable $f(x) setup = (x = aligned(zeros(N))) samples = 10_000 evals = 1 seconds = 3
end

# Loops with more work per iteration, where the hand-off is a smaller part
function f_work_team(x)
    foreach_range(length(x)) do lo, hi
        for i in lo:hi
            @inbounds x[i] = sqrt(x[i] + 1.0) * 1.5
        end
    end
    return
end
function f_work_serial(x)
    for i in eachindex(x)
        @inbounds x[i] = sqrt(x[i] + 1.0) * 1.5
    end
    return
end
for n in (1_000, 100_000)
    SUITE["work"]["foreach_range, $n"] = @benchmarkable f_work_team(x) setup = (x = aligned(zeros($n))) samples = 2_000 evals = 1 seconds = 3
    SUITE["work"]["serial, $n"] = @benchmarkable f_work_serial(x) setup = (x = aligned(zeros($n))) samples = 2_000 evals = 1 seconds = 3
end

# Automatic differentiation with Enzyme: the loop on the team (through the rules of the
# EnzymeCore extension on `run_chunks` and `run_fold`, the derivative chunks run on the team
# too) against a serial loop
function ad_square_sum_team(x, a)
    y = similar(x)
    run_chunks(NW) do c
        for i in chunk_range(c, length(x), NW)
            y[i] = a * x[i]^2
        end
    end
    return sum(y)
end
function ad_square_sum_serial(x, a)
    y = similar(x)
    for i in eachindex(x)
        y[i] = a * x[i]^2
    end
    return sum(y)
end
ad_sum_of_squares_team(x) = run_fold(c -> sum(abs2, view(x, chunk_range(c, length(x), NW))), +, NW)
for n in (1_000, 100_000)
    SUITE["ad"]["reverse run_chunks, $n"] = @benchmarkable Enzyme.gradient(Reverse, ad_square_sum_team, x, a) setup = (x = rand($n); a = 1.7) samples = 500 evals = 1 seconds = 5
    SUITE["ad"]["reverse serial loop, $n"] = @benchmarkable Enzyme.gradient(Reverse, ad_square_sum_serial, x, a) setup = (x = rand($n); a = 1.7) samples = 500 evals = 1 seconds = 5
    # Forward mode differentiates `run_fold`: on Julia 1.12 and later Enzyme can not yet
    # differentiate `run_chunks` with a closure that captures the constant `a` in forward
    # mode (`CallingConventionMismatchError`, EnzymeAD/Enzyme.jl#2707)
    SUITE["ad"]["forward run_fold, $n"] = @benchmarkable autodiff(Forward, ad_sum_of_squares_team, Duplicated(x, dx)) setup = (x = rand($n); dx = ones($n)) samples = 500 evals = 1 seconds = 5
    SUITE["ad"]["reverse run_fold, $n"] = @benchmarkable Enzyme.gradient(Reverse, ad_sum_of_squares_team, x) setup = (x = rand($n)) samples = 500 evals = 1 seconds = 5
end

# Downstream: the loop with which Trixi.jl resets the time derivative (`set_zero!`, timer
# "reset ∂u/∂t") in every evaluation of the right-hand side, on its `:pool` threading
# backend. `du` is a `[variable, node, node, element]` array (a `FixedSizeArray`, as Trixi's
# `wrap_array`) of the 2D linear advection example with 256 elements (polynomial degree 3).
# Each iteration zeros one element (16 values), so the loop takes well under a microsecond
# and its cost is mostly the hand-off. Trixi's `@threaded` passes one chunk per thread to
# OhMyThreads' `tforeach` with the team scheduler, which is one `run_chunks` call with one
# chunk per worker. In Trixi, the loops of a right-hand side follow each other directly:
# "10 in a row" runs ten loops back to back.
function trixi_reset_team!(du)
    n = size(du, 4)
    m = min(NW, n)
    run_chunks(m) do c
        for element in chunk_range(c, n, m)
            @inbounds view(du, :, :, :, element) .= zero(eltype(du))
        end
    end
    return
end
function trixi_reset_serial!(du)
    for element in axes(du, 4)
        @inbounds view(du, :, :, :, element) .= zero(eltype(du))
    end
    return
end
function trixi_reset_static!(du)
    Threads.@threads :static for element in axes(du, 4)
        @inbounds view(du, :, :, :, element) .= zero(eltype(du))
    end
    return
end
function ten_in_a_row(f::F, du) where {F}
    for _ in 1:10
        f(du)
    end
    return
end
trixi_du() = aligned(FixedSizeArray, zeros(1, 4, 4, 256))
SUITE["trixi"]["reset du team"] = @benchmarkable trixi_reset_team!(du) setup = (du = trixi_du()) samples = 10_000 evals = 1 seconds = 3
SUITE["trixi"]["reset du team, 10 in a row"] = @benchmarkable ten_in_a_row(trixi_reset_team!, du) setup = (du = trixi_du()) samples = 10_000 evals = 1 seconds = 3
SUITE["trixi"]["reset du Threads.@threads :static"] = @benchmarkable trixi_reset_static!(du) setup = (du = trixi_du()) samples = 10_000 evals = 1 seconds = 3
SUITE["trixi"]["reset du serial"] = @benchmarkable trixi_reset_serial!(du) setup = (du = trixi_du()) samples = 10_000 evals = 1 seconds = 3
