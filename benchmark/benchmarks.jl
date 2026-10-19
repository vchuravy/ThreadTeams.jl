# Benchmark suite for AirspeedVelocity.jl (`benchpkg`, see .github/workflows/Benchmark.yml):
# the per-loop cost of a loop with 64 trivial iterations on the team, against
# `Threads.@threads` and OhMyThreads' default scheduler. The absolute numbers depend
# strongly on the machine and on how many cores the Julia threads get (see the overhead
# page of the documentation); the suite is for comparing revisions on the same machine.
# `benchmark/overhead/overhead.jl` is the script behind the numbers in the documentation.
using BenchmarkTools
using ThreadTeams
using Enzyme
using FastBroadcast
using FixedSizeArrays: FixedSizeArray
include("aligned.jl")   # `aligned`: copies of the arrays at cache-line-aligned addresses

const SUITE = BenchmarkGroup()

const N = 64
const TEAM = TeamScheduler()

f_tforeach_team(x) = tforeach(i -> (@inbounds x[i] += 1.0), eachindex(x); scheduler = TEAM)
f_tmapreduce_team(x) = tmapreduce(i -> (@inbounds x[i]), +, eachindex(x); scheduler = TEAM)
function f_tasks_team(x)
    @tasks for i in eachindex(x)
        @set scheduler = TEAM
        @inbounds x[i] += 1.0
    end
    return
end
f_greedy(x) = tforeach(i -> (@inbounds x[i] += 1.0), eachindex(x); scheduler = GreedyTeamScheduler())
function f_threads_dynamic(x)
    Threads.@threads :dynamic for i in eachindex(x)
        @inbounds x[i] += 1.0
    end
    return
end
f_omt_dynamic(x) = tforeach(i -> (@inbounds x[i] += 1.0), eachindex(x))
function f_serial(x)
    for i in eachindex(x)
        @inbounds x[i] += 1.0
    end
    return
end

# One sample per loop (`evals = 1`): the cost of one loop is what matters, and the workers'
# spin time (100 µs) covers the gap between samples
for (name, f) in (
        ("tforeach team", f_tforeach_team), ("tmapreduce team", f_tmapreduce_team),
        ("@tasks team", f_tasks_team), ("tforeach greedy team", f_greedy),
        ("Threads.@threads :dynamic", f_threads_dynamic),
        ("OhMyThreads tforeach", f_omt_dynamic), ("serial loop", f_serial),
    )
    SUITE["overhead"][name] = @benchmarkable $f(x) setup = (x = aligned(zeros(N))) samples = 10_000 evals = 1 seconds = 3
end

# Loops with more work per iteration, where the hand-off is a smaller part
function f_work_team(x)
    tforeach(eachindex(x); scheduler = TEAM) do i
        @inbounds x[i] = sqrt(x[i] + 1.0) * 1.5
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
    SUITE["work"]["tforeach team, $n"] = @benchmarkable f_work_team(x) setup = (x = aligned(zeros($n))) samples = 2_000 evals = 1 seconds = 3
    SUITE["work"]["serial, $n"] = @benchmarkable f_work_serial(x) setup = (x = aligned(zeros($n))) samples = 2_000 evals = 1 seconds = 3
end

# Automatic differentiation with Enzyme: the loop on the team (through the rules of the
# EnzymeCore extension, the derivative chunks run on the team too) against a serial loop
function ad_square_sum_team(x, a)
    y = similar(x)
    tforeach(i -> (y[i] = a * x[i]^2), eachindex(x); scheduler = TEAM)
    return sum(y)
end
function ad_square_sum_serial(x, a)
    y = similar(x)
    for i in eachindex(x)
        y[i] = a * x[i]^2
    end
    return sum(y)
end
ad_sum_of_squares_team(x) = tmapreduce(abs2, +, x; scheduler = TEAM)
for n in (1_000, 100_000)
    SUITE["ad"]["reverse tforeach team, $n"] = @benchmarkable Enzyme.gradient(Reverse, ad_square_sum_team, x, a) setup = (x = rand($n); a = 1.7) samples = 500 evals = 1 seconds = 5
    SUITE["ad"]["reverse serial loop, $n"] = @benchmarkable Enzyme.gradient(Reverse, ad_square_sum_serial, x, a) setup = (x = rand($n); a = 1.7) samples = 500 evals = 1 seconds = 5
    SUITE["ad"]["forward tforeach team, $n"] = @benchmarkable autodiff(Forward, ad_square_sum_team, Duplicated(x, dx), Const(a)) setup = (x = rand($n); dx = ones($n); a = 1.7) samples = 500 evals = 1 seconds = 5
    SUITE["ad"]["reverse tmapreduce team, $n"] = @benchmarkable Enzyme.gradient(Reverse, ad_sum_of_squares_team, x) setup = (x = rand($n)) samples = 500 evals = 1 seconds = 5
end

# Downstream: FastBroadcast's `@..` with `thread = true` splits the last axis of the
# destination into chunks that run on the team (FastBroadcast's ThreadTeams extension),
# against the serial `@..` and Base's broadcast
fb_log_thread!(dst, src) = (@.. thread = true dst = log(src); nothing)
fb_log_serial!(dst, src) = (@.. thread = false dst = log(src); nothing)
fb_log_base!(dst, src) = (dst .= log.(src); nothing)
for n in (1_000, 100_000)
    SUITE["fastbroadcast"]["log thread, $n"] = @benchmarkable fb_log_thread!(dst, src) setup = (src = aligned(rand($n)); dst = aligned(similar(src))) samples = 2_000 evals = 1 seconds = 3
    SUITE["fastbroadcast"]["log serial, $n"] = @benchmarkable fb_log_serial!(dst, src) setup = (src = aligned(rand($n)); dst = aligned(similar(src))) samples = 2_000 evals = 1 seconds = 3
    SUITE["fastbroadcast"]["log Base, $n"] = @benchmarkable fb_log_base!(dst, src) setup = (src = aligned(rand($n)); dst = aligned(similar(src))) samples = 2_000 evals = 1 seconds = 3
end
# Many arguments over a 3-dimensional array (the example of FastBroadcast's README)
function fb_axpy9_thread!(a, b, c, d, e, f, g, h, i)
    @.. thread = true a = b + 0.1 * (0.2c + 0.3d + 0.4e + 0.5f + 0.6g + 0.6h + 0.6i)
    return
end
function fb_axpy9_serial!(a, b, c, d, e, f, g, h, i)
    @.. thread = false a = b + 0.1 * (0.2c + 0.3d + 0.4e + 0.5f + 0.6g + 0.6h + 0.6i)
    return
end
SUITE["fastbroadcast"]["9 arguments thread, 100x100x8"] = @benchmarkable fb_axpy9_thread!(xs...) setup = (xs = [aligned(rand(100, 100, 8)) for _ in 1:9]) samples = 2_000 evals = 1 seconds = 3
SUITE["fastbroadcast"]["9 arguments serial, 100x100x8"] = @benchmarkable fb_axpy9_serial!(xs...) setup = (xs = [aligned(rand(100, 100, 8)) for _ in 1:9]) samples = 2_000 evals = 1 seconds = 3

# Downstream: the loop with which Trixi.jl resets the time derivative (`set_zero!`, timer
# "reset ∂u/∂t") in every evaluation of the right-hand side, on its `:pool` threading
# backend. `du` is a `[variable, node, node, element]` array (a `FixedSizeArray`, as Trixi's
# `wrap_array`) of the 2D linear advection example with 256 elements (polynomial degree 3).
# Each iteration zeros one element (16 values), so the loop takes well under a microsecond
# and its cost is mostly the hand-off. Trixi's `@threaded` passes one chunk per thread
# (`chunks(...; n = nthreads())`) to `tforeach`. In Trixi, the loops of a right-hand side
# follow each other directly: "10 in a row" runs ten loops back to back.
function trixi_reset_team!(du)
    n = size(du, 4)
    parts = chunks(1:n; n = min(Threads.nthreads(), n))
    tforeach(parts; scheduler = TEAM) do part
        for element in part
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
