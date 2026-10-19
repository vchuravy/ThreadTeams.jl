# Differentiating parallel operations on the team with Enzyme (ext/ThreadTeamsEnzymeCoreExt.jl)
using Enzyme

function ad_square_sum(x, a)
    y = similar(x)
    tforeach(i -> (y[i] = a * x[i]^2), eachindex(x); scheduler = team)
    return sum(y)
end
function ad_stencil(x)
    y = zeros(length(x))
    tforeach(i -> (y[i] = x[i - 1] * x[i + 1]), 2:(length(x) - 1); scheduler = team)
    return sum(y)
end
ad_stencil_serial(x) = sum(x[i - 1] * x[i + 1] for i in 2:(length(x) - 1))
function ad_constant_scalar(x)
    y = similar(x)
    c = 2.0
    tforeach(i -> (y[i] = c * x[i]^2), eachindex(x); scheduler = team)
    return sum(y)
end
function ad_nested(x, a)
    y = zeros(length(x), 4)
    tforeach(1:4; scheduler = team) do j
        tforeach(i -> (y[i, j] = a * j * x[i]), eachindex(x); scheduler = team)
    end
    return sum(y)
end
ad_reduction(x) = tmapreduce(i -> x[i]^2, +, eachindex(x); scheduler = team)
function ad_two_reductions(x)
    s, m = tmapreduce(
        i -> (x[i], x[i]), (a, b) -> (a[1] + b[1], max(a[2], b[2])),
        eachindex(x); scheduler = team
    )
    return s + 2m
end
ad_tmapreduce(x) = tmapreduce(abs2, +, x; scheduler = team)
ad_greedy(x) = tmapreduce(abs2, +, x; scheduler = GreedyTeamScheduler(chunksize = 7))
ad_active_closure(a) = tmapreduce(i -> a * i, +, 1:100; scheduler = team)
# The operator reads a value that changes after the reduction: `w^(n - 1) * prod(abs2, x)`
const AD_WEIGHT = Ref(2.0)
function ad_stateful_op(x)
    AD_WEIGHT[] = 2.0
    s = tmapreduce(abs2, (a, b) -> AD_WEIGHT[] * a * b, x; scheduler = team)
    AD_WEIGHT[] = 3.0
    return s
end
function ad_tasks(x)
    return @tasks for i in eachindex(x)
        @set scheduler = team
        @set reducer = +
        x[i]^3
    end
end
function ad_inplace!(y, x)
    tforeach(eachindex(x); scheduler = team) do i
        y[i] = sin(x[i]) * x[i]
    end
    return nothing
end

@testset "Enzyme" begin
    x = collect(range(0.1, 2.0; length = 103))
    a = 1.7
    @testset "reverse" begin
        dx, da = Enzyme.gradient(Reverse, ad_square_sum, x, a)
        @test dx ≈ 2a .* x
        @test da ≈ sum(x .^ 2)
        @test Enzyme.gradient(Reverse, ad_stencil, x)[1] ≈
            Enzyme.gradient(Reverse, ad_stencil_serial, x)[1]
        @test Enzyme.gradient(Reverse, ad_constant_scalar, x)[1] ≈ 4x
        dx, da = Enzyme.gradient(Reverse, ad_nested, x, a)
        @test dx ≈ fill(10a, length(x))
        @test da ≈ 10sum(x)
    end
    @testset "reductions" begin
        @test Enzyme.gradient(Reverse, ad_reduction, x)[1] ≈ 2x
        @test Enzyme.gradient(Reverse, ad_two_reductions, x)[1] ≈
            [1.0 + (i == argmax(x) ? 2.0 : 0.0) for i in eachindex(x)]
        @test Enzyme.gradient(Reverse, ad_tmapreduce, x)[1] ≈ 2x
        @test Enzyme.gradient(Reverse, ad_greedy, x)[1] ≈ 2x
        @test Enzyme.gradient(Reverse, ad_active_closure, a)[1] ≈ sum(1:100)
        @test Enzyme.gradient(Reverse, ad_tasks, x)[1] ≈ 3x .^ 2
        y = collect(range(0.5, 1.5; length = 13))
        @test Enzyme.gradient(Reverse, ad_stateful_op, y)[1] ≈
            2 .* (2.0^12 * prod(abs2, y)) ./ y
        # a constant result
        dx = zero(x)
        autodiff(Reverse, ad_tmapreduce, Const, Duplicated(x, dx))
        @test iszero(dx)
    end
    @testset "forward" begin
        dx = ones(length(x))
        @test autodiff(Forward, ad_square_sum, Duplicated(x, dx), Const(a))[1] ≈ sum(2a .* x)
        @test autodiff(Forward, ad_reduction, Duplicated(x, dx))[1] ≈ sum(2x)
        @test autodiff(ForwardWithPrimal, ad_reduction, Duplicated(x, dx))[2] ≈ sum(x .^ 2)
        @test autodiff(Forward, ad_active_closure, Duplicated(a, 1.0))[1] ≈ sum(1:100)
    end
    @testset "batched" begin
        dx = ones(length(x))
        r = autodiff(Forward, ad_square_sum, BatchDuplicated(x, (dx, 2dx)), Const(a))[1]
        @test r[1] ≈ sum(2a .* x)
        @test r[2] ≈ 2sum(2a .* x)
        y = zeros(length(x))
        dys = (ones(length(x)), fill(2.0, length(x)))
        dxs = (zeros(length(x)), zeros(length(x)))
        autodiff(Reverse, ad_inplace!, Const, BatchDuplicated(y, dys), BatchDuplicated(x, dxs))
        @test y ≈ sin.(x) .* x
        @test dxs[1] ≈ cos.(x) .* x .+ sin.(x)
        @test dxs[2] ≈ 2 .* (cos.(x) .* x .+ sin.(x))
    end
end
