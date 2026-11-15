@testset "tmapreduce / treduce" begin
    x = rand(1001)
    for sched in (team, :team)
        @test tmapreduce(i -> x[i], max, eachindex(x); scheduler = sched) == maximum(x)
        @test treduce(+, 1:0; scheduler = sched, init = 0) == 0
        @test treduce(+, [1]; scheduler = sched) == 1
        @test tmapreduce(identity, +, 1:10; scheduler = sched, init = 0.0) === 55.0
        @test treduce(+, Int[]; scheduler = sched) == 0
        @test_throws Exception treduce(max, Int[]; scheduler = sched)
        @test treduce(*, 1:20; scheduler = sched) == prod(1:20)   # wraps on 32-bit
        @test treduce(xor, 1:100; scheduler = sched) == reduce(xor, 1:100)
        # several reductions at once, with a tuple
        v = randn(503)
        mx, mn = tmapreduce(
            i -> (v[i], v[i]), (a, b) -> (max(a[1], b[1]), min(a[2], b[2])),
            eachindex(v); scheduler = sched
        )
        @test (mx, mn) == (maximum(v), minimum(v))
    end
    # `init` is used once per chunk, so it must be neutral; a non-neutral init shows up
    # once per chunk
    n = team_nchunks(1:100)
    @test tmapreduce(identity, +, 1:100; scheduler = team, init = 1) == 5050 + n
    # @tasks reductions
    s = @tasks for i in 1:100
        @set scheduler = team
        @set reducer = +
        i
    end
    @test s == 5050
    # the type of the accumulator can differ from the element type
    @test tmapreduce(i -> 0.5, +, 1:100; scheduler = team) === 50.0
    # deterministic results
    f = rand(10_000)
    @test all(_ -> psum(f) == psum(f), 1:100)
    @test psum(f) ≈ sum(f)
end
