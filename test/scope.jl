@testset "dynamic scope and world age" begin
    r = zeros(Int, 64)
    # ScopedValues.jl (`Base.ScopedValues` on Julia ≥ 1.11)
    sv = ScopedValues.ScopedValue(0)
    ScopedValues.with(sv => 42) do
        tforeach(i -> (r[i] = sv[]), eachindex(r); scheduler = team)
    end
    @test all(==(42), r)
    tforeach(i -> (r[i] = sv[]), eachindex(r); scheduler = team)
    @test all(==(0), r)
    @test ScopedValues.with(() -> tmapreduce(_ -> sv[], +, 1:100; scheduler = team), sv => 3) == 300
    # the logger (in the task's scope on Julia ≥ 1.11, in its `logstate` on 1.10)
    logger = Base.CoreLogging.NullLogger()
    ok = zeros(Bool, 64)
    Base.CoreLogging.with_logger(logger) do
        tforeach(eachindex(ok); scheduler = team) do i
            ok[i] = Base.CoreLogging.current_logger() === logger
        end
    end
    @test all(ok)
    @test Base.CoreLogging.current_logger() !== logger
    # world age
    @test all(==(1), pw_world_h())
    w = Base.get_world_counter()
    @eval pw_world_g(::Int) = 2   # a new, more specific method
    @test all(==(1), Base.invoke_in_world(w, pw_world_h))
    @test all(==(2), Base.invokelatest(pw_world_h))
    @test all(==(1), Base.invoke_in_world(w, pw_world_h))
end

@testset "workers are created without a scope" begin
    # A new task inherits the scope of the task that creates it; the workers must not keep
    # the scope of the first caller (nor install a handler for later callers without one)
    # (Julia ≥ 1.11, where a task keeps its scope in the `scope` field)
    if NW > 1 && isdefined(Core, :current_scope)
        ThreadTeams.shutdown!()
        sv = ScopedValues.ScopedValue(0)
        r = zeros(Int, 64)
        ScopedValues.with(sv => 7) do
            tforeach(i -> (r[i] = sv[]), eachindex(r); scheduler = team)
        end
        @test all(==(7), r)
        pool = @atomic ThreadTeams.POOL.pool
        @test all(t -> getfield(t, :scope) === nothing, pool.tasks)
        tforeach(i -> (r[i] = sv[]), eachindex(r); scheduler = team)
        @test all(==(0), r)
    end
end
