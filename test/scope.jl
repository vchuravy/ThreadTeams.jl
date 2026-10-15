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
    # The closure that runs the chunks is one per chunk function type, created in the
    # caller's world age and reused while that is the current one (`job_closure`)
    f = let r = r
        i -> (r[i] = pw_world_g(Int8(i)))
    end
    slot = ThreadTeams.job_slot(typeof(f))
    slot[] = f
    job = ThreadTeams.job_closure(slot)
    @test job isa ThreadTeams.JobClosure
    @test job.world == ThreadTeams.world_age()
    @test ThreadTeams.job_closure(slot) === job
    # called as by worker 1, from its own slot
    ThreadTeams.worker_slots(typeof(f), 1)[1][] = f
    job(1, 5)
    @test r[5] == 1
    @eval pw_world_g(::Int8) = 3
    job2 = Base.invokelatest(ThreadTeams.job_closure, slot)
    @test job2 !== job
    @test job2.world > job.world
    wslot = ThreadTeams.worker_slots(typeof(f), 1)[1]   # emptied after every chunk
    wslot[] = f
    job2(1, 5)
    @test r[5] == 3
    wslot[] = f
    job(1, 5)   # the older closure still runs in its world
    @test r[5] == 1
    @test !isassigned(wslot)
    ThreadTeams.clear_slot!(slot)
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
