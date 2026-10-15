# After a loop, the team keeps no reference to the loop body, the arrays it captured or the
# partial results of a reduction
function loop_over_temporary()
    x = zeros(10^6)
    w = WeakRef(x)
    tforeach(i -> (@inbounds x[i] += 1.0), eachindex(x); scheduler = team)
    s = tmapreduce(i -> (@inbounds x[i]), +, eachindex(x); scheduler = team)
    v = tmapreduce(i -> fill(1.0, 1000), (a, b) -> a .+ b, 1:64; scheduler = team)
    return w, s, WeakRef(v)
end

# Also when a chunk throws: the exception and the array it refers to are dropped. (No
# `@inbounds` in these bodies: with it, Julia 1.10 running with code coverage keeps the
# array alive after the exception, through nothing the team holds; not on Julia ≥ 1.11,
# without coverage, or with a plain loop.)
function failing_loop_over_temporary()
    x = zeros(10^6)
    w = WeakRef(x)
    try
        tforeach(
            i -> (x[i] += 1.0; i == 10^6 && throw(DomainError(x))),
            eachindex(x); scheduler = team
        )
    catch
    end
    try
        tforeach(
            i -> (x[i] += 1.0; i == 10^6 && throw(DomainError(x))),
            eachindex(x); scheduler = GreedyTeamScheduler(chunksize = 1000)
        )
    catch
    end
    return w
end

# `GC.gc()` does nothing when another thread is collecting at that moment: it only waits for
# that collection, which can be an incremental one that keeps old objects. Tasks of the test
# runner allocate on other threads, so collect until the objects are gone (or give up).
function collected(ws::WeakRef...)
    for _ in 1:10
        GC.gc()
        all(w -> w.value === nothing, ws) && return true
    end
    return false
end

@testset "no references after a loop" begin
    w, s, wv = loop_over_temporary()
    @test s == 10^6
    @test collected(w, wv)
    w = failing_loop_over_temporary()
    @test collected(w)
    if NW > 1
        pool = @atomic ThreadTeams.POOL.pool
        @test all(isnothing, pool.errs)
        @test all(isnothing, pool.greedy_errs)
    end
    if NW > 1
        # the partials buffers of the reductions above are empty again
        @test isempty(ThreadTeams.static_buffer(Float64))
        @test isempty(ThreadTeams.static_buffer(Vector{Float64}))
        # and a cleared slot is undefined again, as a fresh `RefValue{F}()` is
        f = let v = zeros(1)
            i -> v[i]
        end
        slot = ThreadTeams.job_slot(typeof(f))
        slot[] = f
        @test isdefined(slot, :x)
        ThreadTeams.clear_slot!(slot)
        @test !isdefined(slot, :x)
    end
end
