@testset "exceptions" begin
    # one chunk throws: the exception of the first failing iteration in serial
    # order is rethrown after all chunks finished
    for bad in (1, 50, 100)
        done = zeros(Int, 100)
        err = try
            tforeach(1:100; scheduler = team) do i
                i == bad && throw(DomainError(i))
                done[i] = 1
            end
            nothing
        catch e
            e
        end
        @test err == DomainError(bad)
        # everything a serial loop would have done before the error was done
        @test all(==(1), done[1:(bad - 1)])
    end
    # all chunks throw
    err = try
        tforeach(i -> throw(ArgumentError("chunk starting at $i")), 1:100; scheduler = team)
    catch e
        e
    end
    @test err == ArgumentError("chunk starting at 1")
    err = try
        tforeach(i -> i > 10 && error("bad $i"), 1:100; scheduler = team)
    catch e
        e
    end
    @test err == ErrorException("bad 11")
    err = try
        treduce((a, b) -> b > 60 ? error("bad") : a + b, 1:100; scheduler = team)
    catch e
        e
    end
    @test err isa ErrorException
    # the pool still works afterwards
    for _ in 1:100
        @test psum(collect(1:100)) == 5050
    end
    # backtrace of a worker chunk is kept
    if NW > 1
        excs = try
            tforeach(i -> i == 100 && error("last"), 1:100; scheduler = team)
        catch
            current_exceptions()
        end
        i = findfirst(e -> e.exception isa ThreadTeams.ChunkFailedException, excs)
        @test i !== nothing
        cause = excs[i].exception
        @test cause.exception == ErrorException("last")
        # the raw backtrace is kept, and only looked up when shown
        @test !(cause.backtrace isa Vector{Base.StackTraces.StackFrame})
        shown = sprint(showerror, cause)
        @test occursin("nested chunk error: last", shown)
        @test occursin("exceptions.jl", shown)
    end
end

@testset "exception thrown into a waiting worker" begin
    if NW > 1
        old = ThreadTeams.spin_time()
        ThreadTeams.set_spin_time!(0)
        @test psum(collect(1:100)) == 5050
        pool = @atomic ThreadTeams.POOL.pool
        k = NW
        @test timedwait(() -> (@atomic pool.sleepflags[k].sleeping), 10) === :ok
        # An interrupt on the worker's thread before it read its command (here `STOP`,
        # not notified) must not erase the command
        @atomic pool.workers[k].cmd = ThreadTeams.STOP
        schedule(pool.tasks[k], InterruptException(); error = true)
        @test timedwait(() -> istaskdone(pool.tasks[k]), 10) === :ok
        ThreadTeams.shutdown!()
        ThreadTeams.set_spin_time!(old)
        @test psum(collect(1:100)) == 5050
    end
end

@testset "interrupted while waiting for the workers" begin
    if NW > 1
        main = current_task()
        running = Threads.Atomic{Bool}(false)
        gate = Threads.Atomic{Bool}(false)
        returned = Threads.Atomic{Bool}(false)
        finished = Threads.Atomic{Bool}(false)
        for sched in (team, GPS(chunksize = 1))
            running[] = false
            gate[] = false
            returned[] = false
            finished[] = false
            # Interrupt the caller while a worker runs a chunk, again while the caller waits
            # for that chunk, and only then let the chunk finish
            observer = @async begin
                t = time()
                while !running[] && time() - t < 10
                    yield()
                end
                # no `sleep`: a caller that waits by yielding keeps the timers from running
                schedule(main, InterruptException(); error = true)
                yield()
                t = time()
                while time() - t < 0.2
                    yield()
                end
                returned[] || schedule(main, InterruptException(); error = true)
                yield()
                t = time()
                while time() - t < 0.2
                    yield()
                end
                gate[] = true
            end
            err = try
                tforeach(1:(4NW); scheduler = sched) do i
                    t = time()
                    if current_task() === main
                        # don't take back the chunks of workers that are still waking up
                        while !running[] && time() - t < 10
                        end
                    elseif !Threads.atomic_xchg!(running, true)
                        while !gate[] && time() - t < 10
                        end
                        finished[] = true
                    end
                end
                nothing
            catch e
                e
            end
            returned[] = true
            # the loop returned only after its running chunk
            @test finished[]
            wait(observer)
            @test err isa InterruptException
            done = zeros(Int, 1000)
            tforeach(i -> (done[i] += 1), 1:1000; scheduler = sched)
            @test all(==(1), done)
        end
    end
end
