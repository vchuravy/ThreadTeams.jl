# The closure the workers call (`job_closure` in src/pool.jl) is created at run time only.
# It holds the code pointers of its session, which an image could not store, and the
# world age it was created in. Parallel loops run serially during precompilation, so no
# closure is created then; the precompile workload only specializes the per-type cache
# (`job_closure_cache`), which must come out of the image empty. Checked in fresh
# processes, since the test process has run loops already: one that loads ThreadTeams
# from its image, and one that precompiles and loads a package whose own precompile
# workload runs loops on the team.

# Run `code` in a fresh process with the threads of this one and the test environment,
# and collect the `key=value` lines it prints
function run_fresh(code; env = Dict{String, String}())
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) --threads=$(Threads.nthreads(:default)) -e $code`
    cmd = addenv(cmd, env)
    out = IOBuffer()
    ok = success(pipeline(cmd; stdout = out, stderr = out))
    text = String(take!(out))
    ok || @error "the process failed" text
    @test ok
    values = Dict{String, String}()
    for line in eachline(IOBuffer(text))
        m = match(r"^([a-z_]+)=(.*)$", line)
        m === nothing || (values[m[1]] = m[2])
    end
    return values
end

# The chunk function types that the cache was specialized for, and how many of the
# caches hold a closure: a report at the end of the fresh process' code. (A
# specialization on `::Type{F}` is keyed `Type{F}`, on Julia ≥ 1.14 `Core.TypeEgal{F}`, a
# kind of type of its own.)
const REPORT = """
function cached_types()
    Fs = Any[]
    for mi in Base.specializations(only(methods(ThreadTeams.job_closure_cache)))
        st = mi.specTypes
        st isa DataType || continue
        T = st.parameters[2]
        ((T isa DataType && T <: Type) || (isdefined(Core, :TypeEgal) && T isa Core.TypeEgal)) ||
            continue
        isconcretetype(T.parameters[1]) || continue
        push!(Fs, T.parameters[1])
    end
    return Fs
end
jobs(Fs) = Any[job for job in (ThreadTeams.job_closure_cache(F)[] for F in Fs) if job !== nothing]
closures(Fs) = length(jobs(Fs))
worlds_ok(jobs, w) = all(job isa ThreadTeams.JobClosure && job.world == w for job in jobs)
"""

@testset "precompilation" begin
    # Loaded from its image, ThreadTeams has caches for the loop bodies of its precompile
    # workload (`run_fold`, `dispatch!` and the cache are specialized for them), all empty.
    # The first loop of a session creates the closure, in the session's world age, and
    # the next loop with the same body type reuses it. (Both loops run in one function:
    # on Julia ≥ 1.12 a top-level assignment that creates a global is a new world age.)
    code = REPORT * """
        using ThreadTeams
        Fs = cached_types()
        println("caches=", length(Fs))
        println("closures_before=", closures(Fs))
        f(x) = tforeach(i -> (x[i] = Float64(i)), eachindex(x); scheduler = TeamScheduler())
        function two_loops()
            x = zeros(64)
            f(x)
            ok1 = x == 1:64
            w = Base.get_world_counter()
            jobs1 = jobs(cached_types())
            fill!(x, 0.0)
            f(x)
            ok2 = x == 1:64
            jobs2 = jobs(cached_types())
            return ok1 && ok2, length(jobs1), worlds_ok(jobs1, w), jobs1 == jobs2
        end
        result_ok, closures_after, world_ok, reuse_ok = two_loops()
        println("result_ok=", result_ok)
        println("closures_after=", closures_after)
        println("world_ok=", world_ok)
        println("reuse_ok=", reuse_ok)
        """
    v = run_fresh(code)
    @test parse(Int, v["caches"]) >= 1
    @test v["closures_before"] == "0"
    @test v["result_ok"] == "true"
    # With a single thread, loops run serially and need no closure
    @test v["closures_after"] == (NW > 1 ? "1" : "0")
    @test v["world_ok"] == "true"
    @test v["reuse_ok"] == "true"

    # A package whose precompile workload runs loops on the team: its image holds caches
    # for its loop bodies, and no closure
    mktempdir() do dir
        pkg = joinpath(dir, "TTPrecompileTest")
        mkpath(joinpath(pkg, "src"))
        write(
            joinpath(pkg, "Project.toml"), """
            name = "TTPrecompileTest"
            uuid = "0d2a5a7e-5a5e-4d2d-9b7e-3c1f2e4a6b8c"
            version = "0.1.0"

            [deps]
            ThreadTeams = "100744f9-5cbb-42a9-938e-f15a20f71997"
            """
        )
        write(
            joinpath(pkg, "src", "TTPrecompileTest.jl"), """
            module TTPrecompileTest
            using ThreadTeams
            fill_index!(x) = tforeach(i -> (x[i] = Float64(i)), eachindex(x); scheduler = TeamScheduler())
            sum_squares(x) = tmapreduce(abs2, +, x; scheduler = TeamScheduler())
            # Precompile workload (loops run serially during precompilation)
            let x = zeros(8)
                fill_index!(x)
                sum_squares(x)
            end
            end
            """
        )
        # Pkg resolves the package's manifest (offline, from the installed versions) and
        # precompiles it; its cache files go to a temporary depot, in front of the others
        depot = joinpath(dir, "depot")
        env = Dict("JULIA_DEPOT_PATH" => join([depot; DEPOT_PATH], Sys.iswindows() ? ";" : ":"))
        code = REPORT * """
            using Pkg
            Pkg.offline(true)
            Pkg.activate($(repr(pkg)); io = devnull)
            Pkg.develop(; path = $(repr(pkgdir(ThreadTeams))), io = devnull)
            using ThreadTeams
            using TTPrecompileTest
            Fs = cached_types()
            println("caches=", length(Fs))
            println("closures_before=", closures(Fs))
            x = zeros(64)
            TTPrecompileTest.fill_index!(x)
            println("result_ok=", x == 1:64 && TTPrecompileTest.sum_squares(x) == sum(abs2, 1:64))
            println("closures_after=", closures(cached_types()))
            """
        v = run_fresh(code; env)
        cache = joinpath(depot, "compiled", "v$(VERSION.major).$(VERSION.minor)", "TTPrecompileTest")
        @test isdir(cache) && !isempty(readdir(cache))
        @test parse(Int, v["caches"]) >= 2   # the workload's `tforeach` and `tmapreduce`
        @test v["closures_before"] == "0"
        @test v["result_ok"] == "true"
        @test v["closures_after"] == (NW > 1 ? "2" : "0")
    end
end
