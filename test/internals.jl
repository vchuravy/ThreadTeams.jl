# The scheduler internals that idle workers use to go to sleep when another task is ready
# (`other_work`). They are guarded with `isdefined`, so nothing fails when they move; this
# test tells when that happens on a supported release.
@testset "scheduler internals" begin
    if VERSION < v"1.14-"
        @test isdefined(Base, :workqueue_for)
        @test isdefined(Base, :Partr) && isdefined(Base.Partr, :multiq_check_empty)
        @test ThreadTeams.other_work() isa Bool
    end
end

# The cache line of each hot atomic field lies within its (16-byte aligned) object, so
# that neither the other fields of the pool nor a neighbouring object share it
@testset "cache line padding" begin
    L = ThreadTeams.CACHE_LINE
    @test L == (Sys.ARCH === :aarch64 ? 128 : 64)
    for (T, field) in [
            (ThreadTeams.ChunkCounter, :next), (ThreadTeams.WorkerState, :cmd),
            (ThreadTeams.SleepFlag, :sleeping),
        ]
        off = Int(fieldoffset(T, findfirst(==(field), fieldnames(T))))
        for base in 0:16:(L - 16)   # every alignment of the object within a line
            start = base + off - (base + off) % L
            @test start >= base
            @test start + L <= base + sizeof(T)
        end
    end
    # A worker's command and its closure form a two-word pair: always on the same line
    W = ThreadTeams.WorkerState
    off = Int(fieldoffset(W, findfirst(==(:cmd), fieldnames(W))))
    @test off % (2 * sizeof(Int)) == 0
    @test Int(fieldoffset(W, findfirst(==(:job), fieldnames(W)))) == off + sizeof(Int)
end

@testset "command encoding" begin
    for (n, skip, scope) in [(2, 0, false), (64, 3, true), (ThreadTeams.CMD_MASK, 17, true)]
        base = ThreadTeams.cmd_base(n, skip, scope)
        for chunk in (2, n)
            cmd = base | chunk
            @test cmd > 0
            @test ThreadTeams.cmd_chunk(cmd) == chunk
            @test ThreadTeams.cmd_nchunks(cmd) == n
            @test ThreadTeams.cmd_skip(cmd) == skip
            @test ThreadTeams.cmd_has_scope(cmd) == scope
        end
    end
end

# Chunk functions are stored into the workers' own slots if they are small (`COPY_BYTES`),
# else as the dispatcher's slot. Both work, also for the kinds of fields a closure can have,
# and leave no reference behind in the slots of the workers.
struct BigJob
    x::Vector{Float64}
    pad::NTuple{40, Float64}   # larger than `COPY_BYTES`
end
(j::BigJob)(i) = (j.x[i] += j.pad[1]; nothing)
mutable struct MutableJob
    x::Vector{Float64}
end
(j::MutableJob)(i) = (j.x[i] += 1.0; nothing)
struct UnionJob
    x::Vector{Float64}
    a::Union{Nothing, Float64}
    b::Int8
end
(j::UnionJob)(i) = (j.x[i] += something(j.a, 0.0) + j.b; nothing)

# `run_chunks` calls the chunk function with the chunk index
function slot_loops(n)
    x = zeros(n)
    ThreadTeams.run_chunks(BigJob(x, ntuple(_ -> 1.0, 40)), n, team)
    ThreadTeams.run_chunks(MutableJob(x), n, team)
    ThreadTeams.run_chunks(UnionJob(x, 0.5, Int8(2)), n, team)
    ThreadTeams.run_chunks(UnionJob(x, nothing, Int8(-1)), n, team)
    return x
end

# A slot's element has at least a cache line of unused elements on each side, and is empty
# when created
@testset "slot padding" begin
    for F in (Int, MutableJob, UnionJob, BigJob)
        slot = ThreadTeams.Slot{F}()
        mem = ThreadTeams.slot_memory(slot)
        i = ThreadTeams.slot_index(slot)
        sz = max(Base.aligned_sizeof(F), 1)
        @test (i - 1) * sz >= ThreadTeams.CACHE_LINE
        @test (length(mem) - i) * sz >= ThreadTeams.CACHE_LINE
        isbitstype(F) || @test !isassigned(slot)   # an isbits element is always assigned
    end
end

# A worker the last region saw idle is handed its next command without a check of its line
# (`seen_idle`); a region that fails clears the flags, so that the next one checks again
@testset "seen idle" begin
    if NW > 1
        # (the flags are kept only where the check is skipped, which is compiled in only on
        # ARM; the pool sets the setting when it starts, so start it first)
        tforeach(identity, 1:NW; scheduler = team)
        old = ThreadTeams.TRUST_SEEN_IDLE[]
        ThreadTeams.TRUST_SEEN_IDLE[] = true
        try
            kept = ThreadTeams.SKIPS_IDLE_CHECK ? NW - 1 : 0   # every worker but the caller's thread's
            tforeach(identity, 1:NW; scheduler = team)
            pool = ThreadTeams.POOL.pool
            @test count(pool.seen_idle) == kept
            @test_throws ErrorException tforeach(i -> i == NW && error("chunk $i"), 1:NW; scheduler = team)
            @test !any(pool.seen_idle)
            @test all(isnothing, pool.errs) && !pool.failed
            tforeach(identity, 1:NW; scheduler = team)
            @test count(pool.seen_idle) == kept
        finally
            ThreadTeams.TRUST_SEEN_IDLE[] = old
        end
        # chosen from the architecture, the environment variable overrides it
        @test ThreadTeams.trusts_seen_idle(:aarch64, "")
        @test !ThreadTeams.trusts_seen_idle(:x86_64, "")
        @test ThreadTeams.trusts_seen_idle(:x86_64, "unseen")
        @test !ThreadTeams.trusts_seen_idle(:aarch64, "always")
        @test (@test_logs (:warn,) ThreadTeams.trusts_seen_idle(:x86_64, "never")) == false
        @test ThreadTeams.TRUST_SEEN_IDLE[] == ThreadTeams.IDLE_CHECK()
        # and the results stay right over many regions, with and without the check
        for trust in (false, true)
            old = ThreadTeams.TRUST_SEEN_IDLE[]
            ThreadTeams.TRUST_SEEN_IDLE[] = trust
            try
                y = zeros(Int, NW)
                for _ in 1:1000
                    tforeach(i -> (y[i] += 1), 1:NW; scheduler = team)
                end
                @test all(==(1000), y)
            finally
                ThreadTeams.TRUST_SEEN_IDLE[] = old
            end
        end
        x = zeros(Int, NW)
        for _ in 1:1000
            tforeach(i -> (x[i] += 1), 1:NW; scheduler = team)
        end
        @test all(==(1000), x)
    end
end

# Who empties the workers' slots: chosen once per process from the CPU, the environment
# variable overrides it
@testset "slot clearing" begin
    clears = ThreadTeams.dispatcher_clears_slots
    @test clears("cascadelake", :x86_64, "")
    @test clears("broadwell", :x86_64, "")
    @test clears("alderlake", :x86_64, "")
    @test !clears("znver3", :x86_64, "")
    @test !clears("znver5", :x86_64, "")
    @test !clears("bdver2", :x86_64, "")
    @test !clears("generic", :x86_64, "")
    @test !clears("x86-64-v3", :x86_64, "")
    @test !clears("neoverse-v2", :aarch64, "")
    @test !clears("apple-m1", :aarch64, "")
    @test clears("apple-m1", :aarch64, "dispatcher")
    @test !clears("cascadelake", :x86_64, "worker")
    @test (@test_logs (:warn,) clears("cascadelake", :x86_64, "both")) == true
    # the pool reads it when it starts (with one thread there is no pool: loops run serially)
    if NW > 1
        tforeach(identity, 1:NW; scheduler = team)
        @test ThreadTeams.DISPATCHER_CLEARS[] == ThreadTeams.SLOT_CLEARING()
    end
end

# The slot tests below in both modes: `DISPATCHER_CLEARS` is read once per region, so it
# can change between regions
const SLOT_MODES = (false, true)
function with_slot_mode(f, mode)
    old = ThreadTeams.DISPATCHER_CLEARS[]
    ThreadTeams.DISPATCHER_CLEARS[] = mode
    return try
        f()
    finally
        ThreadTeams.DISPATCHER_CLEARS[] = old
    end
end

@testset "own slots, dispatcher clears: $mode" for mode in SLOT_MODES
    with_slot_mode(mode) do
        @test !ThreadTeams.own_slots(BigJob)
        @test ThreadTeams.own_slots(MutableJob) && ThreadTeams.own_slots(UnionJob)
        for _ in 1:100
            @test all(==(3.5), slot_loops(NW))   # 1 + 1 + 2.5 - 1
        end
        if NW > 1
            # a small chunk function has a slot per worker, a large one none
            for (F, nslots) in ((BigJob, 0), (MutableJob, NW), (UnionJob, NW))
                @test !isassigned(ThreadTeams.job_slot(F))
                slots = ThreadTeams.worker_slot_vector(F)
                @test length(slots) == nslots
                @test !any(isassigned, slots)
            end
        end
    end
end

# A chunk that throws on a worker leaves the chunk function in the worker's slot; the
# dispatcher drops it at the end of the region
struct ThrowingJob
    x::Vector{Float64}
end
(j::ThrowingJob)(i) = i == length(j.x) ? throw(ArgumentError("chunk $i")) : (j.x[i] = 1.0; nothing)

@testset "own slots: exceptions, dispatcher clears: $mode" for mode in SLOT_MODES
    with_slot_mode(mode) do
        if NW > 1
            x = zeros(NW)
            @test_throws ArgumentError ThreadTeams.run_chunks(ThrowingJob(x), NW, team)
            @test !isassigned(ThreadTeams.job_slot(ThrowingJob))
            @test !any(isassigned, ThreadTeams.worker_slot_vector(ThrowingJob))
            # The next loop with the same chunk function type works
            y = zeros(NW + 1)
            ThreadTeams.run_chunks(ThrowingJob(y), NW, team)
            @test all(==(1.0), y[1:NW])
            @test !any(isassigned, ThreadTeams.worker_slot_vector(ThrowingJob))
        end
    end
end
