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
            (ThreadTeams.SleepFlag, :sleeping), (ThreadTeams.Slot{Int}, :x),
        ]
        off = Int(fieldoffset(T, findfirst(==(field), fieldnames(T))))
        for base in 0:16:(L - 16)   # every alignment of the object within a line
            start = base + off - (base + off) % L
            @test start >= base
            @test start + L <= base + sizeof(T)
        end
    end
    # A worker's command and its closure form a 16-byte pair: always on the same line
    W = ThreadTeams.WorkerState
    off = Int(fieldoffset(W, findfirst(==(:cmd), fieldnames(W))))
    @test off % 16 == 0
    @test Int(fieldoffset(W, findfirst(==(:job), fieldnames(W)))) == off + 8
end

@testset "command encoding" begin
    for (n, skip, scope) in [(2, 0, false), (64, 3, true), (ThreadTeams.CMD_MASK, 17, true)]
        base = ThreadTeams.cmd_base(n, skip, scope, !scope)
        for chunk in (2, n)
            cmd = base | chunk
            @test cmd > 0
            @test ThreadTeams.cmd_chunk(cmd) == chunk
            @test ThreadTeams.cmd_nchunks(cmd) == n
            @test ThreadTeams.cmd_skip(cmd) == skip
            @test ThreadTeams.cmd_has_scope(cmd) == scope
            @test ThreadTeams.cmd_has_payload(cmd) == !scope
        end
    end
end

# The payload buffer of a worker is a whole number of cache lines of its own
@testset "payload buffer" begin
    L = ThreadTeams.CACHE_LINE
    ws = ThreadTeams.WorkerState()
    p = UInt(ws.payload_ptr)
    start = UInt(pointer_from_objref(ws.payload))
    @test p % L == 0
    @test start <= p
    @test p + ThreadTeams.PAYLOAD_BYTES <= start + sizeof(ThreadTeams.PayloadBuffer)
end

# Chunk functions are handed to the workers as bytes if they fit into the payload buffer,
# else as the dispatcher's slot. Both work, also for the kinds of fields a closure can have,
# and leave no reference behind in the slots of the workers.
struct BigJob
    x::Vector{Float64}
    pad::NTuple{40, Float64}   # does not fit into the payload buffer
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
function payload_loops(n)
    x = zeros(n)
    ThreadTeams.run_chunks(BigJob(x, ntuple(_ -> 1.0, 40)), n, team)
    ThreadTeams.run_chunks(MutableJob(x), n, team)
    ThreadTeams.run_chunks(UnionJob(x, 0.5, Int8(2)), n, team)
    ThreadTeams.run_chunks(UnionJob(x, nothing, Int8(-1)), n, team)
    return x
end

@testset "payload hand-off" begin
    @test sizeof(BigJob) > ThreadTeams.PAYLOAD_BYTES
    for _ in 1:100
        @test all(==(3.5), payload_loops(NW))   # 1 + 1 + 2.5 - 1
    end
    if NW > 1
        for F in (BigJob, MutableJob, UnionJob)
            @test !isdefined(ThreadTeams.job_slot(F), :x)
            slots = ThreadTeams.worker_slot_vector(F)
            @test length(slots) == NW
            @test !any(slot -> isdefined(slot, :x), slots)
        end
    end
end

# A chunk that throws on a worker leaves its copy of the chunk function in the worker's
# slot; the dispatcher drops it at the end of the region
struct ThrowingJob
    x::Vector{Float64}
end
(j::ThrowingJob)(i) = i == length(j.x) ? throw(ArgumentError("chunk $i")) : (j.x[i] = 1.0; nothing)

@testset "payload hand-off: exceptions" begin
    if NW > 1
        x = zeros(NW)
        @test_throws ArgumentError ThreadTeams.run_chunks(ThrowingJob(x), NW, team)
        @test !isdefined(ThreadTeams.job_slot(ThrowingJob), :x)
        @test !any(slot -> isdefined(slot, :x), ThreadTeams.worker_slot_vector(ThrowingJob))
        # The next loop with the same chunk function type works
        y = zeros(NW + 1)
        ThreadTeams.run_chunks(ThrowingJob(y), NW, team)
        @test all(==(1.0), y[1:NW])
        @test !any(slot -> isdefined(slot, :x), ThreadTeams.worker_slot_vector(ThrowingJob))
    end
end
