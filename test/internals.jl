# Check the scheduler internals used by `local_work`, `queued_work` and `run_queued_task`.
# Their `isdefined` guards allow a fallback; this test catches changes on supported releases.
@testset "scheduler internals" begin
    if VERSION < v"1.14-"
        @test isdefined(Base, :workqueue_for)
        @test ThreadTeams.HAS_WORKQUEUE && ThreadTeams.HAS_MULTIQ
        @test ThreadTeams.local_work() isa Bool
        @test ThreadTeams.queued_work() isa Bool
    end
end

# The cache line of each hot atomic field lies within its (16-byte aligned) object, so
# that neither the other fields of the pool nor a neighbouring object share it
@testset "cache line padding" begin
    L = ThreadTeams.CACHE_LINE
    @test L == (Sys.ARCH === :aarch64 ? 128 : 64)
    for (T, field) in [
            (ThreadTeams.ChunkCounter, :next), (ThreadTeams.WorkerState, :cmd),
            (ThreadTeams.SleepFlag, :sleeping), (ThreadTeams.RangeBlock, :word),
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
    # So do the failure and stop flag of `foreach_range`
    for (T, a, b) in [(ThreadTeams.RangeState, :err, :stop)]
        off = Int(fieldoffset(T, findfirst(==(a), fieldnames(T))))
        @test off % (2 * sizeof(Int)) == 0
        @test Int(fieldoffset(T, findfirst(==(b), fieldnames(T)))) == off + sizeof(Int)
        i = findfirst(==(:_pad1), fieldnames(T))
        stop = Int(fieldoffset(T, i)) + sizeof(fieldtype(T, i))
        for base in 0:16:(L - 16)
            start = base + off - (base + off) % L
            @test start >= base
            @test start + L <= base + stop
        end
    end
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

# Oversubscription from the threads' affinity masks (e.g. after ThreadPinning.jl's
# `pinthreads`): Julia threads that have to share CPUs
@testset "oversubscribed affinity masks" begin
    mask(cpus...; n = 8) = [i - 1 in cpus for i in 1:n]
    over = ThreadTeams.oversubscribed_masks
    @test !over([mask(0), mask(1), mask(2), mask(3)])                 # pinthreads(:cores)
    @test over([mask(0), mask(0), mask(1), mask(1)])                  # pinthreads([0, 0, 1, 1])
    @test !over([mask(0:7...), mask(0), mask(1)])                     # unpinned main thread
    @test over([mask(0, 1), mask(0, 1), mask(0, 1)])                  # taskset -c 0,1, -t 3
    @test !over([mask(0, 1), mask(0, 1), mask(2)])
    @test over([mask(0:7...), mask(0), mask(0)])
    @test !over([mask(0:7...) for _ in 1:8])
    @test over([mask(0:7...) for _ in 1:9])
    @test ThreadTeams.is_oversubscribed() isa Bool
end

# Victim orders of steal regions on synthetic topologies: per worker an LLC id, a node
# (index into the distance matrix)
@testset "victim orders" begin
    orders = ThreadTeams.victim_orders
    # Each order lists the other workers once, its local part on the worker's node, and
    # restricted to the blocks of a smaller team (`p < nworkers()`) still covers them
    function valid(vo, node)
        nw = length(vo.order)
        return all(1:nw) do w
            o = vo.order[w]
            sort(o) == setdiff(1:nw, w) && all(v -> node[v] == node[w], o[1:vo.nlocal[w]]) &&
                all(p -> sort(filter(<=(p), o)) == setdiff(1:p, w), w:nw)
        end
    end
    d2 = [10 20; 20 10]
    # 2 nodes x 2 LLCs, compact (`pinthreads(:cores)`): the LLC, the rest of the node
    # (after w, wrapping), then the remote node from w's rank in its own
    llc = [0, 0, 2, 2, 4, 4, 6, 6]
    node = [1, 1, 1, 1, 2, 2, 2, 2]
    vo = orders(llc, node, d2)
    @test valid(vo, node)
    @test vo.order[1] == [2, 3, 4, 5, 6, 7, 8] && vo.nlocal[1] == 3
    @test vo.order[4] == [3, 1, 2, 8, 5, 6, 7] && vo.nlocal[4] == 3
    @test vo.order[6] == [5, 7, 8, 2, 3, 4, 1] && vo.nlocal[6] == 3
    # round-robin over the nodes (`pinthreads(:numa)`)
    llc = [0, 1, 0, 1, 4, 5, 4, 5]
    node = [1, 2, 1, 2, 1, 2, 1, 2]
    vo = orders(llc, node, d2)
    @test valid(vo, node)
    @test vo.order[1] == [3, 5, 7, 2, 4, 6, 8] && vo.nlocal[1] == 3
    @test vo.order[8] == [6, 2, 4, 7, 1, 3, 5] && vo.nlocal[8] == 3
    # sub-NUMA clustering: an LLC spans the two nodes of a socket, which are nearer to
    # each other than to the other socket's (ties by node)
    llc = [0, 0, 0, 0, 4, 4, 4, 4]
    node = [1, 1, 2, 2, 3, 3, 4, 4]
    d4 = [10 12 21 21; 12 10 21 21; 21 21 10 12; 21 21 12 10]
    vo = orders(llc, node, d4)
    @test valid(vo, node)
    @test vo.order[2] == [1, 4, 3, 6, 5, 8, 7] && vo.nlocal[2] == 1
    @test vo.order[5] == [6, 7, 8, 1, 2, 3, 4] && vo.nlocal[5] == 1
    # one node with two LLCs (a Ryzen with two CCDs)
    vo = orders([0, 0, 2, 2], ones(Int, 4), fill(10, 1, 1))
    @test vo.order[1] == [2, 3, 4] && vo.order[3] == [4, 1, 2] && vo.nlocal == fill(3, 4)
    # nothing to order by: cyclic
    @test orders([0, 1, 2, 3], ones(Int, 4), fill(10, 1, 1)) === nothing   # no shared cache
    @test orders(zeros(Int, 4), ones(Int, 4), fill(10, 1, 1)) === nothing
    @test orders([0], [1], fill(10, 1, 1)) === nothing
end

# The topology from a fake sysfs tree
@testset "sysfs topology" begin
    @test ThreadTeams.parse_cpulist("0-7,16-23\n") == [0:7; 16:23]
    @test ThreadTeams.parse_cpulist("3\n") == [3] && isempty(ThreadTeams.parse_cpulist("\n"))
    function fake(root; ncpu = 8, llc = 4, nodes = [0:3, 4:7], distance = true, l3 = true)
        for c in 0:(ncpu - 1)
            caches = [(1, "Data", "$c"), (1, "Instruction", "$c"), (2, "Unified", "$c")]
            first = c - c % llc
            l3 && push!(caches, (3, "Unified", "$first-$(first + llc - 1)"))
            for (i, (level, type, shared)) in enumerate(caches)
                dir = mkpath(joinpath(root, "cpu", "cpu$c", "cache", "index$(i - 1)"))
                write(joinpath(dir, "level"), "$level\n")
                write(joinpath(dir, "type"), "$type\n")
                write(joinpath(dir, "shared_cpu_list"), "$shared\n")
            end
        end
        for (i, cpus) in enumerate(nodes)
            dir = mkpath(joinpath(root, "node", "node$(i - 1)"))
            write(joinpath(dir, "cpulist"), "$(first(cpus))-$(last(cpus))\n")
            distance && write(
                joinpath(dir, "distance"),
                join([j == i ? 10 : 10 + 11abs(i - j) for j in eachindex(nodes)], ' ') * "\n"
            )
        end
        mkpath(joinpath(root, "node"))
        write(joinpath(root, "node", "online"), "0-$(length(nodes) - 1)\n")
        return root
    end
    read_topology = ThreadTeams.read_topology
    mktempdir() do dir
        root = fake(joinpath(dir, "a"); llc = 2)
        @test read_topology([0, 1, 2, 5, 7], root) == ([0, 0, 2, 4, 6], [1, 1, 1, 2, 2], [10 21; 21 10])
        # no shared cache reported (Ampere Altra): each CPU its own domain
        root = fake(joinpath(dir, "b"); l3 = false)
        @test read_topology([0, 1, 4], root)[1] == [0, 1, 4]
        # emulated NUMA with overlapping nodes (Raspberry Pi 5): one node, LLCs kept
        root = fake(joinpath(dir, "c"); ncpu = 4, llc = 4, nodes = [0:3 for _ in 1:8])
        @test read_topology(0:3, root) == ([0, 0, 0, 0], ones(Int, 4), fill(10, 1, 1))
        # without node information or distances
        root = fake(joinpath(dir, "d"); nodes = [], distance = false)
        @test read_topology([0, 4], root)[2:3] == ([1, 1], fill(10, 1, 1))
        root = fake(joinpath(dir, "e"); distance = false, nodes = [0:1, 2:3, 4:7])
        @test read_topology([0, 2, 4], root) == ([0, 0, 4], [1, 2, 3], [10 20 20; 20 10 20; 20 20 10])
        # a CPU outside the nodes: one node
        root = fake(joinpath(dir, "f"); nodes = [0:3])
        @test read_topology([0, 4], root)[2] == [1, 1]
        if Sys.islinux()
            mask(cpus...) = [i - 1 in cpus for i in 1:8]
            wvo = ThreadTeams.worker_victim_orders
            root = fake(joinpath(dir, "g"); llc = 2)
            vo = wvo([mask(c) for c in 0:7], root)
            @test vo.order[1] == [2, 3, 4, 5, 6, 7, 8] && vo.nlocal[1] == 3
            # unpinned, shared or missing masks, an unreadable tree: cyclic
            @test wvo([mask(0:7...) for _ in 1:8], root) === ThreadTeams.CYCLIC_VICTIMS
            @test wvo([mask(0), mask(0)], root) === ThreadTeams.CYCLIC_VICTIMS
            @test wvo(nothing, root) === ThreadTeams.CYCLIC_VICTIMS
            rm(joinpath(root, "cpu", "cpu1", "cache", "index3", "type"))
            @test wvo([mask(c) for c in 0:7], root) === ThreadTeams.CYCLIC_VICTIMS
        end
    end
end
