# Victim orders for steal regions (`run_steal`), from the cache and NUMA topology of the
# workers' CPUs. Computed once per pool, only when every worker is pinned to a CPU of its
# own (Linux); a performance heuristic, so anything unexpected selects cyclic order.

# `order[w]`: the other workers, nearest first; its first `nlocal[w]` entries share w's
# NUMA node. No orders: cyclic order (`CYCLIC_VICTIMS`).
struct VictimOrders
    order::Vector{Vector{Int32}}
    nlocal::Vector{Int32}
end
const CYCLIC_VICTIMS = VictimOrders(Vector{Int32}[], Int32[])

# `ws` (sorted) starting after `w`, wrapping around
rotate_after(ws, w) = vcat(filter(>(w), ws), filter(<(w), ws))

# Tiers of worker w's victims: its LLC within its node, the rest of its node, then the
# other nodes by distance (ties by node). Local tiers start after w, as cyclic order does;
# a remote node starts at w's rank within its own node, so that the thieves of a node
# spread over the remote victims. `llc[w]` and `node[w]` are domain ids of worker w,
# `node[w]` indexes `dist`. `nothing` if the result is cyclic order.
function victim_orders(llc::AbstractVector{<:Integer}, node::AbstractVector{<:Integer}, dist::AbstractMatrix{<:Integer})
    nw = length(llc)
    order = Vector{Vector{Int32}}(undef, nw)
    nlocal = zeros(Int32, nw)
    for w in 1:nw
        mates = filter(v -> node[v] == node[w], 1:nw)
        o = Int32[]
        append!(o, rotate_after(filter(v -> llc[v] == llc[w], mates), w))
        append!(o, rotate_after(filter(v -> llc[v] != llc[w], mates), w))
        nlocal[w] = length(o)
        rank = searchsortedfirst(mates, w) - 1
        remote = sort!(unique(node[v] for v in 1:nw if node[v] != node[w]); by = m -> (dist[node[w], m], m))
        for m in remote
            ws = filter(v -> node[v] == m, 1:nw)
            append!(o, circshift(ws, -(rank % length(ws))))
        end
        order[w] = o
    end
    all(w -> nlocal[w] == nw - 1 && order[w] == rotate_after(1:nw, w), 1:nw) && return nothing
    return VictimOrders(order, nlocal)
end

# CPUs of a cpulist such as "0-7,16-23"
function parse_cpulist(s::AbstractString)
    cpus = Int[]
    for part in split(strip(s), ',', keepempty = false)
        r = split(part, '-')
        lo = parse(Int, r[1])
        append!(cpus, lo:(length(r) == 1 ? lo : parse(Int, r[2])))
    end
    return cpus
end

# Per CPU in `cpus` (kernel numbering, as in affinity masks): an LLC id and a NUMA node
# (an index into the distance matrix). Throws on unreadable files.
# Read directly: SysInfo.jl (ThreadPinning's) has no cache-sharing domains or node
# distances, and it and Hwloc.jl cost every user 50-250 ms; this takes ~0.6 ms and works
# whichever tool pinned the threads (taskset, numactl, Slurm).
function read_topology(cpus::AbstractVector{<:Integer}, root::AbstractString = "/sys/devices/system")
    llc = [cache_domain(root, c) for c in cpus]
    node, dist = numa_nodes(root, cpus)
    return llc, node, dist
end

# First CPU sharing the highest-level unified or data cache of `cpu`; itself without
# cache information. Not `index3`: ARM servers may report no shared cache, or other levels.
function cache_domain(root::AbstractString, cpu::Integer)
    dir = joinpath(root, "cpu", "cpu$cpu", "cache")
    isdir(dir) || return Int(cpu)
    level = 0
    domain = Int(cpu)
    for d in readdir(dir)
        startswith(d, "index") || continue
        type = strip(read(joinpath(dir, d, "type"), String))
        (type == "Unified" || type == "Data") || continue
        l = parse(Int, read(joinpath(dir, d, "level"), String))
        if l > level
            level = l
            domain = minimum(parse_cpulist(read(joinpath(dir, d, "shared_cpu_list"), String)))
        end
    end
    return domain
end

# Node index of each CPU and the node distances. One node if there is no node
# information, or the nodes overlap: emulated NUMA, e.g. 8 nodes of 4 CPUs each on a
# Raspberry Pi 5.
function numa_nodes(root::AbstractString, cpus::AbstractVector{<:Integer})
    single = (ones(Int, length(cpus)), fill(10, 1, 1))
    dir = joinpath(root, "node")
    isdir(dir) || return single
    ids = sort!([parse(Int, d[5:end]) for d in readdir(dir) if occursin(r"^node\d+$", d)])
    isempty(ids) && return single
    nodeof = Dict{Int, Int}()
    for (i, id) in enumerate(ids)
        for c in parse_cpulist(read(joinpath(dir, "node$id", "cpulist"), String))
            haskey(nodeof, c) && return single
            nodeof[c] = i
        end
    end
    all(c -> haskey(nodeof, c), cpus) || return single
    # Rows list the distances to the nodes in increasing order
    dist = [i == j ? 10 : 20 for i in eachindex(ids), j in eachindex(ids)]
    for (i, id) in enumerate(ids)
        file = joinpath(dir, "node$id", "distance")
        isfile(file) || continue
        row = parse.(Int, split(read(file, String)))
        length(row) == length(ids) && (dist[i, :] = row)
    end
    return [nodeof[c] for c in cpus], dist
end

# The CPU of each mask if each has exactly one and they differ, else `nothing`
function pinned_cpus(masks::AbstractVector{<:AbstractVector{Bool}})
    all(m -> count(m) == 1, masks) || return nothing
    cpus = [findfirst(m) - 1 for m in masks]
    return allunique(cpus) ? cpus : nothing
end

# Victim orders of the pool's workers from their affinity masks (`thread_affinity_masks`),
# cyclic unless all are pinned to distinct CPUs on Linux. Pinning after the pool exists
# leaves its orders stale (`ThreadTeams.shutdown!()` rebuilds them).
function worker_victim_orders(masks, root::AbstractString = "/sys/devices/system")
    (Sys.islinux() && masks !== nothing) || return CYCLIC_VICTIMS
    cpus = pinned_cpus(masks)
    cpus === nothing && return CYCLIC_VICTIMS
    topology = try
        read_topology(cpus, root)
    catch
        return CYCLIC_VICTIMS
    end
    orders = victim_orders(topology...)
    return orders === nothing ? CYCLIC_VICTIMS : orders
end
