# Markdown tables from the `results.csv` files of `run.sh`, one table per benchmark and metric
# with a row per machine, Julia version and thread count and a column per mechanism, as in
# https://github.com/vchuravy/ThreadTeams.jl/issues/46. With several rounds (`REPEAT`) or
# files for the same configuration, a cell shows the range "min–max" of the values.
#
#     julia benchmark/sweep/summarize.jl benchmark/sweep/results/*/results.csv
#     julia benchmark/sweep/summarize.jl --bench overhead --metric min_us results.csv
#
# Options: `--bench B` and `--metric M` (repeatable) select the tables (default: the ones in
# `DEFAULT_TABLES`, each with a selection of the columns), `--mechanism REGEX` the columns,
# `--digits D` the rounding.
using Printf

# (benchmark, metric, columns): the tables printed without `--bench` and `--metric`
const DEFAULT_TABLES = [
    ("overhead", "median_us", r"^(?!.*FSV)"), ("overhead", "min_us", r"^(?!.*FSV)"),
    ("syncbench", "overhead_median_us", r""),
    # Of EPCC's tests, the ones that the Julia mechanisms can be compared with
    ("epcc", "overhead_median_us", r"^(PARALLEL FOR|BARRIER|REDUCTION) "),
    ("wakeup", "added_median_us", r"busy gap=(200|1000|10000)us"),
    ("wakeup", "median_us", r"spin0"),
    ("counters", "caller:cycles", r""), ("counters", "caller:instr", r""),
    ("counters", "caller:L1d_miss", r""), ("counters", "caller:LLC_miss", r""),
    ("counters", "all:ctx_switches", r""),
    ("steal", "mean_ns", r"w=(0|1000|10000)ns"), ("steal", "stolen_pct", r"w=(0|1000|10000)ns"),
]

function parse_csv_line(line)
    fields = String[]
    buf = IOBuffer()
    inquote = false
    i = firstindex(line)
    while i <= lastindex(line)
        c = line[i]
        if inquote
            if c == '"'
                j = nextind(line, i)
                if j <= lastindex(line) && line[j] == '"'
                    write(buf, '"')
                    i = j
                else
                    inquote = false
                end
            else
                write(buf, c)
            end
        elseif c == '"'
            inquote = true
        elseif c == ','
            push!(fields, String(take!(buf)))
        else
            write(buf, c)
        end
        i = nextind(line, i)
    end
    push!(fields, String(take!(buf)))
    return fields
end

function read_rows(paths)
    rows = Dict{String, String}[]
    for path in paths
        lines = readlines(path)
        isempty(lines) && continue
        header = parse_csv_line(lines[1])
        for line in lines[2:end]
            startswith(line, "host,") && continue   # header of an appended file
            f = parse_csv_line(line)
            length(f) == length(header) && push!(rows, Dict(zip(header, f)))
        end
    end
    return rows
end

fmt(v, digits) = abs(v) >= 100 ? @sprintf("%.0f", v) : Printf.format(Printf.Format("%.$(digits)f"), v)
function cell(vs, digits)
    isempty(vs) && return ""
    lo, hi = extrema(vs)
    a, b = fmt(lo, digits), fmt(hi, digits)
    return a == b ? a : "$(a)–$(b)"
end

function table(io, rows, bench, metric; mechanism = r"", digits = 2)
    sel = filter(r -> r["bench"] == bench && r["metric"] == metric && occursin(mechanism, r["mechanism"]), rows)
    isempty(sel) && return
    mechs = unique(r["mechanism"] for r in sel)
    key(r) = (r["host"], r["julia"], parse(Int, r["threads"]))
    keys_ = sort!(unique(key.(sel)))
    vals = Dict{Tuple{Any, String}, Vector{Float64}}()
    for r in sel
        v = tryparse(Float64, r["value"])
        v === nothing || push!(get!(vals, (key(r), r["mechanism"]), Float64[]), v)
    end
    println(io, "### ", bench, ": ", metric, "\n")
    println(io, "| host | julia | threads | ", join(mechs, " | "), " |")
    println(io, "|---|---|---|", repeat("---|", length(mechs)))
    for k in keys_
        println(io, "| ", k[1], " | ", k[2], " | ", k[3], " | ", join((cell(get(vals, (k, m), Float64[]), digits) for m in mechs), " | "), " |")
    end
    println(io)
    return
end

function main(args)
    paths = String[]
    benches, metrics = String[], String[]
    mechanism = nothing
    digits = 2
    i = 1
    while i <= length(args)
        a = args[i]
        if a in ("--bench", "--metric", "--mechanism", "--digits")
            v = args[i += 1]
            a == "--bench" ? push!(benches, v) : a == "--metric" ? push!(metrics, v) :
                a == "--mechanism" ? (mechanism = Regex(v)) : (digits = parse(Int, v))
        else
            push!(paths, a)
        end
        i += 1
    end
    isempty(paths) && (println(stderr, "usage: summarize.jl [options] results.csv..."); exit(1))
    rows = read_rows(paths)
    if isempty(benches) && isempty(metrics)
        tables = DEFAULT_TABLES
    else
        allpairs = unique((r["bench"], r["metric"]) for r in rows)
        tables = [
            (b, m, r"") for (b, m) in allpairs
                if (isempty(benches) || b in benches) && (isempty(metrics) || m in metrics)
        ]
    end
    for (b, m, cols) in tables
        table(stdout, rows, b, m; mechanism = something(mechanism, cols), digits)
    end
    return
end

main(ARGS)
