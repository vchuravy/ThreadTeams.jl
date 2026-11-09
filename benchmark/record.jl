# Machine-readable results for `benchmark/sweep/run.sh`: `record` appends one row to the CSV
# file named by `ENV["BENCH_CSV"]`, and does nothing if that is not set, so the scripts in
# `benchmark/overhead` run as before on their own. The columns are those of `CSV_HEADER`;
# `host` and `config` come from `BENCH_HOST` and `BENCH_CONFIG` (set by the sweep script).

const CSV_HEADER = "host,julia,threads,config,bench,mechanism,metric,value"

csv_field(s) = (s = string(s); occursin(r"[\",\n]", s) ? "\"" * replace(s, "\"" => "\"\"") * "\"" : s)

function record(bench, mechanism, metric, value)
    path = get(ENV, "BENCH_CSV", "")
    isempty(path) && return
    row = join(
        csv_field.(
            (
                get(ENV, "BENCH_HOST", gethostname()), VERSION,
                Threads.nthreads(:default), get(ENV, "BENCH_CONFIG", ""),
                bench, mechanism, metric, value,
            )
        ), ","
    )
    open(path, "a") do io
        filesize(path) == 0 && println(io, CSV_HEADER)
        println(io, row)
    end
    return
end
