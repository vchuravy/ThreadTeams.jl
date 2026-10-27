#!/usr/bin/env bash
# Runs `run.sh` on remote machines over ssh and collects the results:
#
#     benchmark/sweep/remote.sh start  host1 host2   # copy this checkout, start run.sh
#     benchmark/sweep/remote.sh status host1 host2   # running? last lines of its log
#     benchmark/sweep/remote.sh fetch  host1 host2   # results/ into benchmark/sweep/results
#     julia benchmark/sweep/summarize.jl benchmark/sweep/results/*/results.csv
#
# The hosts are ssh destinations (names from ~/.ssh/config, or user@host). `start` copies
# the checkout, including uncommitted changes and untracked files that git does not
# ignore, to `~/$REMOTE_DIR` (default
# threadteams-sweep) and runs `run.sh` there under `nohup`, with the settings below passed
# on from this shell (see run.sh), so that the ssh connection may close.
set -euo pipefail

SWEEP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SWEEP/../.." && pwd)
REMOTE_DIR=${REMOTE_DIR:-threadteams-sweep}
PASS=(JULIA JULIA_CHANNELS INSTALL_JULIA THREADS CPUS BENCHES REPEAT QUICK LOCK EPCC_CC
    EPCC_CFLAGS COUNTERS_EVENTS)

usage() {
    echo "usage: $0 start|status|fetch|stop host..." >&2
    exit 1
}
(($# >= 2)) || usage
cmd=$1
shift

env_args() {
    local v out=""
    for v in "${PASS[@]}"; do
        [[ -n ${!v+x} ]] && out+=" $v=$(printf '%q' "${!v}")"
    done
    echo "$out"
}

case $cmd in
    start)
        # A tree of the working tree, with untracked files that are not ignored, written
        # through a copy of the index so that the real one stays as it is
        index=$(mktemp)
        cp "$(git -C "$ROOT" rev-parse --path-format=absolute --git-path index)" "$index"
        GIT_INDEX_FILE=$index git -C "$ROOT" add -A
        ref=$(GIT_INDEX_FILE=$index git -C "$ROOT" write-tree)
        rm -f "$index"
        commit=$(git -C "$ROOT" rev-parse HEAD)
        dirty=$([[ $ref == "$(git -C "$ROOT" rev-parse 'HEAD^{tree}')" ]] && echo "" || echo " + local changes")
        for h in "$@"; do
            echo "== $h: copying $commit$dirty to ~/$REMOTE_DIR"
            # shellcheck disable=SC2029  # expanded here on purpose
            ssh "$h" "test ! -f ~/$REMOTE_DIR/sweep.pid || ! kill -0 \$(cat ~/$REMOTE_DIR/sweep.pid) 2>/dev/null" ||
                { echo "   a sweep is still running there (remote.sh stop $h)"; continue; }
            git -C "$ROOT" archive --format=tar "$ref" |
                ssh "$h" "mkdir -p ~/$REMOTE_DIR && tar -xm -C ~/$REMOTE_DIR && echo '$commit$dirty' > ~/$REMOTE_DIR/.sweep-commit"
            # The EPCC sources, if fetched here, so that the host need not reach GitHub
            for t in "$SWEEP"/.cache/epcc-*.tar.gz; do
                [[ -s $t ]] || continue
                ssh "$h" "mkdir -p ~/$REMOTE_DIR/benchmark/sweep/.cache && cat > ~/$REMOTE_DIR/benchmark/sweep/.cache/${t##*/}" <"$t"
            done
            # shellcheck disable=SC2029
            # In a session of its own (if there is `setsid`), so that `stop` can end it all
            ssh "$h" "cd ~/$REMOTE_DIR || exit 1; \$(command -v setsid) nohup env$(env_args) \
                bash benchmark/sweep/run.sh > sweep.log 2>&1 < /dev/null & echo \$! > sweep.pid"
            echo "   started; log in ~/$REMOTE_DIR/sweep.log"
        done
        ;;
    status)
        for h in "$@"; do
            echo "== $h"
            # shellcheck disable=SC2029
            ssh "$h" "cd ~/$REMOTE_DIR 2>/dev/null || { echo '   nothing there'; exit 0; }
                if [ -f sweep.pid ] && kill -0 \$(cat sweep.pid) 2>/dev/null; then echo '   running'; else echo '   not running'; fi
                tail -n 3 sweep.log 2>/dev/null | sed 's/^/   /'"
        done
        ;;
    stop)
        for h in "$@"; do
            # The process group of run.sh, with the benchmark it runs
            # shellcheck disable=SC2029
            if ssh "$h" "cd ~/$REMOTE_DIR && [ -f sweep.pid ] && pid=\$(cat sweep.pid) &&
                    { kill -TERM -- -\$pid 2>/dev/null || kill -TERM \$pid; } && rm -f sweep.pid"; then
                echo "== $h: stopped"
            else
                echo "== $h: not running"
            fi
        done
        ;;
    fetch)
        mkdir -p "$SWEEP/results"
        for h in "$@"; do
            echo "== $h"
            # shellcheck disable=SC2029
            ssh "$h" "cd ~/$REMOTE_DIR/benchmark/sweep && tar -c results" | tar -x -C "$SWEEP"
        done
        ls -d "$SWEEP"/results/*/
        ;;
    *) usage ;;
esac
