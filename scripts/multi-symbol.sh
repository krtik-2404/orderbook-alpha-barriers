#!/usr/bin/env bash
# One collector container per symbol. Nothing inside the collector knows that
# other symbols exist.
#
# That is the entire design. Multi-symbol handling in the collector would mean
# one process, one queue and one writer serving several archives: a crash, an
# OOM kill or a wedged writer then takes every symbol down together, and a
# sequence bug in one stream corrupts partitions belonging to another. Separate
# processes with separate bind mounts cannot do that to each other. The cost is
# a few hundred MB of RAM per symbol, which is the cheapest insurance here.
#
#   ./scripts/multi-symbol.sh                    # ethusdt solusdt dogeusdt
#   ./scripts/multi-symbol.sh ethusdt            # just one
#   ./scripts/multi-symbol.sh --stop             # stop the ones this started
#
# btcusdt is deliberately NOT in the default list: it has been collecting for
# 22 days under the name lobforge-capture and must not be restarted.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"

IMAGE="${LOBF_IMAGE:-lobforge:test}"
DEFAULT_SYMBOLS="ethusdt solusdt dogeusdt"

if [ "${1:-}" = "--stop" ]; then
    shift
    for sym in ${*:-$DEFAULT_SYMBOLS}; do
        docker rm -f "lobforge-$sym" 2>/dev/null && echo "stopped lobforge-$sym" \
            || echo "lobforge-$sym not running"
    done
    exit 0
fi

SYMBOLS="${*:-$DEFAULT_SYMBOLS}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image $IMAGE not found - build it first: docker compose build" >&2
    exit 1
fi
if [ ! -f "$ROOT/.env" ]; then
    echo "no .env - copy .env.example and edit it" >&2
    exit 1
fi

started=()
for sym in $SYMBOLS; do
    name="lobforge-$sym"
    if [ -n "$(docker ps -aq -f "name=^${name}$")" ]; then
        echo "  $name exists already - leaving it alone"
        continue
    fi

    # The image runs as its own uid 10001, but the bind mount is owned by the
    # host user, and a collector that cannot write is not collecting - it burns
    # 50 write failures and exits. Run as the host user rather than chowning the
    # archive to a uid that means nothing outside the container. This is what
    # the btcusdt container has been doing for 22 days.
    mkdir -p "$ROOT/data-$sym"
    docker run -d \
        --name "$name" \
        --restart unless-stopped \
        --stop-timeout 30 \
        --memory 1g \
        --user "$(id -u):$(id -g)" \
        --env-file "$ROOT/.env" \
        -e "LOBF_SYMBOL=$sym" \
        -e "LOBF_DATA_ROOT=/data" \
        --log-driver json-file --log-opt max-size=50m --log-opt max-file=5 \
        -v "$ROOT/data-$sym:/data" \
        "$IMAGE" >/dev/null
    echo "  started $name  ->  data-$sym/"
    started+=("$sym")
done

# ------------------------------------------------------------------- verify
# A subscribed stream that delivers nothing has no gap, no error and no stall:
# the data was never lost, it was never sent. The collector's subscription
# check is the only thing that catches it, so wait for it rather than assuming
# a container that is "Up" is collecting.
[ ${#started[@]} -eq 0 ] && exit 0
echo
echo "waiting for subscription verification (up to 90s each)..."
rc=0
for sym in "${started[@]}"; do
    name="lobforge-$sym"
    state=waiting
    for _ in $(seq 90); do
        if docker logs "$name" 2>&1 | grep -q "subscription verified"; then
            state=ok
            break
        fi
        if [ -z "$(docker ps -q -f "name=^${name}$")" ]; then
            state=dead
            break
        fi
        sleep 1
    done
    case "$state" in
        ok)   echo "  $name  $(docker logs "$name" 2>&1 |
                   grep -m1 'subscription verified')" ;;
        dead) echo "  $name DIED:"
              docker logs --tail 15 "$name" 2>&1 | sed 's/^/      /'; rc=1 ;;
        *)    echo "  $name still unverified after 90s - it is up but may not"
              echo "      be receiving; last lines:"
              docker logs --tail 10 "$name" 2>&1 | sed 's/^/      /'; rc=1 ;;
    esac
done
echo
docker ps --filter name=lobforge --format '  {{.Names}}\t{{.Status}}'
exit $rc
