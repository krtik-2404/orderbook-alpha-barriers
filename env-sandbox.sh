# usage:  source env-sandbox.sh
export LOBF_WS_BASE=ws://127.0.0.1:8765
export LOBF_REST_BASE=http://127.0.0.1:8765
export LOBF_DATA_ROOT=./sandbox-data
export LOBF_SUBSCRIPTION_TIMEOUT_S=5
export LOBF_METRICS_INTERVAL_S=6
export LOBF_STALE_TIMEOUT_S=60
echo "sandbox env -> 127.0.0.1:8765, data in ./sandbox-data"
