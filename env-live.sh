# usage:  source env-live.sh
# Clearing the sandbox overrides is what switches the collector to real Binance.
unset LOBF_WS_BASE LOBF_REST_BASE LOBF_STALE_TIMEOUT_S
export LOBF_DATA_ROOT=./live-data
export LOBF_SUBSCRIPTION_TIMEOUT_S=30
export LOBF_METRICS_INTERVAL_S=30
echo "live env -> real Binance, data in ./live-data"
