FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

# Run unprivileged; the data volume is the only thing we need to write.
RUN useradd -r -u 10001 lobforge && mkdir -p /data && chown lobforge:lobforge /data
USER lobforge

ENV LOBF_DATA_ROOT=/data
VOLUME ["/data"]

# Liveness comes from the heartbeat file, not an in-process check: a wedged event
# loop would still answer an HTTP healthcheck. Gates on the `healthy` flag too,
# so a silently-missing stream marks the container unhealthy.
HEALTHCHECK --interval=60s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import sys,time,os,json; p='/data/heartbeat'; \
sys.exit(0 if os.path.exists(p) and time.time()-os.path.getmtime(p) < 180 \
and json.load(open(p)).get('healthy', True) else 1)"

# SIGTERM triggers the drain-and-seal path; give it room to finish.
STOPSIGNAL SIGTERM
CMD ["lobforge-capture"]
