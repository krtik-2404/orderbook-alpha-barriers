# LOBForge — Deployment Runbook

Kali (dev) → GitHub → Ubuntu 24.04 VPS → Docker → collecting 24/7.
~45 minutes. Run [SANDBOX.md](SANDBOX.md) first — nothing here is urgent until
the fault suite passes.

## 0. Region matters

Binance futures infrastructure sits primarily in AWS Tokyo. RTT drives TCP
retransmits → sequence gaps → unusable dataset windows.

| Region | RTT | Expected gaps |
|---|---|---|
| Tokyo | 5–15 ms | very low |
| Singapore | 35–70 ms | low |
| Germany / EU | 230–260 ms | noticeably higher |
| Mumbai | 120–180 ms | moderate |

Recommended: **Vultr or Linode, Tokyo, ~$12/mo** (1 vCPU, 2 GB, 50 GB SSD).
On a 1 GB box set `LOBF_QUEUE_MAXSIZE=50000`. Image: **Ubuntu 24.04 LTS**, SSH
key added at creation — never password login.

## 1. SSH key (on Kali)

    ssh-keygen -t ed25519 -C "lobforge-vps"
    cat ~/.ssh/id_ed25519.pub        # paste into the provider's SSH Keys field

    cat >> ~/.ssh/config <<'EOF'

    Host lobforge
        HostName YOUR.VPS.IP
        User lobforge
        IdentityFile ~/.ssh/id_ed25519
        ServerAliveInterval 60
    EOF
    chmod 600 ~/.ssh/config

## 2. First login: user, firewall, clock

    ssh root@YOUR.VPS.IP

    adduser --disabled-password --gecos "" lobforge
    usermod -aG sudo lobforge
    rsync --archive --chown=lobforge:lobforge ~/.ssh /home/lobforge/

    sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
    sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
    systemctl restart ssh

    ufw allow OpenSSH && ufw --force enable

    apt update && apt upgrade -y
    apt install -y unattended-upgrades git curl
    timedatectl set-timezone UTC
    timedatectl set-ntp true

**Do not skip the clock.** Every record carries a local nanosecond timestamp; if
NTP drifts, your latency measurements become fiction and you find out at analysis
time.

**Before closing this session**, open a second terminal and confirm
`ssh lobforge` works.

## 3. Docker

    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker $USER
    newgrp docker
    docker run --rm hello-world

Small box — add swap so a memory spike doesn't OOM-kill the collector:

    sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
    sudo mkswap /swapfile && sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

## 4. Ship the code

On Kali:

    cd lobforge
    git init && git add -A
    git commit -m "capture tier + sandbox"
    gh repo create lobforge --private --source=. --push
    # or: git remote add origin git@github.com:<you>/lobforge.git && git push -u origin main

On the VPS:

    git clone git@github.com:<you>/lobforge.git && cd lobforge

Deploys thereafter: `git pull && docker compose up -d --build`.

Alternative without a repo:

    rsync -avz --exclude data --exclude .git ~/lobforge/ lobforge:~/lobforge/

## 5. Launch

    cp .env.example .env
    nano .env
    mkdir -p data
    docker compose up -d --build
    docker compose logs -f

Expect within ~30s: connect line, `snapshot ok (connect) rtt=…`, an `opened …
.part` line, then a counter line every 60s.

## 6. Verify

    docker compose logs --tail 5 capture            # rates non-zero
    cat data/heartbeat; stat -c '%y' data/heartbeat # fresh (<60s)
    du -sh data/*                                   # bytes landing
    zcat data/depth/date=*/hour=*/*.part 2>/dev/null | head -2 | python3 -m json.tool | head -20

After 24 hours, the number that decides everything:

    zcat data/gaps/date=*/hour=*/*.gz data/gaps/date=*/hour=*/*.part 2>/dev/null | wc -l

A handful per day is normal. Dozens per hour means the network path is poor —
change region before committing three weeks. This is why you run a 24-hour trial
first.

## 7. Keep it alive

    sudo tee /etc/cron.hourly/lobforge-disk >/dev/null <<'EOF'
    #!/bin/sh
    USE=$(df --output=pcent /home | tail -1 | tr -dc '0-9')
    [ "$USE" -gt 85 ] && logger -t lobforge "DISK $USE% FULL"
    EOF
    sudo chmod +x /etc/cron.hourly/lobforge-disk

Docker log rotation is capped in `docker-compose.yml`. `restart: unless-stopped`
handles reboots; `stop_grace_period: 30s` gives the drain-and-seal path room.

    docker compose ps                 # health
    docker compose restart capture    # SIGTERM → drain → seal
    docker compose up -d --build      # after a git pull

## 8. Pull data back

    ./sync-data.sh

Only sealed `.jsonl.gz` files are transferred — never `.part`. Safe to run during
collection; that is what the atomic-rename protocol buys. Run weekly so a VPS
failure never costs more than a week.

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `permission denied` on `data/` | container UID 10001 vs host owner | `sudo chown -R 10001:10001 data` |
| No frames, endless reconnects | endpoint blocked / region issue | `curl -sI https://fapi.binance.com/fapi/v1/ping` |
| Container `unhealthy` | heartbeat stale > 180s | `docker compose logs --tail 100 capture` |
| Exit 137 | OOM | lower `LOBF_QUEUE_MAXSIZE`, add swap |
| Rising gap count | network path quality | change region |
| `.part` never sealing | normal — hourly boundary | wait |
