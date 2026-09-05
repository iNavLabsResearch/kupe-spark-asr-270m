#!/usr/bin/env bash
# Terminate TLS on :443 (Let's Encrypt IP cert) and proxy to uvicorn :8000.
# Leaves the ASR process untouched. Open DigitalOcean firewall TCP 80 + 443.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
HOST="${ASR_HOST:-137.184.140.206}"
CADDY="${CADDY_BIN:-$ROOT/.caddy}"

arch="$(uname -m)"
case "$arch" in
  aarch64|arm64) caddy_arch=arm64 ;;
  x86_64|amd64)  caddy_arch=amd64 ;;
  *) echo "unsupported arch: $arch"; exit 1 ;;
esac

if ! command -v "$CADDY" >/dev/null 2>&1 && [[ ! -x "$CADDY" ]]; then
  echo "downloading caddy 2.10.2 ($caddy_arch)…"
  tmp="$(mktemp -d)"
  curl -fsSL "https://github.com/caddyserver/caddy/releases/download/v2.10.2/caddy_2.10.2_linux_${caddy_arch}.tar.gz" \
    | tar -xz -C "$tmp" caddy
  mv "$tmp/caddy" "$CADDY"
  chmod +x "$CADDY"
  rm -rf "$tmp"
fi

if ! curl -fsS --max-time 2 http://127.0.0.1:8000/ >/dev/null; then
  echo "WARNING: uvicorn is not on :8000. In another tmux pane first:"
  echo "  cd ~/kupe-spark-asr-270m && source venv/bin/activate && python server/run.py"
fi

echo "Caddy → https://$HOST  (proxy 127.0.0.1:8000)"
echo "Need DigitalOcean firewall TCP 80 + 443. Leave this process running."
exec "$CADDY" run --config "$ROOT/Caddyfile" --adapter caddyfile
