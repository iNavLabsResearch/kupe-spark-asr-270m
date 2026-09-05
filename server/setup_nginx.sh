#!/usr/bin/env bash
# Run as root ON the droplet (asr-db-fetcher).
# Uvicorn stays on :8000. Nginx terminates TLS for spark-asr.kupe.in → wss://spark-asr.kupe.in/ws
#
# BEFORE this script: add a Hostinger DNS A record (DNS only, not proxied):
#   Type A | Host spark-asr | Points to 137.184.140.206 | TTL 300
#
# Then:
#   cd ~/kupe-spark-asr-270m
#   source venv/bin/activate && python server/run.py     # tmux pane 1
#   sudo bash server/setup_nginx.sh                      # tmux pane 2
set -euo pipefail

DOMAIN="${DOMAIN:-spark-asr.kupe.in}"
ORIGIN="${ORIGIN:-137.184.140.206}"
BACKEND="${BACKEND:-127.0.0.1:8000}"
EMAIL="${EMAIL:-team@kupe.in}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
SITE="/etc/nginx/sites-available/${DOMAIN}"
WEBROOT="/var/www/certbot"

if [ "$(id -u)" -ne 0 ]; then
  echo "run as root:  sudo bash $0"
  exit 1
fi

echo "==> stopping Caddy if it is holding :80/:443"
pkill -x caddy 2>/dev/null || true
sleep 1

echo "==> installing nginx + certbot"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq nginx certbot python3-certbot-nginx dnsutils curl

mkdir -p "$WEBROOT" /etc/nginx/sites-available /etc/nginx/sites-enabled
rm -f /etc/nginx/sites-enabled/default

echo "==> waiting for DNS ${DOMAIN} → ${ORIGIN}"
for i in $(seq 1 36); do
  got="$(dig +short "$DOMAIN" A | head -1 | tr -d '[:space:]')"
  echo "    try $i: $got"
  if [ "$got" = "$ORIGIN" ]; then
    break
  fi
  if [ "$i" -eq 36 ]; then
    echo "DNS is not ready. In Hostinger → DNS → Add record:"
    echo "  A    spark-asr    ${ORIGIN}    TTL 300"
    echo "Then re-run this script."
    exit 1
  fi
  sleep 5
done

echo "==> writing nginx site + websocket map"
cat > /etc/nginx/conf.d/websocket_upgrade.conf <<'NGX'
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}
NGX

cat > "$SITE" <<NGX
server {
    listen 80;
    listen [::]:80;
    server_name ${DOMAIN};

    location /.well-known/acme-challenge/ {
        root ${WEBROOT};
    }

    location / {
        proxy_pass http://${BACKEND};
        proxy_http_version 1.1;
        proxy_set_header Host              \$host;
        proxy_set_header X-Real-IP         \$remote_addr;
        proxy_set_header X-Forwarded-For   \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header Upgrade           \$http_upgrade;
        proxy_set_header Connection        \$connection_upgrade;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
        proxy_buffering off;
    }
}
NGX

ln -sfn "$SITE" "/etc/nginx/sites-enabled/${DOMAIN}"
nginx -t
systemctl enable nginx
systemctl restart nginx

if ! curl -fsS --max-time 3 "http://${BACKEND}/" >/dev/null; then
  echo "WARNING: nothing on ${BACKEND}. Start ASR in another pane:"
  echo "  cd ~/kupe-spark-asr-270m && source venv/bin/activate && python server/run.py"
fi

echo "==> issuing Let's Encrypt cert for ${DOMAIN}"
certbot --nginx -d "$DOMAIN" --email "$EMAIL" --agree-tos --non-interactive --redirect

# Make sure websocket headers survived certbot's rewrite
if ! grep -q "Upgrade" "$SITE"; then
  echo "==> restoring websocket headers after certbot"
  cp "$ROOT/nginx.spark-asr.conf" "$SITE"
  # certbot already wrote the live certs; ssl files exist
  nginx -t && systemctl reload nginx
fi

echo
echo "done. frontend should connect to  wss://${DOMAIN}/ws"
echo "health: curl -sS https://${DOMAIN}/"
echo "keep python server/run.py running on :8000"
