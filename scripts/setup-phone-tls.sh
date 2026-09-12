#!/bin/sh
set -eu
case "${PHONE_HOST:-}" in
  ''|*[!a-zA-Z0-9.-]*) echo 'Set PHONE_HOST in .env to an IPv4 address or DNS hostname (without scheme or port).' >&2; exit 1;;
esac
umask 077
mkdir -p /certs
cd /certs
if [ ! -f ca.key ] || [ ! -f ca.crt ]; then
  if [ -e ca.key ] || [ -e ca.crt ]; then
    echo 'Incomplete existing CA: preserve it and repair the certificate directory before retrying.' >&2; exit 1
  fi
  openssl req -x509 -newkey rsa:3072 -nodes -days 730 -sha256 \
    -keyout ca.key -out ca.crt -subj '/CN=CSI Collection Lab local CA' \
    -addext 'basicConstraints=critical,CA:TRUE' -addext 'keyUsage=critical,keyCertSign,cRLSign'
fi
case "$PHONE_HOST" in
  *[!0-9.]*) phone_san="DNS:$PHONE_HOST";;
  *) phone_san="IP:$PHONE_HOST";;
esac
openssl req -new -newkey rsa:2048 -nodes -keyout .server.key.tmp -out .server.csr.tmp -subj "/CN=$PHONE_HOST"
cat > .server.ext.tmp <<EXT
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=$phone_san,DNS:localhost,IP:127.0.0.1
EXT
openssl x509 -req -in .server.csr.tmp -CA ca.crt -CAkey ca.key -CAcreateserial \
  -out .server.crt.tmp -days 365 -sha256 -extfile .server.ext.tmp
mv .server.key.tmp server.key
mv .server.crt.tmp server.crt
chmod 644 server.crt ca.crt
rm .server.csr.tmp .server.ext.tmp
printf 'Certificate ready for %s. Share only ca.crt with the phone; private keys stay here.\n' "$PHONE_HOST"
