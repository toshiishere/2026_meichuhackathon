FROM node:22.15.0-alpine AS build
WORKDIR /app
COPY apps/frontend/package*.json ./
RUN npm ci
COPY apps/frontend/ ./
RUN npm run build
FROM nginx:1.28.0-alpine
RUN apk add --no-cache openssl
COPY --chmod=755 scripts/setup-phone-tls.sh /usr/local/bin/setup-phone-tls
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
