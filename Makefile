DOCKER ?= docker
COMPOSE = $(DOCKER) compose
MOCK = $(COMPOSE) -f docker-compose.yml -f docker-compose.mock.yml
export PROJECT_GIT_COMMIT := $(shell git rev-parse HEAD)
.PHONY: up mock down logs test test-integration hardware-shell manifest test-ui phone-cert phone phone-off
up:
	$(COMPOSE) up -d --build
mock:
	$(MOCK) up -d --build
down:
	$(COMPOSE) down
logs:
	$(COMPOSE) logs -f --tail=100
test:
	$(MOCK) build hardware-service
	$(MOCK) run --rm --no-deps -e DATA_DIR=/tmp/csi-tests hardware-service python -m pytest tests -m 'not hardware'
test-integration:
	$(MOCK) build hardware-service
	$(MOCK) run --rm --no-deps -e DATA_DIR=/tmp/csi-tests hardware-service python -m pytest tests/integration
hardware-shell:
	$(COMPOSE) exec hardware-service bash
manifest:
	$(COMPOSE) exec hardware-service python scripts/rebuild_manifest.py

test-ui:
	$(DOCKER) build -f docker/browser-tests.Dockerfile -t csi-collection-browser-tests .
	$(DOCKER) run --rm --network host -e BASE_URL -e PHONE_BASE_URL -v "$(CURDIR)/apps/frontend:/app" -w /app csi-collection-browser-tests sh -c 'npm ci && npm run test:e2e'

# Run make up or make mock first. Phone setup preserves the running hardware mode.
phone-cert:
	$(COMPOSE) build frontend
	$(COMPOSE) -f docker-compose.yml -f docker-compose.phone.yml run --rm --no-deps phone-cert
phone:
	$(COMPOSE) -f docker-compose.yml -f docker-compose.phone.yml up -d --no-deps --force-recreate backend frontend
phone-off:
	$(COMPOSE) up -d --no-deps frontend
