# ==============================================================================
# VARIABLES
# ==============================================================================

TARGET_EXEC := app
PORT := 8080
VERSION := local
IMAGE := ghcr.io/raeperd/go-starter-template # Placeholder, update as needed
DOCKER_VERSION := $(if $(VERSION),$(subst /,-,$(VERSION)),latest)

# Go parameters
BINARY_NAME=$(TARGET_EXEC)
CMD_PATH=./cmd/server
GOPATH=$(shell go env GOPATH)
GOIMPORTS=$(GOPATH)/bin/goimports
GOLANGCI_LINT=$(GOPATH)/bin/golangci-lint

# Build parameters
BUILD_PATH=./build

# Go tools
GOLANGCI_LINT_VERSION=v1.64.8

default: clean build lint test format check

download:
	go mod download

build: download
	go build -o $(TARGET_EXEC) -ldflags '-w -X main.Version=$(VERSION)' $(CMD_PATH)

test:
	go test -shuffle=on -race -coverprofile=coverage.txt ./...

init:
	@echo "Installing development tools..."
	@go install github.com/golangci/golangci-lint/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION)
	@go install golang.org/x/tools/cmd/goimports@latest

.PHONY: tidy
tidy: ## Tidy go modules
	@echo "Tidying and verifying dependencies..."
	@go mod tidy
	@go mod verify

.PHONY: lint
lint: ## Run linter
	@echo "Running linter..."
	@$(GOLANGCI_LINT) run ./... --timeout=5m

run: build
	./$(TARGET_EXEC) --port=$(PORT)

watch:
	air

clean:
	rm -rf coverage.txt $(TARGET_EXEC)

docker:
	docker build . --build-arg VERSION=$(VERSION) -t $(IMAGE):$(DOCKER_VERSION)

docker-run: docker
	docker run --rm -p $(PORT):8080 $(IMAGE):$(DOCKER_VERSION)

docker-clean:
	docker image rm -f $(IMAGE):$(DOCKER_VERSION) || true

format: ## Format go files with goimports
	@go mod tidy
	@$(GOIMPORTS) -w -l .

check: ## Run linter
	@$(GOLANGCI_LINT) run ./... --timeout=5m
	@go mod verify
