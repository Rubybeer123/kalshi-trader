.PHONY: test test-unit test-integration test-fast test-coverage lint format clean help
.PHONY: docker-build docker-run docker-stop docker-logs
.PHONY: k8s-deploy k8s-delete k8s-logs k8s-status

# Default target
all: test

# =============================================================================
# Testing
# =============================================================================

# Run all tests
test:
	pytest -v

# Run only unit tests (fast)
test-unit:
	pytest -v -m "not integration"

# Run only integration tests
test-integration:
	pytest -v -m "integration"

# Run fast tests only
test-fast:
	pytest -v -m "not slow"

# Run with coverage report
test-coverage:
	pytest --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=80

# Run specific test modules
test-auth:
	pytest tests/test_auth.py -v

test-client:
	pytest tests/test_client_integration.py -v

test-data:
	pytest tests/test_candle_aggregator.py tests/test_bollinger_bands.py tests/test_market_discovery.py -v

test-strategy:
	pytest tests/test_bollinger_scalper.py tests/test_strategy_base.py -v

test-execution:
	pytest tests/test_execution_engine.py tests/test_risk_manager.py -v

test-paper:
	pytest tests/test_paper_trading.py -v

test-performance:
	pytest tests/test_performance_tracker.py -v

test-integration-bot:
	pytest tests/test_trading_bot_integration.py -v

# =============================================================================
# Code Quality
# =============================================================================

# Lint code
lint:
	ruff check src/ tests/
	mypy src/

# Format code
format:
	black src/ tests/
	ruff check --fix src/ tests/

# Clean generated files
clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Install dependencies
install:
	pip install -r requirements.txt

# =============================================================================
# Local Development
# =============================================================================

# Run the bot
run:
	python -m src.main

# Run paper trading
run-paper:
	KALSHI_ENV=demo PAPER_TRADING=true python -m src.main

# =============================================================================
# Docker Deployment
# =============================================================================

# Build Docker image
docker-build:
	docker build -t kalshi-trader:latest .

# Run with Docker Compose (development)
docker-up:
	docker-compose up -d

# Run with Docker Compose (production)
docker-prod:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Run with monitoring stack
docker-monitor:
	docker-compose --profile monitoring up -d

# Stop all containers
docker-stop:
	docker-compose down

# View logs
docker-logs:
	docker-compose logs -f trader

# Clean Docker resources
docker-clean:
	docker-compose down -v
	docker rmi kalshi-trader:latest || true

# =============================================================================
# Kubernetes Deployment
# =============================================================================

# Deploy to Kubernetes
k8s-deploy:
	kubectl apply -f deploy/k8s/deployment.yaml
	@echo "Remember to create secrets:"
	@echo "  kubectl create secret generic kalshi-trader-secrets --from-literal=KALSHI_API_KEY_ID=YOUR_KEY_ID -n kalshi-trader"
	@echo "  kubectl create secret generic kalshi-private-key --from-file=kalshi-key.pem=/path/to/key.pem -n kalshi-trader"

# Delete Kubernetes deployment
k8s-delete:
	kubectl delete -f deploy/k8s/deployment.yaml

# View Kubernetes logs
k8s-logs:
	kubectl logs -f deployment/kalshi-trader -n kalshi-trader

# Check Kubernetes status
k8s-status:
	kubectl get all -n kalshi-trader

# Port forward for local access
k8s-port-forward:
	kubectl port-forward svc/kalshi-trader 8080:8080 -n kalshi-trader

# =============================================================================
# Secrets Management
# =============================================================================

# Create secrets directory structure
secrets-init:
	mkdir -p secrets logs
	touch secrets/.gitkeep logs/.gitkeep
	chmod 700 secrets
	@echo "Add your kalshi-key.pem to the secrets/ directory"

# Rotate secrets (K8s)
secrets-rotate:
	kubectl create job --from=cronjob/kalshi-key-rotation key-rotation-manual -n kalshi-trader

# =============================================================================
# Monitoring
# =============================================================================

# Open Grafana (requires port-forward or exposed service)
grafana:
	@echo "Grafana available at: http://localhost:3000"
	@echo "Default credentials: admin/admin"

# Check health endpoint
health-check:
	curl -s http://localhost:8080/health | jq .

# Check metrics endpoint
metrics:
	curl -s http://localhost:8080/metrics

# =============================================================================
# Help
# =============================================================================

# Show help
help:
	@echo "Kalshi Trader - Available Commands"
	@echo ""
	@echo "Testing:"
	@echo "  make test              - Run all tests"
	@echo "  make test-unit         - Run unit tests only (fast)"
	@echo "  make test-integration  - Run integration tests"
	@echo "  make test-coverage     - Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint              - Lint code"
	@echo "  make format            - Format code"
	@echo "  make clean             - Clean generated files"
	@echo ""
	@echo "Local Development:"
	@echo "  make run               - Run the bot"
	@echo "  make run-paper         - Run paper trading"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build      - Build Docker image"
	@echo "  make docker-up         - Run with Docker Compose"
	@echo "  make docker-prod       - Run production config"
	@echo "  make docker-monitor    - Run with monitoring"
	@echo "  make docker-logs       - View logs"
	@echo "  make docker-stop       - Stop containers"
	@echo ""
	@echo "Kubernetes:"
	@echo "  make k8s-deploy        - Deploy to Kubernetes"
	@echo "  make k8s-delete        - Delete deployment"
	@echo "  make k8s-logs          - View logs"
	@echo "  make k8s-status        - Check status"
	@echo "  make k8s-port-forward  - Port forward for local access"
	@echo ""
	@echo "Monitoring:"
	@echo "  make health-check      - Check health endpoint"
	@echo "  make metrics           - View Prometheus metrics"
	@echo "  make grafana           - Show Grafana info"
