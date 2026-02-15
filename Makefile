.PHONY: test test-unit test-integration test-fast test-coverage lint format clean help

# Default target
all: test

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

# Run the bot
run:
	python -m src.main

# Run paper trading
run-paper:
	python -m src.main --paper

# Show help
help:
	@echo "Available targets:"
	@echo "  test              - Run all tests"
	@echo "  test-unit         - Run unit tests only (fast)"
	@echo "  test-integration  - Run integration tests"
	@echo "  test-fast         - Run fast tests only"
	@echo "  test-coverage     - Run tests with coverage report"
	@echo "  test-auth         - Run auth tests"
	@echo "  test-client       - Run client tests"
	@echo "  test-data         - Run data tests"
	@echo "  test-strategy     - Run strategy tests"
	@echo "  test-execution    - Run execution tests"
	@echo "  test-paper        - Run paper trading tests"
	@echo "  test-performance  - Run performance tests"
	@echo "  lint              - Lint code"
	@echo "  format            - Format code"
	@echo "  clean             - Clean generated files"
	@echo "  install           - Install dependencies"
	@echo "  run               - Run the bot"
	@echo "  run-paper         - Run paper trading"
