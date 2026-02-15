#!/usr/bin/env python3
"""Test runner for Kalshi Trading Bot."""

import subprocess
import sys
import argparse


def run_tests(
    test_type: str = "all",
    coverage: bool = True,
    verbose: bool = True,
    failfast: bool = False
) -> int:
    """Run tests with specified options."""
    
    cmd = ["pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
    
    if failfast:
        cmd.append("-x")
    
    # Test type filtering
    if test_type == "unit":
        cmd.extend(["-m", "not integration"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    elif test_type == "auth":
        cmd.append("tests/test_auth.py")
    elif test_type == "client":
        cmd.append("tests/test_client_integration.py")
    elif test_type == "data":
        cmd.extend([
            "tests/test_candle_aggregator.py",
            "tests/test_bollinger_bands.py",
            "tests/test_market_discovery.py"
        ])
    elif test_type == "strategy":
        cmd.append("tests/test_bollinger_scalper.py")
    elif test_type == "execution":
        cmd.append("tests/test_execution_engine.py")
    elif test_type == "integration":
        cmd.append("tests/test_trading_bot_integration.py")
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def run_coverage() -> int:
    """Run tests with full coverage report."""
    cmd = [
        "pytest",
        "--cov=src",
        "--cov-report=html:htmlcov",
        "--cov-report=term-missing",
        "--cov-fail-under=80",
        "-v"
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run Kalshi Trading Bot tests")
    parser.add_argument(
        "type",
        nargs="?",
        default="all",
        choices=["all", "unit", "integration", "fast", "auth", "client", "data", "strategy", "execution", "coverage"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage reporting"
    )
    parser.add_argument(
        "--failfast",
        "-x",
        action="store_true",
        help="Stop on first failure"
    )
    
    args = parser.parse_args()
    
    if args.type == "coverage":
        return run_coverage()
    else:
        return run_tests(
            test_type=args.type,
            coverage=not args.no_coverage,
            failfast=args.failfast
        )


if __name__ == "__main__":
    sys.exit(main())
