# Quick Start - Running Tests

## The Problem

Your global Python environment has many conflicting dependencies (difformer, crewai, langchain, etc.) that conflict with our backend requirements.

## The Solution

Create a **clean virtual environment** just for testing the backend.

---

## Step 1: Create Clean Test Environment

```bash
cd webapp/backend

# Create and activate clean virtual environment
./setup_test_env.sh
```

This will:
- Create `venv_test/` directory
- Install only backend dependencies (no conflicts)
- Take ~1-2 minutes

---

## Step 2: Activate Virtual Environment

```bash
# Activate the test environment
source venv_test/bin/activate

# You should see (venv_test) in your prompt
```

---

## Step 3: Run Tests

```bash
# Quick tests (unit only, ~5 seconds)
./run_tests.sh quick

# Full test suite (unit + integration, ~35 seconds)
./run_tests.sh full

# With coverage report
./run_tests.sh coverage

# Performance benchmarks
./run_tests.sh performance
```

---

## If You Get Database Errors

Some tests require a PostgreSQL test database. Create it:

```bash
# Go to database directory
cd ../../database

# Create test database
./setup.sh phonolex_test postgres

# Go back to backend
cd ../webapp/backend

# Run tests again
source venv_test/bin/activate
./run_tests.sh full
```

---

## Alternative: Manual Setup

If the script doesn't work, do it manually:

```bash
cd webapp/backend

# Create virtual environment
python3 -m venv venv_test

# Activate it
source venv_test/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/unit/ -v
```

---

## When You're Done

```bash
# Deactivate virtual environment
deactivate

# To use it again later
cd webapp/backend
source venv_test/bin/activate
```

---

## Quick Test Commands

After activating `venv_test`:

```bash
# All unit tests (fast, no database)
pytest tests/unit/ -v

# Specific test file
pytest tests/unit/test_similarity_service.py -v

# Specific test
pytest tests/unit/test_similarity_service.py::TestSimilarityService::test_precomputed_similarity_lookup -v

# Tests matching pattern
pytest -k "minimal_pairs" -v

# Critical tests only
pytest -m critical -v

# With coverage
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

---

## Troubleshooting

### "pytest: command not found"

```bash
# Make sure venv is activated
source venv_test/bin/activate

# Check pytest is installed
which pytest
# Should show: .../venv_test/bin/pytest
```

### "ModuleNotFoundError"

```bash
# Reinstall dependencies
source venv_test/bin/activate
pip install -r requirements.txt
```

### "Database connection failed"

```bash
# Tests will still run, just skip integration tests
# Or create test database:
cd ../../database
./setup.sh phonolex_test postgres
```

### Starting fresh

```bash
# Delete old venv
rm -rf venv_test

# Create new one
./setup_test_env.sh
source venv_test/bin/activate
```

---

## Summary

```bash
# One-time setup
cd webapp/backend
./setup_test_env.sh

# Every time you want to run tests
source venv_test/bin/activate
./run_tests.sh quick    # or full, coverage, performance

# When done
deactivate
```

That's it! The virtual environment keeps backend dependencies isolated from your global Python packages.
