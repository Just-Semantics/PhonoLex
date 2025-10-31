#!/bin/bash
#
# PhonoLex Backend Test Runner
#
# Runs the complete test suite with options for:
# - Quick tests (unit only)
# - Full tests (unit + integration)
# - Performance benchmarks
# - Coverage reports
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_header() {
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  $1"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "pytest.ini" ]; then
    print_error "Must be run from webapp/backend directory"
    exit 1
fi

# Check/activate virtual environment
check_venv() {
    if [ -d "venv_test" ]; then
        if [ -z "$VIRTUAL_ENV" ]; then
            print_header "Activating test virtual environment..."
            source venv_test/bin/activate
            print_success "Virtual environment activated"
        fi
    else
        print_warning "No test venv found. Run ./setup_test_env.sh first for clean environment"
        echo "Or create venv manually: python3 -m venv venv_test && source venv_test/bin/activate"
    fi
}

# Check if test database exists
check_database() {
    print_header "Checking test database..."

    if psql -lqt | cut -d \| -f 1 | grep -qw phonolex_test; then
        print_success "Test database 'phonolex_test' found"
        return 0
    else
        print_warning "Test database 'phonolex_test' not found"
        echo "Create it with: cd ../../database && ./setup.sh phonolex_test postgres"
        return 1
    fi
}

# Install dependencies
install_deps() {
    print_header "Installing test dependencies..."
    pip install -q -r requirements.txt
    print_success "Dependencies installed"
}

# Run quick tests (unit only)
run_quick() {
    print_header "Running Quick Tests (Unit Tests Only)..."
    pytest tests/unit/ -v --tb=short -m "not slow"
}

# Run full tests (unit + integration)
run_full() {
    print_header "Running Full Test Suite..."
    pytest tests/ -v --tb=short -m "not slow" --ignore=tests/performance/
}

# Run performance benchmarks
run_performance() {
    print_header "Running Performance Benchmarks..."
    pytest tests/performance/ -v --tb=short
}

# Run with coverage
run_coverage() {
    print_header "Running Tests with Coverage..."
    pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing --ignore=tests/performance/
    print_success "Coverage report generated in htmlcov/"
}

# Run critical tests only
run_critical() {
    print_header "Running Critical Tests Only..."
    pytest -v -m critical --tb=short
}

# Run all tests (including slow)
run_all() {
    print_header "Running ALL Tests (including slow)..."
    pytest tests/ -v --tb=short
}

# Show usage
usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  quick       Run quick tests (unit only, fast)"
    echo "  full        Run full test suite (unit + integration)"
    echo "  performance Run performance benchmarks"
    echo "  coverage    Run tests with coverage report"
    echo "  critical    Run critical tests only"
    echo "  all         Run ALL tests (including slow)"
    echo "  check       Check prerequisites (database, dependencies)"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 quick              # Fast unit tests"
    echo "  $0 full               # Complete test suite"
    echo "  $0 coverage           # With coverage report"
    echo "  $0 performance        # Benchmarks only"
    echo ""
}

# Main
main() {
    # Check/activate venv first
    check_venv

    case "${1:-quick}" in
        quick)
            install_deps
            run_quick
            ;;
        full)
            install_deps
            check_database || print_warning "Some integration tests may fail without test database"
            run_full
            ;;
        performance)
            install_deps
            check_database || print_error "Performance tests require test database" && exit 1
            run_performance
            ;;
        coverage)
            install_deps
            check_database || print_warning "Some tests may fail without test database"
            run_coverage
            ;;
        critical)
            install_deps
            run_critical
            ;;
        all)
            install_deps
            check_database || print_warning "Some tests may fail without test database"
            run_all
            ;;
        check)
            print_header "Checking Prerequisites..."
            check_database
            install_deps
            print_success "All prerequisites met!"
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
}

# Run main
main "$@"

# Exit code
if [ $? -eq 0 ]; then
    echo ""
    print_success "All tests passed!"
    exit 0
else
    echo ""
    print_error "Some tests failed"
    exit 1
fi
