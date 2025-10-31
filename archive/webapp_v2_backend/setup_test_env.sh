#!/bin/bash
#
# Setup Clean Test Environment for PhonoLex Backend
#
# This script creates a fresh virtual environment and installs dependencies
# to avoid conflicts with the global Python environment.
#

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Setting up PhonoLex Backend Test Environment            ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo -e "${YELLOW}⚠️  Must be run from webapp/backend directory${NC}"
    exit 1
fi

# Create virtual environment
echo -e "${BLUE}Creating virtual environment...${NC}"
python3 -m venv venv_test

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv_test/bin/activate

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "${BLUE}Installing dependencies...${NC}"
pip install -r requirements.txt

echo ""
echo -e "${GREEN}✅ Test environment setup complete!${NC}"
echo ""
echo "To use this environment:"
echo "  source venv_test/bin/activate"
echo ""
echo "To run tests:"
echo "  source venv_test/bin/activate"
echo "  ./run_tests.sh quick"
echo ""
