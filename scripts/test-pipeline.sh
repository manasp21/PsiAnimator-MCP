#!/bin/bash

# PsiAnimator-MCP Pipeline Test Script
# This script simulates the GitHub Actions pipeline locally for debugging

set -e  # Exit on any error

echo "🚀 PsiAnimator-MCP Pipeline Test"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $1 - PASSED${NC}"
    else
        echo -e "${RED}❌ $1 - FAILED${NC}"
        return 1
    fi
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

# Function to print info
print_info() {
    echo -e "ℹ️ $1"
}

echo ""
echo "1. 🔍 Environment Detection"
echo "----------------------------"

# Check Python version
print_info "Checking Python version..."
python --version
if python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"; then
    print_status "Python version check"
else
    print_warning "Python version should be >= 3.10 for MCP compatibility"
fi

# Check if we're in the right directory
if [[ -f "pyproject.toml" ]] && [[ -d "src/psianimator_mcp" ]]; then
    print_status "Project structure check"
else
    echo -e "${RED}❌ Not in PsiAnimator-MCP root directory or missing files${NC}"
    exit 1
fi

echo ""
echo "2. 📦 Dependencies Installation"
echo "--------------------------------"

print_info "Upgrading pip and installing build tools..."
python -m pip install --upgrade pip setuptools wheel build twine || {
    print_warning "Build tools installation had issues"
}

print_info "Installing package in development mode..."
if python -m pip install -e ".[dev,animation]" > /dev/null 2>&1; then
    print_status "Package installation (with extras)"
elif python -m pip install -e . > /dev/null 2>&1; then
    print_status "Package installation (basic)"
    print_warning "Some optional dependencies may be missing"
else
    echo -e "${RED}❌ Package installation failed${NC}"
    exit 1
fi

print_info "Installing MCP SDK..."
python -m pip install mcp || print_warning "MCP SDK installation failed"

echo ""
echo "3. 🧪 Code Quality Checks"
echo "--------------------------"

# Check if tools are available and run them
if command -v black > /dev/null 2>&1; then
    print_info "Running Black formatting check..."
    if black --check --diff src/ tests/ > /dev/null 2>&1; then
        print_status "Black formatting"
    else
        print_warning "Black formatting issues found (run 'black src/ tests/' to fix)"
    fi
else
    print_warning "Black not available"
fi

if command -v isort > /dev/null 2>&1; then
    print_info "Running isort import sorting check..."
    if isort --check-only src/ tests/ > /dev/null 2>&1; then
        print_status "Import sorting"
    else
        print_warning "Import sorting issues found (run 'isort src/ tests/' to fix)"
    fi
else
    print_warning "isort not available"
fi

if command -v flake8 > /dev/null 2>&1; then
    print_info "Running Flake8 linting..."
    if flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics > /dev/null 2>&1; then
        print_status "Flake8 critical errors"
    else
        print_warning "Flake8 found critical errors"
    fi
else
    print_warning "Flake8 not available"
fi

if command -v mypy > /dev/null 2>&1; then
    print_info "Running MyPy type checking..."
    if mypy src/psianimator_mcp --ignore-missing-imports > /dev/null 2>&1; then
        print_status "MyPy type checking"
    else
        print_warning "MyPy found type issues"
    fi
else
    print_warning "MyPy not available"
fi

echo ""
echo "4. 🧪 Tests"
echo "------------"

if command -v pytest > /dev/null 2>&1; then
    print_info "Running pytest..."
    if pytest tests/ --cov=src/psianimator_mcp --cov-report=term-missing -v; then
        print_status "Unit tests"
    else
        print_warning "Some tests failed"
    fi
else
    print_warning "pytest not available"
fi

echo ""
echo "5. 📦 Build"
echo "------------"

print_info "Building package..."
if python -m build > /dev/null 2>&1; then
    print_status "Package build"
    
    # Check built files
    if [[ -d "dist/" ]] && [[ "$(ls -A dist/)" ]]; then
        print_info "Built files:"
        ls -la dist/
        
        # Check package with twine
        if command -v twine > /dev/null 2>&1; then
            if twine check dist/* > /dev/null 2>&1; then
                print_status "Package validation"
            else
                print_warning "Package validation issues"
            fi
        fi
    else
        print_warning "No built packages found"
    fi
else
    echo -e "${RED}❌ Package build failed${NC}"
fi

echo ""
echo "6. 🔍 MCP Server Validation"
echo "----------------------------"

print_info "Testing MCP server import..."
if python -c "
import sys
import importlib.util
import os

# Try to find the main server file
candidates = ['src/psianimator_mcp/server/mcp_server.py', 'server.py', 'main.py']
for candidate in candidates:
    if os.path.exists(candidate):
        print(f'Found server file: {candidate}')
        spec = importlib.util.spec_from_file_location('server', candidate)
        if spec and spec.loader:
            try:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print('MCP server loaded successfully')
                sys.exit(0)
            except Exception as e:
                print(f'Server validation warning: {e}')
                sys.exit(1)
        break
else:
    print('No main server file found')
    sys.exit(1)
" > /dev/null 2>&1; then
    print_status "MCP server import"
else
    print_warning "MCP server import issues"
fi

# Test CLI if available
if command -v psianimator-mcp > /dev/null 2>&1; then
    print_info "Testing CLI commands..."
    
    if psianimator-mcp --version > /dev/null 2>&1; then
        print_status "CLI version command"
    else
        print_warning "CLI version command failed"
    fi
    
    if psianimator-mcp --help > /dev/null 2>&1; then
        print_status "CLI help command"
    else
        print_warning "CLI help command failed"
    fi
else
    print_warning "CLI not available in PATH"
fi

echo ""
echo "7. 🛡️ Security Checks"
echo "----------------------"

if command -v bandit > /dev/null 2>&1; then
    print_info "Running Bandit security scan..."
    if bandit -r src/psianimator_mcp/ > /dev/null 2>&1; then
        print_status "Bandit security scan"
    else
        print_warning "Bandit found security issues"
    fi
else
    print_warning "Bandit not available"
fi

if command -v safety > /dev/null 2>&1; then
    print_info "Running Safety dependency check..."
    if safety check > /dev/null 2>&1; then
        print_status "Safety dependency check"
    else
        print_warning "Safety found vulnerable dependencies"
    fi
else
    print_warning "Safety not available"
fi

echo ""
echo "🎉 Pipeline Test Complete!"
echo "=========================="
echo ""
echo "💡 Next steps:"
echo "- If all checks passed, your pipeline should work in CI"
echo "- For any warnings, consider installing missing tools"
echo "- Check the individual tool outputs for specific issues"
echo ""
echo "🔧 To install missing tools:"
echo "pip install black isort flake8 mypy pytest pytest-cov bandit safety"