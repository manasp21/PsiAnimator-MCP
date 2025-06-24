#!/bin/bash
# Workflow validation script for PsiAnimator-MCP
# This script checks all GitHub Actions workflows for common issues

set -e

echo "🔍 GitHub Actions Workflow Validator"
echo "==================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL_WORKFLOWS=0
VALID_WORKFLOWS=0
ISSUES_FOUND=0

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
        VALID_WORKFLOWS=$((VALID_WORKFLOWS + 1))
    else
        echo -e "${RED}❌ $2${NC}"
        ISSUES_FOUND=$((ISSUES_FOUND + 1))
    fi
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
}

# Function to print info
print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

# Check if .github/workflows directory exists
if [[ ! -d ".github/workflows" ]]; then
    echo -e "${RED}❌ .github/workflows directory not found${NC}"
    exit 1
fi

print_info "Checking workflows in .github/workflows/"

# Iterate through all YAML files in workflows directory
for workflow in .github/workflows/*.yml .github/workflows/*.yaml; do
    if [[ -f "$workflow" ]]; then
        TOTAL_WORKFLOWS=$((TOTAL_WORKFLOWS + 1))
        echo ""
        echo -e "${BLUE}📄 Checking $workflow${NC}"
        
        # Basic YAML syntax check
        if command -v python > /dev/null 2>&1; then
            python -c "
import yaml
try:
    with open('$workflow', 'r') as f:
        yaml.safe_load(f)
    print('✓ Valid YAML syntax')
    exit(0)
except yaml.YAMLError as e:
    print(f'❌ YAML syntax error: {e}')
    exit(1)
except Exception as e:
    print(f'❌ Error reading file: {e}')
    exit(1)
" && print_status 0 "YAML syntax valid" || print_status 1 "YAML syntax invalid"
        else
            print_warning "Python not available for YAML validation"
        fi
        
        # Check for required fields
        if grep -q "name:" "$workflow"; then
            print_status 0 "Has workflow name"
        else
            print_status 1 "Missing workflow name"
        fi
        
        if grep -q "on:" "$workflow"; then
            print_status 0 "Has trigger configuration"
        else
            print_status 1 "Missing trigger configuration"
        fi
        
        if grep -q "jobs:" "$workflow"; then
            print_status 0 "Has jobs defined"
        else
            print_status 1 "Missing jobs"
        fi
        
        # Check for modern action versions
        if grep -q "actions/checkout@v4" "$workflow"; then
            print_status 0 "Uses modern checkout action (v4)"
        elif grep -q "actions/checkout@v3" "$workflow"; then
            print_warning "Uses older checkout action (v3) - consider upgrading to v4"
        elif grep -q "actions/checkout@" "$workflow"; then
            print_warning "Uses very old checkout action - should upgrade to v4"
        fi
        
        if grep -q "actions/setup-python@v5" "$workflow"; then
            print_status 0 "Uses modern Python setup action (v5)"
        elif grep -q "actions/setup-python@v4" "$workflow"; then
            print_warning "Uses older Python setup action (v4) - consider upgrading to v5"
        elif grep -q "actions/setup-python@" "$workflow"; then
            print_warning "Uses very old Python setup action - should upgrade to v5"
        fi
        
        # Check for timeouts
        if grep -q "timeout-minutes:" "$workflow"; then
            print_status 0 "Has timeout configuration"
        else
            print_warning "Missing timeout configuration - jobs may hang indefinitely"
        fi
        
        # Check for shell specification
        if grep -q "shell:" "$workflow"; then
            print_status 0 "Has shell specification"
        else
            print_warning "Consider adding shell specification for bash scripts"
        fi
        
        # Check for error handling
        if grep -q "continue-on-error:" "$workflow" || grep -q "||" "$workflow"; then
            print_status 0 "Has error handling"
        else
            print_warning "Consider adding error handling for resilient workflows"
        fi
        
        # Check for caching
        if grep -q "cache:" "$workflow"; then
            print_status 0 "Uses dependency caching"
        else
            print_warning "Consider adding dependency caching for faster builds"
        fi
        
        # Check for permissions
        if grep -q "permissions:" "$workflow"; then
            print_status 0 "Has explicit permissions"
        else
            print_warning "Consider adding explicit permissions for security"
        fi
        
        # Check for specific issues based on workflow name
        workflow_basename=$(basename "$workflow")
        case "$workflow_basename" in
            "ci.yml"|"main.yml")
                if grep -q "matrix:" "$workflow"; then
                    print_status 0 "CI workflow uses matrix testing"
                else
                    print_warning "CI workflow might benefit from matrix testing"
                fi
                ;;
            "test-install.yml")
                if grep -q "windows-latest" "$workflow" && grep -q "ubuntu-latest" "$workflow"; then
                    print_status 0 "Installation tests cover multiple platforms"
                else
                    print_warning "Installation tests should cover multiple platforms"
                fi
                ;;
            "code-quality.yml")
                if grep -q "safety" "$workflow" || grep -q "bandit" "$workflow"; then
                    print_status 0 "Code quality includes security checks"
                else
                    print_warning "Code quality should include security checks"
                fi
                ;;
        esac
    fi
done

echo ""
echo "📊 Summary"
echo "=========="
echo -e "Total workflows checked: ${BLUE}$TOTAL_WORKFLOWS${NC}"
echo -e "Workflows with basic validity: ${GREEN}$VALID_WORKFLOWS${NC}"
echo -e "Issues/warnings found: ${YELLOW}$ISSUES_FOUND${NC}"

if [[ $ISSUES_FOUND -eq 0 ]]; then
    echo -e "${GREEN}🎉 All workflows look good!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️ Some issues found. Consider addressing the warnings above.${NC}"
    echo ""
    echo "💡 Tips for better workflows:"
    echo "  • Add timeout-minutes to prevent hanging jobs"
    echo "  • Use continue-on-error for non-critical steps"
    echo "  • Add dependency caching for faster builds"
    echo "  • Specify explicit permissions for security"
    echo "  • Use matrix testing for broader compatibility"
    echo "  • Update to latest action versions"
    exit 0  # Don't fail for warnings
fi