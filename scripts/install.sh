#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Installing PsiAnimator-MCP Server${NC}"
echo -e "${BLUE}Quantum Physics Simulation and Animation Server${NC}"
echo ""

# Check Python version
check_python_version() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: Python 3 is not installed${NC}"
        echo "Please install Python 3.10+ from https://python.org/"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
        echo -e "${RED}Error: Python 3.10+ required${NC}"
        echo "Current version: Python $PYTHON_VERSION"
        echo "Please upgrade Python from https://python.org/"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}"
}

# Check pip availability
check_pip() {
    if ! python3 -m pip --version &> /dev/null; then
        echo -e "${RED}Error: pip is not available${NC}"
        echo "Please install pip: python3 -m ensurepip --upgrade"
        exit 1
    fi
    echo -e "${GREEN}✓ pip is available${NC}"
}

# Install from PyPI
install_from_pypi() {
    echo -e "${YELLOW}Installing PsiAnimator-MCP from PyPI...${NC}"
    
    if python3 -m pip install --upgrade psianimator-mcp; then
        echo -e "${GREEN}✓ Successfully installed from PyPI${NC}"
    else
        echo -e "${RED}✗ PyPI installation failed${NC}"
        echo -e "${YELLOW}This is expected if the package isn't published yet${NC}"
        echo -e "${YELLOW}Please install from source instead${NC}"
        exit 1
    fi
}

# Install from source
install_from_source() {
    echo -e "${YELLOW}Installing PsiAnimator-MCP from source...${NC}"
    
    # Check if we're in the project directory
    if [ ! -f "pyproject.toml" ]; then
        echo -e "${RED}Error: Not in PsiAnimator-MCP source directory${NC}"
        echo "Please run this script from the PsiAnimator-MCP project root"
        exit 1
    fi
    
    # Install in development mode with fallback
    echo "Attempting installation with dev dependencies..."
    if python3 -m pip install -e ".[dev]"; then
        echo -e "${GREEN}✓ Installed with dev dependencies${NC}"
    else
        echo -e "${YELLOW}⚠️ Dev dependencies failed, trying core installation...${NC}"
        if python3 -m pip install -e .; then
            echo -e "${GREEN}✓ Core installation successful${NC}"
            echo -e "${YELLOW}Note: Some features may require additional dependencies${NC}"
        else
            echo -e "${RED}✗ Installation failed${NC}"
            exit 1
        fi
    fi
}

# Install system dependencies (optional)
install_system_deps() {
    echo -e "${YELLOW}Checking system dependencies...${NC}"
    
    # Check for LaTeX (required for Manim)
    if ! command -v latex &> /dev/null; then
        echo -e "${YELLOW}Warning: LaTeX not found${NC}"
        echo "For full Manim functionality, install LaTeX:"
        echo "  Ubuntu/Debian: sudo apt install texlive-full"
        echo "  macOS: brew install --cask mactex"
        echo "  Or install BasicTeX for smaller footprint"
    else
        echo -e "${GREEN}✓ LaTeX found${NC}"
    fi
    
    # Check for ffmpeg (required for video output)
    if ! command -v ffmpeg &> /dev/null; then
        echo -e "${YELLOW}Warning: FFmpeg not found${NC}"
        echo "For video generation, install FFmpeg:"
        echo "  Ubuntu/Debian: sudo apt install ffmpeg"
        echo "  macOS: brew install ffmpeg"
    else
        echo -e "${GREEN}✓ FFmpeg found${NC}"
    fi
}

# Setup configuration
setup_config() {
    CONFIG_DIR="$HOME/.config/psianimator-mcp"
    mkdir -p "$CONFIG_DIR"
    
    if [ ! -f "$CONFIG_DIR/config.json" ]; then
        # Create default configuration
        cat > "$CONFIG_DIR/config.json" << 'EOF'
{
    "quantum": {
        "precision": 1e-12,
        "max_hilbert_dimension": 1024,
        "enable_gpu": false
    },
    "animation": {
        "quality": "medium_quality",
        "frame_rate": 30,
        "output_format": "mp4",
        "cache_animations": true,
        "cache_size_mb": 500
    },
    "server": {
        "log_level": "INFO",
        "enable_logging": true,
        "output_directory": "./output"
    },
    "manim": {
        "renderer": "cairo",
        "background_color": "#000000",
        "tex_template": "default"
    }
}
EOF
        echo -e "${GREEN}✓ Configuration template created at $CONFIG_DIR/config.json${NC}"
        echo -e "${BLUE}You can customize these settings as needed${NC}"
    else
        echo -e "${GREEN}✓ Configuration file already exists${NC}"
    fi
}

# Setup Claude Desktop integration
setup_claude_integration() {
    echo ""
    echo -e "${BLUE}📋 Claude Desktop Integration${NC}"
    echo "To use PsiAnimator-MCP with Claude Desktop, add this to your configuration:"
    echo ""
    
    CLAUDE_CONFIG_PATH=""
    if [[ "$OSTYPE" == "darwin"* ]]; then
        CLAUDE_CONFIG_PATH="~/Library/Application Support/Claude/claude_desktop_config.json"
    else
        CLAUDE_CONFIG_PATH="~/.config/claude-desktop/claude_desktop_config.json"
    fi
    
    echo -e "${YELLOW}File: $CLAUDE_CONFIG_PATH${NC}"
    echo ""
    cat << 'EOF'
{
  "mcpServers": {
    "psianimator-mcp": {
      "command": "python3",
      "args": ["-m", "psianimator_mcp.cli"],
      "env": {
        "PSIANIMATOR_CONFIG": "~/.config/psianimator-mcp/config.json"
      }
    }
  }
}
EOF
    echo ""
}

# Verify installation
verify_installation() {
    echo -e "${YELLOW}Verifying installation...${NC}"
    
    if python3 -c "import psianimator_mcp; print('✓ Package imported successfully')" 2>/dev/null; then
        echo -e "${GREEN}✓ PsiAnimator-MCP installed successfully${NC}"
        
        # Test CLI
        if python3 -m psianimator_mcp.cli --help &> /dev/null; then
            echo -e "${GREEN}✓ CLI is working${NC}"
        else
            echo -e "${YELLOW}Warning: CLI test failed${NC}"
        fi
    else
        echo -e "${RED}✗ Installation verification failed${NC}"
        echo "Please check the error messages above"
        exit 1
    fi
}

# Main installation flow
main() {
    echo "Checking prerequisites..."
    check_python_version
    check_pip
    
    # Determine installation method
    if [ "$1" = "--from-source" ]; then
        install_from_source
    elif [ "$1" = "--from-pypi" ]; then
        install_from_pypi
    else
        # Auto-detect: if pyproject.toml exists, install from source
        if [ -f "pyproject.toml" ] && grep -q "psianimator-mcp" pyproject.toml; then
            echo -e "${BLUE}Detected source directory, installing from source...${NC}"
            install_from_source
        else
            echo -e "${BLUE}Installing from PyPI...${NC}"
            install_from_pypi
        fi
    fi
    
    install_system_deps
    setup_config
    verify_installation
    setup_claude_integration
    
    echo ""
    echo -e "${GREEN}🎉 Installation complete!${NC}"
    echo ""
    echo "Quick start:"
    echo "  python3 -m psianimator_mcp.cli --help"
    echo ""
    echo "For examples and documentation:"
    echo "  https://github.com/manasp21/PsiAnimator-MCP"
}

# Handle script arguments
case "$1" in
    --help|-h)
        echo "PsiAnimator-MCP Installation Script"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --from-source    Install from source code (development)"
        echo "  --from-pypi      Install from PyPI (recommended)"
        echo "  --help, -h       Show this help message"
        echo ""
        echo "If no option is specified, the script will auto-detect the best method."
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac