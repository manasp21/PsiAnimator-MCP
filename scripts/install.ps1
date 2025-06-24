param(
    [switch]$FromSource,
    [switch]$FromPyPI,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Show-Help {
    Write-ColorOutput "PsiAnimator-MCP Installation Script" "Cyan"
    Write-Host ""
    Write-Host "Usage: .\install.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -FromSource      Install from source code (development)"
    Write-Host "  -FromPyPI        Install from PyPI (recommended)"
    Write-Host "  -Help            Show this help message"
    Write-Host ""
    Write-Host "If no option is specified, the script will auto-detect the best method."
    exit 0
}

if ($Help) {
    Show-Help
}

Write-ColorOutput "Installing PsiAnimator-MCP Server" "Green"
Write-ColorOutput "Quantum Physics Simulation and Animation Server" "Blue"
Write-Host ""

function Test-PythonVersion {
    Write-ColorOutput "Checking Python version..." "Yellow"
    
    try {
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -match "Python (\d+)\.(\d+)\.(\d+)") {
            $major = [int]$matches[1]
            $minor = [int]$matches[2]
            
            if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
                throw "Python 3.10+ required. Current: $pythonVersion"
            }
            
            Write-ColorOutput "✓ $pythonVersion detected" "Green"
        } else {
            throw "Could not determine Python version"
        }
    }
    catch {
        Write-ColorOutput "Error: Python 3.10+ is required" "Red"
        Write-ColorOutput "Please install from https://python.org/" "Red"
        Write-ColorOutput "Make sure to add Python to your PATH during installation" "Yellow"
        exit 1
    }
}

function Test-Pip {
    Write-ColorOutput "Checking pip availability..." "Yellow"
    
    try {
        $pipVersion = python -m pip --version 2>&1
        if ($pipVersion -match "pip") {
            Write-ColorOutput "✓ pip is available" "Green"
        } else {
            throw "pip not found"
        }
    }
    catch {
        Write-ColorOutput "Error: pip is not available" "Red"
        Write-ColorOutput "Please run: python -m ensurepip --upgrade" "Yellow"
        exit 1
    }
}

function Install-FromPyPI {
    Write-ColorOutput "Installing PsiAnimator-MCP from PyPI..." "Yellow"
    python -m pip install --upgrade psianimator-mcp
}

function Install-FromSource {
    Write-ColorOutput "Installing PsiAnimator-MCP from source..." "Yellow"
    
    # Check if we're in the project directory
    if (!(Test-Path "pyproject.toml")) {
        Write-ColorOutput "Error: Not in PsiAnimator-MCP source directory" "Red"
        Write-ColorOutput "Please run this script from the PsiAnimator-MCP project root" "Red"
        exit 1
    }
    
    # Install in development mode
    python -m pip install -e ".[dev]"
}

function Test-SystemDependencies {
    Write-ColorOutput "Checking system dependencies..." "Yellow"
    
    # Check for LaTeX
    try {
        $null = latex --version 2>&1
        Write-ColorOutput "✓ LaTeX found" "Green"
    }
    catch {
        Write-ColorOutput "Warning: LaTeX not found" "Yellow"
        Write-ColorOutput "For full Manim functionality, install LaTeX:" "White"
        Write-ColorOutput "  Download MiKTeX from https://miktex.org/" "White"
        Write-ColorOutput "  Or install TeX Live from https://tug.org/texlive/" "White"
    }
    
    # Check for FFmpeg
    try {
        $null = ffmpeg -version 2>&1
        Write-ColorOutput "✓ FFmpeg found" "Green"
    }
    catch {
        Write-ColorOutput "Warning: FFmpeg not found" "Yellow"
        Write-ColorOutput "For video generation, install FFmpeg:" "White"
        Write-ColorOutput "  Download from https://ffmpeg.org/download.html" "White"
        Write-ColorOutput "  Or install via Chocolatey: choco install ffmpeg" "White"
    }
}

function Setup-Configuration {
    $configDir = "$env:USERPROFILE\.config\psianimator-mcp"
    New-Item -ItemType Directory -Path $configDir -Force | Out-Null
    
    $configFile = "$configDir\config.json"
    if (!(Test-Path $configFile)) {
        $configContent = @'
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
'@
        $configContent | Out-File -FilePath $configFile -Encoding UTF8
        Write-ColorOutput "✓ Configuration template created at $configFile" "Green"
        Write-ColorOutput "You can customize these settings as needed" "Blue"
    } else {
        Write-ColorOutput "✓ Configuration file already exists" "Green"
    }
}

function Setup-ClaudeIntegration {
    Write-Host ""
    Write-ColorOutput "📋 Claude Desktop Integration" "Blue"
    Write-Host "To use PsiAnimator-MCP with Claude Desktop, add this to your configuration:"
    Write-Host ""
    
    $claudeConfigPath = "$env:USERPROFILE\AppData\Roaming\Claude\claude_desktop_config.json"
    Write-ColorOutput "File: $claudeConfigPath" "Yellow"
    Write-Host ""
    
    $integrationConfig = @'
{
  "mcpServers": {
    "psianimator-mcp": {
      "command": "python",
      "args": ["-m", "psianimator_mcp.cli"],
      "env": {
        "PSIANIMATOR_CONFIG": "%USERPROFILE%\.config\psianimator-mcp\config.json"
      }
    }
  }
}
'@
    Write-Host $integrationConfig
    Write-Host ""
}

function Test-Installation {
    Write-ColorOutput "Verifying installation..." "Yellow"
    
    try {
        $importTest = python -c "import psianimator_mcp; print('✓ Package imported successfully')" 2>&1
        if ($importTest -match "✓") {
            Write-ColorOutput "✓ PsiAnimator-MCP installed successfully" "Green"
            
            # Test CLI
            try {
                $null = python -m psianimator_mcp.cli --help 2>&1
                Write-ColorOutput "✓ CLI is working" "Green"
            }
            catch {
                Write-ColorOutput "Warning: CLI test failed" "Yellow"
            }
        } else {
            throw "Import test failed"
        }
    }
    catch {
        Write-ColorOutput "✗ Installation verification failed" "Red"
        Write-ColorOutput "Please check the error messages above" "Red"
        exit 1
    }
}

# Main installation flow
function Main {
    Write-Host "Checking prerequisites..."
    Test-PythonVersion
    Test-Pip
    
    # Determine installation method
    if ($FromSource) {
        Install-FromSource
    } elseif ($FromPyPI) {
        Install-FromPyPI
    } else {
        # Auto-detect: if pyproject.toml exists, install from source
        if ((Test-Path "pyproject.toml") -and (Select-String -Path "pyproject.toml" -Pattern "psianimator-mcp" -Quiet)) {
            Write-ColorOutput "Detected source directory, installing from source..." "Blue"
            Install-FromSource
        } else {
            Write-ColorOutput "Installing from PyPI..." "Blue"
            Install-FromPyPI
        }
    }
    
    Test-SystemDependencies
    Setup-Configuration
    Test-Installation
    Setup-ClaudeIntegration
    
    Write-Host ""
    Write-ColorOutput "🎉 Installation complete!" "Green"
    Write-Host ""
    Write-Host "Quick start:"
    Write-Host "  python -m psianimator_mcp.cli --help"
    Write-Host ""
    Write-Host "For examples and documentation:"
    Write-Host "  https://github.com/your-username/PsiAnimator-MCP"
}

# Execute main function
Main