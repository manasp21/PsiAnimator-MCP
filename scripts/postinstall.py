#!/usr/bin/env python3
"""
Post-installation configuration script for PsiAnimator-MCP.

This script sets up the user configuration directory and provides
integration instructions for Claude Desktop.
"""

import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Dict, Any


def get_config_directory() -> Path:
    """Get the appropriate configuration directory for the current platform."""
    system = platform.system().lower()
    home = Path.home()
    
    if system == "windows":
        config_dir = home / ".config" / "psianimator-mcp"
    elif system == "darwin":  # macOS
        config_dir = home / ".config" / "psianimator-mcp"
    else:  # Linux and others
        config_dir = home / ".config" / "psianimator-mcp"
    
    return config_dir


def get_claude_config_paths() -> list[Path]:
    """Get possible Claude Desktop configuration file paths."""
    system = platform.system().lower()
    home = Path.home()
    
    if system == "windows":
        return [
            home / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"
        ]
    elif system == "darwin":  # macOS
        return [
            home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        ]
    else:  # Linux
        return [
            home / ".config" / "claude-desktop" / "claude_desktop_config.json"
        ]


def create_config_directory(config_dir: Path) -> None:
    """Create the configuration directory if it doesn't exist."""
    if not config_dir.exists():
        config_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created configuration directory: {config_dir}")
    else:
        print(f"✓ Configuration directory exists: {config_dir}")


def setup_example_config(config_dir: Path) -> None:
    """Copy example configuration file if it doesn't exist."""
    config_file = config_dir / "config.json"
    
    if config_file.exists():
        print(f"✓ Configuration file already exists: {config_file}")
        return
    
    # Try to find the example config in the package
    example_config_content = {
        "quantum": {
            "precision": 1e-12,
            "max_hilbert_dimension": 1024,
            "enable_gpu": False,
            "state_validation": True,
            "normalization_tolerance": 1e-10
        },
        "animation": {
            "quality": "medium_quality",
            "frame_rate": 30,
            "output_format": "mp4",
            "cache_animations": True,
            "cache_size_mb": 500,
            "resolution": {
                "width": 1920,
                "height": 1080
            }
        },
        "server": {
            "log_level": "INFO",
            "enable_logging": True,
            "output_directory": "./output",
            "max_concurrent_requests": 10,
            "request_timeout": 300
        },
        "manim": {
            "renderer": "cairo",
            "background_color": "#000000",
            "tex_template": "default",
            "preview": False,
            "write_to_movie": True,
            "verbosity": "WARNING"
        },
        "bloch_sphere": {
            "sphere_color": "#0066cc",
            "vector_color": "#ff6600",
            "show_axes": True,
            "show_equator": True,
            "animation_duration": 3.0
        },
        "wigner_function": {
            "grid_resolution": 100,
            "x_range": [-4, 4],
            "p_range": [-4, 4],
            "colormap": "RdBu",
            "contour_levels": 20
        },
        "quantum_circuit": {
            "gate_spacing": 1.0,
            "wire_spacing": 1.0,
            "show_measurements": True,
            "animate_gates": True,
            "gate_duration": 0.5
        },
        "energy_levels": {
            "show_populations": True,
            "show_transitions": True,
            "level_spacing": 1.0,
            "transition_color": "#ff0000",
            "population_color": "#0066ff"
        }
    }
    
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(example_config_content, f, indent=4)
        print(f"✓ Created configuration file: {config_file}")
        print("⚠️  Please edit the configuration file with your settings")
    except Exception as e:
        print(f"❌ Failed to create configuration file: {e}")


def generate_claude_integration_config(config_dir: Path) -> Dict[str, Any]:
    """Generate Claude Desktop integration configuration."""
    system = platform.system().lower()
    
    if system == "windows":
        python_cmd = "python"
        config_path = str(config_dir / "config.json").replace("\\", "\\\\")
    else:
        python_cmd = "python3"
        config_path = str(config_dir / "config.json")
    
    return {
        "mcpServers": {
            "psianimator-mcp": {
                "command": python_cmd,
                "args": ["-m", "psianimator_mcp.cli"],
                "env": {
                    "PSIANIMATOR_CONFIG": config_path
                }
            }
        }
    }


def check_claude_desktop_integration(config_dir: Path) -> None:
    """Check for Claude Desktop and provide integration instructions."""
    claude_config_paths = get_claude_config_paths()
    
    # Check if any Claude Desktop config exists
    existing_config = None
    for path in claude_config_paths:
        if path.exists():
            existing_config = path
            break
    
    integration_config = generate_claude_integration_config(config_dir)
    
    print("\n📋 Claude Desktop Integration")
    print("=" * 50)
    
    if existing_config:
        print(f"✓ Found Claude Desktop configuration: {existing_config}")
        print("\nTo add PsiAnimator-MCP to your existing configuration,")
        print("merge the following into your mcpServers section:")
    else:
        print("To use PsiAnimator-MCP with Claude Desktop, create or update")
        print(f"the configuration file at one of these locations:")
        for path in claude_config_paths:
            print(f"  {path}")
        print("\nConfiguration content:")
    
    print("\n" + json.dumps(integration_config, indent=2))
    
    print(f"\n💡 Tips:")
    print(f"  • Restart Claude Desktop after configuration changes")
    print(f"  • Check Claude Desktop logs if the server doesn't appear")
    print(f"  • Configuration file: {config_dir / 'config.json'}")


def verify_installation() -> bool:
    """Verify that PsiAnimator-MCP is properly installed."""
    try:
        import psianimator_mcp
        print("✓ PsiAnimator-MCP package is importable")
        return True
    except ImportError as e:
        print(f"❌ PsiAnimator-MCP package import failed: {e}")
        return False


def main() -> None:
    """Main post-installation setup function."""
    print("PsiAnimator-MCP Post-Installation Setup")
    print("=" * 50)
    
    try:
        # Verify installation
        if not verify_installation():
            print("\n❌ Installation verification failed!")
            print("Please ensure PsiAnimator-MCP is properly installed.")
            sys.exit(1)
        
        # Setup configuration
        config_dir = get_config_directory()
        create_config_directory(config_dir)
        setup_example_config(config_dir)
        
        # Provide Claude Desktop integration instructions
        check_claude_desktop_integration(config_dir)
        
        print("\n✅ Post-installation setup complete!")
        print(f"\nNext steps:")
        print(f"  1. Edit configuration: {config_dir / 'config.json'}")
        print(f"  2. Test the CLI: python3 -m psianimator_mcp.cli --help")
        print(f"  3. Configure Claude Desktop (see instructions above)")
        print(f"  4. Restart Claude Desktop")
        
    except Exception as e:
        print(f"\n❌ Post-installation setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()