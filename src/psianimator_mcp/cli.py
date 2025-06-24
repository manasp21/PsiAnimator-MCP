"""
Command Line Interface for PsiAnimator-MCP Server

Provides the main entry point for starting the MCP server with various
configuration options.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from .server import MCPServer, MCPConfig

app = typer.Typer(
    name="psianimator-mcp", 
    help="Quantum Physics Simulation and Animation MCP Server",
    no_args_is_help=False
)
console = Console()

def setup_logging(verbose: int = 0) -> None:
    """Setup logging configuration with Rich handler.""" 
    log_level = {
        0: logging.WARNING,
        1: logging.INFO, 
        2: logging.DEBUG
    }.get(verbose, logging.DEBUG)
    
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )

@app.command(name="serve")
def serve_command(
    transport: str = typer.Option(
        "stdio",
        "--transport", "-t", 
        help="Transport protocol: stdio or websocket",
        envvar="PSIANIMATOR_TRANSPORT"
    ),
    host: str = typer.Option(
        "localhost", 
        "--host", "-h",
        help="Host to bind to (websocket only)",
        envvar="PSIANIMATOR_HOST"
    ),
    port: int = typer.Option(
        3000,
        "--port", "-p", 
        help="Port to bind to (websocket only)",
        envvar="PSIANIMATOR_PORT"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration file",
        envvar="PSIANIMATOR_CONFIG"
    ),
    verbose: int = typer.Option(
        0,
        "--verbose", "-v",
        count=True,
        help="Increase verbosity (-v, -vv, -vvv)"
    )
):
    """Start the PsiAnimator-MCP server."""
    setup_logging(verbose)
    
    # Load configuration
    if config_file and config_file.exists():
        config = MCPConfig.from_file(config_file)
    else:
        config = MCPConfig()
    
    # Create and start server
    server = MCPServer(config)
    
    try:
        if transport == "stdio":
            console.print("🚀 Starting PsiAnimator-MCP server with stdio transport...")
            asyncio.run(server.run_stdio())
        elif transport == "websocket":
            console.print(f"🚀 Starting PsiAnimator-MCP server on ws://{host}:{port}...")
            asyncio.run(server.run_websocket(host, port))
        else:
            console.print(f"❌ Unknown transport: {transport}", style="red")
            sys.exit(1)
            
    except KeyboardInterrupt:
        console.print("\n👋 Server stopped by user")
    except Exception as e:
        console.print(f"❌ Server error: {e}", style="red")
        sys.exit(1)

@app.command()
def version():
    """Show version information."""
    from . import __version__
    console.print(f"PsiAnimator-MCP v{__version__}")

@app.command() 
def config(
    config_file: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration file"
    )
):
    """Show current configuration."""
    if config_file and config_file.exists():
        config = MCPConfig.from_file(config_file)
        console.print(f"Configuration from: {config_file}")
    else:
        config = MCPConfig()
        console.print("Using default configuration")
    
    console.print("\n📋 Current Configuration:")
    console.print(f"  Quantum Precision: {config.quantum_precision}")
    console.print(f"  Max Hilbert Dimension: {config.max_hilbert_dimension}")
    console.print(f"  Animation Cache Size: {config.animation_cache_size}")
    console.print(f"  Parallel Workers: {config.parallel_workers}")
    console.print(f"  Render Backend: {config.render_backend}")
    console.print(f"  Output Directory: {config.output_directory}")
    console.print(f"  Enable Logging: {config.enable_logging}")
    console.print(f"  Log Level: {config.log_level}")

@app.command()
def setup():
    """Run post-installation setup."""
    import subprocess
    import sys
    from pathlib import Path
    
    console.print("🔧 Running PsiAnimator-MCP setup...")
    
    # Find the postinstall script
    scripts_dir = Path(__file__).parent.parent.parent / "scripts"
    postinstall_script = scripts_dir / "postinstall.py"
    
    if postinstall_script.exists():
        try:
            result = subprocess.run([sys.executable, str(postinstall_script)], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                console.print("✅ Setup completed successfully!")
                if result.stdout:
                    console.print(result.stdout)
            else:
                console.print("❌ Setup failed!", style="red")
                if result.stderr:
                    console.print(result.stderr, style="red")
        except Exception as e:
            console.print(f"❌ Setup error: {e}", style="red")
    else:
        # Fallback: run setup inline
        console.print("📋 Manual setup instructions:")
        console.print("1. Create config directory: ~/.config/psianimator-mcp/")
        console.print("2. Copy example configuration")
        console.print("3. Configure Claude Desktop integration")

@app.command()
def test():
    """Test the MCP server functionality."""
    console.print("🧪 Testing PsiAnimator-MCP server...")
    
    try:
        # Test imports
        from . import __version__
        from .server import MCPServer
        from .quantum import QuantumStateManager
        console.print("✅ All imports successful")
        
        # Test basic functionality
        config = MCPConfig()
        state_manager = QuantumStateManager(config.max_hilbert_dimension)
        console.print("✅ Core components initialized")
        
        # Test state creation
        state_id = state_manager.create_state(
            state_type="pure",
            system_dims=[2],
            parameters={"state_indices": [0]}
        )
        console.print(f"✅ Created test quantum state: {state_id}")
        
        console.print("🎉 All tests passed!")
        
    except Exception as e:
        console.print(f"❌ Test failed: {e}", style="red")
        sys.exit(1)

@app.command()
def claude_config():
    """Generate Claude Desktop configuration."""
    import platform
    import json
    from pathlib import Path
    
    system = platform.system().lower()
    home = Path.home()
    
    if system == "windows":
        python_cmd = "python"
        config_path = str(home / ".config" / "psianimator-mcp" / "config.json").replace("\\", "\\\\")
    else:
        python_cmd = "python3"
        config_path = str(home / ".config" / "psianimator-mcp" / "config.json")
    
    claude_config = {
        "mcpServers": {
            "psianimator-mcp": {
                "command": python_cmd,
                "args": ["-m", "psianimator_mcp.cli", "serve"],
                "env": {
                    "PSIANIMATOR_CONFIG": config_path
                }
            }
        }
    }
    
    console.print("📋 Claude Desktop Configuration:")
    console.print("Add this to your Claude Desktop configuration file:")
    console.print("")
    console.print(json.dumps(claude_config, indent=2))
    console.print("")
    
    # Show config file locations
    if system == "windows":
        config_locations = [
            home / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"
        ]
    elif system == "darwin":  # macOS
        config_locations = [
            home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        ]
    else:  # Linux
        config_locations = [
            home / ".config" / "claude-desktop" / "claude_desktop_config.json"
        ]
    
    console.print("📍 Configuration file locations:")
    for location in config_locations:
        console.print(f"  {location}")

@app.command()
def examples():
    """Show usage examples."""
    console.print("🎯 PsiAnimator-MCP Usage Examples:")
    console.print("")
    
    console.print("1. Start server with stdio transport (default):")
    console.print("   psianimator-mcp serve")
    console.print("")
    
    console.print("2. Start server with WebSocket transport:")
    console.print("   psianimator-mcp serve --transport websocket --port 3000")
    console.print("")
    
    console.print("3. Use custom configuration:")
    console.print("   psianimator-mcp serve --config ~/.config/psianimator-mcp/config.json")
    console.print("")
    
    console.print("4. Enable verbose logging:")
    console.print("   psianimator-mcp serve -vv")
    console.print("")
    
    console.print("5. Show configuration:")
    console.print("   psianimator-mcp config")
    console.print("")
    
    console.print("6. Run tests:")
    console.print("   psianimator-mcp test")
    console.print("")
    
    console.print("For more examples, see: https://github.com/username/PsiAnimator-MCP/tree/main/examples")

# Create a default callback that runs serve when no command is specified
@app.callback(invoke_without_command=True)
def default_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, 
        "--version", 
        help="Show version and exit"
    )
):
    """PsiAnimator-MCP: Quantum Physics Simulation and Animation MCP Server."""
    if version:
        from . import __version__
        console.print(f"PsiAnimator-MCP v{__version__}")
        return
        
    # If no subcommand is provided, run serve with default options
    if ctx.invoked_subcommand is None:
        # Check for environment variables
        transport = os.getenv("PSIANIMATOR_TRANSPORT", "stdio")
        host = os.getenv("PSIANIMATOR_HOST", "localhost")
        port = int(os.getenv("PSIANIMATOR_PORT", "3000"))
        config_file = os.getenv("PSIANIMATOR_CONFIG")
        
        # Run serve command with defaults
        serve_command(
            transport=transport,
            host=host,
            port=port,
            config_file=Path(config_file) if config_file else None,
            verbose=0
        )

def main():
    """Main entry point for the CLI."""
    app()

if __name__ == "__main__":
    main()