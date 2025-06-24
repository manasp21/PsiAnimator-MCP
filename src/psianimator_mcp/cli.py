"""
Command Line Interface for PsiAnimator-MCP Server

Provides the main entry point for starting the MCP server with various
configuration options.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from .server import MCPServer, MCPConfig

app = typer.Typer(name="psianimator-mcp", help="Quantum Physics Simulation and Animation MCP Server")
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

@app.command()
def serve(
    transport: str = typer.Option(
        "stdio",
        "--transport", "-t", 
        help="Transport protocol: stdio or websocket"
    ),
    host: str = typer.Option(
        "localhost", 
        "--host", "-h",
        help="Host to bind to (websocket only)"
    ),
    port: int = typer.Option(
        3000,
        "--port", "-p", 
        help="Port to bind to (websocket only)"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration file"
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
def config():
    """Show current configuration."""
    config = MCPConfig()
    console.print("Current Configuration:")
    console.print(f"  Quantum Precision: {config.quantum_precision}")
    console.print(f"  Max Hilbert Dimension: {config.max_hilbert_dimension}")
    console.print(f"  Animation Cache Size: {config.animation_cache_size}")
    console.print(f"  Parallel Workers: {config.parallel_workers}")
    console.print(f"  Render Backend: {config.render_backend}")

def main():
    """Main entry point for the CLI."""
    app()

if __name__ == "__main__":
    main()