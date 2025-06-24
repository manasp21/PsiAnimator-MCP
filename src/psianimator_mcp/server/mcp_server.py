"""
Main MCP Server implementation for PsiAnimator-MCP

Provides async MCP server with stdio and WebSocket transport support,
tool registration, and quantum-specific error handling.
"""

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Sequence
import traceback

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.websocket import websocket_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest, 
    ListToolsResult,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource
)

from .config import MCPConfig
from .exceptions import QuantumMCPError, ValidationError, ConfigurationError
from ..tools import (
    create_quantum_state,
    evolve_quantum_system,
    measure_observable,
    animate_quantum_process,
    quantum_gate_sequence,
    calculate_entanglement
)

logger = logging.getLogger(__name__)


class MCPServer:
    """
    Main MCP server for PsiAnimator quantum physics simulations.
    
    Handles tool registration, request processing, and provides both
    stdio and WebSocket transport protocols.
    """
    
    def __init__(self, config: Optional[MCPConfig] = None):
        """Initialize the MCP server with configuration."""
        self.config = config or MCPConfig()
        self.server = Server("psianimator-mcp")
        
        # Create necessary directories
        self.config.create_directories()
        
        # Setup logging
        self._setup_logging()
        
        # Register tools and handlers
        self._register_tools()
        self._register_handlers()
        
        logger.info("PsiAnimator-MCP server initialized")
    
    def _setup_logging(self) -> None:
        """Configure logging based on config settings."""
        if self.config.enable_logging:
            log_level = getattr(logging, self.config.log_level.upper())
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def _register_tools(self) -> None:
        """Register all MCP tools with their schemas."""
        
        # Tool definitions with JSON schemas
        tools = [
            Tool(
                name="create_quantum_state",
                description="Create quantum states (pure/mixed, single/composite systems)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "state_type": {
                            "type": "string",
                            "enum": ["pure", "mixed", "coherent", "squeezed", "thermal", "fock"]
                        },
                        "system_dims": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 2}
                        },
                        "parameters": {"type": "object"},
                        "basis": {
                            "type": "string", 
                            "enum": ["computational", "fock", "spin", "position"],
                            "default": "computational"
                        }
                    },
                    "required": ["state_type", "system_dims"]
                }
            ),
            Tool(
                name="evolve_quantum_system", 
                description="Time evolution using Schrödinger/Master/Stochastic equations",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "state_id": {"type": "string"},
                        "hamiltonian": {"type": "string"},
                        "collapse_operators": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "evolution_type": {
                            "type": "string",
                            "enum": ["unitary", "master", "monte_carlo", "stochastic"]
                        },
                        "time_span": {
                            "type": "array", 
                            "items": {"type": "number"},
                            "minItems": 2
                        },
                        "solver_options": {"type": "object"}
                    },
                    "required": ["state_id", "hamiltonian", "evolution_type", "time_span"]
                }
            ),
            Tool(
                name="measure_observable",
                description="Perform quantum measurements and calculate expectation values", 
                inputSchema={
                    "type": "object",
                    "properties": {
                        "state_id": {"type": "string"},
                        "observable": {"type": "string"},
                        "measurement_type": {
                            "type": "string",
                            "enum": ["expectation", "variance", "probability", "correlation"]
                        },
                        "measurement_basis": {"type": "string"}
                    },
                    "required": ["state_id", "observable", "measurement_type"]
                }
            ),
            Tool(
                name="animate_quantum_process",
                description="Generate Manim animations of quantum processes",
                inputSchema={
                    "type": "object", 
                    "properties": {
                        "animation_type": {
                            "type": "string",
                            "enum": ["bloch_evolution", "wigner_dynamics", "state_tomography", 
                                   "circuit_execution", "energy_levels", "photon_statistics"]
                        },
                        "data_source": {"type": "string"},
                        "render_quality": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "production"],
                            "default": "medium"
                        },
                        "output_format": {
                            "type": "string", 
                            "enum": ["mp4", "gif", "webm"],
                            "default": "mp4"
                        },
                        "frame_rate": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 120,
                            "default": 30
                        },
                        "duration": {"type": "number", "minimum": 0.1},
                        "view_config": {"type": "object"}
                    },
                    "required": ["animation_type", "data_source"]
                }
            ),
            Tool(
                name="quantum_gate_sequence",
                description="Apply sequence of quantum gates with visualization",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "state_id": {"type": "string"},
                        "gates": {
                            "type": "array",
                            "items": {
                                "type": "object", 
                                "properties": {
                                    "name": {"type": "string"},
                                    "qubits": {"type": "array", "items": {"type": "integer"}},
                                    "parameters": {"type": "object"}
                                },
                                "required": ["name", "qubits"]
                            }
                        },
                        "animate_steps": {"type": "boolean", "default": False},
                        "show_intermediate_states": {"type": "boolean", "default": True}
                    },
                    "required": ["state_id", "gates"]
                }
            ),
            Tool(
                name="calculate_entanglement", 
                description="Compute entanglement measures and visualize correlations",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "state_id": {"type": "string"},
                        "measure_type": {
                            "type": "string",
                            "enum": ["von_neumann", "linear_entropy", "concurrence", 
                                   "negativity", "mutual_information"]
                        },
                        "subsystem_partition": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "integer"}}
                        },
                        "visualize_correlations": {"type": "boolean", "default": False}
                    },
                    "required": ["state_id", "measure_type"]
                }
            )
        ]
        
        # Register each tool
        for tool in tools:
            self.server.list_tools().append(tool)
            logger.debug(f"Registered tool: {tool.name}")
    
    def _register_handlers(self) -> None:
        """Register MCP protocol handlers."""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """Return list of available tools."""
            return self.server._tools
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent | ImageContent | EmbeddedResource]:
            """Handle tool execution requests."""
            try:
                logger.info(f"Calling tool: {name} with arguments: {arguments}")
                
                # Route to appropriate tool function
                if name == "create_quantum_state":
                    result = await create_quantum_state(arguments, self.config)
                elif name == "evolve_quantum_system":
                    result = await evolve_quantum_system(arguments, self.config)
                elif name == "measure_observable": 
                    result = await measure_observable(arguments, self.config)
                elif name == "animate_quantum_process":
                    result = await animate_quantum_process(arguments, self.config)
                elif name == "quantum_gate_sequence":
                    result = await quantum_gate_sequence(arguments, self.config)
                elif name == "calculate_entanglement":
                    result = await calculate_entanglement(arguments, self.config)
                else:
                    raise ValidationError(f"Unknown tool: {name}")
                    
                # Convert result to MCP format
                if isinstance(result, dict):
                    return [TextContent(type="text", text=json.dumps(result, indent=2))]
                elif isinstance(result, str):
                    return [TextContent(type="text", text=result)]
                else:
                    return [TextContent(type="text", text=str(result))]
                    
            except QuantumMCPError as e:
                logger.error(f"Tool execution failed: {e}")
                error_response = {
                    "error": True,
                    "error_type": e.__class__.__name__,
                    "message": e.message,
                    "details": e.details
                }
                return [TextContent(type="text", text=json.dumps(error_response, indent=2))]
                
            except Exception as e:
                logger.error(f"Unexpected error in tool {name}: {e}")
                logger.error(traceback.format_exc())
                error_response = {
                    "error": True,
                    "error_type": "UnexpectedError",
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
                return [TextContent(type="text", text=json.dumps(error_response, indent=2))]
    
    async def run_stdio(self) -> None:
        """Run server with stdio transport."""
        logger.info("Starting MCP server with stdio transport")
        try:
            async with stdio_server(self.server) as (read_stream, write_stream):
                await self.server.run(
                    read_stream, 
                    write_stream,
                    options={
                        "name": "psianimator-mcp",
                        "version": "0.1.0"
                    }
                )
        except Exception as e:
            logger.error(f"Stdio server error: {e}")
            raise
    
    async def run_websocket(self, host: str = "localhost", port: int = 3000) -> None:
        """Run server with WebSocket transport."""
        logger.info(f"Starting MCP server with WebSocket transport on {host}:{port}")
        try:
            async with websocket_server(self.server, host, port) as server:
                await server.serve_forever()
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the server."""
        logger.info("Shutting down PsiAnimator-MCP server")
        
        try:
            # Clear any cached quantum states from state managers
            from ..quantum.state_manager import QuantumStateManager
            if hasattr(QuantumStateManager, '_states'):
                QuantumStateManager._states.clear()
                logger.debug("Cleared quantum state cache")
            
            # Clear cached results and evolution data
            from ..tools.quantum_state_tools import get_state_manager
            try:
                state_manager = get_state_manager(self.config)
                state_manager.clear_evolution_results()
                state_manager.clear_measurement_results()
                logger.debug("Cleared evolution and measurement caches")
            except Exception as e:
                logger.warning(f"Could not clear state manager caches: {e}")
            
            # Cancel any pending async tasks
            current_task = asyncio.current_task()
            all_tasks = [task for task in asyncio.all_tasks() if task != current_task]
            
            if all_tasks:
                logger.debug(f"Cancelling {len(all_tasks)} pending tasks")
                for task in all_tasks:
                    task.cancel()
                
                # Wait for tasks to complete cancellation
                await asyncio.gather(*all_tasks, return_exceptions=True)
            
            # Cleanup temporary files in output directory
            import os
            import shutil
            temp_dirs = [
                self.config.output_directory / "temp",
                self.config.output_directory / "cache"
            ]
            
            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    try:
                        shutil.rmtree(temp_dir)
                        logger.debug(f"Cleaned up temporary directory: {temp_dir}")
                    except Exception as e:
                        logger.warning(f"Could not clean up {temp_dir}: {e}")
            
            # Shutdown logging
            logging.shutdown()
            
            logger.info("PsiAnimator-MCP server shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during server shutdown: {e}")
            raise