#!/usr/bin/env python3
"""
HTTP Server for PsiAnimator-MCP deployment on Smithery
Provides HTTP transport for MCP protocol with static tool definitions
"""

import json
import logging
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List, Optional
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Static tool definitions - NO heavy imports during startup
STATIC_TOOLS = [
    {
        "name": "create_quantum_state",
        "description": "Create quantum states (pure/mixed, single/composite systems)",
        "inputSchema": {
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
    },
    {
        "name": "evolve_quantum_system",
        "description": "Time evolution using Schrödinger/Master/Stochastic equations",
        "inputSchema": {
            "type": "object",
            "properties": {
                "state_id": {"type": "string"},
                "evolution_type": {
                    "type": "string",
                    "enum": ["schrodinger", "master", "monte_carlo", "stochastic"]
                },
                "time_span": {
                    "type": "array",
                    "items": {"type": "number"}
                },
                "hamiltonian": {"type": "object"},
                "collapse_operators": {"type": "array"},
                "parameters": {"type": "object"}
            },
            "required": ["state_id", "evolution_type", "time_span"]
        }
    },
    {
        "name": "measure_observable",
        "description": "Perform quantum measurements and calculate expectation values",
        "inputSchema": {
            "type": "object",
            "properties": {
                "state_id": {"type": "string"},
                "observable": {"type": "object"},
                "measurement_type": {
                    "type": "string",
                    "enum": ["expectation", "variance", "projective", "povm"]
                },
                "basis": {"type": "string"},
                "shots": {"type": "integer", "minimum": 1, "default": 1000}
            },
            "required": ["state_id", "observable", "measurement_type"]
        }
    },
    {
        "name": "quantum_gate_sequence",
        "description": "Apply sequence of quantum gates with visualization",
        "inputSchema": {
            "type": "object",
            "properties": {
                "state_id": {"type": "string"},
                "gate_sequence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "gate": {"type": "string"},
                            "qubits": {"type": "array", "items": {"type": "integer"}},
                            "parameters": {"type": "object"}
                        },
                        "required": ["gate", "qubits"]
                    }
                },
                "visualization": {"type": "boolean", "default": True}
            },
            "required": ["state_id", "gate_sequence"]
        }
    },
    {
        "name": "calculate_entanglement",
        "description": "Compute entanglement measures and visualize correlations",
        "inputSchema": {
            "type": "object",
            "properties": {
                "state_id": {"type": "string"},
                "measure": {
                    "type": "string",
                    "enum": ["concurrence", "negativity", "entropy", "mutual_info", "linear_entropy"]
                },
                "subsystems": {"type": "array", "items": {"type": "integer"}},
                "visualization": {"type": "boolean", "default": True}
            },
            "required": ["state_id", "measure"]
        }
    },
    {
        "name": "animate_quantum_process",
        "description": "Generate Manim animations of quantum processes (optional)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "process_type": {
                    "type": "string", 
                    "enum": ["bloch_sphere", "energy_levels", "wigner_function", "photon_statistics"]
                },
                "state_id": {"type": "string"},
                "parameters": {"type": "object"},
                "animation_config": {"type": "object"}
            },
            "required": ["process_type", "state_id"]
        }
    }
]

class MCPHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler implementing MCP protocol over HTTP"""
    
    def log_message(self, format, *args):
        """Custom logging to avoid default HTTP server logging"""
        logger.info(f"{self.address_string()} - {format % args}")
    
    def send_json_response(self, data: Dict[str, Any], status_code: int = 200):
        """Send JSON response with proper headers"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        response_json = json.dumps(data, indent=2)
        self.wfile.write(response_json.encode('utf-8'))
    
    def send_error_response(self, error_code: int, message: str, request_id: Optional[int] = None):
        """Send MCP error response"""
        error_response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": error_code,
                "message": message
            }
        }
        self.send_json_response(error_response, 400)
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == "/":
            self.send_json_response({
                "server": "PsiAnimator-MCP",
                "version": "0.1.0",
                "description": "Quantum Physics Simulation and Animation MCP Server",
                "protocol": "MCP",
                "status": "running"
            })
        elif self.path == "/health":
            self.send_json_response({
                "status": "healthy",
                "server": "psianimator-mcp",
                "quantum_available": self._check_quantum_available(),
                "animation_available": self._check_animation_available()
            })
        elif self.path in ["/mcp", "/mcp/"]:
            self.send_json_response({
                "server": {
                    "name": "psianimator-mcp",
                    "version": "0.1.0"
                },
                "capabilities": {
                    "tools": {},
                    "logging": {},
                    "resources": {},
                    "prompts": {}
                }
            })
        elif self.path == "/mcp/tools":
            self.send_json_response({
                "tools": STATIC_TOOLS
            })
        else:
            self.send_json_response({"error": "Not found"}, 404)
    
    def do_POST(self):
        """Handle POST requests (MCP protocol)"""
        if not (self.path == "/mcp" or self.path.startswith("/mcp")):
            self.send_json_response({"error": "Invalid endpoint"}, 404)
            return
        
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_error_response(-32600, "Invalid Request - No content")
                return
            
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            # Validate JSON-RPC format
            if not isinstance(request_data, dict):
                self.send_error_response(-32600, "Invalid Request - Not a JSON object")
                return
            
            if request_data.get("jsonrpc") != "2.0":
                self.send_error_response(-32600, "Invalid Request - Missing jsonrpc 2.0")
                return
            
            method = request_data.get("method")
            request_id = request_data.get("id", 0)
            
            if not method:
                self.send_error_response(-32600, "Invalid Request - Missing method", request_id)
                return
            
            # Route to appropriate handler
            if method == "initialize":
                self.handle_initialize(request_data)
            elif method == "ping":
                self.handle_ping(request_data)
            elif method == "tools/list":
                self.handle_tools_list(request_data)
            elif method == "tools/call":
                self.handle_tools_call(request_data)
            elif method == "resources/list":
                self.handle_resources_list(request_data)
            elif method == "prompts/list":
                self.handle_prompts_list(request_data)
            else:
                self.send_error_response(-32601, f"Method not found: {method}", request_id)
        
        except json.JSONDecodeError:
            self.send_error_response(-32700, "Parse error - Invalid JSON")
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            logger.error(traceback.format_exc())
            self.send_error_response(-32603, f"Internal error: {str(e)}")
    
    def do_DELETE(self):
        """Handle DELETE requests (connection close)"""
        if self.path == "/mcp" or self.path.startswith("/mcp"):
            self.send_json_response({
                "status": "connection_closed"
            })
        else:
            self.send_json_response({"error": "Not found"}, 404)
    
    def handle_initialize(self, request_data: Dict[str, Any]):
        """Handle MCP initialize request"""
        response = {
            "jsonrpc": "2.0",
            "id": request_data.get("id", 0),
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "logging": {},
                    "resources": {},
                    "prompts": {}
                },
                "serverInfo": {
                    "name": "psianimator-mcp",
                    "version": "0.1.0"
                }
            }
        }
        self.send_json_response(response)
    
    def handle_ping(self, request_data: Dict[str, Any]):
        """Handle MCP ping request - MUST return empty result"""
        response = {
            "jsonrpc": "2.0",
            "id": request_data.get("id", 0),
            "result": {}  # CRITICAL: Must be empty per MCP spec
        }
        self.send_json_response(response)
    
    def handle_tools_list(self, request_data: Dict[str, Any]):
        """Handle tools/list request with static definitions"""
        response = {
            "jsonrpc": "2.0",
            "id": request_data.get("id", 0),
            "result": {
                "tools": STATIC_TOOLS
            }
        }
        self.send_json_response(response)
    
    def handle_tools_call(self, request_data: Dict[str, Any]):
        """Handle tools/call request - lazy load actual implementation"""
        params = request_data.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if not tool_name:
            self.send_error_response(-32602, "Invalid params - Missing tool name", request_data.get("id"))
            return
        
        try:
            # Lazy import and execute tool (only when actually called)
            result = self._execute_tool(tool_name, arguments)
            
            response = {
                "jsonrpc": "2.0",
                "id": request_data.get("id", 0),
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2)
                        }
                    ]
                }
            }
            self.send_json_response(response)
        
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            self.send_error_response(-32603, f"Tool execution failed: {str(e)}", request_data.get("id"))
    
    def handle_resources_list(self, request_data: Dict[str, Any]):
        """Handle resources/list request"""
        response = {
            "jsonrpc": "2.0",
            "id": request_data.get("id", 0),
            "result": {
                "resources": []
            }
        }
        self.send_json_response(response)
    
    def handle_prompts_list(self, request_data: Dict[str, Any]):
        """Handle prompts/list request"""
        response = {
            "jsonrpc": "2.0", 
            "id": request_data.get("id", 0),
            "result": {
                "prompts": []
            }
        }
        self.send_json_response(response)
    
    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Lazy-load and execute tool implementation"""
        try:
            # Only import quantum tools when actually needed
            if tool_name in ["create_quantum_state", "evolve_quantum_system", "measure_observable", 
                           "quantum_gate_sequence", "calculate_entanglement"]:
                
                # Check if quantum functionality is available
                if not self._check_quantum_available():
                    return {
                        "success": False,
                        "error": "quantum_unavailable",
                        "message": "Quantum physics dependencies (NumPy, SciPy, QuTiP) not available",
                        "suggestion": "Install with: pip install 'psianimator-mcp[quantum]'"
                    }
                
                # Import and execute quantum tool
                if tool_name == "create_quantum_state":
                    from .tools.quantum_state_tools import create_quantum_state
                    return asyncio.run(create_quantum_state(arguments, None))
                elif tool_name == "evolve_quantum_system":
                    from .tools.evolution_tools import evolve_quantum_system
                    return asyncio.run(evolve_quantum_system(arguments, None))
                elif tool_name == "measure_observable":
                    from .tools.measurement_tools import measure_observable
                    return asyncio.run(measure_observable(arguments, None))
                elif tool_name == "quantum_gate_sequence":
                    from .tools.gate_tools import quantum_gate_sequence
                    return asyncio.run(quantum_gate_sequence(arguments, None))
                elif tool_name == "calculate_entanglement":
                    from .tools.entanglement_tools import calculate_entanglement
                    return asyncio.run(calculate_entanglement(arguments, None))
            
            elif tool_name == "animate_quantum_process":
                # Check if animation functionality is available
                if not self._check_animation_available():
                    return {
                        "success": False,
                        "error": "animation_unavailable", 
                        "message": "Animation dependencies (Manim) not available",
                        "suggestion": "Install with: pip install 'psianimator-mcp[animation]'"
                    }
                
                from .tools.animation_tools import animate_quantum_process
                return asyncio.run(animate_quantum_process(arguments, None))
            
            else:
                return {
                    "success": False,
                    "error": "unknown_tool",
                    "message": f"Tool '{tool_name}' not implemented"
                }
        
        except ImportError as e:
            return {
                "success": False,
                "error": "import_error",
                "message": f"Required dependencies not available: {str(e)}",
                "suggestion": "Install full dependencies with: pip install 'psianimator-mcp[quantum,animation]'"
            }
        except Exception as e:
            return {
                "success": False,
                "error": "execution_error",
                "message": str(e)
            }
    
    def _check_quantum_available(self) -> bool:
        """Check if quantum dependencies are available - cached"""
        if not hasattr(self, '_quantum_available_cache'):
            try:
                import numpy
                import scipy  
                import qutip
                self._quantum_available_cache = True
            except ImportError:
                self._quantum_available_cache = False
        return self._quantum_available_cache
    
    def _check_animation_available(self) -> bool:
        """Check if animation dependencies are available - cached"""
        if not hasattr(self, '_animation_available_cache'):
            try:
                import manim
                self._animation_available_cache = True
            except ImportError:
                self._animation_available_cache = False
        return self._animation_available_cache

def main():
    """Main entry point for HTTP server"""
    port = int(os.getenv('PORT', 8000))
    
    logger.info(f"Starting PsiAnimator-MCP HTTP server on port {port}")
    logger.info(f"Quantum available: {MCPHTTPHandler(None, None, None)._check_quantum_available()}")
    logger.info(f"Animation available: {MCPHTTPHandler(None, None, None)._check_animation_available()}")
    
    server = HTTPServer(('0.0.0.0', port), MCPHTTPHandler)
    
    try:
        logger.info(f"Server running at http://0.0.0.0:{port}")
        logger.info("Available endpoints:")
        logger.info("  GET  / - Server info")
        logger.info("  GET  /health - Health check")
        logger.info("  GET  /mcp/tools - List tools")
        logger.info("  POST /mcp - MCP protocol endpoint")
        
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

if __name__ == "__main__":
    main()