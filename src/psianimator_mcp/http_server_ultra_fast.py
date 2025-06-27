#!/usr/bin/env python3
"""
Ultra-Fast HTTP Server for PsiAnimator-MCP deployment on Smithery
NO imports during tool scanning - completely static for instant discovery
"""

import json
import logging
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List, Optional
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# STATIC tool definitions - ZERO imports, ZERO computation
STATIC_TOOLS = [
    {
        "name": "create_quantum_state",
        "description": "Create quantum states (pure/mixed, single/composite systems)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "state_type": {"type": "string", "enum": ["pure", "mixed", "coherent", "squeezed", "thermal", "fock"]},
                "system_dims": {"type": "array", "items": {"type": "integer", "minimum": 2}},
                "parameters": {"type": "object"},
                "basis": {"type": "string", "enum": ["computational", "fock", "spin", "position"], "default": "computational"}
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
                "evolution_type": {"type": "string", "enum": ["schrodinger", "master", "monte_carlo", "stochastic"]},
                "time_span": {"type": "array", "items": {"type": "number"}},
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
                "measurement_type": {"type": "string", "enum": ["expectation", "variance", "projective", "povm"]},
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
                "gate_sequence": {"type": "array", "items": {"type": "object", "properties": {"gate": {"type": "string"}, "qubits": {"type": "array", "items": {"type": "integer"}}, "parameters": {"type": "object"}}, "required": ["gate", "qubits"]}},
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
                "measure": {"type": "string", "enum": ["concurrence", "negativity", "entropy", "mutual_info", "linear_entropy"]},
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
                "process_type": {"type": "string", "enum": ["bloch_sphere", "energy_levels", "wigner_function", "photon_statistics"]},
                "state_id": {"type": "string"},
                "parameters": {"type": "object"},
                "animation_config": {"type": "object"}
            },
            "required": ["process_type", "state_id"]
        }
    }
]

class UltraFastMCPHandler(BaseHTTPRequestHandler):
    """Ultra-fast MCP handler with ZERO imports during discovery"""
    
    def log_message(self, format, *args):
        """Custom logging"""
        logger.info(f"{self.address_string()} - {format % args}")
    
    def send_json_response(self, data: Dict[str, Any], status_code: int = 200):
        """Send JSON response with CORS headers"""
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
            "error": {"code": error_code, "message": message}
        }
        self.send_json_response(error_response, 400)
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests - NO DEPENDENCY CHECKS"""
        if self.path == "/":
            self.send_json_response({
                "server": "PsiAnimator-MCP",
                "version": "0.1.0",
                "description": "Quantum Physics Simulation and Animation MCP Server",
                "status": "running"
            })
        elif self.path == "/health":
            # CRITICAL: NO dependency checks during tool scanning
            self.send_json_response({
                "status": "healthy",
                "server": "psianimator-mcp"
            })
        elif self.path in ["/mcp", "/mcp/"]:
            self.send_json_response({
                "server": {"name": "psianimator-mcp", "version": "0.1.0"},
                "capabilities": {"tools": {}}
            })
        elif self.path == "/mcp/tools":
            # INSTANT tool response - NO imports
            self.send_json_response({"tools": STATIC_TOOLS})
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
            
            # Validate JSON-RPC
            if not isinstance(request_data, dict) or request_data.get("jsonrpc") != "2.0":
                self.send_error_response(-32600, "Invalid Request")
                return
            
            method = request_data.get("method")
            request_id = request_data.get("id", 0)
            
            if method == "initialize":
                self.send_json_response({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "psianimator-mcp", "version": "0.1.0"}
                    }
                })
            elif method == "ping":
                self.send_json_response({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {}  # Empty per MCP spec
                })
            elif method == "tools/list":
                # INSTANT response - static tools
                self.send_json_response({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"tools": STATIC_TOOLS}
                })
            elif method == "tools/call":
                # Lazy load only when actually executing
                self._handle_tool_call(request_data)
            elif method in ["resources/list", "prompts/list"]:
                self.send_json_response({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"resources": []} if method == "resources/list" else {"prompts": []}
                })
            else:
                self.send_error_response(-32601, f"Method not found: {method}", request_id)
        
        except json.JSONDecodeError:
            self.send_error_response(-32700, "Parse error")
        except Exception as e:
            logger.error(f"Request error: {e}")
            self.send_error_response(-32603, f"Internal error: {str(e)}")
    
    def do_DELETE(self):
        """Handle DELETE requests"""
        if self.path == "/mcp" or self.path.startswith("/mcp"):
            self.send_json_response({"status": "connection_closed"})
        else:
            self.send_json_response({"error": "Not found"}, 404)
    
    def _handle_tool_call(self, request_data: Dict[str, Any]):
        """Handle tool execution with lazy loading"""
        params = request_data.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        request_id = request_data.get("id", 0)
        
        if not tool_name:
            self.send_error_response(-32602, "Missing tool name", request_id)
            return
        
        try:
            # Only NOW do we import and execute (when actually called)
            result = self._execute_tool_lazy(tool_name, arguments)
            
            self.send_json_response({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
                }
            })
        except Exception as e:
            self.send_error_response(-32603, f"Tool execution failed: {str(e)}", request_id)
    
    def _execute_tool_lazy(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool with lazy imports (only when called)"""
        try:
            # Add the src directory to path for imports
            src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            
            # Quantum tools
            if tool_name in ["create_quantum_state", "evolve_quantum_system", "measure_observable", 
                           "quantum_gate_sequence", "calculate_entanglement"]:
                
                # Check if quantum deps available
                try:
                    import numpy, scipy, qutip
                except ImportError:
                    return {
                        "success": False,
                        "error": "quantum_unavailable",
                        "message": "Quantum dependencies not available",
                        "suggestion": "Install with: pip install numpy scipy qutip"
                    }
                
                # Import and execute tool
                import asyncio
                if tool_name == "create_quantum_state":
                    from psianimator_mcp.tools.quantum_state_tools import create_quantum_state
                    return asyncio.run(create_quantum_state(arguments, None))
                elif tool_name == "evolve_quantum_system":
                    from psianimator_mcp.tools.evolution_tools import evolve_quantum_system
                    return asyncio.run(evolve_quantum_system(arguments, None))
                elif tool_name == "measure_observable":
                    from psianimator_mcp.tools.measurement_tools import measure_observable
                    return asyncio.run(measure_observable(arguments, None))
                elif tool_name == "quantum_gate_sequence":
                    from psianimator_mcp.tools.gate_tools import quantum_gate_sequence
                    return asyncio.run(quantum_gate_sequence(arguments, None))
                elif tool_name == "calculate_entanglement":
                    from psianimator_mcp.tools.entanglement_tools import calculate_entanglement
                    return asyncio.run(calculate_entanglement(arguments, None))
            
            elif tool_name == "animate_quantum_process":
                try:
                    import manim
                except ImportError:
                    return {
                        "success": False,
                        "error": "animation_unavailable",
                        "message": "Animation dependencies not available",
                        "suggestion": "Install with: pip install manim"
                    }
                
                import asyncio
                from psianimator_mcp.tools.animation_tools import animate_quantum_process
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
                "message": f"Dependencies not available: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": "execution_error",
                "message": str(e)
            }

def main():
    """Main entry point"""
    port = int(os.getenv('PORT', 8000))
    
    logger.info(f"Starting Ultra-Fast PsiAnimator-MCP server on port {port}")
    logger.info("Tool discovery optimized for instant scanning")
    
    server = HTTPServer(('0.0.0.0', port), UltraFastMCPHandler)
    
    try:
        logger.info(f"Server running at http://0.0.0.0:{port}")
        logger.info("Endpoints: GET /health, GET /mcp/tools, POST /mcp")
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

if __name__ == "__main__":
    main()