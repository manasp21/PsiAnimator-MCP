# 🚀 Complete Guide: Deploying MCP Servers to Smithery

## **The Definitive Guide to Avoiding Common Pitfalls**

Based on real-world experience resolving persistent deployment issues, this guide provides a bulletproof approach to deploying MCP servers on Smithery.

---

## 📋 **Quick Checklist - Do This First!**

Before starting development, ensure you have:

- [ ] **Container Runtime**: Use `runtime: "container"` in `smithery.yaml`
- [ ] **Correct Dockerfile CMD**: Points to your actual server entry point
- [ ] **Standard Library Approach**: Minimize dependencies for reliability
- [ ] **Flexible Path Handling**: Support `/mcp` and `/mcp/*` paths
- [ ] **Complete MCP Protocol**: All required methods implemented
- [ ] **Fast Tool Discovery**: Static tool definitions, no heavy imports

---

## 🎯 **Step-by-Step Deployment Process**

### **Step 1: Choose Your Architecture**

#### **✅ Recommended: Standard Library Server**
```python
# Use Python's built-in http.server for maximum reliability
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

# Static tool definitions - no computation during discovery
TOOLS = [
    {
        "name": "your_tool",
        "description": "Tool description",
        "inputSchema": {
            "type": "object",
            "properties": {
                "param": {"type": "string"}
            },
            "required": ["param"]
        }
    }
]
```

#### **⚠️ Alternative: FastAPI (More Complex)**
```python
# If you must use FastAPI, import conditionally
def create_app():
    from fastapi import FastAPI  # Import inside function
    app = FastAPI()
    return app
```

### **Step 2: Implement Complete MCP Protocol**

```python
class MCPRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/mcp" or self.path.startswith("/mcp"):
            data = json.loads(self.rfile.read(...))
            method = data.get("method")
            
            if method == "initialize":
                return self.mcp_initialize(data)
            elif method == "ping":
                return self.mcp_ping(data)
            elif method == "tools/list":
                return self.mcp_tools_list(data)
            elif method == "tools/call":
                return self.mcp_tools_call(data)
            elif method == "resources/list":
                return self.mcp_resources_list(data)
            elif method == "prompts/list":
                return self.mcp_prompts_list(data)
            else:
                return self.mcp_unknown_method(data)
    
    def mcp_initialize(self, data):
        return {
            "jsonrpc": "2.0",
            "id": data.get("id", 0),
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "your-server-name",
                    "version": "1.0.0"
                }
            }
        }
    
    def mcp_ping(self, data):
        # CRITICAL: Must return empty result per MCP spec
        return {
            "jsonrpc": "2.0",
            "id": data.get("id", 0),
            "result": {}
        }
    
    def mcp_tools_list(self, data):
        return {
            "jsonrpc": "2.0",
            "id": data.get("id", 0),
            "result": {"tools": TOOLS}  # Use static definitions
        }
```

### **Step 3: Configure Container Deployment**

#### **smithery.yaml**
```yaml
version: 1
runtime: "container"  # CRITICAL: Use container runtime
build:
  dockerfile: "Dockerfile"
  dockerBuildPath: "."
startCommand:
  type: "http"
  port: 8000
  command: "python"
  args: ["your_server_entry_point.py"]  # Must match Dockerfile CMD
```

#### **Dockerfile**
```dockerfile
FROM python:3.11-slim

# Install minimal dependencies only
RUN pip install --no-cache-dir <your-essential-deps>

# Copy your server code
COPY src/ ./src/
COPY your_server_entry_point.py ./

# CRITICAL: CMD must match smithery.yaml args
CMD ["python", "your_server_entry_point.py"]
```

### **Step 4: Handle All HTTP Methods**

```python
def do_GET(self):
    if self.path == "/":
        return {"status": "ok", "server": "your-server"}
    elif self.path == "/health":
        return {"status": "healthy"}
    elif self.path == "/mcp" or self.path.startswith("/mcp"):
        return {"server": {"name": "your-server"}}

def do_DELETE(self):
    if self.path == "/mcp" or self.path.startswith("/mcp"):
        return {"status": "connection_closed"}

def do_OPTIONS(self):
    # CORS support
    self.send_response(200)
    self.send_header('Access-Control-Allow-Origin', '*')
    self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
    self.end_headers()
```

---

## ⚠️ **Critical Pitfalls to Avoid**

### **❌ Pitfall 1: Dockerfile/Config Mismatch**
```yaml
# smithery.yaml
args: ["run_server.py"]

# Dockerfile  
CMD ["python", "different_server.py"]  # ← MISMATCH CAUSES ISSUES
```
**✅ Solution**: Ensure Dockerfile CMD matches smithery.yaml args exactly.

### **❌ Pitfall 2: Heavy Imports During Discovery**
```python
# BAD: Imports heavy libraries at module level
import tensorflow as tf
import torch
from your_heavy_physics_lib import compute_everything

def list_tools():
    return tools  # Smithery times out during import
```
**✅ Solution**: Use static tool definitions, import heavy libs only during tool execution.

### **❌ Pitfall 3: Missing MCP Methods**
```python
# BAD: Only implements tools/list
if method == "tools/list":
    return tools
else:
    raise Exception("Unknown method")  # ← Causes HTTP 500s
```
**✅ Solution**: Implement ALL MCP methods (ping, resources/list, prompts/list).

### **❌ Pitfall 4: Inflexible Path Handling**
```python
# BAD: Only handles exact path
if self.path == "/mcp":
    handle_request()
else:
    return 404  # ← Smithery might use /mcp/something
```
**✅ Solution**: Use `self.path.startswith("/mcp")` for flexibility.

### **❌ Pitfall 5: Non-Compliant Ping Response**
```python
# BAD: Custom ping response
def ping():
    return {"status": "pong", "timestamp": time.time()}  # ← Schema validation error
```
**✅ Solution**: Ping must return empty result: `{"result": {}}`

---

## 🔧 **Debugging Common Issues**

### **Issue: "Request timed out" during tool scanning**
**Cause**: Heavy imports or computation during discovery
**Solution**: 
```python
# Move ALL heavy imports inside functions
def execute_tool():
    import heavy_library  # ← Only import when actually needed
    return heavy_library.compute()
```

### **Issue: "HTTP 404 Not Found"**
**Cause**: Smithery can't find your MCP endpoint
**Solution**: 
```python
# Handle multiple path patterns
if self.path in ["/mcp", "/mcp/", "/mcp/tools"] or self.path.startswith("/mcp"):
    handle_mcp_request()
```

### **Issue: "Internal error while deploying"**
**Cause**: Container startup failure
**Solution**: 
1. Add comprehensive error handling in entry point
2. Use standard library instead of complex frameworks
3. Test container locally: `docker build . && docker run -p 8000:8000 image_name`

### **Issue: "Schema validation errors"**
**Cause**: Non-compliant MCP responses
**Solution**: 
```python
# Follow exact MCP schema
{
    "jsonrpc": "2.0",
    "id": request_id,
    "result": {}  # Must be empty for ping
}
```

---

## 🧪 **Testing Your Server Locally**

```bash
# 1. Test container build
docker build -t test-mcp-server .

# 2. Run container locally
docker run -p 8000:8000 test-mcp-server

# 3. Test MCP endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "ping", "id": 1}'

curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 2}'
```

---

## 📊 **Deployment Success Checklist**

Before deploying to Smithery:

- [ ] Container builds without errors
- [ ] Server starts and responds to health checks
- [ ] All MCP methods return valid responses
- [ ] Tool discovery is fast (<1 second)
- [ ] Dockerfile CMD matches smithery.yaml
- [ ] No heavy imports at module level
- [ ] CORS headers included
- [ ] Error handling is comprehensive

---

## 🚀 **Template Server Structure**

```
your-mcp-server/
├── smithery.yaml           # Smithery config
├── Dockerfile             # Container definition
├── requirements.txt       # Minimal dependencies
├── server_entry.py        # Main entry point
├── src/
│   ├── mcp_handler.py     # MCP protocol implementation
│   ├── tools.py           # Static tool definitions
│   └── business_logic.py  # Heavy computation (imported lazily)
└── README.md
```

---

## 💡 **Pro Tips**

1. **Start Simple**: Begin with standard library, add complexity later
2. **Test Early**: Verify each component before integration
3. **Static Everything**: Make discovery phase computation-free
4. **Log Everything**: Comprehensive logging saves debugging time
5. **Plan for Scale**: Design tool loading to support many tools

---

## 🎯 **Success Metrics**

You know your deployment is correct when:

✅ `Deployment successful.`  
✅ `Scanning for tools...`  
✅ `Server tools successfully scanned.`

**Follow this guide and you'll avoid the common pitfalls that cause MCP deployment failures!**

---

*Based on successful deployment of the Rabi MCP Server - Advanced Atomic, Molecular and Optical Physics simulations on Smithery.*