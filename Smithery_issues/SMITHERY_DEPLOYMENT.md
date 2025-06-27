# 🚀 Smithery Deployment Fix Guide

## ❌ **Issue: Connection Timeout on Smithery**

**Problem**: "Sorry, the connection timed out. Please try again later."

**Root Cause**: The MCP server was configured for stdio transport but Smithery Cloud requires HTTP transport.

## ✅ **Solution: HTTP-Compatible MCP Server**

### 1. **Fixed Files Created:**
- ✅ `src/http_server.py` - HTTP-compatible MCP server
- ✅ Updated `smithery.yaml` - Changed to HTTP transport
- ✅ Updated `Dockerfile` - Added health checks and HTTP server
- ✅ `.gitignore` & `.gitattributes` - Fixed Git line ending issues

### 2. **Updated Configuration:**

**Smithery.yaml:**
```yaml
startCommand:
  type: "http"           # Changed from "mcp"
  port: 8000
  command: "python"
  args: ["-m", "src.http_server"]  # Changed from src.mcp_server
```

**Dockerfile:**
```dockerfile
# Added health check and HTTP endpoints
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "-m", "src.http_server"]
```

### 3. **Deploy to Smithery:**

```bash
# Method 1: Direct deploy
npx @smithery/cli deploy https://github.com/manasp21/rabi-mcp.git

# Method 2: Fork and deploy your own
git clone https://github.com/manasp21/rabi-mcp.git
cd rabi-mcp
# Push to your GitHub
npx @smithery/cli deploy https://github.com/YOUR_USERNAME/rabi-mcp.git
```

### 4. **Test Endpoints:**

After deployment, these endpoints should work:
- `GET /` - Server info
- `GET /health` - Health check 
- `GET /mcp/tools` - List physics tools
- `POST /mcp/tools/{tool_name}` - Execute physics tools
- `POST /mcp` - JSON-RPC MCP endpoint

### 5. **Expected Behavior:**

✅ **Before Fix:**
- Connection timeout
- "Please configure the server to list available tools"
- No tools visible

✅ **After Fix:**
- Immediate connection
- 25+ physics tools visible
- Interactive tool testing available
- HTTP health checks passing

## 🧪 **Test the Fix:**

### Option A: Test Locally First
```bash
# Clone the repo
git clone https://github.com/manasp21/rabi-mcp.git
cd rabi-mcp

# Build and test Docker container
docker build -t rabi-mcp-test .
docker run -p 8000:8000 rabi-mcp-test

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/mcp/tools
```

### Option B: Deploy to Smithery
```bash
# Deploy latest version
npx @smithery/cli deploy https://github.com/manasp21/rabi-mcp.git

# Wait for build to complete (2-3 minutes)
# Test in Smithery dashboard
```

## 🔬 **Physics Tools Available:**

After the fix, you should see all these tools:
- `simulate_two_level_atom` - Quantum dynamics
- `rabi_oscillations` - Coherent oscillations  
- `multi_level_atom` - Complex atomic systems
- `absorption_spectrum` - Spectroscopic analysis
- `laser_atom_interaction` - Strong-field physics
- `plot_bloch_sphere` - 3D quantum visualization
- `plot_population_dynamics` - Time evolution
- `bec_simulation` - Bose-Einstein condensates
- `cavity_qed` - Quantum optics

## 🐛 **If Still Having Issues:**

### Check Deployment Logs:
1. Go to Smithery dashboard
2. Click on "Deployments" tab
3. Check build logs for errors

### Common Issues:
- **Build timeout**: Container takes >10 minutes (normal for first build)
- **Memory issues**: Reduce `MAX_HILBERT_DIM` in config
- **Dependency errors**: Check `requirements.txt` compatibility

### Alternative: Use Pre-built Docker Image
```yaml
# In smithery.yaml, use pre-built image
build:
  image: "manasp21/rabi-mcp-server:latest"
```

## ✅ **Verification Steps:**

1. **Health Check**: `GET /health` returns 200
2. **Tools List**: `GET /mcp/tools` returns physics tools
3. **Tool Execution**: Physics calculations work
4. **Interactive UI**: Smithery dashboard shows tools
5. **No Timeouts**: Immediate connection

The fix addresses the core issue of transport protocol mismatch and should resolve the connection timeout completely! 🚀