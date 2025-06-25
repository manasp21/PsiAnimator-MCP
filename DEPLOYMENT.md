# PsiAnimator-MCP Deployment Guide

## Smithery Deployment

This repository is configured for deployment on [Smithery](https://smithery.ai) with simplified, reliable configuration.

### Quick Deploy

1. **Push to GitHub**: Ensure your code is pushed to the repository
2. **Connect to Smithery**: Import your repository in Smithery
3. **Deploy**: Smithery will automatically detect the configuration and deploy

### Configuration Files

- **`Dockerfile`**: Simplified build optimized for cloud deployment
- **`Dockerfile.minimal`**: Ultra-lightweight version (core MCP only)
- **`smithery.yaml`**: Simplified Smithery deployment configuration
- **`config/default_config.json`**: Default server configuration

### Build Options (if you encounter timeouts)

**Option 1: Default** (Recommended)
```yaml
# Current smithery.yaml uses this
dockerfile: "./Dockerfile"
```

**Option 2: Minimal Build** (If timeout issues persist)
```yaml
# Update smithery.yaml to use:
dockerfile: "./Dockerfile.minimal"
```

**Option 3: Conda-based** (Alternative approach)
```yaml
# Update smithery.yaml to use:
dockerfile: "./Dockerfile.light"
```

### Troubleshooting Smithery Errors

**"Unexpected internal error or timeout":**

1. **Try minimal build** - Update `smithery.yaml`:
   ```yaml
   spec:
     build:
       dockerfile: "./Dockerfile.minimal"
   ```

2. **Reduce complexity** - The minimal Dockerfile skips heavy scientific dependencies:
   - ✅ Core MCP server functionality
   - ✅ Basic quantum operations  
   - ⚠️ Advanced physics simulations may be limited

3. **Check logs** - Look for specific timeout or memory issues

**Build Taking Too Long:**
- Default: Uses prebuilt wheels (faster)
- Minimal: Only essential dependencies
- Light: Uses conda-forge (may be slower but more reliable)

### Local Testing

Test builds locally to debug:

```bash
# Test default build
docker build -t test-default .

# Test minimal build (fastest)
docker build -f Dockerfile.minimal -t test-minimal .

# Test locally
docker run --rm test-default python -c "import psianimator_mcp; print('Success')"
```

### Environment Variables

Configure in Smithery:
- `PSIANIMATOR_TRANSPORT`: `stdio` (default) or `websocket`
- `PYTHONUNBUFFERED`: `1` (for proper logging)

### Resource Requirements

**Minimal:**
- Memory: 256Mi
- CPU: 100m

**Default:**
- Memory: 1Gi  
- CPU: 500m

### Production Notes

- Start with minimal build for fastest deployment
- Upgrade to full scientific stack once running
- MCP server has conditional imports for graceful degradation
- Heavy physics computations optional

### If All Else Fails

Create a minimal `Dockerfile` with just:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install mcp pydantic typer rich
RUN pip install -e . --no-deps
CMD ["python", "-m", "psianimator_mcp.cli", "serve"]
```

The MCP server is designed to work with or without heavy dependencies!