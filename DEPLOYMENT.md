# PsiAnimator-MCP Deployment Guide

## Smithery Deployment

This repository is configured for deployment on [Smithery](https://smithery.ai) with all necessary configuration files.

### Quick Deploy

1. **Push to GitHub**: Ensure your code is pushed to the repository
2. **Connect to Smithery**: Import your repository in Smithery
3. **Deploy**: Smithery will automatically detect the configuration and deploy

### Configuration Files

- **`Dockerfile`**: Production-optimized container with scientific dependencies
- **`Dockerfile.light`**: Lightweight alternative using conda-forge (faster builds)
- **`smithery.yaml`**: Smithery deployment configuration
- **`environment.yml`**: Conda environment for lightweight builds
- **`config/default_config.json`**: Default server configuration

### Key Features Handled

✅ **Heavy Scientific Dependencies**: Pre-installed NumPy, SciPy, QuTiP with proper system libraries  
✅ **System Dependencies**: BLAS, LAPACK, Fortran compilers included  
✅ **MCP Protocol**: Full Model Context Protocol server implementation  
✅ **Security**: Non-root user, minimal attack surface  
✅ **Health Checks**: Built-in container health monitoring  
✅ **Multi-transport**: Supports both stdio and WebSocket transports  

### Environment Variables

Set these in Smithery for custom configuration:

- `PSIANIMATOR_TRANSPORT`: `stdio` (default) or `websocket`
- `PSIANIMATOR_LOG_LEVEL`: `DEBUG`, `INFO`, `WARNING`, `ERROR`
- `PSIANIMATOR_CONFIG`: Path to custom config file

### Resource Requirements

**Minimum:**
- Memory: 512Mi
- CPU: 250m

**Recommended:**
- Memory: 2Gi
- CPU: 1000m

### Build Alternatives

1. **Standard Build** (`Dockerfile`): Full scientific stack with system compilation
2. **Fast Build** (`Dockerfile.light`): Uses conda-forge prebuilt packages (recommended for faster deployment)

To use the lightweight build, update `smithery.yaml`:
```yaml
spec:
  build:
    dockerfile: "./Dockerfile.light"
```

### Troubleshooting

**Build Issues:**
- Heavy dependencies require significant build time
- Use `Dockerfile.light` for faster builds
- Ensure adequate memory allocation during build

**Runtime Issues:**
- Check health endpoints
- Verify MCP protocol connectivity
- Review logs for import errors

### Local Testing

Test the Docker build locally:

```bash
# Standard build
docker build -t psianimator-mcp .

# Lightweight build  
docker build -f Dockerfile.light -t psianimator-mcp-light .

# Test run
docker run --rm psianimator-mcp
```

### Production Considerations

- Use persistent volumes for output data
- Configure log aggregation
- Set up monitoring for quantum computations
- Consider GPU acceleration for large simulations
- Implement proper backup strategies for quantum state data

For more details, see the [Smithery documentation](https://smithery.ai/docs).