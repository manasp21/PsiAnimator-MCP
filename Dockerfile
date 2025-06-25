# Simple Dockerfile for PsiAnimator-MCP on Smithery
# Optimized for cloud build environments

FROM python:3.11-slim

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest version
RUN pip install --no-cache-dir --upgrade pip

# Install core dependencies using prebuilt wheels (fastest approach)
RUN pip install --no-cache-dir --only-binary=all \
    numpy \
    scipy \
    matplotlib || \
    pip install --no-cache-dir \
    numpy \
    scipy \
    matplotlib

# Install QuTiP separately (may need fallback)
RUN pip install --no-cache-dir qutip || \
    pip install --no-cache-dir --no-binary qutip qutip

# Set working directory
WORKDIR /app

# Copy only essential files first
COPY pyproject.toml ./
COPY src/ ./src/

# Install MCP and lightweight dependencies
RUN pip install --no-cache-dir \
    mcp \
    aiohttp \
    websockets \
    pydantic \
    typer \
    rich \
    loguru

# Install the application
RUN pip install --no-cache-dir -e .

# Copy configuration
COPY config/ ./config/ 

# Create config directory and copy default config
RUN mkdir -p /root/.config/psianimator-mcp
COPY config/default_config.json /root/.config/psianimator-mcp/config.json

# Set environment variables
ENV PYTHONPATH=/app/src \
    PSIANIMATOR_CONFIG=/root/.config/psianimator-mcp/config.json \
    PYTHONUNBUFFERED=1

# Expose port
EXPOSE 3000

# Simple health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import psianimator_mcp" || exit 1

# Run server
CMD ["python", "-m", "psianimator_mcp.cli", "serve"]