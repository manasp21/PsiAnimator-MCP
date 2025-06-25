# Optimized Dockerfile for PsiAnimator-MCP on Smithery
# Handles heavy scientific dependencies efficiently

FROM python:3.11-slim

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_NO_CACHE_DIR=1
ARG PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for scientific computing
RUN apt-get update && apt-get install -y \
    # Essential build tools
    build-essential \
    gcc \
    g++ \
    gfortran \
    # BLAS/LAPACK libraries for NumPy/SciPy
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    # Additional scientific libraries  
    libfftw3-dev \
    libhdf5-dev \
    pkg-config \
    # Utilities
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install build essentials
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install NumPy first (required by other scientific packages)
RUN pip install --no-cache-dir "numpy>=1.24.0"

# Install scientific computing stack in optimal order
RUN pip install --no-cache-dir \
    "scipy>=1.11.0" \
    "matplotlib>=3.7.0" \
    "qutip>=4.7.0"

# Set working directory  
WORKDIR /app

# Copy project files (order optimized for layer caching)
COPY pyproject.toml ./
COPY src/ ./src/

# Install remaining MCP and utility dependencies
RUN pip install --no-cache-dir \
    "mcp>=1.0.0" \
    "aiohttp>=3.8.0" \
    "websockets>=11.0.0" \
    "pydantic>=2.0.0" \
    "typer>=0.9.0" \
    "rich>=13.0.0" \
    "loguru>=0.7.0"

# Install the application without dependencies (heavy deps already installed)
RUN pip install --no-cache-dir -e . --no-deps

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash psianimator && \
    mkdir -p /home/psianimator/.config/psianimator-mcp && \
    chown -R psianimator:psianimator /app /home/psianimator

# Copy configuration and scripts
COPY config/ ./config/
COPY --chown=psianimator:psianimator config/default_config.json /home/psianimator/.config/psianimator-mcp/config.json

# Switch to non-root user
USER psianimator

# Set environment variables
ENV PYTHONPATH=/app/src \
    PSIANIMATOR_CONFIG=/home/psianimator/.config/psianimator-mcp/config.json \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose port for WebSocket server
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import psianimator_mcp; print('Health check passed')" || exit 1

# Default command - run MCP server with stdio transport
CMD ["python", "-m", "psianimator_mcp.cli", "serve"]