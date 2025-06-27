# Production Dockerfile for PsiAnimator-MCP on Smithery
# Full scientific stack with optimized build

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for scientific computing
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install core dependencies first
RUN pip install --no-cache-dir --upgrade pip

# Install MCP and utility dependencies
RUN pip install --no-cache-dir \
    mcp \
    aiohttp \
    websockets \
    pydantic \
    typer \
    rich \
    loguru

# Install scientific packages with prebuilt wheels
RUN pip install --no-cache-dir --only-binary=all \
    numpy \
    scipy \
    matplotlib \
    qutip || \
    pip install --no-cache-dir \
    numpy \
    scipy \
    matplotlib

# Install the application
RUN pip install --no-cache-dir -e .

# Copy configuration
COPY config/ ./config/
RUN mkdir -p /root/.config/psianimator-mcp
COPY config/default_config.json /root/.config/psianimator-mcp/config.json

# Environment variables
ENV PYTHONPATH=/app/src \
    PSIANIMATOR_CONFIG=/root/.config/psianimator-mcp/config.json \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Expose port for HTTP server
EXPOSE 8000

# Health check for HTTP server
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run HTTP server (matches smithery.yaml exactly)
CMD ["python", "-m", "psianimator_mcp.http_server"]