# Optimized Dockerfile for Smithery deployment
# Fast, reliable build with prebuilt wheels only

FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install dependencies using prebuilt wheels only (fastest, most reliable)
RUN pip install --no-cache-dir --only-binary=all \
    mcp \
    aiohttp \
    websockets \
    pydantic \
    typer \
    rich \
    loguru \
    numpy \
    matplotlib

# Try to install scientific packages with prebuilt wheels only
RUN pip install --no-cache-dir --only-binary=all scipy || echo "Scipy wheel not available"
RUN pip install --no-cache-dir --only-binary=all qutip || echo "QuTiP wheel not available"

# Install the application without dependencies
RUN pip install --no-cache-dir -e . --no-deps

# Copy configuration
COPY config/ ./config/
RUN mkdir -p /root/.config/psianimator-mcp
COPY config/default_config.json /root/.config/psianimator-mcp/config.json

# Environment
ENV PYTHONPATH=/app/src \
    PSIANIMATOR_CONFIG=/root/.config/psianimator-mcp/config.json \
    PYTHONUNBUFFERED=1

EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=2 \
    CMD python -c "import psianimator_mcp" || exit 1

# Run server
CMD ["python", "-m", "psianimator_mcp.cli", "serve"]