"""
Configuration management for PsiAnimator-MCP Server

Handles server configuration with quantum-specific settings and
support for loading from files or environment variables.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

from pydantic import BaseModel, validator


class MCPConfig(BaseModel):
    """Configuration class for MCP server with quantum-specific settings."""
    
    # Quantum computation settings
    quantum_precision: float = 1e-12
    max_hilbert_dimension: int = 1024
    enable_gpu_acceleration: bool = False
    
    # Animation settings
    animation_cache_size: int = 100
    render_backend: str = "cairo"  # "cairo" or "opengl"
    default_render_quality: str = "medium"  # "low", "medium", "high", "production"
    default_frame_rate: int = 30
    
    # Server settings
    parallel_workers: int = 4
    max_memory_usage_mb: int = 4096
    enable_logging: bool = True
    log_level: str = "INFO"
    
    # Output settings
    output_directory: str = "./output"
    enable_caching: bool = True
    cache_directory: str = "./cache"
    
    @validator("render_backend")
    def validate_render_backend(cls, v):
        if v not in ["cairo", "opengl"]:
            raise ValueError("render_backend must be 'cairo' or 'opengl'")
        return v
    
    @validator("default_render_quality")
    def validate_render_quality(cls, v):
        if v not in ["low", "medium", "high", "production"]:
            raise ValueError("render_quality must be one of: low, medium, high, production")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        if v not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("log_level must be a valid logging level")
        return v
    
    @validator("parallel_workers")
    def validate_workers(cls, v):
        if v < 1:
            raise ValueError("parallel_workers must be at least 1")
        return min(v, os.cpu_count() or 4)
    
    @classmethod
    def from_file(cls, config_path: Path) -> "MCPConfig":
        """Load configuration from a JSON file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, "r") as f:
            config_data = json.load(f)
        
        return cls(**config_data)
    
    @classmethod
    def from_env(cls) -> "MCPConfig":
        """Load configuration from environment variables."""
        config_data = {}
        
        # Map environment variables to config fields
        env_mapping = {
            "PSIANIMATOR_QUANTUM_PRECISION": "quantum_precision",
            "PSIANIMATOR_MAX_HILBERT_DIM": "max_hilbert_dimension", 
            "PSIANIMATOR_GPU_ACCELERATION": "enable_gpu_acceleration",
            "PSIANIMATOR_ANIMATION_CACHE_SIZE": "animation_cache_size",
            "PSIANIMATOR_RENDER_BACKEND": "render_backend",
            "PSIANIMATOR_RENDER_QUALITY": "default_render_quality",
            "PSIANIMATOR_FRAME_RATE": "default_frame_rate",
            "PSIANIMATOR_PARALLEL_WORKERS": "parallel_workers",
            "PSIANIMATOR_MAX_MEMORY_MB": "max_memory_usage_mb",
            "PSIANIMATOR_LOG_LEVEL": "log_level",
            "PSIANIMATOR_OUTPUT_DIR": "output_directory",
            "PSIANIMATOR_CACHE_DIR": "cache_directory"
        }
        
        for env_var, config_key in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if config_key in ["quantum_precision"]:
                    config_data[config_key] = float(value)
                elif config_key in ["max_hilbert_dimension", "animation_cache_size", 
                                   "default_frame_rate", "parallel_workers", "max_memory_usage_mb"]:
                    config_data[config_key] = int(value)
                elif config_key in ["enable_gpu_acceleration", "enable_logging", "enable_caching"]:
                    config_data[config_key] = value.lower() in ["true", "1", "yes", "on"]
                else:
                    config_data[config_key] = value
        
        return cls(**config_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.dict()
    
    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to a JSON file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def create_directories(self) -> None:
        """Create necessary directories for operation."""
        Path(self.output_directory).mkdir(parents=True, exist_ok=True)
        if self.enable_caching:
            Path(self.cache_directory).mkdir(parents=True, exist_ok=True)