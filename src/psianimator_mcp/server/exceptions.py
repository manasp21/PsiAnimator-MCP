"""
Custom exception classes for PsiAnimator-MCP

Provides quantum physics and MCP-specific error handling with
detailed error messages and proper error hierarchies.
"""

from typing import Optional, Any, Dict


class QuantumMCPError(Exception):
    """Base exception class for all PsiAnimator-MCP errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }


class QuantumStateError(QuantumMCPError):
    """Raised when quantum state operations fail."""
    
    def __init__(
        self, 
        message: str, 
        state_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        details = {"state_info": state_info} if state_info else {}
        details.update(kwargs.get("details", {}))
        super().__init__(message, details=details, **kwargs)


class QuantumOperationError(QuantumMCPError):
    """Raised when quantum operations fail."""
    
    def __init__(
        self, 
        message: str,
        operation: Optional[str] = None,
        operator_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        details = {
            "operation": operation,
            "operator_info": operator_info
        }
        details.update(kwargs.get("details", {}))
        super().__init__(message, details=details, **kwargs)


class QuantumSystemError(QuantumMCPError):
    """Raised when quantum system modeling fails."""
    
    def __init__(
        self, 
        message: str,
        system_type: Optional[str] = None,
        dimensions: Optional[list] = None,
        **kwargs
    ):
        details = {
            "system_type": system_type,
            "dimensions": dimensions
        }
        details.update(kwargs.get("details", {}))
        super().__init__(message, details=details, **kwargs)


class QuantumEvolutionError(QuantumMCPError):
    """Raised when time evolution calculations fail."""
    
    def __init__(
        self, 
        message: str,
        evolution_type: Optional[str] = None,
        time_span: Optional[tuple] = None,
        solver_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        details = {
            "evolution_type": evolution_type,
            "time_span": time_span,
            "solver_info": solver_info
        }
        details.update(kwargs.get("details", {}))
        super().__init__(message, details=details, **kwargs)


class QuantumMeasurementError(QuantumMCPError):
    """Raised when quantum measurement operations fail."""
    
    def __init__(
        self, 
        message: str,
        measurement_type: Optional[str] = None,
        observable_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        details = {
            "measurement_type": measurement_type,
            "observable_info": observable_info
        }
        details.update(kwargs.get("details", {}))
        super().__init__(message, details=details, **kwargs)


class AnimationError(QuantumMCPError):
    """Raised when animation generation fails."""
    
    def __init__(
        self, 
        message: str,
        animation_type: Optional[str] = None,
        render_settings: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        details = {
            "animation_type": animation_type,
            "render_settings": render_settings
        }
        details.update(kwargs.get("details", {}))
        super().__init__(message, details=details, **kwargs)


class ValidationError(QuantumMCPError):
    """Raised when input validation fails."""
    
    def __init__(
        self, 
        message: str,
        field: Optional[str] = None,
        expected_type: Optional[str] = None,
        received_value: Optional[Any] = None,
        **kwargs
    ):
        details = {
            "field": field,
            "expected_type": expected_type,
            "received_value": str(received_value) if received_value is not None else None
        }
        details.update(kwargs.get("details", {}))
        super().__init__(message, details=details, **kwargs)


class ConfigurationError(QuantumMCPError):
    """Raised when server configuration is invalid."""
    
    def __init__(
        self, 
        message: str,
        config_field: Optional[str] = None,
        **kwargs
    ):
        details = {"config_field": config_field}
        details.update(kwargs.get("details", {}))
        super().__init__(message, details=details, **kwargs)


class ResourceError(QuantumMCPError):
    """Raised when system resources are insufficient."""
    
    def __init__(
        self, 
        message: str,
        resource_type: Optional[str] = None,
        requested: Optional[Any] = None,
        available: Optional[Any] = None,
        **kwargs
    ):
        details = {
            "resource_type": resource_type,
            "requested": requested,
            "available": available
        }
        details.update(kwargs.get("details", {}))
        super().__init__(message, details=details, **kwargs)


class DimensionError(QuantumMCPError):
    """Raised when Hilbert space dimensions are incompatible."""
    
    def __init__(
        self, 
        message: str,
        expected_dims: Optional[list] = None,
        actual_dims: Optional[list] = None,
        **kwargs
    ):
        details = {
            "expected_dimensions": expected_dims,
            "actual_dimensions": actual_dims
        }
        details.update(kwargs.get("details", {}))
        super().__init__(message, details=details, **kwargs)


class NormalizationError(QuantumStateError):
    """Raised when quantum state normalization fails."""
    
    def __init__(
        self, 
        message: str = "Quantum state is not properly normalized",
        norm_value: Optional[float] = None,
        tolerance: Optional[float] = None,
        **kwargs
    ):
        details = {
            "norm_value": norm_value,
            "tolerance": tolerance
        }
        details.update(kwargs.get("details", {}))
        super().__init__(message, details=details, **kwargs)


class HermitianError(QuantumOperationError):
    """Raised when an operator is expected to be Hermitian but is not."""
    
    def __init__(
        self, 
        message: str = "Operator is not Hermitian",
        max_deviation: Optional[float] = None,
        **kwargs
    ):
        details = {"max_deviation": max_deviation}
        details.update(kwargs.get("details", {}))
        super().__init__(message, details=details, **kwargs)


class UnitarityError(QuantumOperationError):
    """Raised when an operator is expected to be unitary but is not."""
    
    def __init__(
        self, 
        message: str = "Operator is not unitary",
        max_deviation: Optional[float] = None,
        **kwargs
    ):
        details = {"max_deviation": max_deviation}
        details.update(kwargs.get("details", {}))
        super().__init__(message, details=details, **kwargs)