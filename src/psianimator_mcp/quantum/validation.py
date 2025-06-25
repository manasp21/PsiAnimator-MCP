"""
Quantum state and operator validation utilities

Provides functions to validate quantum states, operators, and ensure
they satisfy required quantum mechanical properties.
"""

from __future__ import annotations

import numpy as np
from typing import Union, List, Tuple, Optional
import qutip as qt

from ..server.exceptions import (
    NormalizationError,
    HermitianError, 
    UnitarityError,
    DimensionError,
    QuantumStateError
)


def validate_quantum_state(
    state: Union[qt.Qobj, np.ndarray], 
    tolerance: float = 1e-10
) -> qt.Qobj:
    """
    Validate that a quantum state is properly normalized.
    
    Parameters
    ----------
    state : qt.Qobj or np.ndarray
        The quantum state to validate
    tolerance : float, optional
        Numerical tolerance for normalization check
        
    Returns
    -------
    qt.Qobj
        Validated quantum state as QuTip object
        
    Raises
    ------
    NormalizationError
        If state is not properly normalized
    QuantumStateError
        If state has invalid properties
    """
    # Convert to QuTip object if needed
    if isinstance(state, np.ndarray):
        if state.ndim == 1:
            state = qt.Qobj(state.reshape(-1, 1))
        else:
            state = qt.Qobj(state)
    
    if not isinstance(state, qt.Qobj):
        raise QuantumStateError("State must be a QuTip Qobj or numpy array")
    
    # Check if it's a valid quantum state type
    if state.type not in ['ket', 'bra', 'oper']:
        raise QuantumStateError(f"Invalid state type: {state.type}")
    
    # For ket states, check normalization
    if state.type == 'ket':
        norm = state.norm()
        if abs(norm - 1.0) > tolerance:
            raise NormalizationError(
                f"State norm is {norm:.10f}, expected 1.0",
                norm_value=norm,
                tolerance=tolerance
            )
    
    # For density matrices, check trace and positive semidefiniteness
    elif state.type == 'oper':
        trace_val = state.tr()
        if abs(trace_val - 1.0) > tolerance:
            raise NormalizationError(
                f"Density matrix trace is {trace_val:.10f}, expected 1.0",
                norm_value=trace_val,
                tolerance=tolerance
            )
        
        # Check if positive semidefinite (all eigenvalues >= 0)
        eigenvals = state.eigenenergies()
        if np.any(eigenvals < -tolerance):
            min_eigenval = np.min(eigenvals)
            raise QuantumStateError(
                f"Density matrix has negative eigenvalue: {min_eigenval:.10f}",
                state_info={"min_eigenvalue": min_eigenval}
            )
    
    return state


def validate_hermitian(
    operator: Union[qt.Qobj, np.ndarray],
    tolerance: float = 1e-10
) -> qt.Qobj:
    """
    Validate that an operator is Hermitian.
    
    Parameters
    ----------
    operator : qt.Qobj or np.ndarray
        The operator to validate
    tolerance : float, optional
        Numerical tolerance for Hermiticity check
        
    Returns
    -------
    qt.Qobj
        Validated Hermitian operator
        
    Raises
    ------
    HermitianError
        If operator is not Hermitian
    """
    # Convert to QuTip object if needed
    if isinstance(operator, np.ndarray):
        operator = qt.Qobj(operator)
    
    if not isinstance(operator, qt.Qobj):
        raise HermitianError("Operator must be a QuTip Qobj or numpy array")
    
    # Check if operator is square
    if operator.shape[0] != operator.shape[1]:
        raise HermitianError(
            f"Operator must be square, got shape {operator.shape}"
        )
    
    # Check Hermiticity: A = A†
    hermitian_diff = operator - operator.dag()
    max_deviation = np.max(np.abs(hermitian_diff.full()))
    
    if max_deviation > tolerance:
        raise HermitianError(
            f"Operator is not Hermitian, max deviation: {max_deviation:.2e}",
            max_deviation=max_deviation
        )
    
    return operator


def validate_unitary(
    operator: Union[qt.Qobj, np.ndarray],
    tolerance: float = 1e-10
) -> qt.Qobj:
    """
    Validate that an operator is unitary.
    
    Parameters
    ----------
    operator : qt.Qobj or np.ndarray
        The operator to validate
    tolerance : float, optional
        Numerical tolerance for unitarity check
        
    Returns
    -------
    qt.Qobj
        Validated unitary operator
        
    Raises
    ------
    UnitarityError
        If operator is not unitary
    """
    # Convert to QuTip object if needed  
    if isinstance(operator, np.ndarray):
        operator = qt.Qobj(operator)
    
    if not isinstance(operator, qt.Qobj):
        raise UnitarityError("Operator must be a QuTip Qobj or numpy array")
    
    # Check if operator is square
    if operator.shape[0] != operator.shape[1]:
        raise UnitarityError(
            f"Operator must be square, got shape {operator.shape}"
        )
    
    # Check unitarity: U†U = I
    identity_test = operator.dag() * operator
    identity_expected = qt.qeye(operator.shape[0])
    
    unity_diff = identity_test - identity_expected
    max_deviation = np.max(np.abs(unity_diff.full()))
    
    if max_deviation > tolerance:
        raise UnitarityError(
            f"Operator is not unitary, max deviation: {max_deviation:.2e}",
            max_deviation=max_deviation
        )
    
    return operator


def validate_dimensions(
    *objects: Union[qt.Qobj, List[int]],
    operation: str = "operation"
) -> None:
    """
    Validate that quantum objects have compatible dimensions.
    
    Parameters
    ----------
    *objects : qt.Qobj or list of int
        Quantum objects or dimension lists to check
    operation : str, optional
        Description of the operation for error messages
        
    Raises
    ------
    DimensionError
        If dimensions are incompatible
    """
    dims = []
    
    for i, obj in enumerate(objects):
        if isinstance(obj, qt.Qobj):
            dims.append(obj.dims)
        elif isinstance(obj, (list, tuple)):
            dims.append(list(obj))
        else:
            raise DimensionError(
                f"Object {i} must be QuTip Qobj or dimension list, got {type(obj)}"
            )
    
    # Check if all dimensions are compatible
    if len(set(str(d) for d in dims)) > 1:
        raise DimensionError(
            f"Incompatible dimensions for {operation}",
            expected_dims=dims[0] if dims else None,
            actual_dims=dims
        )


def check_hilbert_space_dimension(
    dimension: int,
    max_dimension: int = 1024
) -> None:
    """
    Check if Hilbert space dimension is within allowed limits.
    
    Parameters
    ----------
    dimension : int
        The Hilbert space dimension to check
    max_dimension : int, optional
        Maximum allowed dimension
        
    Raises
    ------
    DimensionError
        If dimension exceeds limits
    """
    if dimension < 2:
        raise DimensionError(
            f"Hilbert space dimension must be at least 2, got {dimension}"
        )
    
    if dimension > max_dimension:
        raise DimensionError(
            f"Hilbert space dimension {dimension} exceeds maximum {max_dimension}",
            expected_dims=[f"<= {max_dimension}"],
            actual_dims=[dimension]
        )


def normalize_state(
    state: Union[qt.Qobj, np.ndarray]
) -> qt.Qobj:
    """
    Normalize a quantum state.
    
    Parameters
    ----------
    state : qt.Qobj or np.ndarray
        The quantum state to normalize
        
    Returns
    -------
    qt.Qobj
        Normalized quantum state
    """
    # Convert to QuTip object if needed
    if isinstance(state, np.ndarray):
        if state.ndim == 1:
            state = qt.Qobj(state.reshape(-1, 1))
        else:
            state = qt.Qobj(state)
    
    if state.type == 'ket':
        return state.unit()
    elif state.type == 'oper':
        # For density matrices, normalize by trace
        return state / state.tr()
    else:
        raise QuantumStateError(f"Cannot normalize state of type: {state.type}")


def ensure_qobj(
    obj: Union[qt.Qobj, np.ndarray, complex, float],
    obj_type: Optional[str] = None
) -> qt.Qobj:
    """
    Ensure object is a QuTip Qobj with optional type validation.
    
    Parameters
    ----------
    obj : qt.Qobj, np.ndarray, complex, or float
        Object to convert to QuTip Qobj
    obj_type : str, optional
        Expected QuTip object type ('ket', 'bra', 'oper', 'super')
        
    Returns
    -------
    qt.Qobj
        QuTip object
        
    Raises
    ------
    QuantumStateError
        If conversion fails or type doesn't match
    """
    if isinstance(obj, qt.Qobj):
        qobj = obj
    elif isinstance(obj, np.ndarray):
        qobj = qt.Qobj(obj)
    elif isinstance(obj, (complex, float, int)):
        qobj = qt.Qobj([[obj]])
    else:
        raise QuantumStateError(f"Cannot convert {type(obj)} to QuTip Qobj")
    
    if obj_type and qobj.type != obj_type:
        raise QuantumStateError(
            f"Expected QuTip object type '{obj_type}', got '{qobj.type}'"
        )
    
    return qobj