"""
Quantum Operations for PsiAnimator-MCP

Handles unitary transformations, measurement operations, quantum channels,
and general quantum operations on states and operators.
"""

import numpy as np
import qutip as qt
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import logging

from .validation import (
    validate_quantum_state,
    validate_hermitian,
    validate_unitary,
    validate_dimensions,
    ensure_qobj
)
from ..server.exceptions import (
    QuantumOperationError,
    QuantumMeasurementError,
    ValidationError,
    DimensionError
)

logger = logging.getLogger(__name__)


class QuantumOperations:
    """
    Provides quantum operations including unitary transformations,
    measurements, quantum channels, and operator manipulations.
    """
    
    def __init__(self):
        """Initialize the quantum operations handler."""
        self._measurement_cache: Dict[str, Any] = {}
        logger.info("QuantumOperations initialized")
    
    def apply_unitary(
        self,
        state: qt.Qobj,
        unitary: Union[qt.Qobj, str, Dict[str, Any]],
        subsystems: Optional[List[int]] = None
    ) -> qt.Qobj:
        """
        Apply a unitary transformation to a quantum state.
        
        Parameters
        ----------
        state : qt.Qobj
            Input quantum state
        unitary : qt.Qobj, str, or dict
            Unitary operator, gate name, or gate specification
        subsystems : list of int, optional
            Subsystems to apply the operation to (for composite systems)
            
        Returns
        -------
        qt.Qobj
            Transformed quantum state
            
        Raises
        ------
        QuantumOperationError
            If operation fails
        """
        try:
            # Validate input state
            state = validate_quantum_state(state)
            
            # Parse unitary operator
            if isinstance(unitary, str):
                unitary_op = self._get_named_gate(unitary, state.dims)
            elif isinstance(unitary, dict):
                unitary_op = self._construct_gate_from_spec(unitary, state.dims)
            else:
                unitary_op = ensure_qobj(unitary, 'oper')
            
            # Validate unitarity
            unitary_op = validate_unitary(unitary_op)
            
            # Apply to specific subsystems if specified
            if subsystems is not None:
                unitary_op = self._expand_operator_to_subsystems(
                    unitary_op, subsystems, state.dims
                )
            
            # Check dimension compatibility
            validate_dimensions(state, unitary_op, operation="unitary transformation")
            
            # Apply transformation
            if state.type == 'ket':
                result = unitary_op * state
            elif state.type == 'oper':
                result = unitary_op * state * unitary_op.dag()
            else:
                raise QuantumOperationError(f"Cannot apply unitary to state type: {state.type}")
            
            logger.debug(f"Applied unitary transformation to {state.type} state")
            return result
            
        except Exception as e:
            raise QuantumOperationError(
                f"Failed to apply unitary transformation: {str(e)}",
                operation="apply_unitary",
                operator_info={"type": "unitary", "subsystems": subsystems}
            )
    
    def measure_observable(
        self,
        state: qt.Qobj,
        observable: Union[qt.Qobj, str, Dict[str, Any]],
        measurement_type: str = "expectation"
    ) -> Union[float, complex, Dict[str, Any]]:
        """
        Perform quantum measurement of an observable.
        
        Parameters
        ----------
        state : qt.Qobj
            Quantum state to measure
        observable : qt.Qobj, str, or dict
            Observable operator, name, or specification
        measurement_type : str
            Type of measurement ('expectation', 'variance', 'probability', 'correlation')
            
        Returns
        -------
        float, complex, or dict
            Measurement result
            
        Raises
        ------
        QuantumMeasurementError
            If measurement fails
        """
        try:
            # Validate input state
            state = validate_quantum_state(state)
            
            # Parse observable
            if isinstance(observable, str):
                obs_op = self._get_named_observable(observable, state.dims)
            elif isinstance(observable, dict):
                obs_op = self._construct_observable_from_spec(observable, state.dims)
            else:
                obs_op = ensure_qobj(observable, 'oper')
            
            # Validate observable (should be Hermitian)
            obs_op = validate_hermitian(obs_op)
            
            # Check dimension compatibility
            validate_dimensions(state, obs_op, operation="measurement")
            
            if measurement_type == "expectation":
                return self._calculate_expectation_value(state, obs_op)
            elif measurement_type == "variance":
                return self._calculate_variance(state, obs_op)
            elif measurement_type == "probability":
                return self._calculate_probability_distribution(state, obs_op)
            elif measurement_type == "correlation":
                return self._calculate_correlation_functions(state, obs_op)
            else:
                raise ValidationError(
                    f"Unknown measurement type: {measurement_type}",
                    field="measurement_type",
                    expected_type="one of: expectation, variance, probability, correlation"
                )
                
        except Exception as e:
            raise QuantumMeasurementError(
                f"Failed to measure observable: {str(e)}",
                measurement_type=measurement_type,
                observable_info={"type": type(observable).__name__}
            )
    
    def apply_quantum_channel(
        self,
        state: qt.Qobj,
        channel_type: str,
        parameters: Dict[str, Any]
    ) -> qt.Qobj:
        """
        Apply a quantum channel (CPTP map) to a quantum state.
        
        Parameters
        ----------
        state : qt.Qobj
            Input quantum state
        channel_type : str
            Type of quantum channel
        parameters : dict
            Channel-specific parameters
            
        Returns
        -------
        qt.Qobj
            State after channel application
            
        Raises
        ------
        QuantumOperationError
            If channel application fails
        """
        try:
            state = validate_quantum_state(state)
            
            if channel_type == "depolarizing":
                return self._apply_depolarizing_channel(state, parameters)
            elif channel_type == "amplitude_damping":
                return self._apply_amplitude_damping_channel(state, parameters)
            elif channel_type == "phase_damping":
                return self._apply_phase_damping_channel(state, parameters)
            elif channel_type == "bit_flip":
                return self._apply_bit_flip_channel(state, parameters)
            elif channel_type == "phase_flip":
                return self._apply_phase_flip_channel(state, parameters)
            elif channel_type == "kraus":
                return self._apply_kraus_channel(state, parameters)
            else:
                raise ValidationError(f"Unknown channel type: {channel_type}")
                
        except Exception as e:
            raise QuantumOperationError(
                f"Failed to apply quantum channel: {str(e)}",
                operation="apply_quantum_channel",
                operator_info={"channel_type": channel_type, "parameters": parameters}
            )
    
    def _get_named_gate(self, gate_name: str, dims: List[List[int]]) -> qt.Qobj:
        """Get a named quantum gate."""
        
        gate_name = gate_name.lower()
        
        # Single-qubit gates
        if gate_name in ['x', 'pauli_x', 'sigmax']:
            return qt.sigmax()
        elif gate_name in ['y', 'pauli_y', 'sigmay']:
            return qt.sigmay()
        elif gate_name in ['z', 'pauli_z', 'sigmaz']:
            return qt.sigmaz()
        elif gate_name in ['h', 'hadamard']:
            return qt.hadamard_transform()
        elif gate_name in ['s', 'phase']:
            return qt.phasegate(np.pi/2)
        elif gate_name in ['t']:
            return qt.phasegate(np.pi/4)
        elif gate_name in ['rx']:
            # Rotation around X-axis (requires angle parameter)
            return qt.rx(np.pi)  # Default π rotation
        elif gate_name in ['ry']:
            return qt.ry(np.pi)
        elif gate_name in ['rz']:
            return qt.rz(np.pi)
        
        # Two-qubit gates
        elif gate_name in ['cnot', 'cx']:
            return qt.cnot()
        elif gate_name in ['cz']:
            return qt.cphase(np.pi)
        elif gate_name in ['swap']:
            return qt.swap()
        
        # Multi-level system operators
        elif gate_name in ['create', 'a_dag']:
            dim = dims[0][0] if dims else 10
            return qt.create(dim)
        elif gate_name in ['destroy', 'a']:
            dim = dims[0][0] if dims else 10
            return qt.destroy(dim)
        elif gate_name in ['number', 'num']:
            dim = dims[0][0] if dims else 10
            return qt.num(dim)
        
        else:
            raise ValidationError(f"Unknown gate name: {gate_name}")
    
    def _get_named_observable(self, obs_name: str, dims: List[List[int]]) -> qt.Qobj:
        """Get a named observable operator."""
        
        obs_name = obs_name.lower()
        
        # Pauli operators
        if obs_name in ['x', 'sigma_x', 'pauli_x']:
            return qt.sigmax()
        elif obs_name in ['y', 'sigma_y', 'pauli_y']:
            return qt.sigmay()
        elif obs_name in ['z', 'sigma_z', 'pauli_z']:
            return qt.sigmaz()
        
        # Harmonic oscillator operators
        elif obs_name in ['position', 'x_op']:
            dim = dims[0][0] if dims else 10
            return (qt.create(dim) + qt.destroy(dim)) / np.sqrt(2)
        elif obs_name in ['momentum', 'p_op']:
            dim = dims[0][0] if dims else 10
            return 1j * (qt.create(dim) - qt.destroy(dim)) / np.sqrt(2)
        elif obs_name in ['number', 'n_op', 'photon_number']:
            dim = dims[0][0] if dims else 10
            return qt.num(dim)
        
        # Spin operators
        elif obs_name in ['spin_x', 'sx']:
            return qt.jmat(1/2, 'x')  # Spin-1/2 by default
        elif obs_name in ['spin_y', 'sy']:
            return qt.jmat(1/2, 'y')
        elif obs_name in ['spin_z', 'sz']:
            return qt.jmat(1/2, 'z')
        
        else:
            raise ValidationError(f"Unknown observable name: {obs_name}")
    
    def _construct_gate_from_spec(self, spec: Dict[str, Any], dims: List[List[int]]) -> qt.Qobj:
        """Construct quantum gate from specification dictionary."""
        
        gate_type = spec.get("type", "").lower()
        
        if gate_type == "rotation":
            axis = spec.get("axis", "z").lower()
            angle = spec.get("angle", 0.0)
            
            if axis == "x":
                return qt.rx(angle)
            elif axis == "y":
                return qt.ry(angle)
            elif axis == "z":
                return qt.rz(angle)
            else:
                raise ValidationError(f"Unknown rotation axis: {axis}")
        
        elif gate_type == "phase":
            phase = spec.get("phase", 0.0)
            return qt.phasegate(phase)
        
        elif gate_type == "custom":
            matrix = spec.get("matrix")
            if matrix is None:
                raise ValidationError("Custom gate requires 'matrix' parameter")
            return qt.Qobj(np.array(matrix))
        
        else:
            raise ValidationError(f"Unknown gate type: {gate_type}")
    
    def _construct_observable_from_spec(self, spec: Dict[str, Any], dims: List[List[int]]) -> qt.Qobj:
        """Construct observable from specification dictionary."""
        
        obs_type = spec.get("type", "").lower()
        
        if obs_type == "pauli":
            direction = spec.get("direction", [0, 0, 1])  # Default Z direction
            return (direction[0] * qt.sigmax() + 
                   direction[1] * qt.sigmay() + 
                   direction[2] * qt.sigmaz())
        
        elif obs_type == "spin":
            spin = spec.get("spin", 1/2)
            direction = spec.get("direction", "z")
            return qt.jmat(spin, direction)
        
        elif obs_type == "quadrature":
            dim = dims[0][0] if dims else 10
            phase = spec.get("phase", 0.0)
            a = qt.destroy(dim)
            return (a * np.exp(-1j * phase) + a.dag() * np.exp(1j * phase)) / np.sqrt(2)
        
        elif obs_type == "custom":
            matrix = spec.get("matrix")
            if matrix is None:
                raise ValidationError("Custom observable requires 'matrix' parameter")
            return qt.Qobj(np.array(matrix))
        
        else:
            raise ValidationError(f"Unknown observable type: {obs_type}")
    
    def _calculate_expectation_value(self, state: qt.Qobj, observable: qt.Qobj) -> complex:
        """Calculate expectation value of observable."""
        if state.type == 'ket':
            return qt.expect(observable, state)
        elif state.type == 'oper':
            return (observable * state).tr()
        else:
            raise QuantumMeasurementError(f"Cannot calculate expectation for state type: {state.type}")
    
    def _calculate_variance(self, state: qt.Qobj, observable: qt.Qobj) -> float:
        """Calculate variance of observable."""
        exp_val = self._calculate_expectation_value(state, observable)
        exp_val_squared = self._calculate_expectation_value(state, observable * observable)
        return np.real(exp_val_squared - exp_val * np.conj(exp_val))
    
    def _calculate_probability_distribution(self, state: qt.Qobj, observable: qt.Qobj) -> Dict[str, Any]:
        """Calculate probability distribution for observable measurement."""
        
        # Get eigenvalues and eigenstates
        eigenvals, eigenstates = observable.eigenstates()
        
        probabilities = []
        if state.type == 'ket':
            for eigenstate in eigenstates:
                prob = abs(eigenstate.dag() * state)**2
                probabilities.append(float(np.real(prob[0, 0])))
        elif state.type == 'oper':
            for eigenstate in eigenstates:
                proj = eigenstate * eigenstate.dag()
                prob = (proj * state).tr()
                probabilities.append(float(np.real(prob)))
        
        return {
            "eigenvalues": [float(np.real(val)) for val in eigenvals],
            "probabilities": probabilities,
            "measurement_outcomes": list(zip(eigenvals, probabilities))
        }
    
    def _calculate_correlation_functions(self, state: qt.Qobj, observable: qt.Qobj) -> Dict[str, Any]:
        """Calculate correlation functions for the observable."""
        
        # This is a simplified version - full correlation functions would require
        # time evolution or multiple observables
        exp_val = self._calculate_expectation_value(state, observable)
        variance = self._calculate_variance(state, observable)
        
        return {
            "expectation_value": complex(exp_val),
            "variance": float(variance),
            "standard_deviation": float(np.sqrt(variance))
        }
    
    def _expand_operator_to_subsystems(
        self, 
        operator: qt.Qobj, 
        subsystems: List[int], 
        full_dims: List[List[int]]
    ) -> qt.Qobj:
        """Expand operator to act on specific subsystems of composite system."""
        
        # This is a simplified implementation
        # Full implementation would require proper tensor product expansion
        total_subsystems = len(full_dims[0])
        
        if len(subsystems) == 1 and total_subsystems > 1:
            # Single subsystem operation in composite system
            subsystem_idx = subsystems[0]
            
            identity_ops = []
            for i in range(total_subsystems):
                if i == subsystem_idx:
                    identity_ops.append(operator)
                else:
                    dim = full_dims[0][i]
                    identity_ops.append(qt.qeye(dim))
            
            return qt.tensor(*identity_ops)
        else:
            return operator
    
    def _apply_depolarizing_channel(self, state: qt.Qobj, params: Dict[str, Any]) -> qt.Qobj:
        """Apply depolarizing channel."""
        p = params.get("probability", 0.1)
        dim = state.shape[0]
        
        if state.type == 'ket':
            state = state * state.dag()  # Convert to density matrix
        
        identity = qt.qeye(dim) / dim
        return (1 - p) * state + p * identity
    
    def _apply_amplitude_damping_channel(self, state: qt.Qobj, params: Dict[str, Any]) -> qt.Qobj:
        """Apply amplitude damping channel."""
        gamma = params.get("gamma", 0.1)
        
        # Kraus operators for amplitude damping
        E0 = qt.Qobj([[1, 0], [0, np.sqrt(1 - gamma)]])
        E1 = qt.Qobj([[0, np.sqrt(gamma)], [0, 0]])
        
        if state.type == 'ket':
            state = state * state.dag()
        
        return E0 * state * E0.dag() + E1 * state * E1.dag()
    
    def _apply_phase_damping_channel(self, state: qt.Qobj, params: Dict[str, Any]) -> qt.Qobj:
        """Apply phase damping channel."""
        gamma = params.get("gamma", 0.1)
        
        # Kraus operators for phase damping
        E0 = qt.Qobj([[1, 0], [0, np.sqrt(1 - gamma)]])
        E1 = qt.Qobj([[0, 0], [0, np.sqrt(gamma)]])
        
        if state.type == 'ket':
            state = state * state.dag()
        
        return E0 * state * E0.dag() + E1 * state * E1.dag()
    
    def _apply_bit_flip_channel(self, state: qt.Qobj, params: Dict[str, Any]) -> qt.Qobj:
        """Apply bit flip channel."""
        p = params.get("probability", 0.1)
        
        if state.type == 'ket':
            state = state * state.dag()
        
        X = qt.sigmax()
        return (1 - p) * state + p * X * state * X.dag()
    
    def _apply_phase_flip_channel(self, state: qt.Qobj, params: Dict[str, Any]) -> qt.Qobj:
        """Apply phase flip channel."""
        p = params.get("probability", 0.1)
        
        if state.type == 'ket':
            state = state * state.dag()
        
        Z = qt.sigmaz()
        return (1 - p) * state + p * Z * state * Z.dag()
    
    def _apply_kraus_channel(self, state: qt.Qobj, params: Dict[str, Any]) -> qt.Qobj:
        """Apply general Kraus channel."""
        kraus_ops = params.get("kraus_operators", [])
        
        if not kraus_ops:
            raise ValidationError("Kraus channel requires 'kraus_operators' parameter")
        
        if state.type == 'ket':
            state = state * state.dag()
        
        result = sum(qt.Qobj(np.array(K)) * state * qt.Qobj(np.array(K)).dag() 
                    for K in kraus_ops)
        
        return result