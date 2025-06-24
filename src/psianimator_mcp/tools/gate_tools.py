"""
Quantum Gate Tools for PsiAnimator-MCP

Implements the quantum_gate_sequence MCP tool for applying sequences of quantum gates
with visualization support and intermediate state tracking.
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import qutip as qt

from ..quantum.state_manager import QuantumStateManager
from ..quantum.operations import QuantumOperations
from ..quantum.validation import validate_quantum_state, validate_unitary
from ..server.config import MCPConfig
from ..server.exceptions import (
    QuantumOperationError,
    ValidationError,
    QuantumStateError,
    DimensionError
)
from .quantum_state_tools import get_state_manager

logger = logging.getLogger(__name__)

# Global quantum operations instance
_quantum_operations = None


def get_quantum_operations() -> QuantumOperations:
    """Get or create the global quantum operations instance."""
    global _quantum_operations
    if _quantum_operations is None:
        _quantum_operations = QuantumOperations()
    return _quantum_operations


async def quantum_gate_sequence(arguments: Dict[str, Any], config: MCPConfig) -> Dict[str, Any]:
    """
    Apply a sequence of quantum gates to a quantum state.
    
    Parameters
    ----------
    arguments : dict
        Tool arguments containing:
        - state_id: str - ID of the initial quantum state
        - gates: list - List of gate specifications
        - animate_steps: bool - Whether to animate gate application
        - show_intermediate_states: bool - Whether to show/store intermediate states
        - gate_visualization: str - Type of visualization ('circuit', 'bloch', 'matrix')
        - measurement_at_end: dict - Optional measurement specification
    config : MCPConfig
        Server configuration
        
    Returns
    -------
    dict
        Gate sequence results with final state and intermediate information
    """
    try:
        logger.info(f"Applying quantum gate sequence with arguments: {arguments}")
        
        # Validate required arguments
        required_args = ['state_id', 'gates']
        for arg in required_args:
            if arg not in arguments:
                raise ValidationError(f"{arg} is required", field=arg)
        
        state_id = arguments['state_id']
        gates = arguments['gates']
        animate_steps = arguments.get('animate_steps', False)
        show_intermediate_states = arguments.get('show_intermediate_states', True)
        gate_visualization = arguments.get('gate_visualization', 'circuit')
        measurement_at_end = arguments.get('measurement_at_end', None)
        
        # Validate gates
        if not isinstance(gates, list):
            raise ValidationError("gates must be a list", field="gates")
        
        if len(gates) == 0:
            raise ValidationError("gates list cannot be empty", field="gates")
        
        # Validate each gate specification
        for i, gate_spec in enumerate(gates):
            validate_gate_specification(gate_spec, i)
        
        # Get initial state
        state_manager = get_state_manager(config)
        
        if state_id not in state_manager.list_states():
            raise QuantumStateError(f"State with ID '{state_id}' not found")
        
        initial_state = state_manager.get_state(state_id)
        initial_state = validate_quantum_state(initial_state)
        
        # Determine number of qubits/subsystems
        state_dims = initial_state.dims[0]
        n_qubits = len(state_dims) if len(state_dims) > 1 else int(np.log2(initial_state.shape[0]))
        
        # Get quantum operations handler
        quantum_ops = get_quantum_operations()
        
        # Apply gate sequence
        result = await apply_gate_sequence(
            initial_state, gates, quantum_ops, show_intermediate_states, state_manager, state_id, config
        )
        
        # Perform final measurement if specified
        if measurement_at_end:
            measurement_result = await perform_final_measurement(
                result['final_state'], measurement_at_end, quantum_ops
            )
            result['final_measurement'] = measurement_result
        
        # Generate visualization data if requested
        if animate_steps:
            visualization_data = generate_gate_visualization_data(
                initial_state, gates, result['intermediate_states'], gate_visualization
            )
            result['visualization_data'] = visualization_data
        
        # Calculate circuit properties
        circuit_properties = analyze_circuit_properties(gates, initial_state, result['final_state'])
        result['circuit_analysis'] = circuit_properties
        
        # Prepare final result
        gate_result = {
            'success': True,
            'initial_state_id': state_id,
            'n_gates_applied': len(gates),
            'n_qubits': n_qubits,
            'gate_sequence': gates,
            'animate_steps': animate_steps,
            'show_intermediate_states': show_intermediate_states,
            'gate_application_results': result,
            'message': f"Successfully applied {len(gates)} gates to state {state_id}"
        }
        
        logger.info(f"Successfully applied gate sequence to state {state_id}")
        return gate_result
        
    except (ValidationError, QuantumOperationError, QuantumStateError, DimensionError) as e:
        logger.error(f"Gate sequence application failed: {e}")
        return {
            'success': False,
            'error': e.__class__.__name__,
            'message': str(e),
            'details': e.details if hasattr(e, 'details') else {}
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in quantum_gate_sequence: {e}")
        return {
            'success': False,
            'error': 'UnexpectedError',
            'message': f"An unexpected error occurred: {str(e)}",
            'details': {}
        }


def validate_gate_specification(gate_spec: Dict[str, Any], gate_index: int) -> None:
    """
    Validate a single gate specification.
    
    Parameters
    ----------
    gate_spec : dict
        Gate specification dictionary
    gate_index : int
        Index of gate in sequence (for error reporting)
    """
    if not isinstance(gate_spec, dict):
        raise ValidationError(
            f"Gate {gate_index} must be a dictionary",
            field=f"gates[{gate_index}]"
        )
    
    if 'name' not in gate_spec:
        raise ValidationError(
            f"Gate {gate_index} must have 'name' field",
            field=f"gates[{gate_index}].name"
        )
    
    if 'qubits' not in gate_spec:
        raise ValidationError(
            f"Gate {gate_index} must have 'qubits' field",
            field=f"gates[{gate_index}].qubits"
        )
    
    gate_name = gate_spec['name']
    qubits = gate_spec['qubits']
    
    # Validate qubit list
    if not isinstance(qubits, list):
        raise ValidationError(
            f"Gate {gate_index} 'qubits' must be a list",
            field=f"gates[{gate_index}].qubits"
        )
    
    if len(qubits) == 0:
        raise ValidationError(
            f"Gate {gate_index} 'qubits' list cannot be empty",
            field=f"gates[{gate_index}].qubits"
        )
    
    if not all(isinstance(q, int) and q >= 0 for q in qubits):
        raise ValidationError(
            f"Gate {gate_index} 'qubits' must contain non-negative integers",
            field=f"gates[{gate_index}].qubits"
        )
    
    # Validate gate name and qubit count compatibility
    single_qubit_gates = ['X', 'Y', 'Z', 'H', 'S', 'T', 'RX', 'RY', 'RZ', 'U', 'PHASE']
    two_qubit_gates = ['CNOT', 'CX', 'CZ', 'SWAP', 'CY', 'CPHASE']
    three_qubit_gates = ['TOFFOLI', 'FREDKIN']
    
    if gate_name.upper() in single_qubit_gates and len(qubits) != 1:
        raise ValidationError(
            f"Gate {gate_index} '{gate_name}' requires exactly 1 qubit, got {len(qubits)}",
            field=f"gates[{gate_index}].qubits"
        )
    
    if gate_name.upper() in two_qubit_gates and len(qubits) != 2:
        raise ValidationError(
            f"Gate {gate_index} '{gate_name}' requires exactly 2 qubits, got {len(qubits)}",
            field=f"gates[{gate_index}].qubits"
        )
    
    if gate_name.upper() in three_qubit_gates and len(qubits) != 3:
        raise ValidationError(
            f"Gate {gate_index} '{gate_name}' requires exactly 3 qubits, got {len(qubits)}",
            field=f"gates[{gate_index}].qubits"
        )
    
    # Validate parameters for parameterized gates
    parameters = gate_spec.get('parameters', {})
    if gate_name.upper() in ['RX', 'RY', 'RZ', 'PHASE', 'CPHASE', 'U']:
        if not parameters:
            raise ValidationError(
                f"Gate {gate_index} '{gate_name}' requires parameters",
                field=f"gates[{gate_index}].parameters"
            )
        
        if gate_name.upper() in ['RX', 'RY', 'RZ', 'PHASE', 'CPHASE']:
            if 'angle' not in parameters:
                raise ValidationError(
                    f"Gate {gate_index} '{gate_name}' requires 'angle' parameter",
                    field=f"gates[{gate_index}].parameters.angle"
                )
        
        if gate_name.upper() == 'U':
            required_params = ['theta', 'phi', 'lambda']
            for param in required_params:
                if param not in parameters:
                    raise ValidationError(
                        f"Gate {gate_index} 'U' gate requires '{param}' parameter",
                        field=f"gates[{gate_index}].parameters.{param}"
                    )


async def apply_gate_sequence(initial_state: qt.Qobj, gates: List[Dict[str, Any]], quantum_ops: QuantumOperations,
                            show_intermediate: bool, state_manager: QuantumStateManager, 
                            initial_state_id: str, config: MCPConfig) -> Dict[str, Any]:
    """
    Apply sequence of gates to quantum state.
    
    Parameters
    ----------
    initial_state : qt.Qobj
        Initial quantum state
    gates : list
        List of gate specifications
    quantum_ops : QuantumOperations
        Quantum operations handler
    show_intermediate : bool
        Whether to store intermediate states
    state_manager : QuantumStateManager
        State manager for storing intermediate states
    initial_state_id : str
        ID of initial state
    config : MCPConfig
        Server configuration
        
    Returns
    -------
    dict
        Gate application results
    """
    try:
        current_state = initial_state
        intermediate_states = []
        intermediate_state_ids = []
        gate_fidelities = []
        applied_gates = []
        
        for i, gate_spec in enumerate(gates):
            gate_name = gate_spec['name']
            qubits = gate_spec['qubits']
            parameters = gate_spec.get('parameters', {})
            
            # Create gate operator
            gate_operator = create_gate_operator(gate_spec, current_state.dims)
            
            # Apply gate
            previous_state = current_state
            current_state = quantum_ops.apply_unitary(current_state, gate_operator, qubits)
            
            # Calculate gate fidelity
            if previous_state.type == 'ket' and current_state.type == 'ket':
                fidelity = abs(previous_state.dag() * current_state)**2
            else:
                # For mixed states, use quantum fidelity
                if previous_state.type == 'ket':
                    rho1 = previous_state * previous_state.dag()
                else:
                    rho1 = previous_state
                
                if current_state.type == 'ket':
                    rho2 = current_state * current_state.dag()
                else:
                    rho2 = current_state
                
                fidelity = qt.fidelity(rho1, rho2)**2
            
            gate_fidelities.append(float(np.real(fidelity)))
            
            # Store intermediate state if requested
            if show_intermediate:
                intermediate_state_id = f"{initial_state_id}_after_gate_{i}_{gate_name}"
                
                try:
                    # Store in state manager
                    state_manager._states[intermediate_state_id] = current_state
                    state_manager._state_metadata[intermediate_state_id] = {
                        'state_type': 'intermediate',
                        'parent_state_id': initial_state_id,
                        'gate_index': i,
                        'gate_name': gate_name,
                        'gate_qubits': qubits,
                        'gate_parameters': parameters,
                        'system_dims': current_state.dims[0],
                        'hilbert_dim': current_state.shape[0]
                    }
                    intermediate_state_ids.append(intermediate_state_id)
                except Exception as e:
                    logger.warning(f"Could not store intermediate state after gate {i}: {e}")
            
            intermediate_states.append(current_state)
            applied_gates.append({
                'index': i,
                'name': gate_name,
                'qubits': qubits,
                'parameters': parameters,
                'fidelity': gate_fidelities[-1]
            })
            
            logger.debug(f"Applied gate {i}: {gate_name} on qubits {qubits}")
        
        # Calculate final state properties
        final_state_properties = analyze_final_state(initial_state, current_state)
        
        return {
            'final_state': current_state,
            'intermediate_states': intermediate_states,
            'intermediate_state_ids': intermediate_state_ids,
            'applied_gates': applied_gates,
            'gate_fidelities': gate_fidelities,
            'average_fidelity': float(np.mean(gate_fidelities)),
            'min_fidelity': float(np.min(gate_fidelities)),
            'final_state_properties': final_state_properties
        }
        
    except Exception as e:
        raise QuantumOperationError(
            f"Failed to apply gate sequence: {str(e)}",
            operation="gate_sequence"
        )


def create_gate_operator(gate_spec: Dict[str, Any], state_dims: List[List[int]]) -> qt.Qobj:
    """
    Create quantum gate operator from specification.
    
    Parameters
    ----------
    gate_spec : dict
        Gate specification
    state_dims : list
        State dimensions
        
    Returns
    -------
    qt.Qobj
        Gate operator
    """
    gate_name = gate_spec['name'].upper()
    parameters = gate_spec.get('parameters', {})
    
    # Single-qubit gates
    if gate_name == 'X':
        return qt.sigmax()
    elif gate_name == 'Y':
        return qt.sigmay()
    elif gate_name == 'Z':
        return qt.sigmaz()
    elif gate_name == 'H':
        return qt.hadamard_transform()
    elif gate_name == 'S':
        return qt.phasegate(np.pi/2)
    elif gate_name == 'T':
        return qt.phasegate(np.pi/4)
    elif gate_name == 'RX':
        angle = parameters.get('angle', 0.0)
        return qt.rx(angle)
    elif gate_name == 'RY':
        angle = parameters.get('angle', 0.0)
        return qt.ry(angle)
    elif gate_name == 'RZ':
        angle = parameters.get('angle', 0.0)
        return qt.rz(angle)
    elif gate_name == 'PHASE':
        angle = parameters.get('angle', 0.0)
        return qt.phasegate(angle)
    elif gate_name == 'U':
        theta = parameters.get('theta', 0.0)
        phi = parameters.get('phi', 0.0)
        lam = parameters.get('lambda', 0.0)
        return qt.Qobj([[np.cos(theta/2), -np.exp(1j*lam)*np.sin(theta/2)],
                       [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lam))*np.cos(theta/2)]])
    
    # Two-qubit gates
    elif gate_name in ['CNOT', 'CX']:
        return qt.cnot()
    elif gate_name == 'CZ':
        return qt.cphase(np.pi)
    elif gate_name == 'CY':
        return qt.Qobj([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, -1j],
                       [0, 0, 1j, 0]])
    elif gate_name == 'SWAP':
        return qt.swap()
    elif gate_name == 'CPHASE':
        angle = parameters.get('angle', 0.0)
        return qt.cphase(angle)
    
    # Three-qubit gates
    elif gate_name == 'TOFFOLI':
        return qt.toffoli()
    elif gate_name == 'FREDKIN':
        return qt.fredkin()
    
    # Custom gate
    elif gate_name == 'CUSTOM':
        if 'matrix' not in parameters:
            raise ValidationError("Custom gate requires 'matrix' parameter")
        matrix = np.array(parameters['matrix'], dtype=complex)
        gate = qt.Qobj(matrix)
        return validate_unitary(gate)
    
    else:
        raise ValidationError(f"Unknown gate: {gate_name}")


async def perform_final_measurement(final_state: qt.Qobj, measurement_spec: Dict[str, Any], 
                                  quantum_ops: QuantumOperations) -> Dict[str, Any]:
    """
    Perform measurement on final state.
    
    Parameters
    ----------
    final_state : qt.Qobj
        Final quantum state
    measurement_spec : dict
        Measurement specification
    quantum_ops : QuantumOperations
        Quantum operations handler
        
    Returns
    -------
    dict
        Measurement results
    """
    try:
        observable = measurement_spec.get('observable', 'sigmaz')
        measurement_type = measurement_spec.get('type', 'expectation')
        
        # Create observable operator
        if observable == 'sigmaz':
            obs = qt.sigmaz()
        elif observable == 'sigmax':
            obs = qt.sigmax()
        elif observable == 'sigmay':
            obs = qt.sigmay()
        elif observable == 'num':
            obs = qt.num(final_state.shape[0])
        else:
            # Try to parse as expression
            obs = quantum_ops._get_named_observable(observable, final_state.dims)
        
        # Perform measurement
        if measurement_type == 'expectation':
            result = quantum_ops._calculate_expectation_value(final_state, obs)
            return {
                'type': 'expectation',
                'observable': observable,
                'value': complex(result)
            }
        elif measurement_type == 'probability':
            prob_dist = quantum_ops._calculate_probability_distribution(final_state, obs)
            return {
                'type': 'probability',
                'observable': observable,
                'distribution': prob_dist
            }
        else:
            raise ValidationError(f"Unknown measurement type: {measurement_type}")
            
    except Exception as e:
        logger.warning(f"Final measurement failed: {e}")
        return {
            'type': 'error',
            'message': str(e)
        }


def generate_gate_visualization_data(initial_state: qt.Qobj, gates: List[Dict[str, Any]], 
                                   intermediate_states: List[qt.Qobj], 
                                   visualization_type: str) -> Dict[str, Any]:
    """
    Generate data for gate sequence visualization.
    
    Parameters
    ----------
    initial_state : qt.Qobj
        Initial state
    gates : list
        Gate specifications
    intermediate_states : list
        Intermediate states after each gate
    visualization_type : str
        Type of visualization
        
    Returns
    -------
    dict
        Visualization data
    """
    try:
        if visualization_type == 'circuit':
            return {
                'type': 'circuit',
                'n_qubits': int(np.log2(initial_state.shape[0])),
                'gates': gates,
                'layout': 'horizontal'
            }
        
        elif visualization_type == 'bloch' and initial_state.shape[0] == 2:
            # For qubit, generate Bloch sphere trajectory
            bloch_points = []
            all_states = [initial_state] + intermediate_states
            
            for state in all_states:
                if state.type == 'ket':
                    rho = state * state.dag()
                else:
                    rho = state
                
                # Calculate Bloch vector
                x = np.real((qt.sigmax() * rho).tr())
                y = np.real((qt.sigmay() * rho).tr()) 
                z = np.real((qt.sigmaz() * rho).tr())
                
                bloch_points.append([float(x), float(y), float(z)])
            
            return {
                'type': 'bloch',
                'trajectory': bloch_points,
                'gates': gates
            }
        
        elif visualization_type == 'matrix':
            # Matrix representation evolution
            matrices = []
            all_states = [initial_state] + intermediate_states
            
            for state in all_states:
                if state.type == 'ket':
                    matrix = (state * state.dag()).full()
                else:
                    matrix = state.full()
                
                # Convert to serializable format
                matrix_real = np.real(matrix).tolist()
                matrix_imag = np.imag(matrix).tolist()
                
                matrices.append({
                    'real': matrix_real,
                    'imag': matrix_imag
                })
            
            return {
                'type': 'matrix',
                'matrices': matrices,
                'gates': gates
            }
        
        else:
            return {
                'type': 'unsupported',
                'message': f"Visualization type '{visualization_type}' not supported for this state"
            }
            
    except Exception as e:
        logger.warning(f"Visualization data generation failed: {e}")
        return {
            'type': 'error',
            'message': str(e)
        }


def analyze_circuit_properties(gates: List[Dict[str, Any]], initial_state: qt.Qobj, 
                             final_state: qt.Qobj) -> Dict[str, Any]:
    """
    Analyze properties of the quantum circuit.
    
    Parameters
    ----------
    gates : list
        Gate specifications
    initial_state : qt.Qobj
        Initial state
    final_state : qt.Qobj
        Final state
        
    Returns
    -------
    dict
        Circuit analysis results
    """
    try:
        analysis = {}
        
        # Basic circuit statistics
        analysis['total_gates'] = len(gates)
        analysis['unique_gate_types'] = len(set(gate['name'].upper() for gate in gates))
        
        # Count gate types
        gate_counts = {}
        for gate in gates:
            gate_name = gate['name'].upper()
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
        
        analysis['gate_counts'] = gate_counts
        
        # Circuit depth (simplified - assumes all gates can be parallelized optimally)
        qubit_usage = {}
        depth = 0
        current_time = {}
        
        for gate in gates:
            qubits = gate['qubits']
            max_time = max((current_time.get(q, 0) for q in qubits), default=0)
            
            for q in qubits:
                current_time[q] = max_time + 1
                qubit_usage[q] = qubit_usage.get(q, 0) + 1
        
        analysis['circuit_depth'] = max(current_time.values()) if current_time else 0
        analysis['qubit_usage'] = qubit_usage
        
        # State fidelity
        if initial_state.type == final_state.type:
            if initial_state.type == 'ket':
                fidelity = abs(initial_state.dag() * final_state)**2
            else:
                fidelity = qt.fidelity(initial_state, final_state)**2
            
            analysis['initial_final_fidelity'] = float(np.real(fidelity))
        
        # Entanglement analysis (for multi-qubit systems)
        if final_state.shape[0] > 2 and final_state.shape[0] == 2**int(np.log2(final_state.shape[0])):
            n_qubits = int(np.log2(final_state.shape[0]))
            
            if n_qubits >= 2:
                # Calculate entanglement entropy for bipartition
                if final_state.type == 'ket':
                    rho = final_state * final_state.dag()
                else:
                    rho = final_state
                
                # Partial trace over second half
                subsystem_a = list(range(n_qubits // 2))
                rho_a = rho.ptrace(subsystem_a)
                
                # Von Neumann entropy
                eigenvals = rho_a.eigenenergies()
                entropy = -np.sum(eigenvals * np.log2(eigenvals + 1e-16))
                
                analysis['entanglement_entropy'] = float(np.real(entropy))
                analysis['is_entangled'] = entropy > 0.01  # Threshold for numerical precision
        
        return analysis
        
    except Exception as e:
        logger.warning(f"Circuit analysis failed: {e}")
        return {'error': str(e)}


def analyze_final_state(initial_state: qt.Qobj, final_state: qt.Qobj) -> Dict[str, Any]:
    """
    Analyze properties of the final state.
    
    Parameters
    ----------
    initial_state : qt.Qobj
        Initial state
    final_state : qt.Qobj
        Final state
        
    Returns
    -------
    dict
        Final state analysis
    """
    try:
        properties = {}
        
        # Basic properties
        properties['state_type'] = final_state.type
        properties['dimensions'] = final_state.shape
        properties['hilbert_space_dim'] = final_state.shape[0]
        
        # Normalization
        if final_state.type == 'ket':
            norm = final_state.norm()
            properties['norm'] = float(norm)
            properties['is_normalized'] = abs(norm - 1.0) < 1e-10
        else:
            trace = final_state.tr()
            properties['trace'] = complex(trace)
            properties['is_normalized'] = abs(trace - 1.0) < 1e-10
        
        # Purity
        if final_state.type == 'ket':
            properties['purity'] = 1.0
        else:
            purity = np.real((final_state * final_state).tr())
            properties['purity'] = float(purity)
        
        properties['is_pure'] = properties['purity'] > 0.99
        
        # For qubits, add Bloch vector
        if final_state.shape[0] == 2:
            if final_state.type == 'ket':
                rho = final_state * final_state.dag()
            else:
                rho = final_state
            
            bloch_x = np.real((qt.sigmax() * rho).tr())
            bloch_y = np.real((qt.sigmay() * rho).tr())
            bloch_z = np.real((qt.sigmaz() * rho).tr())
            
            properties['bloch_vector'] = [float(bloch_x), float(bloch_y), float(bloch_z)]
            properties['bloch_vector_length'] = float(np.sqrt(bloch_x**2 + bloch_y**2 + bloch_z**2))
        
        return properties
        
    except Exception as e:
        logger.warning(f"Final state analysis failed: {e}")
        return {'error': str(e)}