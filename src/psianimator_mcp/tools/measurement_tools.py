"""
Quantum Measurement Tools for PsiAnimator-MCP

Implements the measure_observable MCP tool for performing quantum measurements,
calculating expectation values, variances, and probability distributions.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import qutip as qt

from ..quantum.state_manager import QuantumStateManager
from ..quantum.operations import QuantumOperations
from ..quantum.validation import validate_quantum_state, validate_hermitian
from ..server.config import MCPConfig
from ..server.exceptions import (
    QuantumMeasurementError,
    ValidationError,
    QuantumStateError
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


async def measure_observable(arguments: Dict[str, Any], config: MCPConfig) -> Dict[str, Any]:
    """
    Perform quantum measurement of an observable on a quantum state.
    
    Parameters
    ----------
    arguments : dict
        Tool arguments containing:
        - state_id: str - ID of the quantum state to measure
        - observable: str - Observable specification (name, expression, or matrix)
        - measurement_type: str - Type of measurement ('expectation', 'variance', 'probability', 'correlation')
        - measurement_basis: str (optional) - Measurement basis specification
        - n_measurements: int (optional) - Number of measurement repetitions for statistics
        - store_results: bool (optional) - Whether to store measurement results
    config : MCPConfig
        Server configuration
        
    Returns
    -------
    dict
        Measurement results including values, uncertainties, and statistics
    """
    try:
        logger.info(f"Performing quantum measurement with arguments: {arguments}")
        
        # Validate required arguments
        required_args = ['state_id', 'observable', 'measurement_type']
        for arg in required_args:
            if arg not in arguments:
                raise ValidationError(f"{arg} is required", field=arg)
        
        state_id = arguments['state_id']
        observable_spec = arguments['observable']
        measurement_type = arguments['measurement_type']
        measurement_basis = arguments.get('measurement_basis', None)
        n_measurements = arguments.get('n_measurements', 1)
        store_results = arguments.get('store_results', False)
        
        # Validate measurement type
        valid_types = ['expectation', 'variance', 'probability', 'correlation']
        if measurement_type not in valid_types:
            raise ValidationError(
                f"Invalid measurement_type: {measurement_type}",
                field="measurement_type",
                expected_type=f"one of: {', '.join(valid_types)}"
            )
        
        # Validate n_measurements
        if not isinstance(n_measurements, int) or n_measurements < 1:
            raise ValidationError(
                "n_measurements must be a positive integer",
                field="n_measurements"
            )
        
        # Get quantum state
        state_manager = get_state_manager(config)
        
        if state_id not in state_manager.list_states():
            raise QuantumStateError(f"State with ID '{state_id}' not found")
        
        quantum_state = state_manager.get_state(state_id)
        quantum_state = validate_quantum_state(quantum_state)
        
        # Parse observable
        observable = parse_observable(observable_spec, quantum_state.dims, config)
        
        # Validate observable dimensions
        if observable.shape != quantum_state.shape:
            raise ValidationError(
                f"Observable dimensions {observable.shape} do not match state dimensions {quantum_state.shape}"
            )
        
        # Get quantum operations handler
        quantum_ops = get_quantum_operations()
        
        # Perform measurement based on type
        if measurement_type == 'expectation':
            result = await measure_expectation_value(
                quantum_state, observable, quantum_ops, n_measurements
            )
            
        elif measurement_type == 'variance':
            result = await measure_variance(
                quantum_state, observable, quantum_ops, n_measurements
            )
            
        elif measurement_type == 'probability':
            result = await measure_probability_distribution(
                quantum_state, observable, quantum_ops, measurement_basis
            )
            
        elif measurement_type == 'correlation':
            result = await measure_correlations(
                quantum_state, observable, quantum_ops, measurement_basis
            )
        
        # Add measurement metadata
        measurement_result = {
            'success': True,
            'state_id': state_id,
            'observable_spec': observable_spec,
            'measurement_type': measurement_type,
            'measurement_basis': measurement_basis,
            'n_measurements': n_measurements,
            'quantum_state_info': {
                'type': quantum_state.type,
                'dimensions': quantum_state.dims,
                'hilbert_space_dim': quantum_state.shape[0],
                'is_pure': quantum_state.type == 'ket' or (
                    quantum_state.type == 'oper' and 
                    abs((quantum_state * quantum_state).tr() - 1.0) < 1e-10
                )
            },
            'observable_info': {
                'dimensions': observable.shape,
                'is_hermitian': check_hermiticity(observable),
                'eigenvalue_range': get_eigenvalue_range(observable)
            },
            'measurement_results': result,
            'timestamp': logger.info.__defaults__[0] if hasattr(logger.info, '__defaults__') else None
        }
        
        # Store results if requested
        if store_results:
            measurement_id = f"measurement_{state_id}_{measurement_type}_{hash(str(arguments)) % 10000}"
            
            # Store in state manager
            try:
                state_manager.store_measurement_result(
                    measurement_id,
                    state_id,
                    observable,
                    measurement_type,
                    measurement_result['measurement_results'],
                    {
                        'measurement_basis': measurement_basis,
                        'n_measurements': n_measurements,
                        'arguments': arguments
                    }
                )
                measurement_result['measurement_id'] = measurement_id
                logger.info(f"Stored measurement result with ID: {measurement_id}")
                
            except Exception as e:
                logger.warning(f"Could not store measurement result: {e}")
                measurement_result['measurement_id'] = None
        
        logger.info(f"Successfully performed {measurement_type} measurement on state {state_id}")
        return measurement_result
        
    except (ValidationError, QuantumMeasurementError, QuantumStateError) as e:
        logger.error(f"Quantum measurement failed: {e}")
        return {
            'success': False,
            'error': e.__class__.__name__,
            'message': str(e),
            'details': e.details if hasattr(e, 'details') else {}
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in measure_observable: {e}")
        return {
            'success': False,
            'error': 'UnexpectedError',
            'message': f"An unexpected error occurred: {str(e)}",
            'details': {}
        }


def parse_observable(observable_spec: str, state_dims: List[List[int]], config: MCPConfig) -> qt.Qobj:
    """
    Parse observable specification into QuTip operator.
    
    Parameters
    ----------
    observable_spec : str
        Observable specification
    state_dims : list
        State dimensions
    config : MCPConfig
        Server configuration
        
    Returns
    -------
    qt.Qobj
        Observable operator
    """
    try:
        # Get quantum operations handler
        quantum_ops = get_quantum_operations()
        
        # Try to parse as named observable
        try:
            return quantum_ops._get_named_observable(observable_spec, state_dims)
        except ValidationError:
            pass
        
        # Try to parse as matrix
        if observable_spec.startswith('[') and observable_spec.endswith(']'):
            try:
                matrix_data = json.loads(observable_spec)
                obs = qt.Qobj(np.array(matrix_data, dtype=complex))
                return validate_hermitian(obs)
            except (json.JSONDecodeError, ValueError) as e:
                raise ValidationError(f"Invalid matrix format: {e}")
        
        # Try to parse as operator expression
        elif '+' in observable_spec or '*' in observable_spec or any(op in observable_spec for op in ['sigma', 'num', 'create', 'destroy']):
            return parse_observable_expression(observable_spec, state_dims)
        
        else:
            raise ValidationError(f"Unknown observable specification: {observable_spec}")
            
    except Exception as e:
        raise QuantumMeasurementError(
            f"Failed to parse observable: {str(e)}",
            measurement_type="observable_parsing",
            observable_info={'spec': observable_spec}
        )


def parse_observable_expression(expr: str, state_dims: List[List[int]]) -> qt.Qobj:
    """
    Parse observable expression like 'sigmax + sigmay' or 'num + 0.5*identity'.
    
    Parameters
    ----------
    expr : str
        Observable expression
    state_dims : list
        System dimensions
        
    Returns
    -------
    qt.Qobj
        Parsed observable operator
    """
    try:
        if len(state_dims[0]) == 1:
            dim = state_dims[0][0]
            
            # Create common operators
            operators = {
                'identity': qt.qeye(dim),
                'num': qt.num(dim),
                'destroy': qt.destroy(dim),
                'create': qt.create(dim),
                'x': (qt.destroy(dim) + qt.create(dim)) / np.sqrt(2),
                'p': 1j * (qt.create(dim) - qt.destroy(dim)) / np.sqrt(2),
                'position': (qt.destroy(dim) + qt.create(dim)) / np.sqrt(2),
                'momentum': 1j * (qt.create(dim) - qt.destroy(dim)) / np.sqrt(2)
            }
            
            # For qubits, add Pauli operators
            if dim == 2:
                operators.update({
                    'sigmax': qt.sigmax(),
                    'sigmay': qt.sigmay(),
                    'sigmaz': qt.sigmaz(),
                    'sigmap': qt.sigmap(),
                    'sigmam': qt.sigmam(),
                    'pauli_x': qt.sigmax(),
                    'pauli_y': qt.sigmay(),
                    'pauli_z': qt.sigmaz()
                })
            
            # Replace operator names in expression
            eval_expr = expr
            for name, op in operators.items():
                eval_expr = eval_expr.replace(name, f'operators["{name}"]')
            
            # Define constants and functions
            constants = {
                'np': np,
                'operators': operators,
                'sqrt': np.sqrt,
                'exp': np.exp,
                'sin': np.sin,
                'cos': np.cos,
                'pi': np.pi
            }
            
            # Evaluate the expression
            result = eval(eval_expr, constants)
            
            if isinstance(result, qt.Qobj):
                return result
            else:
                # If result is a scalar, multiply by identity
                return result * qt.qeye(dim)
        
        else:
            raise ValidationError("Complex observable expressions not yet supported for composite systems")
            
    except Exception as e:
        raise ValidationError(f"Failed to parse observable expression '{expr}': {str(e)}")


async def measure_expectation_value(state: qt.Qobj, observable: qt.Qobj, quantum_ops: QuantumOperations, n_measurements: int) -> Dict[str, Any]:
    """
    Calculate expectation value of observable.
    
    Parameters
    ----------
    state : qt.Qobj
        Quantum state
    observable : qt.Qobj
        Observable operator
    quantum_ops : QuantumOperations
        Quantum operations handler
    n_measurements : int
        Number of measurements for statistical analysis
        
    Returns
    -------
    dict
        Expectation value results
    """
    try:
        # Calculate exact expectation value
        exact_expectation = quantum_ops._calculate_expectation_value(state, observable)
        
        # Calculate variance for uncertainty
        variance = quantum_ops._calculate_variance(state, observable)
        uncertainty = np.sqrt(variance)
        
        # For multiple measurements, simulate shot noise
        if n_measurements > 1:
            # Simulate measurement statistics (simplified)
            # In reality, this would involve actual quantum measurement simulation
            measurements = []
            for _ in range(n_measurements):
                # Add shot noise based on uncertainty
                noise = np.random.normal(0, uncertainty / np.sqrt(n_measurements))
                measurement = exact_expectation + noise
                measurements.append(complex(measurement))
            
            # Calculate statistics from simulated measurements
            measured_mean = np.mean(measurements)
            measured_std = np.std(measurements)
            statistical_uncertainty = measured_std / np.sqrt(n_measurements)
            
            return {
                'exact_expectation_value': complex(exact_expectation),
                'measured_mean': complex(measured_mean),
                'theoretical_uncertainty': float(np.real(uncertainty)),
                'statistical_uncertainty': float(np.real(statistical_uncertainty)),
                'measured_std': float(np.real(measured_std)),
                'variance': float(variance),
                'individual_measurements': measurements if n_measurements <= 100 else measurements[:100],
                'measurement_statistics': {
                    'n_measurements': n_measurements,
                    'relative_error': float(abs(measured_mean - exact_expectation) / abs(exact_expectation)) if abs(exact_expectation) > 1e-10 else 0.0
                }
            }
        else:
            return {
                'expectation_value': complex(exact_expectation),
                'uncertainty': float(np.real(uncertainty)),
                'variance': float(variance),
                'standard_deviation': float(np.real(uncertainty))
            }
            
    except Exception as e:
        raise QuantumMeasurementError(
            f"Failed to calculate expectation value: {str(e)}",
            measurement_type="expectation"
        )


async def measure_variance(state: qt.Qobj, observable: qt.Qobj, quantum_ops: QuantumOperations, n_measurements: int) -> Dict[str, Any]:
    """
    Calculate variance and higher moments of observable.
    
    Parameters
    ----------
    state : qt.Qobj
        Quantum state
    observable : qt.Qobj
        Observable operator
    quantum_ops : QuantumOperations
        Quantum operations handler
    n_measurements : int
        Number of measurements
        
    Returns
    -------
    dict
        Variance and moment results
    """
    try:
        # Calculate expectation value and variance
        expectation = quantum_ops._calculate_expectation_value(state, observable)
        variance = quantum_ops._calculate_variance(state, observable)
        
        # Calculate higher moments
        obs_squared = observable * observable
        obs_cubed = obs_squared * observable
        obs_fourth = obs_cubed * observable
        
        second_moment = quantum_ops._calculate_expectation_value(state, obs_squared)
        third_moment = quantum_ops._calculate_expectation_value(state, obs_cubed)
        fourth_moment = quantum_ops._calculate_expectation_value(state, obs_fourth)
        
        # Central moments
        central_third_moment = third_moment - 3 * expectation * second_moment + 2 * expectation**3
        central_fourth_moment = (fourth_moment - 4 * expectation * third_moment + 
                               6 * expectation**2 * second_moment - 3 * expectation**4)
        
        # Skewness and kurtosis
        if variance > 1e-10:
            skewness = central_third_moment / (variance**(3/2))
            kurtosis = central_fourth_moment / (variance**2)
        else:
            skewness = 0.0
            kurtosis = 0.0
        
        return {
            'expectation_value': complex(expectation),
            'variance': float(np.real(variance)),
            'standard_deviation': float(np.real(np.sqrt(variance))),
            'second_moment': complex(second_moment),
            'third_moment': complex(third_moment),
            'fourth_moment': complex(fourth_moment),
            'central_third_moment': complex(central_third_moment),
            'central_fourth_moment': complex(central_fourth_moment),
            'skewness': float(np.real(skewness)),
            'kurtosis': float(np.real(kurtosis)),
            'coefficient_of_variation': float(np.real(np.sqrt(variance) / expectation)) if abs(expectation) > 1e-10 else float('inf')
        }
        
    except Exception as e:
        raise QuantumMeasurementError(
            f"Failed to calculate variance: {str(e)}",
            measurement_type="variance"
        )


async def measure_probability_distribution(state: qt.Qobj, observable: qt.Qobj, quantum_ops: QuantumOperations, measurement_basis: Optional[str]) -> Dict[str, Any]:
    """
    Calculate probability distribution for observable measurement.
    
    Parameters
    ----------
    state : qt.Qobj
        Quantum state
    observable : qt.Qobj
        Observable operator
    quantum_ops : QuantumOperations
        Quantum operations handler
    measurement_basis : str, optional
        Measurement basis specification
        
    Returns
    -------
    dict
        Probability distribution results
    """
    try:
        # Get probability distribution from quantum operations
        prob_dist = quantum_ops._calculate_probability_distribution(state, observable)
        
        eigenvalues = prob_dist['eigenvalues']
        probabilities = prob_dist['probabilities']
        
        # Calculate distribution statistics
        mean_value = sum(val * prob for val, prob in zip(eigenvalues, probabilities))
        variance_value = sum((val - mean_value)**2 * prob for val, prob in zip(eigenvalues, probabilities))
        
        # Find most probable outcome
        max_prob_idx = np.argmax(probabilities)
        most_probable_value = eigenvalues[max_prob_idx]
        max_probability = probabilities[max_prob_idx]
        
        # Calculate cumulative distribution
        sorted_indices = np.argsort(eigenvalues)
        sorted_eigenvalues = [eigenvalues[i] for i in sorted_indices]
        sorted_probabilities = [probabilities[i] for i in sorted_indices]
        cumulative_probabilities = np.cumsum(sorted_probabilities).tolist()
        
        # Shannon entropy of distribution
        shannon_entropy = -sum(p * np.log2(p + 1e-16) for p in probabilities if p > 1e-16)
        
        result = {
            'eigenvalues': eigenvalues,
            'probabilities': probabilities,
            'measurement_outcomes': prob_dist['measurement_outcomes'],
            'distribution_statistics': {
                'mean': float(np.real(mean_value)),
                'variance': float(np.real(variance_value)),
                'standard_deviation': float(np.real(np.sqrt(variance_value))),
                'most_probable_value': float(np.real(most_probable_value)),
                'max_probability': float(max_probability),
                'shannon_entropy': float(np.real(shannon_entropy)),
                'number_of_outcomes': len(eigenvalues)
            },
            'cumulative_distribution': {
                'sorted_eigenvalues': sorted_eigenvalues,
                'cumulative_probabilities': cumulative_probabilities
            }
        }
        
        # Add basis-specific information
        if measurement_basis:
            result['measurement_basis'] = measurement_basis
            
            # Add basis-specific interpretations
            if measurement_basis.lower() in ['computational', 'z']:
                result['basis_interpretation'] = {
                    'basis_type': 'computational',
                    'basis_states': [f"|{i}⟩" for i in range(len(eigenvalues))],
                    'physical_meaning': 'Computational basis measurement (Z-basis for qubits)'
                }
            elif measurement_basis.lower() in ['x', 'hadamard']:
                result['basis_interpretation'] = {
                    'basis_type': 'x_basis',
                    'basis_states': ["|+⟩", "|-⟩"] if len(eigenvalues) == 2 else [f"|x_{i}⟩" for i in range(len(eigenvalues))],
                    'physical_meaning': 'X-basis measurement (superposition basis)'
                }
            elif measurement_basis.lower() in ['y']:
                result['basis_interpretation'] = {
                    'basis_type': 'y_basis', 
                    'basis_states': ["|+i⟩", "|-i⟩"] if len(eigenvalues) == 2 else [f"|y_{i}⟩" for i in range(len(eigenvalues))],
                    'physical_meaning': 'Y-basis measurement'
                }
        
        return result
        
    except Exception as e:
        raise QuantumMeasurementError(
            f"Failed to calculate probability distribution: {str(e)}",
            measurement_type="probability"
        )


async def measure_correlations(state: qt.Qobj, observable: qt.Qobj, quantum_ops: QuantumOperations, measurement_basis: Optional[str]) -> Dict[str, Any]:
    """
    Calculate correlation functions and related measures.
    
    Parameters
    ----------
    state : qt.Qobj
        Quantum state
    observable : qt.Qobj
        Observable operator
    quantum_ops : QuantumOperations
        Quantum operations handler
    measurement_basis : str, optional
        Measurement basis specification
        
    Returns
    -------
    dict
        Correlation function results
    """
    try:
        # Get basic correlation information
        corr_info = quantum_ops._calculate_correlation_functions(state, observable)
        
        expectation_value = corr_info['expectation_value']
        variance = corr_info['variance']
        
        # Calculate autocorrelation (⟨A†A⟩)
        obs_dag = observable.dag()
        autocorr = quantum_ops._calculate_expectation_value(state, obs_dag * observable)
        
        # For composite systems, calculate cross-correlations
        correlation_results = {
            'expectation_value': expectation_value,
            'variance': variance,
            'standard_deviation': corr_info['standard_deviation'],
            'autocorrelation': complex(autocorr)
        }
        
        # If observable is creation/destruction operator, calculate g^(1) and g^(2)
        if 'destroy' in str(observable) or 'create' in str(observable) or observable.shape[0] > 2:
            # Assume this is a bosonic system
            a = qt.destroy(observable.shape[0])
            a_dag = qt.create(observable.shape[0])
            
            # First-order coherence g^(1)
            g1_numerator = quantum_ops._calculate_expectation_value(state, a_dag * a)
            g1_denominator = abs(quantum_ops._calculate_expectation_value(state, a))**2
            
            if abs(g1_denominator) > 1e-10:
                g1 = g1_numerator / g1_denominator
            else:
                g1 = 0.0
            
            # Second-order coherence g^(2)(0)
            g2_numerator = quantum_ops._calculate_expectation_value(state, a_dag * a_dag * a * a)
            mean_n = quantum_ops._calculate_expectation_value(state, a_dag * a)
            g2_denominator = mean_n**2
            
            if abs(g2_denominator) > 1e-10:
                g2 = g2_numerator / g2_denominator
            else:
                g2 = 0.0
            
            correlation_results.update({
                'first_order_coherence_g1': complex(g1),
                'second_order_coherence_g2': complex(g2),
                'mean_photon_number': complex(mean_n),
                'coherence_classification': {
                    'g1_coherent': abs(g1 - 1.0) < 0.1,
                    'g2_antibunched': np.real(g2) < 1.0,
                    'g2_bunched': np.real(g2) > 1.0,
                    'g2_coherent': abs(np.real(g2) - 1.0) < 0.1
                }
            })
        
        # For qubit systems, calculate Pauli correlations
        if observable.shape[0] == 2:
            pauli_ops = {
                'x': qt.sigmax(),
                'y': qt.sigmay(),
                'z': qt.sigmaz()
            }
            
            pauli_correlations = {}
            for name, pauli_op in pauli_ops.items():
                pauli_exp = quantum_ops._calculate_expectation_value(state, pauli_op)
                pauli_var = quantum_ops._calculate_variance(state, pauli_op)
                
                # Cross-correlation with main observable
                cross_corr = quantum_ops._calculate_expectation_value(state, observable * pauli_op)
                
                pauli_correlations[f'pauli_{name}'] = {
                    'expectation': complex(pauli_exp),
                    'variance': float(np.real(pauli_var)),
                    'cross_correlation': complex(cross_corr)
                }
            
            correlation_results['pauli_correlations'] = pauli_correlations
            
            # Bloch vector for qubit
            if state.type == 'oper':
                rho = state
            else:
                rho = state * state.dag()
            
            bloch_vector = [
                np.real((pauli_ops['x'] * rho).tr()),
                np.real((pauli_ops['y'] * rho).tr()),
                np.real((pauli_ops['z'] * rho).tr())
            ]
            
            correlation_results['bloch_vector'] = bloch_vector
            correlation_results['bloch_vector_length'] = float(np.linalg.norm(bloch_vector))
        
        return correlation_results
        
    except Exception as e:
        raise QuantumMeasurementError(
            f"Failed to calculate correlations: {str(e)}",
            measurement_type="correlation"
        )


def check_hermiticity(operator: qt.Qobj, tolerance: float = 1e-10) -> bool:
    """Check if operator is Hermitian."""
    try:
        hermitian_diff = operator - operator.dag()
        max_deviation = np.max(np.abs(hermitian_diff.full()))
        return max_deviation < tolerance
    except Exception:
        return False


def get_eigenvalue_range(operator: qt.Qobj) -> Dict[str, float]:
    """Get eigenvalue range of operator."""
    try:
        eigenvals = operator.eigenenergies()
        return {
            'min_eigenvalue': float(np.min(np.real(eigenvals))),
            'max_eigenvalue': float(np.max(np.real(eigenvals))),
            'eigenvalue_spread': float(np.max(np.real(eigenvals)) - np.min(np.real(eigenvals))),
            'number_of_eigenvalues': len(eigenvals)
        }
    except Exception:
        return {
            'min_eigenvalue': 0.0,
            'max_eigenvalue': 0.0,
            'eigenvalue_spread': 0.0,
            'number_of_eigenvalues': 0
        }