"""
Entanglement Analysis Tools for PsiAnimator-MCP

Implements the calculate_entanglement MCP tool for computing various entanglement measures
and visualizing quantum correlations in multi-particle systems.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import qutip as qt

from ..quantum.state_manager import QuantumStateManager
from ..quantum.validation import validate_quantum_state
from ..server.config import MCPConfig
from ..server.exceptions import (
    QuantumMeasurementError,
    ValidationError,
    QuantumStateError,
    DimensionError
)
from .quantum_state_tools import get_state_manager

logger = logging.getLogger(__name__)


async def calculate_entanglement(arguments: Dict[str, Any], config: MCPConfig) -> Dict[str, Any]:
    """
    Calculate various entanglement measures for quantum states.
    
    Parameters
    ----------
    arguments : dict
        Tool arguments containing:
        - state_id: str - ID of the quantum state to analyze
        - measure_type: str - Type of entanglement measure
        - subsystem_partition: list - How to partition the system
        - visualize_correlations: bool - Whether to generate correlation visualization
        - detect_entanglement: bool - Whether to perform entanglement detection
        - witnesses: list - Entanglement witnesses to evaluate
    config : MCPConfig
        Server configuration
        
    Returns
    -------
    dict
        Entanglement analysis results with measures and correlations
    """
    try:
        logger.info(f"Calculating entanglement with arguments: {arguments}")
        
        # Validate required arguments
        required_args = ['state_id', 'measure_type']
        for arg in required_args:
            if arg not in arguments:
                raise ValidationError(f"{arg} is required", field=arg)
        
        state_id = arguments['state_id']
        measure_type = arguments['measure_type']
        subsystem_partition = arguments.get('subsystem_partition', None)
        visualize_correlations = arguments.get('visualize_correlations', False)
        detect_entanglement = arguments.get('detect_entanglement', True)
        witnesses = arguments.get('witnesses', [])
        
        # Validate measure type
        valid_measures = ['von_neumann', 'linear_entropy', 'concurrence', 'negativity', 'mutual_information']
        if measure_type not in valid_measures:
            raise ValidationError(
                f"Invalid measure_type: {measure_type}",
                field="measure_type",
                expected_type=f"one of: {', '.join(valid_measures)}"
            )
        
        # Get quantum state
        state_manager = get_state_manager(config)
        
        if state_id not in state_manager.list_states():
            raise QuantumStateError(f"State with ID '{state_id}' not found")
        
        quantum_state = state_manager.get_state(state_id)
        quantum_state = validate_quantum_state(quantum_state)
        
        # Convert to density matrix if needed
        if quantum_state.type == 'ket':
            rho = quantum_state * quantum_state.dag()
        else:
            rho = quantum_state
        
        # Determine system structure
        system_info = analyze_system_structure(rho, subsystem_partition)
        
        # Calculate requested entanglement measure
        if measure_type == 'von_neumann':
            result = await calculate_von_neumann_entropy(rho, system_info)
        elif measure_type == 'linear_entropy':
            result = await calculate_linear_entropy(rho, system_info)
        elif measure_type == 'concurrence':
            result = await calculate_concurrence(rho, system_info)
        elif measure_type == 'negativity':
            result = await calculate_negativity(rho, system_info)
        elif measure_type == 'mutual_information':
            result = await calculate_mutual_information(rho, system_info)
        
        # Perform entanglement detection if requested
        if detect_entanglement:
            detection_results = await perform_entanglement_detection(rho, system_info)
            result['entanglement_detection'] = detection_results
        
        # Evaluate entanglement witnesses if provided
        if witnesses:
            witness_results = await evaluate_entanglement_witnesses(rho, witnesses)
            result['witness_evaluation'] = witness_results
        
        # Generate correlation analysis
        correlation_analysis = await analyze_quantum_correlations(rho, system_info)
        result['correlation_analysis'] = correlation_analysis
        
        # Visualization data if requested
        if visualize_correlations:
            visualization_data = generate_correlation_visualization_data(rho, system_info, result)
            result['visualization_data'] = visualization_data
        
        # Prepare final result
        entanglement_result = {
            'success': True,
            'state_id': state_id,
            'measure_type': measure_type,
            'system_info': system_info,
            'entanglement_analysis': result,
            'quantum_state_properties': {
                'type': quantum_state.type,
                'dimensions': quantum_state.dims,
                'hilbert_space_dim': quantum_state.shape[0],
                'is_pure': quantum_state.type == 'ket' or abs((rho * rho).tr() - 1.0) < 1e-10
            },
            'message': f"Successfully calculated {measure_type} entanglement measure for state {state_id}"
        }
        
        logger.info(f"Successfully calculated entanglement for state {state_id}")
        return entanglement_result
        
    except (ValidationError, QuantumMeasurementError, QuantumStateError, DimensionError) as e:
        logger.error(f"Entanglement calculation failed: {e}")
        return {
            'success': False,
            'error': e.__class__.__name__,
            'message': str(e),
            'details': e.details if hasattr(e, 'details') else {}
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in calculate_entanglement: {e}")
        return {
            'success': False,
            'error': 'UnexpectedError',
            'message': f"An unexpected error occurred: {str(e)}",
            'details': {}
        }


def analyze_system_structure(rho: qt.Qobj, subsystem_partition: Optional[List[List[int]]]) -> Dict[str, Any]:
    """
    Analyze the structure of the quantum system.
    
    Parameters
    ----------
    rho : qt.Qobj
        Density matrix of the system
    subsystem_partition : list of list of int, optional
        How to partition the system into subsystems
        
    Returns
    -------
    dict
        System structure information
    """
    try:
        total_dim = rho.shape[0]
        
        # Try to determine if this is a multi-qubit system
        if total_dim > 1 and (total_dim & (total_dim - 1)) == 0:  # Power of 2
            n_qubits = int(np.log2(total_dim))
            is_qubit_system = True
            subsystem_dims = [2] * n_qubits
        else:
            # General multi-level system
            is_qubit_system = False
            n_qubits = 0
            # Try to factorize dimension
            subsystem_dims = factorize_dimension(total_dim)
        
        # Use provided partition or default bipartition
        if subsystem_partition is None:
            if is_qubit_system and n_qubits >= 2:
                # Default: split in half
                partition_a = list(range(n_qubits // 2))
                partition_b = list(range(n_qubits // 2, n_qubits))
                subsystem_partition = [partition_a, partition_b]
            elif len(subsystem_dims) >= 2:
                # For general systems, partition by subsystem index
                partition_a = [0]
                partition_b = list(range(1, len(subsystem_dims)))
                subsystem_partition = [partition_a, partition_b]
            else:
                # Single subsystem - no entanglement possible
                subsystem_partition = [[0]]
        
        return {
            'total_dimension': total_dim,
            'is_qubit_system': is_qubit_system,
            'n_qubits': n_qubits,
            'subsystem_dimensions': subsystem_dims,
            'subsystem_partition': subsystem_partition,
            'n_partitions': len(subsystem_partition),
            'is_bipartite': len(subsystem_partition) == 2
        }
        
    except Exception as e:
        logger.warning(f"System structure analysis failed: {e}")
        return {
            'total_dimension': rho.shape[0],
            'is_qubit_system': False,
            'n_qubits': 0,
            'subsystem_dimensions': [rho.shape[0]],
            'subsystem_partition': [[0]],
            'n_partitions': 1,
            'is_bipartite': False,
            'error': str(e)
        }


def factorize_dimension(dim: int) -> List[int]:
    """
    Factorize Hilbert space dimension into subsystem dimensions.
    
    Parameters
    ----------
    dim : int
        Total Hilbert space dimension
        
    Returns
    -------
    list of int
        List of subsystem dimensions
    """
    factors = []
    d = dim
    
    # Try small primes first
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        while d % p == 0:
            factors.append(p)
            d = d // p
        if d == 1:
            break
    
    # If not fully factorized, add remaining factor
    if d > 1:
        factors.append(d)
    
    return factors if factors else [dim]


async def calculate_von_neumann_entropy(rho: qt.Qobj, system_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate von Neumann entropy for entanglement quantification.
    
    Parameters
    ----------
    rho : qt.Qobj
        Density matrix
    system_info : dict
        System structure information
        
    Returns
    -------
    dict
        Von Neumann entropy results
    """
    try:
        results = {
            'measure_type': 'von_neumann',
            'entanglement_values': {},
            'subsystem_entropies': {},
            'interpretation': {}
        }
        
        if not system_info['is_bipartite'] or len(system_info['subsystem_partition']) < 2:
            return {
                'measure_type': 'von_neumann',
                'error': 'Von Neumann entropy requires bipartite system',
                'entanglement_values': {},
                'subsystem_entropies': {}
            }
        
        partition = system_info['subsystem_partition']
        
        # Calculate reduced density matrices and their entropies
        for i, subsystem in enumerate(partition):
            try:
                rho_reduced = rho.ptrace(subsystem)
                eigenvals = rho_reduced.eigenenergies()
                
                # Calculate von Neumann entropy
                entropy = -np.sum(eigenvals * np.log2(eigenvals + 1e-16))
                
                results['subsystem_entropies'][f'subsystem_{i}'] = {
                    'entropy': float(np.real(entropy)),
                    'eigenvalues': [float(np.real(val)) for val in eigenvals],
                    'subsystem_indices': subsystem,
                    'reduced_dimension': rho_reduced.shape[0]
                }
                
            except Exception as e:
                logger.warning(f"Could not calculate entropy for subsystem {i}: {e}")
                results['subsystem_entropies'][f'subsystem_{i}'] = {'error': str(e)}
        
        # For bipartite systems, entanglement entropy is the entropy of either subsystem
        if len(results['subsystem_entropies']) >= 2:
            entropies = [info.get('entropy', 0) for info in results['subsystem_entropies'].values() 
                        if 'entropy' in info]
            
            if entropies:
                entanglement_entropy = entropies[0]  # Should be same for both subsystems in pure states
                results['entanglement_values']['entanglement_entropy'] = entanglement_entropy
                
                # Interpretation
                if entanglement_entropy < 0.01:
                    interpretation = "Separable (no entanglement)"
                elif entanglement_entropy < 1.0:
                    interpretation = "Weakly entangled"
                elif entanglement_entropy < 2.0:
                    interpretation = "Moderately entangled"
                else:
                    interpretation = "Highly entangled"
                
                results['interpretation']['entanglement_level'] = interpretation
                results['interpretation']['max_possible_entropy'] = np.log2(min(
                    results['subsystem_entropies']['subsystem_0'].get('reduced_dimension', 1),
                    results['subsystem_entropies']['subsystem_1'].get('reduced_dimension', 1)
                ))
        
        return results
        
    except Exception as e:
        raise QuantumMeasurementError(
            f"Failed to calculate von Neumann entropy: {str(e)}",
            measurement_type="von_neumann"
        )


async def calculate_linear_entropy(rho: qt.Qobj, system_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate linear entropy (1 - Tr(ρ²)) for entanglement quantification.
    
    Parameters
    ----------
    rho : qt.Qobj
        Density matrix
    system_info : dict
        System structure information
        
    Returns
    -------
    dict
        Linear entropy results
    """
    try:
        results = {
            'measure_type': 'linear_entropy',
            'entanglement_values': {},
            'subsystem_entropies': {},
            'interpretation': {}
        }
        
        if not system_info['is_bipartite'] or len(system_info['subsystem_partition']) < 2:
            # Calculate global linear entropy
            purity = np.real((rho * rho).tr())
            linear_entropy = 1 - purity
            
            results['entanglement_values']['global_linear_entropy'] = float(linear_entropy)
            results['interpretation']['purity'] = float(purity)
            
            return results
        
        partition = system_info['subsystem_partition']
        
        # Calculate linear entropy for each subsystem
        for i, subsystem in enumerate(partition):
            try:
                rho_reduced = rho.ptrace(subsystem)
                purity = np.real((rho_reduced * rho_reduced).tr())
                linear_entropy = 1 - purity
                
                results['subsystem_entropies'][f'subsystem_{i}'] = {
                    'linear_entropy': float(linear_entropy),
                    'purity': float(purity),
                    'subsystem_indices': subsystem,
                    'reduced_dimension': rho_reduced.shape[0]
                }
                
            except Exception as e:
                logger.warning(f"Could not calculate linear entropy for subsystem {i}: {e}")
                results['subsystem_entropies'][f'subsystem_{i}'] = {'error': str(e)}
        
        # Entanglement measure based on average subsystem linear entropy
        if len(results['subsystem_entropies']) >= 2:
            linear_entropies = [info.get('linear_entropy', 0) for info in results['subsystem_entropies'].values() 
                              if 'linear_entropy' in info]
            
            if linear_entropies:
                avg_linear_entropy = np.mean(linear_entropies)
                results['entanglement_values']['average_subsystem_linear_entropy'] = float(avg_linear_entropy)
                
                # Interpretation
                if avg_linear_entropy < 0.01:
                    interpretation = "Separable (no entanglement)"
                elif avg_linear_entropy < 0.3:
                    interpretation = "Weakly entangled"
                elif avg_linear_entropy < 0.7:
                    interpretation = "Moderately entangled"
                else:
                    interpretation = "Highly entangled"
                
                results['interpretation']['entanglement_level'] = interpretation
        
        return results
        
    except Exception as e:
        raise QuantumMeasurementError(
            f"Failed to calculate linear entropy: {str(e)}",
            measurement_type="linear_entropy"
        )


async def calculate_concurrence(rho: qt.Qobj, system_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate concurrence for two-qubit entanglement quantification.
    
    Parameters
    ----------
    rho : qt.Qobj
        Density matrix
    system_info : dict
        System structure information
        
    Returns
    -------
    dict
        Concurrence results
    """
    try:
        results = {
            'measure_type': 'concurrence',
            'entanglement_values': {},
            'interpretation': {}
        }
        
        # Concurrence is defined for two-qubit systems
        if not (system_info['is_qubit_system'] and system_info['n_qubits'] == 2):
            return {
                'measure_type': 'concurrence',
                'error': 'Concurrence is only defined for two-qubit systems',
                'entanglement_values': {}
            }
        
        # Calculate concurrence using Wootters formula
        # C(ρ) = max(0, λ₁ - λ₂ - λ₃ - λ₄)
        # where λᵢ are square roots of eigenvalues of ρ * ρ̃ in decreasing order
        # ρ̃ = (σy ⊗ σy) ρ* (σy ⊗ σy)
        
        sigma_y = qt.sigmay()
        flip_op = qt.tensor(sigma_y, sigma_y)
        
        # Complex conjugate of density matrix
        rho_conj = qt.Qobj(np.conj(rho.full()))
        
        # Flipped density matrix
        rho_tilde = flip_op * rho_conj * flip_op
        
        # Calculate eigenvalues of ρ * ρ̃
        R = rho * rho_tilde
        eigenvals = R.eigenenergies()
        
        # Take square roots and sort in decreasing order
        sqrt_eigenvals = np.sqrt(np.maximum(0, np.real(eigenvals)))
        sqrt_eigenvals = np.sort(sqrt_eigenvals)[::-1]
        
        # Calculate concurrence
        if len(sqrt_eigenvals) >= 4:
            concurrence = max(0, sqrt_eigenvals[0] - sqrt_eigenvals[1] - sqrt_eigenvals[2] - sqrt_eigenvals[3])
        else:
            concurrence = 0
        
        results['entanglement_values']['concurrence'] = float(concurrence)
        results['entanglement_values']['sqrt_eigenvalues'] = [float(val) for val in sqrt_eigenvals]
        
        # Calculate entanglement of formation from concurrence
        if concurrence > 0:
            x = (1 + np.sqrt(1 - concurrence**2)) / 2
            if x > 0:
                entanglement_of_formation = -x * np.log2(x) - (1-x) * np.log2(1-x + 1e-16)
            else:
                entanglement_of_formation = 0
        else:
            entanglement_of_formation = 0
        
        results['entanglement_values']['entanglement_of_formation'] = float(entanglement_of_formation)
        
        # Interpretation
        if concurrence < 0.01:
            interpretation = "Separable (no entanglement)"
        elif concurrence < 0.3:
            interpretation = "Weakly entangled"
        elif concurrence < 0.7:
            interpretation = "Moderately entangled"
        else:
            interpretation = "Highly entangled"
        
        results['interpretation']['entanglement_level'] = interpretation
        results['interpretation']['max_concurrence'] = 1.0
        
        return results
        
    except Exception as e:
        raise QuantumMeasurementError(
            f"Failed to calculate concurrence: {str(e)}",
            measurement_type="concurrence"
        )


async def calculate_negativity(rho: qt.Qobj, system_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate negativity for entanglement quantification.
    
    Parameters
    ----------
    rho : qt.Qobj
        Density matrix
    system_info : dict
        System structure information
        
    Returns
    -------
    dict
        Negativity results
    """
    try:
        results = {
            'measure_type': 'negativity',
            'entanglement_values': {},
            'interpretation': {}
        }
        
        if not system_info['is_bipartite'] or len(system_info['subsystem_partition']) < 2:
            return {
                'measure_type': 'negativity',
                'error': 'Negativity requires bipartite system',
                'entanglement_values': {}
            }
        
        partition = system_info['subsystem_partition']
        
        # Calculate partial transpose and its eigenvalues
        # For simplicity, assume we're doing partial transpose with respect to first subsystem
        try:
            rho_pt = rho.ptrace(partition[0]).dag() * rho.ptrace(partition[1])  # Simplified - needs proper partial transpose
            
            # Actual partial transpose calculation for qubits
            if system_info['is_qubit_system']:
                # For 2-qubit system, partial transpose is straightforward
                if system_info['n_qubits'] == 2:
                    rho_matrix = rho.full()
                    # Partial transpose with respect to second qubit
                    rho_pt_matrix = np.array([
                        [rho_matrix[0,0], rho_matrix[0,2], rho_matrix[2,0], rho_matrix[2,2]],
                        [rho_matrix[0,1], rho_matrix[0,3], rho_matrix[2,1], rho_matrix[2,3]],
                        [rho_matrix[1,0], rho_matrix[1,2], rho_matrix[3,0], rho_matrix[3,2]], 
                        [rho_matrix[1,1], rho_matrix[1,3], rho_matrix[3,1], rho_matrix[3,3]]
                    ])
                    
                    rho_pt = qt.Qobj(rho_pt_matrix)
                    eigenvals = rho_pt.eigenenergies()
                    
                    # Negativity is sum of absolute values of negative eigenvalues
                    negative_eigenvals = eigenvals[eigenvals < 0]
                    negativity = np.sum(np.abs(negative_eigenvals))
                    
                    # Logarithmic negativity
                    log_negativity = np.log2(2 * negativity + 1) if negativity > 0 else 0
                    
                    results['entanglement_values']['negativity'] = float(negativity)
                    results['entanglement_values']['logarithmic_negativity'] = float(log_negativity)
                    results['entanglement_values']['eigenvalues_pt'] = [float(np.real(val)) for val in eigenvals]
                    results['entanglement_values']['negative_eigenvalues'] = [float(val) for val in negative_eigenvals]
                    
                    # Interpretation
                    if negativity < 0.01:
                        interpretation = "Separable (no entanglement)"
                    elif negativity < 0.1:
                        interpretation = "Weakly entangled"
                    elif negativity < 0.3:
                        interpretation = "Moderately entangled"
                    else:
                        interpretation = "Highly entangled"
                    
                    results['interpretation']['entanglement_level'] = interpretation
                    results['interpretation']['ppt_criterion'] = negativity > 0.01  # PPT = Positive Partial Transpose
                
                else:
                    results['error'] = f"Partial transpose not implemented for {system_info['n_qubits']}-qubit systems"
            else:
                results['error'] = "Partial transpose not implemented for general multi-level systems"
        
        except Exception as e:
            results['error'] = f"Partial transpose calculation failed: {str(e)}"
        
        return results
        
    except Exception as e:
        raise QuantumMeasurementError(
            f"Failed to calculate negativity: {str(e)}",
            measurement_type="negativity"
        )


async def calculate_mutual_information(rho: qt.Qobj, system_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate quantum mutual information.
    
    Parameters
    ----------
    rho : qt.Qobj
        Density matrix
    system_info : dict
        System structure information
        
    Returns
    -------
    dict
        Mutual information results
    """
    try:
        results = {
            'measure_type': 'mutual_information',
            'entanglement_values': {},
            'subsystem_entropies': {},
            'interpretation': {}
        }
        
        if not system_info['is_bipartite'] or len(system_info['subsystem_partition']) < 2:
            return {
                'measure_type': 'mutual_information',
                'error': 'Mutual information requires bipartite system',
                'entanglement_values': {}
            }
        
        partition = system_info['subsystem_partition']
        
        # Calculate entropies: S(A), S(B), S(AB)
        # Mutual information I(A:B) = S(A) + S(B) - S(AB)
        
        # Total system entropy
        eigenvals_total = rho.eigenenergies()
        S_total = -np.sum(eigenvals_total * np.log2(eigenvals_total + 1e-16))
        
        subsystem_entropies = []
        
        # Calculate subsystem entropies
        for i, subsystem in enumerate(partition):
            try:
                rho_reduced = rho.ptrace(subsystem)
                eigenvals = rho_reduced.eigenenergies()
                entropy = -np.sum(eigenvals * np.log2(eigenvals + 1e-16))
                
                subsystem_entropies.append(entropy)
                results['subsystem_entropies'][f'subsystem_{i}'] = {
                    'entropy': float(np.real(entropy)),
                    'eigenvalues': [float(np.real(val)) for val in eigenvals],
                    'subsystem_indices': subsystem
                }
                
            except Exception as e:
                logger.warning(f"Could not calculate entropy for subsystem {i}: {e}")
                subsystem_entropies.append(0)
                results['subsystem_entropies'][f'subsystem_{i}'] = {'error': str(e)}
        
        # Calculate mutual information
        if len(subsystem_entropies) >= 2:
            S_A, S_B = subsystem_entropies[0], subsystem_entropies[1]
            mutual_information = S_A + S_B - S_total
            
            results['entanglement_values']['mutual_information'] = float(np.real(mutual_information))
            results['entanglement_values']['total_entropy'] = float(np.real(S_total))
            results['entanglement_values']['subsystem_A_entropy'] = float(np.real(S_A))
            results['entanglement_values']['subsystem_B_entropy'] = float(np.real(S_B))
            
            # Interpretation
            if mutual_information < 0.01:
                interpretation = "No quantum correlations"
            elif mutual_information < 0.5:
                interpretation = "Weak quantum correlations"
            elif mutual_information < 1.5:
                interpretation = "Moderate quantum correlations"
            else:
                interpretation = "Strong quantum correlations"
            
            results['interpretation']['correlation_level'] = interpretation
            results['interpretation']['max_possible_mutual_information'] = float(np.real(min(S_A, S_B)))
        
        return results
        
    except Exception as e:
        raise QuantumMeasurementError(
            f"Failed to calculate mutual information: {str(e)}",
            measurement_type="mutual_information"
        )


async def perform_entanglement_detection(rho: qt.Qobj, system_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform various entanglement detection tests.
    
    Parameters
    ----------
    rho : qt.Qobj
        Density matrix
    system_info : dict
        System structure information
        
    Returns
    -------
    dict
        Entanglement detection results
    """
    try:
        detection_results = {
            'tests_performed': [],
            'entanglement_detected': False,
            'separability_tests': {}
        }
        
        # PPT (Positive Partial Transpose) test for 2-qubit systems
        if system_info['is_qubit_system'] and system_info['n_qubits'] == 2:
            try:
                # Simple PPT test
                rho_matrix = rho.full()
                rho_pt_matrix = np.array([
                    [rho_matrix[0,0], rho_matrix[0,2], rho_matrix[2,0], rho_matrix[2,2]],
                    [rho_matrix[0,1], rho_matrix[0,3], rho_matrix[2,1], rho_matrix[2,3]],
                    [rho_matrix[1,0], rho_matrix[1,2], rho_matrix[3,0], rho_matrix[3,2]], 
                    [rho_matrix[1,1], rho_matrix[1,3], rho_matrix[3,1], rho_matrix[3,3]]
                ])
                
                eigenvals_pt = np.linalg.eigvals(rho_pt_matrix)
                has_negative_eigenvals = np.any(np.real(eigenvals_pt) < -1e-10)
                
                detection_results['separability_tests']['ppt_test'] = {
                    'test_name': 'Positive Partial Transpose',
                    'entanglement_detected': has_negative_eigenvals,
                    'min_eigenvalue': float(np.min(np.real(eigenvals_pt))),
                    'interpretation': 'Entangled' if has_negative_eigenvals else 'PPT (possibly separable)'
                }
                
                detection_results['tests_performed'].append('ppt_test')
                if has_negative_eigenvals:
                    detection_results['entanglement_detected'] = True
                    
            except Exception as e:
                detection_results['separability_tests']['ppt_test'] = {'error': str(e)}
        
        # Realignment criterion (for comparison)
        try:
            # Simplified realignment test
            # This would require more sophisticated implementation for full accuracy
            purity = np.real((rho * rho).tr())
            
            detection_results['separability_tests']['purity_test'] = {
                'test_name': 'Purity Check',
                'purity': float(purity),
                'interpretation': 'Pure state' if purity > 0.99 else 'Mixed state'
            }
            
            detection_results['tests_performed'].append('purity_test')
            
        except Exception as e:
            detection_results['separability_tests']['purity_test'] = {'error': str(e)}
        
        # Schmidt decomposition for pure states
        if abs((rho * rho).tr() - 1.0) < 1e-10:  # Pure state
            try:
                # For pure bipartite states, entanglement can be detected via Schmidt rank
                if system_info['is_bipartite'] and len(system_info['subsystem_partition']) == 2:
                    partition_a = system_info['subsystem_partition'][0]
                    rho_a = rho.ptrace(partition_a)
                    eigenvals_a = rho_a.eigenenergies()
                    
                    # Schmidt rank is number of non-zero eigenvalues
                    schmidt_rank = np.sum(eigenvals_a > 1e-10)
                    
                    detection_results['separability_tests']['schmidt_decomposition'] = {
                        'test_name': 'Schmidt Decomposition',
                        'schmidt_rank': int(schmidt_rank),
                        'entanglement_detected': schmidt_rank > 1,
                        'interpretation': 'Entangled' if schmidt_rank > 1 else 'Separable (product state)'
                    }
                    
                    detection_results['tests_performed'].append('schmidt_decomposition')
                    if schmidt_rank > 1:
                        detection_results['entanglement_detected'] = True
                        
            except Exception as e:
                detection_results['separability_tests']['schmidt_decomposition'] = {'error': str(e)}
        
        return detection_results
        
    except Exception as e:
        logger.warning(f"Entanglement detection failed: {e}")
        return {'error': str(e)}


async def evaluate_entanglement_witnesses(rho: qt.Qobj, witnesses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate entanglement witnesses on the quantum state.
    
    Parameters
    ----------
    rho : qt.Qobj
        Density matrix
    witnesses : list
        List of witness specifications
        
    Returns
    -------
    dict
        Witness evaluation results
    """
    try:
        witness_results = {
            'witnesses_evaluated': [],
            'entanglement_detected': False,
            'witness_values': {}
        }
        
        for i, witness_spec in enumerate(witnesses):
            try:
                witness_name = witness_spec.get('name', f'witness_{i}')
                witness_matrix = witness_spec.get('matrix', None)
                
                if witness_matrix is None:
                    witness_results['witness_values'][witness_name] = {
                        'error': 'No witness matrix provided'
                    }
                    continue
                
                # Create witness operator
                W = qt.Qobj(np.array(witness_matrix, dtype=complex))
                
                # Calculate expectation value
                witness_value = np.real((W * rho).tr())
                
                # Witness detects entanglement if expectation value is negative
                entanglement_detected = witness_value < 0
                
                witness_results['witness_values'][witness_name] = {
                    'expectation_value': float(witness_value),
                    'entanglement_detected': entanglement_detected,
                    'interpretation': 'Entangled' if entanglement_detected else 'Witness inconclusive'
                }
                
                witness_results['witnesses_evaluated'].append(witness_name)
                
                if entanglement_detected:
                    witness_results['entanglement_detected'] = True
                    
            except Exception as e:
                witness_results['witness_values'][f'witness_{i}'] = {'error': str(e)}
        
        return witness_results
        
    except Exception as e:
        logger.warning(f"Witness evaluation failed: {e}")
        return {'error': str(e)}


async def analyze_quantum_correlations(rho: qt.Qobj, system_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze various types of quantum correlations.
    
    Parameters
    ----------
    rho : qt.Qobj
        Density matrix
    system_info : dict
        System structure information
        
    Returns
    -------
    dict
        Correlation analysis results
    """
    try:
        correlation_analysis = {
            'correlation_types': {},
            'classical_correlations': {},
            'quantum_correlations': {}
        }
        
        if system_info['is_bipartite'] and len(system_info['subsystem_partition']) == 2:
            # For bipartite systems, analyze different correlation measures
            
            # Total correlations (mutual information)
            partition = system_info['subsystem_partition']
            
            # Calculate subsystem entropies
            S_A = 0
            S_B = 0
            
            try:
                rho_A = rho.ptrace(partition[0])
                eigenvals_A = rho_A.eigenenergies()
                S_A = -np.sum(eigenvals_A * np.log2(eigenvals_A + 1e-16))
                
                rho_B = rho.ptrace(partition[1])
                eigenvals_B = rho_B.eigenenergies()
                S_B = -np.sum(eigenvals_B * np.log2(eigenvals_B + 1e-16))
                
                # Total system entropy
                eigenvals_total = rho.eigenenergies()
                S_total = -np.sum(eigenvals_total * np.log2(eigenvals_total + 1e-16))
                
                # Total correlations (mutual information)
                total_correlations = S_A + S_B - S_total
                
                correlation_analysis['correlation_types']['total_correlations'] = {
                    'mutual_information': float(np.real(total_correlations)),
                    'interpretation': 'Measures all correlations (classical + quantum)'
                }
                
                # Quantum discord (simplified estimation)
                # This is a simplified calculation - full quantum discord requires optimization
                discord_estimate = S_A - S_total  # Simplified approximation
                
                correlation_analysis['quantum_correlations']['discord_estimate'] = {
                    'value': float(np.real(discord_estimate)),
                    'interpretation': 'Approximate quantum discord (requires optimization for exact value)'
                }
                
            except Exception as e:
                correlation_analysis['error'] = f"Correlation calculation failed: {str(e)}"
        
        return correlation_analysis
        
    except Exception as e:
        logger.warning(f"Correlation analysis failed: {e}")
        return {'error': str(e)}


def generate_correlation_visualization_data(rho: qt.Qobj, system_info: Dict[str, Any], 
                                          results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate data for correlation visualization.
    
    Parameters
    ----------
    rho : qt.Qobj
        Density matrix
    system_info : dict
        System structure information
    results : dict
        Entanglement analysis results
        
    Returns
    -------
    dict
        Visualization data
    """
    try:
        viz_data = {
            'visualization_type': 'correlation_matrix',
            'data': {}
        }
        
        if system_info['is_qubit_system'] and system_info['n_qubits'] <= 3:
            # For small qubit systems, create correlation matrix visualization
            
            # Calculate Pauli correlation matrix
            pauli_ops = [qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
            pauli_names = ['I', 'X', 'Y', 'Z']
            
            n_qubits = system_info['n_qubits']
            
            if n_qubits == 2:
                correlations = {}
                
                for i, (op1, name1) in enumerate(zip(pauli_ops, pauli_names)):
                    for j, (op2, name2) in enumerate(zip(pauli_ops, pauli_names)):
                        if i == 0 and j == 0:  # Skip identity-identity
                            continue
                        
                        # Create tensor product operator
                        pauli_tensor = qt.tensor(op1, op2)
                        
                        # Calculate expectation value
                        correlation = np.real((pauli_tensor * rho).tr())
                        
                        correlations[f"{name1}{name2}"] = float(correlation)
                
                viz_data['data']['pauli_correlations'] = correlations
                viz_data['visualization_type'] = 'pauli_correlation_matrix'
        
        # Add entanglement measure visualization
        if 'entanglement_values' in results:
            viz_data['data']['entanglement_measures'] = results['entanglement_values']
        
        return viz_data
        
    except Exception as e:
        logger.warning(f"Visualization data generation failed: {e}")
        return {'error': str(e)}