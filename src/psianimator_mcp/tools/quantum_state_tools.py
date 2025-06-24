"""
Quantum State Creation Tools for PsiAnimator-MCP

Implements the create_quantum_state MCP tool for creating various types
of quantum states including pure, mixed, coherent, squeezed, and Fock states.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Union, Any

import numpy as np
import qutip as qt

from ..quantum.state_manager import QuantumStateManager
from ..quantum.validation import validate_quantum_state
from ..server.config import MCPConfig
from ..server.exceptions import (
    QuantumStateError,
    ValidationError,
    DimensionError
)

logger = logging.getLogger(__name__)

# Global state manager instance
_state_manager = None


def get_state_manager(config: MCPConfig) -> QuantumStateManager:
    """Get or create the global state manager instance."""
    global _state_manager
    if _state_manager is None:
        _state_manager = QuantumStateManager(max_dimension=config.max_hilbert_dimension)
    return _state_manager


async def create_quantum_state(arguments: Dict[str, Any], config: MCPConfig) -> Dict[str, Any]:
    """
    Create a quantum state with specified parameters.
    
    This is the main MCP tool function for quantum state creation.
    Supports creation of pure states, mixed states, coherent states,
    squeezed states, thermal states, and Fock states.
    
    Parameters
    ----------
    arguments : dict
        Tool arguments containing:
        - state_type: str - Type of state ('pure', 'mixed', 'coherent', 'squeezed', 'thermal', 'fock')
        - system_dims: list - Dimensions of each subsystem
        - parameters: dict - State-specific parameters
        - basis: str - Basis for state creation ('computational', 'fock', 'spin', 'position')
        - state_id: str (optional) - Custom state identifier
    config : MCPConfig
        Server configuration
        
    Returns
    -------
    dict
        Result containing state_id, state_info, and success status
        
    Raises
    ------
    ValidationError
        If input arguments are invalid
    QuantumStateError
        If state creation fails
    """
    try:
        logger.info(f"Creating quantum state with arguments: {arguments}")
        
        # Validate required arguments
        if 'state_type' not in arguments:
            raise ValidationError("state_type is required", field="state_type")
        
        if 'system_dims' not in arguments:
            raise ValidationError("system_dims is required", field="system_dims")
        
        state_type = arguments['state_type']
        system_dims = arguments['system_dims']
        parameters = arguments.get('parameters', {})
        basis = arguments.get('basis', 'computational')
        state_id = arguments.get('state_id', None)
        
        # Validate state_type
        valid_types = ['pure', 'mixed', 'coherent', 'squeezed', 'thermal', 'fock']
        if state_type not in valid_types:
            raise ValidationError(
                f"Invalid state_type: {state_type}",
                field="state_type",
                expected_type=f"one of: {', '.join(valid_types)}"
            )
        
        # Validate system_dims
        if not isinstance(system_dims, list):
            raise ValidationError(
                "system_dims must be a list",
                field="system_dims",
                expected_type="list of integers"
            )
        
        if not all(isinstance(dim, int) and dim >= 2 for dim in system_dims):
            raise ValidationError(
                "All system dimensions must be integers >= 2",
                field="system_dims",
                expected_type="list of integers >= 2"
            )
        
        # Validate basis
        valid_bases = ['computational', 'fock', 'spin', 'position']
        if basis not in valid_bases:
            raise ValidationError(
                f"Invalid basis: {basis}",
                field="basis",
                expected_type=f"one of: {', '.join(valid_bases)}"
            )
        
        # Get state manager
        state_manager = get_state_manager(config)
        
        # Create the quantum state
        created_state_id = state_manager.create_state(
            state_type=state_type,
            system_dims=system_dims,
            parameters=parameters,
            basis=basis,
            state_id=state_id
        )
        
        # Get state information
        state_info = state_manager.get_state_info(created_state_id)
        
        # Get the actual quantum state for additional analysis
        quantum_state = state_manager.get_state(created_state_id)
        
        # Add quantum mechanical properties
        additional_props = {}
        
        if quantum_state.type == 'ket':
            # Pure state properties
            additional_props['purity'] = 1.0
            additional_props['von_neumann_entropy'] = 0.0
            
            # Calculate overlap with computational basis states
            if state_type == 'pure' and basis == 'computational':
                amplitudes = quantum_state.full().flatten()
                basis_overlaps = {}
                for i, amp in enumerate(amplitudes):
                    if abs(amp) > 1e-10:
                        basis_overlaps[f"|{i}>"] = {
                            'amplitude': complex(amp),
                            'probability': float(abs(amp)**2)
                        }
                additional_props['basis_decomposition'] = basis_overlaps
        
        elif quantum_state.type == 'oper':
            # Mixed state properties
            eigenvals = quantum_state.eigenenergies()
            purity = np.real(quantum_state.tr(quantum_state * quantum_state))
            von_neumann_entropy = -np.sum(eigenvals * np.log2(eigenvals + 1e-16))
            
            additional_props['purity'] = float(purity)
            additional_props['von_neumann_entropy'] = float(von_neumann_entropy)
            additional_props['eigenvalues'] = [float(np.real(val)) for val in eigenvals]
            
            # Check if state is separable (simplified test)
            if len(system_dims) > 1:
                # For now, just indicate if it's a product state or potentially entangled
                total_dim = np.prod(system_dims)
                if quantum_state.shape[0] == total_dim:
                    # Simplified entanglement detection based on purity
                    is_product = purity > 0.99  # Very pure states are likely product states
                    additional_props['likely_product_state'] = is_product
                    additional_props['potentially_entangled'] = not is_product
        
        # Special properties for specific state types
        if state_type == 'coherent':
            alpha = parameters.get('alpha', 1.0)
            additional_props['coherent_amplitude'] = complex(alpha)
            additional_props['mean_photon_number'] = float(abs(alpha)**2)
            
        elif state_type == 'squeezed':
            r = parameters.get('r', 1.0)
            theta = parameters.get('theta', 0.0)
            additional_props['squeezing_parameter'] = float(r)
            additional_props['squeezing_angle'] = float(theta)
            
        elif state_type == 'thermal':
            n_avg = parameters.get('n_avg', 1.0)
            additional_props['mean_photon_number'] = float(n_avg)
            
        elif state_type == 'fock':
            n = parameters.get('n', 0)
            additional_props['photon_number'] = int(n)
        
        # Update state_info with additional properties
        state_info.update(additional_props)
        
        result = {
            'success': True,
            'state_id': created_state_id,
            'state_info': state_info,
            'message': f"Successfully created {state_type} state with ID: {created_state_id}",
            'quantum_properties': {
                'hilbert_space_dimension': state_info['hilbert_dim'],
                'state_type': quantum_state.type,
                'is_normalized': state_info['norm'] > 0.99,
                'system_dimensions': system_dims,
                'basis_used': basis
            }
        }
        
        logger.info(f"Successfully created quantum state: {created_state_id}")
        return result
        
    except (ValidationError, QuantumStateError, DimensionError) as e:
        logger.error(f"Quantum state creation failed: {e}")
        return {
            'success': False,
            'error': e.__class__.__name__,
            'message': str(e),
            'details': e.details if hasattr(e, 'details') else {}
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in create_quantum_state: {e}")
        return {
            'success': False,
            'error': 'UnexpectedError',
            'message': f"An unexpected error occurred: {str(e)}",
            'details': {}
        }


async def list_quantum_states(config: MCPConfig) -> Dict[str, Any]:
    """
    List all stored quantum states.
    
    Parameters
    ----------
    config : MCPConfig
        Server configuration
        
    Returns
    -------
    dict
        List of state IDs and their basic information
    """
    try:
        state_manager = get_state_manager(config)
        state_ids = state_manager.list_states()
        
        states_info = {}
        for state_id in state_ids:
            try:
                info = state_manager.get_state_info(state_id)
                states_info[state_id] = {
                    'state_type': info.get('state_type', 'unknown'),
                    'hilbert_dim': info.get('hilbert_dim', 0),
                    'system_dims': info.get('system_dims', []),
                    'is_pure': info.get('is_pure', False),
                    'norm': info.get('norm', 0.0)
                }
            except Exception as e:
                logger.warning(f"Could not get info for state {state_id}: {e}")
                states_info[state_id] = {'error': str(e)}
        
        return {
            'success': True,
            'total_states': len(state_ids),
            'states': states_info
        }
        
    except Exception as e:
        logger.error(f"Error listing quantum states: {e}")
        return {
            'success': False,
            'error': str(e),
            'states': {}
        }


async def get_quantum_state_info(state_id: str, config: MCPConfig) -> Dict[str, Any]:
    """
    Get detailed information about a specific quantum state.
    
    Parameters
    ----------
    state_id : str
        State identifier
    config : MCPConfig
        Server configuration
        
    Returns
    -------
    dict
        Detailed state information
    """
    try:
        state_manager = get_state_manager(config)
        
        if state_id not in state_manager.list_states():
            return {
                'success': False,
                'error': 'StateNotFound',
                'message': f"State with ID '{state_id}' not found"
            }
        
        state_info = state_manager.get_state_info(state_id)
        quantum_state = state_manager.get_state(state_id)
        
        # Add detailed quantum mechanical analysis
        detailed_info = state_info.copy()
        
        # State vector or density matrix representation
        if config.max_hilbert_dimension <= 8:  # Only for small systems
            if quantum_state.type == 'ket':
                amplitudes = quantum_state.full().flatten()
                detailed_info['state_vector'] = [complex(amp) for amp in amplitudes]
            elif quantum_state.type == 'oper':
                matrix = quantum_state.full()
                detailed_info['density_matrix'] = [[complex(matrix[i, j]) for j in range(matrix.shape[1])] 
                                                 for i in range(matrix.shape[0])]
        
        return {
            'success': True,
            'state_id': state_id,
            'detailed_info': detailed_info
        }
        
    except Exception as e:
        logger.error(f"Error getting state info for {state_id}: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f"Could not retrieve information for state {state_id}"
        }


async def delete_quantum_state(state_id: str, config: MCPConfig) -> Dict[str, Any]:
    """
    Delete a quantum state from storage.
    
    Parameters
    ----------
    state_id : str
        State identifier to delete
    config : MCPConfig
        Server configuration
        
    Returns
    -------
    dict
        Deletion result
    """
    try:
        state_manager = get_state_manager(config)
        
        if state_id not in state_manager.list_states():
            return {
                'success': False,
                'error': 'StateNotFound',
                'message': f"State with ID '{state_id}' not found"
            }
        
        state_manager.delete_state(state_id)
        
        return {
            'success': True,
            'message': f"Successfully deleted state {state_id}"
        }
        
    except Exception as e:
        logger.error(f"Error deleting state {state_id}: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f"Could not delete state {state_id}"
        }