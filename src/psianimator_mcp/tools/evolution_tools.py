"""
Quantum Evolution Tools for PsiAnimator-MCP

Implements the evolve_quantum_system MCP tool for time evolution of quantum systems
using Schrödinger equation, master equation, and stochastic methods.
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
from ..quantum.systems import QuantumSystems
from ..quantum.validation import validate_quantum_state, validate_hermitian
from ..server.config import MCPConfig
from ..server.exceptions import (
    QuantumStateError,
    QuantumEvolutionError,
    ValidationError,
    DimensionError
)
from .quantum_state_tools import get_state_manager

logger = logging.getLogger(__name__)

# Global instances
_quantum_operations = None
_quantum_systems = None


def get_quantum_operations() -> QuantumOperations:
    """Get or create the global quantum operations instance."""
    global _quantum_operations
    if _quantum_operations is None:
        _quantum_operations = QuantumOperations()
    return _quantum_operations


def get_quantum_systems(config: MCPConfig) -> QuantumSystems:
    """Get or create the global quantum systems instance."""
    global _quantum_systems
    if _quantum_systems is None:
        _quantum_systems = QuantumSystems(max_dimension=config.max_hilbert_dimension)
    return _quantum_systems


def parse_hamiltonian(hamiltonian_spec: str, state_dims: List[List[int]], config: MCPConfig) -> qt.Qobj:
    """
    Parse Hamiltonian specification string into QuTip operator.
    
    Parameters
    ----------
    hamiltonian_spec : str
        Hamiltonian specification (LaTeX, system name, or matrix)
    state_dims : list
        Dimensions of the quantum system
    config : MCPConfig
        Server configuration
        
    Returns
    -------
    qt.Qobj
        Hamiltonian operator
    """
    try:
        quantum_systems = get_quantum_systems(config)
        
        # Check if it's a known system type
        if hamiltonian_spec.lower() in ['harmonic_oscillator', 'ho']:
            if len(state_dims[0]) == 1:
                dim = state_dims[0][0]
                operators = quantum_systems.create_harmonic_oscillator(dim)
                return operators['Hamiltonian']
            else:
                raise ValidationError("Harmonic oscillator requires single subsystem")
        
        elif hamiltonian_spec.lower() in ['spin', 'spin_1/2']:
            if len(state_dims[0]) == 1 and state_dims[0][0] == 2:
                operators = quantum_systems.create_spin_system(0.5)
                return operators['Hamiltonian']
            else:
                raise ValidationError("Spin-1/2 system requires 2-dimensional Hilbert space")
        
        elif hamiltonian_spec.lower() in ['jaynes_cummings', 'jc']:
            if len(state_dims[0]) == 2:
                N_photons, N_atom = state_dims[0]
                if N_atom == 2:  # Two-level atom
                    operators = quantum_systems.create_jaynes_cummings_model(N_photons)
                    return operators['Hamiltonian']
            raise ValidationError("Jaynes-Cummings model requires [N_photons, 2] dimensions")
        
        elif hamiltonian_spec.lower() in ['rabi_model', 'rabi']:
            if len(state_dims[0]) == 2:
                N_photons, N_atom = state_dims[0]
                if N_atom == 2:
                    operators = quantum_systems.create_rabi_model(N_photons)
                    return operators['Hamiltonian']
            raise ValidationError("Rabi model requires [N_photons, 2] dimensions")
        
        # Try to parse as matrix specification
        elif hamiltonian_spec.startswith('[') and hamiltonian_spec.endswith(']'):
            try:
                # Parse as JSON matrix
                matrix_data = json.loads(hamiltonian_spec)
                H = qt.Qobj(np.array(matrix_data, dtype=complex))
                return validate_hermitian(H)
            except (json.JSONDecodeError, ValueError) as e:
                raise ValidationError(f"Invalid matrix format: {e}")
        
        # Try to parse as operator expression
        elif '+' in hamiltonian_spec or '*' in hamiltonian_spec:
            return parse_operator_expression(hamiltonian_spec, state_dims)
        
        else:
            raise ValidationError(f"Unknown Hamiltonian specification: {hamiltonian_spec}")
            
    except Exception as e:
        raise QuantumEvolutionError(
            f"Failed to parse Hamiltonian: {str(e)}",
            evolution_type="hamiltonian_parsing"
        )


def parse_operator_expression(expr: str, state_dims: List[List[int]]) -> qt.Qobj:
    """
    Parse operator expression like 'omega*num + g*(destroy + create)'.
    
    Parameters
    ----------
    expr : str
        Operator expression
    state_dims : list
        System dimensions
        
    Returns
    -------
    qt.Qobj
        Parsed operator
    """
    try:
        # This is a simplified parser for common operators
        if len(state_dims[0]) == 1:
            dim = state_dims[0][0]
            
            # Create common operators
            operators = {
                'num': qt.num(dim),
                'destroy': qt.destroy(dim),
                'create': qt.create(dim),
                'identity': qt.qeye(dim),
                'x': (qt.destroy(dim) + qt.create(dim)) / np.sqrt(2),
                'p': 1j * (qt.create(dim) - qt.destroy(dim)) / np.sqrt(2)
            }
            
            # For qubits, add Pauli operators
            if dim == 2:
                operators.update({
                    'sigmax': qt.sigmax(),
                    'sigmay': qt.sigmay(),
                    'sigmaz': qt.sigmaz(),
                    'sigmap': qt.sigmap(),
                    'sigmam': qt.sigmam()
                })
            
            # Simple expression evaluation (very basic)
            # Replace operator names with variables
            eval_expr = expr
            for name, op in operators.items():
                eval_expr = eval_expr.replace(name, f'operators["{name}"]')
            
            # Define constants
            constants = {
                'omega': 1.0,
                'g': 0.1,
                'gamma': 0.01,
                'np': np,
                'operators': operators
            }
            
            # Evaluate the expression
            result = eval(eval_expr, constants)
            
            if isinstance(result, qt.Qobj):
                return result
            else:
                # If result is a scalar, multiply by identity
                return result * qt.qeye(dim)
        
        else:
            raise ValidationError("Complex operator expressions not yet supported for composite systems")
            
    except Exception as e:
        raise ValidationError(f"Failed to parse operator expression '{expr}': {str(e)}")


def parse_collapse_operators(collapse_specs: List[str], state_dims: List[List[int]]) -> List[qt.Qobj]:
    """
    Parse collapse operator specifications.
    
    Parameters
    ----------
    collapse_specs : list of str
        List of collapse operator specifications
    state_dims : list
        System dimensions
        
    Returns
    -------
    list of qt.Qobj
        Collapse operators
    """
    collapse_ops = []
    
    for spec in collapse_specs:
        try:
            if len(state_dims[0]) == 1:
                dim = state_dims[0][0]
                
                if spec.lower() in ['destroy', 'a']:
                    collapse_ops.append(qt.destroy(dim))
                elif spec.lower() in ['sigmam', 'sigma_minus']:
                    if dim == 2:
                        collapse_ops.append(qt.sigmam())
                    else:
                        raise ValidationError(f"sigma_minus only valid for qubits, got dimension {dim}")
                elif spec.lower() in ['dephasing', 'sigmaz']:
                    if dim == 2:
                        collapse_ops.append(qt.sigmaz())
                    else:
                        raise ValidationError(f"sigmaz only valid for qubits, got dimension {dim}")
                else:
                    # Try to parse as operator expression
                    collapse_ops.append(parse_operator_expression(spec, state_dims))
            else:
                raise ValidationError("Collapse operators for composite systems not yet implemented")
                
        except Exception as e:
            logger.warning(f"Could not parse collapse operator '{spec}': {e}")
    
    return collapse_ops


async def evolve_quantum_system(arguments: Dict[str, Any], config: MCPConfig) -> Dict[str, Any]:
    """
    Evolve a quantum system in time using various methods.
    
    Parameters
    ----------
    arguments : dict
        Tool arguments containing:
        - state_id: str - ID of the initial state
        - hamiltonian: str - Hamiltonian specification
        - evolution_type: str - Type of evolution ('unitary', 'master', 'monte_carlo', 'stochastic')
        - time_span: list - [t_start, t_end] or [t_start, t_end, n_steps]
        - collapse_operators: list (optional) - Collapse operators for open system evolution
        - solver_options: dict (optional) - Solver-specific options
    config : MCPConfig
        Server configuration
        
    Returns
    -------
    dict
        Evolution results with time points and evolved states
    """
    try:
        logger.info(f"Evolving quantum system with arguments: {arguments}")
        
        # Validate required arguments
        required_args = ['state_id', 'hamiltonian', 'evolution_type', 'time_span']
        for arg in required_args:
            if arg not in arguments:
                raise ValidationError(f"{arg} is required", field=arg)
        
        state_id = arguments['state_id']
        hamiltonian_spec = arguments['hamiltonian']
        evolution_type = arguments['evolution_type']
        time_span = arguments['time_span']
        collapse_operators_specs = arguments.get('collapse_operators', [])
        solver_options = arguments.get('solver_options', {})
        
        # Validate evolution type
        valid_types = ['unitary', 'master', 'monte_carlo', 'stochastic']
        if evolution_type not in valid_types:
            raise ValidationError(
                f"Invalid evolution_type: {evolution_type}",
                field="evolution_type",
                expected_type=f"one of: {', '.join(valid_types)}"
            )
        
        # Validate time span
        if not isinstance(time_span, list) or len(time_span) < 2:
            raise ValidationError(
                "time_span must be a list with at least 2 elements [t_start, t_end]",
                field="time_span"
            )
        
        t_start, t_end = time_span[0], time_span[1]
        n_steps = time_span[2] if len(time_span) > 2 else 100
        
        if t_end <= t_start:
            raise ValidationError("t_end must be greater than t_start", field="time_span")
        
        if n_steps < 2:
            raise ValidationError("Number of time steps must be at least 2", field="time_span")
        
        # Get initial state
        state_manager = get_state_manager(config)
        
        if state_id not in state_manager.list_states():
            raise QuantumStateError(f"State with ID '{state_id}' not found")
        
        initial_state = state_manager.get_state(state_id)
        initial_state = validate_quantum_state(initial_state)
        
        # Parse Hamiltonian
        hamiltonian = parse_hamiltonian(hamiltonian_spec, initial_state.dims, config)
        
        # Validate Hamiltonian dimensions
        if hamiltonian.shape != initial_state.shape:
            raise DimensionError(
                f"Hamiltonian dimensions {hamiltonian.shape} do not match state dimensions {initial_state.shape}"
            )
        
        # Create time array
        times = np.linspace(t_start, t_end, n_steps)
        
        # Perform evolution based on type
        if evolution_type == 'unitary':
            result = await evolve_unitary(initial_state, hamiltonian, times, solver_options)
            
        elif evolution_type == 'master':
            # Parse collapse operators
            collapse_ops = parse_collapse_operators(collapse_operators_specs, initial_state.dims)
            result = await evolve_master_equation(initial_state, hamiltonian, collapse_ops, times, solver_options)
            
        elif evolution_type == 'monte_carlo':
            # Parse collapse operators
            collapse_ops = parse_collapse_operators(collapse_operators_specs, initial_state.dims)
            n_trajectories = solver_options.get('n_trajectories', 100)
            result = await evolve_monte_carlo(initial_state, hamiltonian, collapse_ops, times, n_trajectories, solver_options)
            
        elif evolution_type == 'stochastic':
            # Stochastic evolution (simplified implementation)
            result = await evolve_stochastic(initial_state, hamiltonian, times, solver_options)
        
        # Store evolved states if requested
        store_states = solver_options.get('store_evolved_states', False)
        stored_state_ids = []
        evolution_id = None
        
        if store_states:
            evolution_id = f"evolution_{str(uuid.uuid4())[:8]}"
            
            # Store complete evolution result
            evolution_data = {
                'states': result['states'],
                'times': times.tolist(),
                'expectation_values': result.get('expectation_values', {}),
                'solver_info': result.get('solver_info', {})
            }
            
            evolution_parameters = {
                'hamiltonian_spec': hamiltonian_spec,
                'evolution_type': evolution_type,
                'time_span': [t_start, t_end],
                'n_time_steps': n_steps,
                'solver_options': solver_options
            }
            
            if evolution_type in ['master', 'monte_carlo']:
                evolution_parameters['collapse_operators'] = collapse_operators_specs
            if evolution_type == 'monte_carlo':
                evolution_parameters['n_trajectories'] = n_trajectories
            
            try:
                state_manager.store_evolution_result(
                    evolution_id,
                    evolution_data,
                    state_id,
                    evolution_type,
                    evolution_parameters
                )
                
                # The state manager automatically creates state IDs for each time step
                stored_state_ids = [f"{evolution_id}_t{i}" for i in range(len(result['states']))]
                
            except Exception as e:
                logger.warning(f"Could not store evolution result: {e}")
                evolution_id = None
        
        # Prepare result
        evolution_result = {
            'success': True,
            'initial_state_id': state_id,
            'evolution_type': evolution_type,
            'hamiltonian_spec': hamiltonian_spec,
            'time_span': [t_start, t_end],
            'n_time_steps': n_steps,
            'times': times.tolist(),
            'evolution_data': result,
            'stored_state_ids': stored_state_ids if store_states else [],
            'evolution_id': evolution_id,
            'message': f"Successfully evolved quantum system using {evolution_type} evolution"
        }
        
        # Add evolution-specific information
        if evolution_type in ['master', 'monte_carlo']:
            evolution_result['n_collapse_operators'] = len(collapse_ops)
            evolution_result['collapse_operators_specs'] = collapse_operators_specs
        
        if evolution_type == 'monte_carlo':
            evolution_result['n_trajectories'] = n_trajectories
        
        logger.info(f"Successfully evolved quantum system: {state_id}")
        return evolution_result
        
    except (ValidationError, QuantumStateError, QuantumEvolutionError, DimensionError) as e:
        logger.error(f"Quantum evolution failed: {e}")
        return {
            'success': False,
            'error': e.__class__.__name__,
            'message': str(e),
            'details': e.details if hasattr(e, 'details') else {}
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in evolve_quantum_system: {e}")
        return {
            'success': False,
            'error': 'UnexpectedError',
            'message': f"An unexpected error occurred: {str(e)}",
            'details': {}
        }


async def evolve_unitary(initial_state: qt.Qobj, hamiltonian: qt.Qobj, times: np.ndarray, options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform unitary time evolution using Schrödinger equation.
    
    Parameters
    ----------
    initial_state : qt.Qobj
        Initial quantum state
    hamiltonian : qt.Qobj
        System Hamiltonian
    times : np.ndarray
        Time points for evolution
    options : dict
        Solver options
        
    Returns
    -------
    dict
        Evolution results
    """
    try:
        # Use QuTip's Schrödinger equation solver
        solver_method = options.get('method', 'adams')
        rtol = options.get('rtol', 1e-8)
        atol = options.get('atol', 1e-10)
        
        # Convert to density matrix if needed for consistency
        if initial_state.type == 'ket':
            initial_rho = initial_state * initial_state.dag()
        else:
            initial_rho = initial_state
        
        # Solve using mesolve with no collapse operators (unitary evolution)
        result = qt.mesolve(
            hamiltonian, 
            initial_rho, 
            times,
            [],  # No collapse operators for unitary evolution
            options={'method': solver_method, 'rtol': rtol, 'atol': atol}
        )
        
        # Calculate expectation values for analysis
        observables = {
            'energy': hamiltonian,
        }
        
        if hamiltonian.shape[0] == 2:  # Qubit system
            observables.update({
                'sigma_x': qt.sigmax(),
                'sigma_y': qt.sigmay(), 
                'sigma_z': qt.sigmaz()
            })
        elif hamiltonian.shape[0] > 2:  # Harmonic oscillator-like
            observables.update({
                'number': qt.num(hamiltonian.shape[0]),
                'position': (qt.destroy(hamiltonian.shape[0]) + qt.create(hamiltonian.shape[0])) / np.sqrt(2),
                'momentum': 1j * (qt.create(hamiltonian.shape[0]) - qt.destroy(hamiltonian.shape[0])) / np.sqrt(2)
            })
        
        expectation_values = {}
        for name, obs in observables.items():
            try:
                exp_vals = qt.expect(obs, result.states)
                expectation_values[name] = [complex(val) for val in exp_vals]
            except Exception as e:
                logger.warning(f"Could not calculate expectation value for {name}: {e}")
        
        return {
            'states': result.states,
            'expectation_values': expectation_values,
            'solver_info': {
                'method': solver_method,
                'rtol': rtol,
                'atol': atol,
                'num_states': len(result.states)
            }
        }
        
    except Exception as e:
        raise QuantumEvolutionError(
            f"Unitary evolution failed: {str(e)}",
            evolution_type="unitary",
            solver_info={'method': options.get('method', 'adams')}
        )


async def evolve_master_equation(initial_state: qt.Qobj, hamiltonian: qt.Qobj, collapse_ops: List[qt.Qobj], times: np.ndarray, options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform open system evolution using master equation.
    
    Parameters
    ----------
    initial_state : qt.Qobj
        Initial quantum state (density matrix)
    hamiltonian : qt.Qobj
        System Hamiltonian
    collapse_ops : list of qt.Qobj
        Collapse operators
    times : np.ndarray
        Time points for evolution
    options : dict
        Solver options
        
    Returns
    -------
    dict
        Evolution results
    """
    try:
        # Convert to density matrix if needed
        if initial_state.type == 'ket':
            initial_rho = initial_state * initial_state.dag()
        else:
            initial_rho = initial_state
        
        solver_method = options.get('method', 'adams')
        rtol = options.get('rtol', 1e-8)
        atol = options.get('atol', 1e-10)
        
        # Solve master equation
        result = qt.mesolve(
            hamiltonian,
            initial_rho,
            times,
            collapse_ops,
            options={'method': solver_method, 'rtol': rtol, 'atol': atol}
        )
        
        # Calculate purity evolution
        purity_evolution = []
        for state in result.states:
            purity = np.real((state * state).tr())
            purity_evolution.append(float(purity))
        
        # Calculate von Neumann entropy evolution
        entropy_evolution = []
        for state in result.states:
            eigenvals = state.eigenenergies()
            entropy = -np.sum(eigenvals * np.log2(eigenvals + 1e-16))
            entropy_evolution.append(float(np.real(entropy)))
        
        return {
            'states': result.states,
            'purity_evolution': purity_evolution,
            'entropy_evolution': entropy_evolution,
            'solver_info': {
                'method': solver_method,
                'rtol': rtol,
                'atol': atol,
                'num_collapse_ops': len(collapse_ops),
                'num_states': len(result.states)
            }
        }
        
    except Exception as e:
        raise QuantumEvolutionError(
            f"Master equation evolution failed: {str(e)}",
            evolution_type="master",
            solver_info={'num_collapse_ops': len(collapse_ops)}
        )


async def evolve_monte_carlo(initial_state: qt.Qobj, hamiltonian: qt.Qobj, collapse_ops: List[qt.Qobj], times: np.ndarray, n_trajectories: int, options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform Monte Carlo quantum trajectory evolution.
    
    Parameters
    ----------
    initial_state : qt.Qobj
        Initial quantum state
    hamiltonian : qt.Qobj
        System Hamiltonian
    collapse_ops : list of qt.Qobj
        Collapse operators
    times : np.ndarray
        Time points for evolution
    n_trajectories : int
        Number of Monte Carlo trajectories
    options : dict
        Solver options
        
    Returns
    -------
    dict
        Evolution results
    """
    try:
        solver_method = options.get('method', 'adams')
        rtol = options.get('rtol', 1e-8)
        atol = options.get('atol', 1e-10)
        
        # Use QuTip's Monte Carlo solver
        result = qt.mcsolve(
            hamiltonian,
            initial_state,
            times,
            collapse_ops,
            ntraj=n_trajectories,
            options={'method': solver_method, 'rtol': rtol, 'atol': atol}
        )
        
        # Average the trajectories to get ensemble average
        averaged_states = []
        for i in range(len(times)):
            # Average over all trajectories at time i
            avg_state = sum(traj[i] * traj[i].dag() for traj in result.states) / n_trajectories
            averaged_states.append(avg_state)
        
        # Calculate trajectory statistics
        trajectory_stats = {
            'n_trajectories': n_trajectories,
            'trajectories_completed': len(result.states),
            'average_jumps_per_trajectory': 0  # Could be calculated from jump times
        }
        
        return {
            'states': averaged_states,
            'individual_trajectories': result.states if options.get('return_trajectories', False) else [],
            'trajectory_statistics': trajectory_stats,
            'solver_info': {
                'method': solver_method,
                'rtol': rtol,
                'atol': atol,
                'num_collapse_ops': len(collapse_ops),
                'n_trajectories': n_trajectories
            }
        }
        
    except Exception as e:
        raise QuantumEvolutionError(
            f"Monte Carlo evolution failed: {str(e)}",
            evolution_type="monte_carlo",
            solver_info={'n_trajectories': n_trajectories}
        )


async def evolve_stochastic(initial_state: qt.Qobj, hamiltonian: qt.Qobj, times: np.ndarray, options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform stochastic evolution (simplified implementation).
    
    Parameters
    ----------
    initial_state : qt.Qobj
        Initial quantum state
    hamiltonian : qt.Qobj
        System Hamiltonian  
    times : np.ndarray
        Time points for evolution
    options : dict
        Solver options
        
    Returns
    -------
    dict
        Evolution results
    """
    try:
        # This is a simplified stochastic evolution
        # In a full implementation, this would include stochastic Schrödinger equations
        
        noise_strength = options.get('noise_strength', 0.01)
        
        # Convert to density matrix
        if initial_state.type == 'ket':
            current_state = initial_state * initial_state.dag()
        else:
            current_state = initial_state
        
        states = [current_state]
        dt = times[1] - times[0]
        
        for i in range(1, len(times)):
            # Unitary evolution step
            U = (-1j * hamiltonian * dt).expm()
            current_state = U * current_state * U.dag()
            
            # Add stochastic noise (simplified)
            noise_op = qt.rand_herm(current_state.shape[0]) * noise_strength
            current_state = current_state + dt * noise_op
            
            # Renormalize
            current_state = current_state / current_state.tr()
            
            states.append(current_state)
        
        return {
            'states': states,
            'noise_strength': noise_strength,
            'solver_info': {
                'method': 'stochastic_euler',
                'noise_strength': noise_strength,
                'time_step': dt
            }
        }
        
    except Exception as e:
        raise QuantumEvolutionError(
            f"Stochastic evolution failed: {str(e)}",
            evolution_type="stochastic"
        )