"""
Quantum State Management for PsiAnimator-MCP

Handles creation, manipulation, and analysis of quantum states including
pure states, mixed states, composite systems, and time evolution.
"""

import numpy as np
import qutip as qt
from typing import Dict, List, Optional, Union, Tuple, Any
import uuid
import logging

from .validation import (
    validate_quantum_state,
    validate_dimensions,
    check_hilbert_space_dimension,
    normalize_state,
    ensure_qobj
)
from ..server.exceptions import (
    QuantumStateError,
    DimensionError,
    ValidationError
)

logger = logging.getLogger(__name__)


class QuantumStateManager:
    """
    Manages quantum states with creation, storage, and manipulation capabilities.
    
    Supports pure states (ket vectors), mixed states (density matrices),
    composite systems (tensor products), and provides state analysis tools.
    """
    
    def __init__(self, max_dimension: int = 1024):
        """
        Initialize the quantum state manager.
        
        Parameters
        ----------
        max_dimension : int, optional
            Maximum allowed Hilbert space dimension
        """
        self.max_dimension = max_dimension
        self._states: Dict[str, qt.Qobj] = {}
        self._state_metadata: Dict[str, Dict[str, Any]] = {}
        self._evolution_results: Dict[str, Dict[str, Any]] = {}
        self._measurement_results: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"QuantumStateManager initialized with max_dimension={max_dimension}")
    
    def create_state(
        self,
        state_type: str,
        system_dims: List[int],
        parameters: Optional[Dict[str, Any]] = None,
        basis: str = "computational",
        state_id: Optional[str] = None
    ) -> str:
        """
        Create a quantum state with specified parameters.
        
        Parameters
        ----------
        state_type : str
            Type of state to create ('pure', 'mixed', 'coherent', 'squeezed', 'thermal', 'fock')
        system_dims : list of int
            Dimensions of each subsystem
        parameters : dict, optional
            State-specific parameters
        basis : str, optional
            Basis for state creation ('computational', 'fock', 'spin', 'position')
        state_id : str, optional
            Custom state identifier (auto-generated if None)
            
        Returns
        -------
        str
            Unique identifier for the created state
            
        Raises
        ------
        QuantumStateError
            If state creation fails
        DimensionError
            If dimensions are invalid
        """
        if state_id is None:
            state_id = str(uuid.uuid4())
        
        # Validate dimensions
        total_dim = np.prod(system_dims)
        check_hilbert_space_dimension(total_dim, self.max_dimension)
        
        parameters = parameters or {}
        
        try:
            if state_type == "pure":
                state = self._create_pure_state(system_dims, parameters, basis)
            elif state_type == "mixed":
                state = self._create_mixed_state(system_dims, parameters, basis)
            elif state_type == "coherent":
                state = self._create_coherent_state(system_dims, parameters)
            elif state_type == "squeezed":
                state = self._create_squeezed_state(system_dims, parameters)
            elif state_type == "thermal":
                state = self._create_thermal_state(system_dims, parameters)
            elif state_type == "fock":
                state = self._create_fock_state(system_dims, parameters)
            else:
                raise ValidationError(
                    f"Unknown state type: {state_type}",
                    field="state_type",
                    expected_type="one of: pure, mixed, coherent, squeezed, thermal, fock"
                )
            
            # Validate and store state
            state = validate_quantum_state(state)
            self._states[state_id] = state
            
            # Store metadata
            self._state_metadata[state_id] = {
                "state_type": state_type,
                "system_dims": system_dims,
                "parameters": parameters,
                "basis": basis,
                "creation_time": logger.info.__defaults__[0] if hasattr(logger.info, '__defaults__') else None,
                "hilbert_dim": total_dim
            }
            
            logger.info(f"Created {state_type} state with ID: {state_id}")
            return state_id
            
        except Exception as e:
            raise QuantumStateError(
                f"Failed to create {state_type} state: {str(e)}",
                state_info={
                    "state_type": state_type,
                    "system_dims": system_dims,
                    "parameters": parameters
                }
            )
    
    def _create_pure_state(
        self,
        system_dims: List[int],
        parameters: Dict[str, Any],
        basis: str
    ) -> qt.Qobj:
        """Create a pure quantum state."""
        
        if basis == "computational":
            # Create computational basis state
            state_indices = parameters.get("state_indices", [0] * len(system_dims))
            
            if len(state_indices) != len(system_dims):
                raise ValidationError("state_indices length must match system_dims length")
            
            # Create tensor product of basis states
            state_parts = []
            for i, (dim, idx) in enumerate(zip(system_dims, state_indices)):
                if idx >= dim:
                    raise ValidationError(f"State index {idx} exceeds dimension {dim} for subsystem {i}")
                state_parts.append(qt.basis(dim, idx))
            
            return qt.tensor(*state_parts)
            
        elif basis == "spin":
            # Create spin state (assumes spin-1/2 unless specified)
            if len(system_dims) != 1:
                raise ValidationError("Spin basis currently supports single spin systems only")
            
            spin_value = parameters.get("spin", 0.5)
            theta = parameters.get("theta", 0.0)  # Polar angle
            phi = parameters.get("phi", 0.0)     # Azimuthal angle
            
            return qt.spin_state(spin_value, theta, phi)
            
        elif basis == "position":
            # Create position eigenstate (approximate)
            if len(system_dims) != 1:
                raise ValidationError("Position basis currently supports single particle systems only")
            
            x0 = parameters.get("position", 0.0)
            sigma = parameters.get("width", 1.0)
            N = system_dims[0]
            
            # Create Gaussian wavepacket centered at x0
            x = np.linspace(-5*sigma, 5*sigma, N)
            psi = np.exp(-(x - x0)**2 / (2*sigma**2))
            psi = psi / np.sqrt(np.trapz(np.abs(psi)**2, x))
            
            return qt.Qobj(psi.reshape(-1, 1))
        
        else:
            # Custom state from coefficients
            coefficients = parameters.get("coefficients")
            if coefficients is None:
                raise ValidationError("coefficients required for custom pure state")
            
            total_dim = np.prod(system_dims)
            if len(coefficients) != total_dim:
                raise ValidationError(f"coefficients length {len(coefficients)} must match total dimension {total_dim}")
            
            state = qt.Qobj(np.array(coefficients).reshape(-1, 1))
            return normalize_state(state)
    
    def _create_mixed_state(
        self,
        system_dims: List[int],
        parameters: Dict[str, Any],
        basis: str
    ) -> qt.Qobj:
        """Create a mixed quantum state (density matrix)."""
        
        mixture_type = parameters.get("mixture_type", "random")
        
        if mixture_type == "random":
            # Random mixed state
            total_dim = np.prod(system_dims)
            purity = parameters.get("purity", 0.5)  # 0 = maximally mixed, 1 = pure
            
            # Generate random density matrix with specified purity
            random_unitary = qt.rand_unitary(total_dim)
            eigenvals = self._generate_eigenvalues(total_dim, purity)
            
            # Construct density matrix
            rho = sum(eigenvals[i] * random_unitary[:, i] * random_unitary[:, i].dag() 
                     for i in range(total_dim))
            
            return rho
            
        elif mixture_type == "statistical":
            # Statistical mixture of pure states
            pure_states = parameters.get("pure_states", [])
            probabilities = parameters.get("probabilities", [])
            
            if not pure_states:
                raise ValidationError("pure_states required for statistical mixture")
            
            if not probabilities:
                probabilities = [1.0 / len(pure_states)] * len(pure_states)
            
            if len(probabilities) != len(pure_states):
                raise ValidationError("probabilities length must match pure_states length")
            
            if abs(sum(probabilities) - 1.0) > 1e-10:
                raise ValidationError("probabilities must sum to 1.0")
            
            # Create mixture
            rho = sum(prob * (state * state.dag() if isinstance(state, qt.Qobj) else 
                             self._create_pure_state(system_dims, state, basis) * 
                             self._create_pure_state(system_dims, state, basis).dag())
                     for prob, state in zip(probabilities, pure_states))
            
            return rho
        
        else:
            raise ValidationError(f"Unknown mixture_type: {mixture_type}")
    
    def _create_coherent_state(
        self,
        system_dims: List[int],
        parameters: Dict[str, Any]
    ) -> qt.Qobj:
        """Create a coherent state for harmonic oscillator."""
        
        if len(system_dims) != 1:
            raise ValidationError("Coherent states currently support single oscillator systems only")
        
        N = system_dims[0]
        alpha = parameters.get("alpha", 1.0)  # Complex amplitude
        
        return qt.coherent(N, alpha)
    
    def _create_squeezed_state(
        self,
        system_dims: List[int],
        parameters: Dict[str, Any]
    ) -> qt.Qobj:
        """Create a squeezed state for harmonic oscillator."""
        
        if len(system_dims) != 1:
            raise ValidationError("Squeezed states currently support single oscillator systems only")
        
        N = system_dims[0]
        alpha = parameters.get("alpha", 0.0)  # Displacement
        r = parameters.get("r", 1.0)          # Squeezing parameter
        theta = parameters.get("theta", 0.0)   # Squeezing angle
        
        return qt.squeeze(N, alpha, r, theta)
    
    def _create_thermal_state(
        self,
        system_dims: List[int],
        parameters: Dict[str, Any]
    ) -> qt.Qobj:
        """Create a thermal state for harmonic oscillator."""
        
        if len(system_dims) != 1:
            raise ValidationError("Thermal states currently support single oscillator systems only")
        
        N = system_dims[0]
        n_avg = parameters.get("n_avg", 1.0)  # Average photon number
        
        return qt.thermal_dm(N, n_avg)
    
    def _create_fock_state(
        self,
        system_dims: List[int],
        parameters: Dict[str, Any]
    ) -> qt.Qobj:
        """Create a Fock (number) state for harmonic oscillator."""
        
        if len(system_dims) != 1:
            raise ValidationError("Fock states currently support single oscillator systems only")
        
        N = system_dims[0]
        n = parameters.get("n", 0)  # Photon number
        
        if n >= N:
            raise ValidationError(f"Photon number {n} must be less than dimension {N}")
        
        return qt.fock(N, n)
    
    def _generate_eigenvalues(self, dim: int, purity: float) -> np.ndarray:
        """Generate eigenvalues for mixed state with specified purity."""
        
        if purity == 1.0:
            # Pure state
            eigenvals = np.zeros(dim)
            eigenvals[0] = 1.0
        elif purity == 0.0:
            # Maximally mixed state
            eigenvals = np.ones(dim) / dim
        else:
            # Intermediate purity
            # Use random distribution and adjust to achieve target purity
            eigenvals = np.random.exponential(1.0, dim)
            eigenvals = eigenvals / np.sum(eigenvals)
            
            # Adjust purity by mixing with maximally mixed state
            max_mixed = np.ones(dim) / dim
            eigenvals = purity * eigenvals + (1 - purity) * max_mixed
            eigenvals = eigenvals / np.sum(eigenvals)
        
        return eigenvals
    
    def get_state(self, state_id: str) -> qt.Qobj:
        """
        Retrieve a quantum state by ID.
        
        Parameters
        ----------
        state_id : str
            State identifier
            
        Returns
        -------
        qt.Qobj
            The quantum state
            
        Raises
        ------
        QuantumStateError
            If state not found
        """
        if state_id not in self._states:
            raise QuantumStateError(f"State with ID '{state_id}' not found")
        
        return self._states[state_id]
    
    def get_state_info(self, state_id: str) -> Dict[str, Any]:
        """
        Get metadata about a quantum state.
        
        Parameters
        ----------
        state_id : str
            State identifier
            
        Returns
        -------
        dict
            State metadata and properties
        """
        if state_id not in self._states:
            raise QuantumStateError(f"State with ID '{state_id}' not found")
        
        state = self._states[state_id]
        metadata = self._state_metadata[state_id].copy()
        
        # Add computed properties
        metadata.update({
            "type": state.type,
            "shape": state.shape,
            "dims": state.dims,
            "norm": state.norm() if state.type == 'ket' else state.tr(),
            "is_pure": state.type == 'ket' or (state.type == 'oper' and abs(state.tr(state * state) - 1.0) < 1e-10)
        })
        
        if state.type == 'oper':
            # Additional properties for density matrices
            eigenvals = state.eigenenergies()
            metadata.update({
                "eigenvalues": eigenvals.tolist(),
                "purity": np.real(state.tr(state * state)),
                "von_neumann_entropy": -np.sum(eigenvals * np.log2(eigenvals + 1e-16))
            })
        
        return metadata
    
    def list_states(self) -> List[str]:
        """
        List all stored state IDs.
        
        Returns
        -------
        list of str
            List of state identifiers
        """
        return list(self._states.keys())
    
    def delete_state(self, state_id: str) -> None:
        """
        Delete a stored quantum state.
        
        Parameters
        ----------
        state_id : str
            State identifier to delete
        """
        if state_id in self._states:
            del self._states[state_id]
            del self._state_metadata[state_id]
            logger.info(f"Deleted state: {state_id}")
        else:
            logger.warning(f"Attempted to delete non-existent state: {state_id}")
    
    def clear_all_states(self) -> None:
        """Delete all stored quantum states."""
        count = len(self._states)
        self._states.clear()
        self._state_metadata.clear()
        logger.info(f"Cleared {count} quantum states")
    
    def tensor_product(self, state_ids: List[str], new_state_id: Optional[str] = None) -> str:
        """
        Create tensor product of multiple quantum states.
        
        Parameters
        ----------
        state_ids : list of str
            List of state identifiers to combine
        new_state_id : str, optional
            ID for the resulting state
            
        Returns
        -------
        str
            ID of the tensor product state
        """
        if not state_ids:
            raise ValidationError("At least one state ID required")
        
        states = [self.get_state(sid) for sid in state_ids]
        
        # Create tensor product
        result_state = qt.tensor(*states)
        
        # Generate new state ID
        if new_state_id is None:
            new_state_id = str(uuid.uuid4())
        
        # Store result
        self._states[new_state_id] = result_state
        self._state_metadata[new_state_id] = {
            "state_type": "tensor_product",
            "component_states": state_ids,
            "system_dims": [s.shape[0] for s in states],
            "creation_time": None,
            "hilbert_dim": result_state.shape[0]
        }
        
        logger.info(f"Created tensor product state: {new_state_id}")
        return new_state_id
    
    def store_evolution_result(
        self,
        evolution_id: str,
        evolution_data: Dict[str, Any],
        initial_state_id: str,
        evolution_type: str,
        parameters: Dict[str, Any]
    ) -> None:
        """
        Store evolution results for later retrieval.
        
        Parameters
        ----------
        evolution_id : str
            Unique identifier for the evolution result
        evolution_data : dict
            Evolution data including states, times, expectation values
        initial_state_id : str
            ID of the initial state
        evolution_type : str
            Type of evolution ('unitary', 'master', 'monte_carlo', 'stochastic')
        parameters : dict
            Evolution parameters (hamiltonian, time_span, etc.)
        """
        # Convert QuTip states to serializable format for storage
        stored_states = []
        if 'states' in evolution_data:
            for i, state in enumerate(evolution_data['states']):
                state_id = f"{evolution_id}_t{i}"
                self._states[state_id] = state
                stored_states.append(state_id)
        
        self._evolution_results[evolution_id] = {
            'evolution_type': evolution_type,
            'initial_state_id': initial_state_id,
            'parameters': parameters,
            'stored_state_ids': stored_states,
            'expectation_values': evolution_data.get('expectation_values', {}),
            'times': evolution_data.get('times', []),
            'solver_info': evolution_data.get('solver_info', {}),
            'creation_time': None
        }
        
        logger.info(f"Stored evolution result: {evolution_id} with {len(stored_states)} time steps")
    
    def get_evolution_result(self, evolution_id: str) -> Dict[str, Any]:
        """
        Retrieve stored evolution results.
        
        Parameters
        ----------
        evolution_id : str
            Evolution result identifier
            
        Returns
        -------
        dict
            Evolution result data
            
        Raises
        ------
        QuantumStateError
            If evolution result not found
        """
        if evolution_id not in self._evolution_results:
            raise QuantumStateError(f"Evolution result '{evolution_id}' not found")
        
        result = self._evolution_results[evolution_id].copy()
        
        # Reconstruct states from stored IDs
        if 'stored_state_ids' in result:
            states = []
            for state_id in result['stored_state_ids']:
                if state_id in self._states:
                    states.append(self._states[state_id])
            result['states'] = states
        
        return result
    
    def list_evolution_results(self) -> List[str]:
        """
        List all stored evolution result IDs.
        
        Returns
        -------
        list of str
            List of evolution result identifiers
        """
        return list(self._evolution_results.keys())
    
    def clear_evolution_results(self) -> None:
        """Clear all stored evolution results."""
        # Remove stored evolution states
        for evolution_data in self._evolution_results.values():
            for state_id in evolution_data.get('stored_state_ids', []):
                if state_id in self._states:
                    del self._states[state_id]
                if state_id in self._state_metadata:
                    del self._state_metadata[state_id]
        
        self._evolution_results.clear()
        logger.info("Cleared all evolution results")
    
    def store_measurement_result(
        self,
        measurement_id: str,
        state_id: str,
        observable_spec: str,
        measurement_type: str,
        measurement_results: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> None:
        """
        Store measurement results for later retrieval.
        
        Parameters
        ----------
        measurement_id : str
            Unique identifier for the measurement result
        state_id : str
            ID of the measured state
        observable_spec : str
            Observable specification string
        measurement_type : str
            Type of measurement ('expectation', 'variance', 'probability', etc.)
        measurement_results : dict
            The actual measurement results
        parameters : dict
            Measurement parameters
        """
        self._measurement_results[measurement_id] = {
            'state_id': state_id,
            'observable_spec': observable_spec,
            'measurement_type': measurement_type,
            'measurement_results': measurement_results,
            'parameters': parameters,
            'timestamp': None
        }
        
        logger.info(f"Stored measurement result: {measurement_id}")
    
    def get_measurement_result(self, measurement_id: str) -> Dict[str, Any]:
        """
        Retrieve stored measurement results.
        
        Parameters
        ----------
        measurement_id : str
            Measurement result identifier
            
        Returns
        -------
        dict
            Measurement result data
            
        Raises
        ------
        QuantumStateError
            If measurement result not found
        """
        if measurement_id not in self._measurement_results:
            raise QuantumStateError(f"Measurement result '{measurement_id}' not found")
        
        return self._measurement_results[measurement_id].copy()
    
    def list_measurement_results(self) -> List[str]:
        """
        List all stored measurement result IDs.
        
        Returns
        -------
        list of str
            List of measurement result identifiers
        """
        return list(self._measurement_results.keys())
    
    def clear_measurement_results(self) -> None:
        """Clear all stored measurement results."""
        self._measurement_results.clear()
        logger.info("Cleared all measurement results")