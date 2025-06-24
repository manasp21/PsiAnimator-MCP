"""
Quantum Systems for PsiAnimator-MCP

Provides specialized quantum systems including spin systems, harmonic oscillators,
multi-level atoms, cavity QED systems, and quantum optics models.
"""

import numpy as np
import qutip as qt
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import logging

from .validation import (
    validate_quantum_state,
    validate_hermitian,
    check_hilbert_space_dimension,
    ensure_qobj
)
from ..server.exceptions import (
    QuantumSystemError,
    ValidationError,
    DimensionError
)

logger = logging.getLogger(__name__)


class QuantumSystems:
    """
    Provides quantum system models and Hamiltonians for various
    physical systems commonly studied in quantum mechanics.
    """
    
    def __init__(self, max_dimension: int = 1024):
        """
        Initialize quantum systems handler.
        
        Parameters
        ----------
        max_dimension : int, optional
            Maximum allowed Hilbert space dimension
        """
        self.max_dimension = max_dimension
        self._system_cache: Dict[str, qt.Qobj] = {}
        logger.info(f"QuantumSystems initialized with max_dimension={max_dimension}")
    
    def create_spin_system(
        self,
        spin: float,
        magnetic_field: Optional[Tuple[float, float, float]] = None,
        coupling_strength: Optional[float] = None
    ) -> Dict[str, qt.Qobj]:
        """
        Create a spin system with optional magnetic field.
        
        Parameters
        ----------
        spin : float
            Spin quantum number (1/2, 1, 3/2, ...)
        magnetic_field : tuple of float, optional
            Magnetic field components (Bx, By, Bz)
        coupling_strength : float, optional
            Coupling strength for magnetic field interaction
            
        Returns
        -------
        dict
            Dictionary containing Hamiltonian and spin operators
            
        Raises
        ------
        QuantumSystemError
            If system creation fails
        """
        try:
            # Validate spin value
            if spin <= 0 or not (spin * 2).is_integer():
                raise ValidationError(f"Invalid spin value: {spin}")
            
            dim = int(2 * spin + 1)
            check_hilbert_space_dimension(dim, self.max_dimension)
            
            # Create spin operators
            Sx = qt.jmat(spin, 'x')
            Sy = qt.jmat(spin, 'y') 
            Sz = qt.jmat(spin, 'z')
            Sp = qt.jmat(spin, '+')
            Sm = qt.jmat(spin, '-')
            
            operators = {
                'Sx': Sx,
                'Sy': Sy,
                'Sz': Sz,
                'S_plus': Sp,
                'S_minus': Sm,
                'S_squared': Sx*Sx + Sy*Sy + Sz*Sz
            }
            
            # Create Hamiltonian
            if magnetic_field is not None:
                Bx, By, Bz = magnetic_field
                g = coupling_strength or 1.0  # gyromagnetic ratio
                
                H = g * (Bx * Sx + By * Sy + Bz * Sz)
                operators['Hamiltonian'] = H
            else:
                operators['Hamiltonian'] = qt.qzero_like(Sx)
            
            logger.debug(f"Created spin-{spin} system with dimension {dim}")
            return operators
            
        except Exception as e:
            raise QuantumSystemError(
                f"Failed to create spin system: {str(e)}",
                system_type="spin",
                dimensions=[dim]
            )
    
    def create_harmonic_oscillator(
        self,
        N: int,
        omega: float = 1.0,
        displacement: Optional[complex] = None,
        squeezing: Optional[Dict[str, float]] = None
    ) -> Dict[str, qt.Qobj]:
        """
        Create a quantum harmonic oscillator system.
        
        Parameters
        ----------
        N : int
            Number of Fock states (dimension of Hilbert space)
        omega : float, optional
            Oscillator frequency
        displacement : complex, optional
            Displacement parameter for displaced oscillator
        squeezing : dict, optional
            Squeezing parameters {'r': squeezing_strength, 'phi': squeezing_phase}
            
        Returns
        -------
        dict
            Dictionary containing Hamiltonian and ladder operators
        """
        try:
            check_hilbert_space_dimension(N, self.max_dimension)
            
            # Create ladder operators
            a = qt.destroy(N)
            a_dag = qt.create(N)
            n = qt.num(N)
            
            # Position and momentum operators  
            x = (a + a_dag) / np.sqrt(2)
            p = 1j * (a_dag - a) / np.sqrt(2)
            
            operators = {
                'a': a,
                'a_dag': a_dag, 
                'n': n,
                'x': x,
                'p': p
            }
            
            # Create Hamiltonian
            H = omega * (n + 0.5)  # Standard harmonic oscillator
            
            # Add displacement if specified
            if displacement is not None:
                alpha = displacement
                D = qt.displace(N, alpha)
                H = D.dag() * H * D
                operators['displacement'] = D
            
            # Add squeezing if specified
            if squeezing is not None:
                r = squeezing.get('r', 0.0)
                phi = squeezing.get('phi', 0.0)
                S = qt.squeeze(N, r * np.exp(1j * phi))
                H = S.dag() * H * S
                operators['squeezing'] = S
            
            operators['Hamiltonian'] = H
            
            logger.debug(f"Created harmonic oscillator with N={N}, ω={omega}")
            return operators
            
        except Exception as e:
            raise QuantumSystemError(
                f"Failed to create harmonic oscillator: {str(e)}",
                system_type="harmonic_oscillator",
                dimensions=[N]
            )
    
    def create_multi_level_atom(
        self,
        energy_levels: List[float],
        transition_dipoles: Optional[Dict[Tuple[int, int], complex]] = None,
        decay_rates: Optional[Dict[Tuple[int, int], float]] = None
    ) -> Dict[str, qt.Qobj]:
        """
        Create a multi-level atomic system.
        
        Parameters
        ----------
        energy_levels : list of float
            Energy of each atomic level
        transition_dipoles : dict, optional
            Transition dipole matrix elements {(i,j): dipole_ij}
        decay_rates : dict, optional
            Spontaneous decay rates {(i,j): gamma_ij}
            
        Returns
        -------
        dict
            Dictionary containing Hamiltonian and atomic operators
        """
        try:
            N = len(energy_levels)
            check_hilbert_space_dimension(N, self.max_dimension)
            
            # Create atomic Hamiltonian (diagonal in energy eigenbasis)
            H_atom = qt.Qobj(np.diag(energy_levels))
            
            operators = {'Hamiltonian': H_atom}
            
            # Create projection operators for each level
            for i in range(N):
                proj = qt.basis(N, i) * qt.basis(N, i).dag()
                operators[f'P_{i}'] = proj
            
            # Create transition operators
            if transition_dipoles is not None:
                for (i, j), dipole in transition_dipoles.items():
                    if i < N and j < N and i != j:
                        sigma_ij = qt.basis(N, i) * qt.basis(N, j).dag()
                        operators[f'sigma_{i}_{j}'] = dipole * sigma_ij
                        
                        # Also create Hermitian conjugate
                        operators[f'sigma_{j}_{i}'] = np.conj(dipole) * sigma_ij.dag()
            
            # Store decay information for later use in master equations
            if decay_rates is not None:
                operators['_decay_rates'] = decay_rates
            
            logger.debug(f"Created {N}-level atomic system")
            return operators
            
        except Exception as e:
            raise QuantumSystemError(
                f"Failed to create multi-level atom: {str(e)}",
                system_type="multi_level_atom",
                dimensions=[len(energy_levels)]
            )
    
    def create_jaynes_cummings_model(
        self,
        N_photons: int,
        N_atom_levels: int = 2,
        omega_c: float = 1.0,
        omega_a: float = 1.0,
        g: float = 0.1,
        delta: Optional[float] = None
    ) -> Dict[str, qt.Qobj]:
        """
        Create Jaynes-Cummings model (atom-cavity interaction).
        
        Parameters
        ----------
        N_photons : int
            Maximum number of photons in cavity mode
        N_atom_levels : int, optional
            Number of atomic levels (default: 2 for two-level atom)
        omega_c : float, optional
            Cavity frequency
        omega_a : float, optional 
            Atomic transition frequency
        g : float, optional
            Atom-cavity coupling strength
        delta : float, optional
            Detuning (omega_a - omega_c), computed if None
            
        Returns
        -------
        dict
            Dictionary containing JC Hamiltonian and operators
        """
        try:
            total_dim = N_photons * N_atom_levels
            check_hilbert_space_dimension(total_dim, self.max_dimension)
            
            # Create cavity operators
            a = qt.tensor(qt.destroy(N_photons), qt.qeye(N_atom_levels))
            a_dag = qt.tensor(qt.create(N_photons), qt.qeye(N_atom_levels))
            n_c = a_dag * a
            
            # Create atomic operators (assume two-level atom)
            if N_atom_levels == 2:
                sigma_z = qt.tensor(qt.qeye(N_photons), qt.sigmaz())
                sigma_plus = qt.tensor(qt.qeye(N_photons), qt.sigmap())
                sigma_minus = qt.tensor(qt.qeye(N_photons), qt.sigmam())
            else:
                # General multi-level case
                sz = qt.Qobj(np.diag(np.arange(N_atom_levels) - (N_atom_levels-1)/2))
                sigma_z = qt.tensor(qt.qeye(N_photons), sz)
                sigma_plus = qt.tensor(qt.qeye(N_photons), 
                                     qt.basis(N_atom_levels, N_atom_levels-1) * 
                                     qt.basis(N_atom_levels, 0).dag())
                sigma_minus = sigma_plus.dag()
            
            # Detuning
            if delta is None:
                delta = omega_a - omega_c
            
            # Jaynes-Cummings Hamiltonian
            H_JC = (omega_c * n_c + 
                   (omega_a / 2) * sigma_z + 
                   g * (a * sigma_plus + a_dag * sigma_minus))
            
            operators = {
                'Hamiltonian': H_JC,
                'a': a,
                'a_dag': a_dag,
                'n_photons': n_c,
                'sigma_z': sigma_z,
                'sigma_plus': sigma_plus,
                'sigma_minus': sigma_minus,
                'interaction': g * (a * sigma_plus + a_dag * sigma_minus)
            }
            
            logger.debug(f"Created Jaynes-Cummings model: N_photons={N_photons}, g={g}")
            return operators
            
        except Exception as e:
            raise QuantumSystemError(
                f"Failed to create Jaynes-Cummings model: {str(e)}",
                system_type="jaynes_cummings",
                dimensions=[N_photons, N_atom_levels]
            )
    
    def create_cavity_qed_system(
        self,
        N_modes: int,
        N_atoms: int,
        mode_frequencies: List[float],
        atom_frequencies: List[float],
        coupling_matrix: np.ndarray,
        decay_rates: Optional[Dict[str, float]] = None
    ) -> Dict[str, qt.Qobj]:
        """
        Create a general cavity QED system with multiple modes and atoms.
        
        Parameters
        ----------
        N_modes : int
            Number of cavity modes
        N_atoms : int
            Number of atoms
        mode_frequencies : list of float
            Frequency of each cavity mode
        atom_frequencies : list of float
            Frequency of each atomic transition
        coupling_matrix : np.ndarray
            Coupling strengths between atoms and modes [N_atoms × N_modes]
        decay_rates : dict, optional
            Decay rates for modes and atoms
            
        Returns
        -------
        dict
            Dictionary containing cavity QED Hamiltonian and operators
        """
        try:
            if len(mode_frequencies) != N_modes:
                raise ValidationError("mode_frequencies length must match N_modes")
            if len(atom_frequencies) != N_atoms:
                raise ValidationError("atom_frequencies length must match N_atoms")
            if coupling_matrix.shape != (N_atoms, N_modes):
                raise ValidationError(f"coupling_matrix shape must be ({N_atoms}, {N_modes})")
            
            # For simplicity, assume each mode has same dimension and each atom is two-level
            N_photons_per_mode = 10  # Could be made configurable
            
            total_dim = (N_photons_per_mode ** N_modes) * (2 ** N_atoms)
            if total_dim > self.max_dimension:
                # Reduce complexity or raise error
                raise DimensionError(f"System too large: dimension {total_dim} > {self.max_dimension}")
            
            # Create mode operators
            mode_ops = []
            H_modes = 0
            
            for i in range(N_modes):
                # Create operators for mode i
                ops_i = [qt.qeye(N_photons_per_mode if j == i 
                              else N_photons_per_mode if j < N_modes 
                              else 2) for j in range(N_modes + N_atoms)]
                
                ops_i[i] = qt.destroy(N_photons_per_mode)
                a_i = qt.tensor(*ops_i)
                
                ops_i[i] = qt.create(N_photons_per_mode)  
                a_dag_i = qt.tensor(*ops_i)
                
                mode_ops.append({'a': a_i, 'a_dag': a_dag_i})
                H_modes += mode_frequencies[i] * a_dag_i * a_i
            
            # Create atomic operators
            atom_ops = []
            H_atoms = 0
            
            for j in range(N_atoms):
                # Create operators for atom j
                ops_j = [qt.qeye(N_photons_per_mode if k < N_modes 
                               else 2 if k == N_modes + j 
                               else 2) for k in range(N_modes + N_atoms)]
                
                ops_j[N_modes + j] = qt.sigmaz()
                sigma_z_j = qt.tensor(*ops_j)
                
                ops_j[N_modes + j] = qt.sigmap()
                sigma_plus_j = qt.tensor(*ops_j)
                
                ops_j[N_modes + j] = qt.sigmam()
                sigma_minus_j = qt.tensor(*ops_j)
                
                atom_ops.append({
                    'sigma_z': sigma_z_j,
                    'sigma_plus': sigma_plus_j,
                    'sigma_minus': sigma_minus_j
                })
                
                H_atoms += (atom_frequencies[j] / 2) * sigma_z_j
            
            # Create interaction Hamiltonian
            H_int = sum(coupling_matrix[j, i] * 
                       (mode_ops[i]['a'] * atom_ops[j]['sigma_plus'] +
                        mode_ops[i]['a_dag'] * atom_ops[j]['sigma_minus'])
                       for i in range(N_modes) for j in range(N_atoms))
            
            # Total Hamiltonian
            H_total = H_modes + H_atoms + H_int
            
            operators = {
                'Hamiltonian': H_total,
                'H_modes': H_modes,
                'H_atoms': H_atoms,
                'H_interaction': H_int,
                'mode_operators': mode_ops,
                'atom_operators': atom_ops
            }
            
            if decay_rates is not None:
                operators['_decay_rates'] = decay_rates
            
            logger.debug(f"Created cavity QED system: {N_modes} modes, {N_atoms} atoms")
            return operators
            
        except Exception as e:
            raise QuantumSystemError(
                f"Failed to create cavity QED system: {str(e)}",
                system_type="cavity_qed",
                dimensions=[N_modes, N_atoms]
            )
    
    def create_rabi_model(
        self,
        N_photons: int,
        omega_c: float = 1.0,
        omega_a: float = 1.0, 
        g: float = 0.1
    ) -> Dict[str, qt.Qobj]:
        """
        Create quantum Rabi model (without rotating wave approximation).
        
        Parameters
        ----------
        N_photons : int
            Maximum number of photons in cavity mode
        omega_c : float, optional
            Cavity frequency
        omega_a : float, optional
            Atomic transition frequency  
        g : float, optional
            Atom-cavity coupling strength
            
        Returns
        -------
        dict
            Dictionary containing Rabi Hamiltonian and operators
        """
        try:
            total_dim = N_photons * 2  # 2 for two-level atom
            check_hilbert_space_dimension(total_dim, self.max_dimension)
            
            # Create operators
            a = qt.tensor(qt.destroy(N_photons), qt.qeye(2))
            a_dag = qt.tensor(qt.create(N_photons), qt.qeye(2))
            n_c = a_dag * a
            
            sigma_x = qt.tensor(qt.qeye(N_photons), qt.sigmax())
            sigma_z = qt.tensor(qt.qeye(N_photons), qt.sigmaz())
            
            # Rabi Hamiltonian (no rotating wave approximation)
            H_Rabi = (omega_c * n_c + 
                     (omega_a / 2) * sigma_z + 
                     g * sigma_x * (a + a_dag))
            
            operators = {
                'Hamiltonian': H_Rabi,
                'a': a,
                'a_dag': a_dag,
                'n_photons': n_c,
                'sigma_x': sigma_x,
                'sigma_z': sigma_z,
                'interaction': g * sigma_x * (a + a_dag)
            }
            
            logger.debug(f"Created Rabi model: N_photons={N_photons}, g={g}")
            return operators
            
        except Exception as e:
            raise QuantumSystemError(
                f"Failed to create Rabi model: {str(e)}",
                system_type="rabi_model",
                dimensions=[N_photons, 2]
            )
    
    def create_transmon_qubit(
        self,
        N_levels: int = 5,
        E_C: float = 0.2,
        E_J: float = 10.0,
        d: float = 0.0
    ) -> Dict[str, qt.Qobj]:
        """
        Create a transmon qubit model.
        
        Parameters
        ----------
        N_levels : int, optional
            Number of transmon energy levels to include
        E_C : float, optional
            Charging energy
        E_J : float, optional
            Josephson energy
        d : float, optional
            Asymmetry parameter
            
        Returns
        -------
        dict
            Dictionary containing transmon Hamiltonian and operators
        """
        try:
            check_hilbert_space_dimension(N_levels, self.max_dimension)
            
            # Create charge operator (momentum conjugate to phase)
            n = qt.Qobj(np.diag(np.arange(N_levels) - N_levels//2))
            
            # Create phase operators (approximate for finite Hilbert space)
            phi = qt.Qobj(np.zeros((N_levels, N_levels)))
            for i in range(N_levels-1):
                phi[i, i+1] = 1.0
                phi[i+1, i] = 1.0
            phi = phi * np.pi / (2 * np.sqrt(E_J / E_C))
            
            # Transmon Hamiltonian
            H_transmon = 4 * E_C * n * n - E_J * qt.Qobj(qt.cosm(phi))
            
            if d != 0:
                # Add asymmetry term
                H_transmon -= d * E_J * qt.Qobj(qt.sinm(phi))
            
            # Transition operators (approximately like harmonic oscillator for low levels)
            a = qt.destroy(N_levels)
            a_dag = qt.create(N_levels)
            
            operators = {
                'Hamiltonian': H_transmon,
                'n': n,
                'phi': phi,
                'a': a,  # Approximate lowering operator
                'a_dag': a_dag  # Approximate raising operator
            }
            
            logger.debug(f"Created transmon qubit: N_levels={N_levels}, E_C={E_C}, E_J={E_J}")
            return operators
            
        except Exception as e:
            raise QuantumSystemError(
                f"Failed to create transmon qubit: {str(e)}",
                system_type="transmon_qubit",
                dimensions=[N_levels]
            )
    
    def get_system_parameters(self, system_type: str) -> Dict[str, Any]:
        """
        Get default parameters for a quantum system type.
        
        Parameters
        ----------
        system_type : str
            Type of quantum system
            
        Returns
        -------
        dict
            Default parameters for the system
        """
        defaults = {
            "spin": {
                "spin": 0.5,
                "magnetic_field": (0.0, 0.0, 1.0),
                "coupling_strength": 1.0
            },
            "harmonic_oscillator": {
                "N": 20,
                "omega": 1.0,
                "displacement": None,
                "squeezing": None
            },
            "multi_level_atom": {
                "energy_levels": [0.0, 1.0, 2.0],
                "transition_dipoles": {(0, 1): 1.0, (1, 2): 0.8},
                "decay_rates": {(1, 0): 0.1, (2, 1): 0.05}
            },
            "jaynes_cummings": {
                "N_photons": 10,
                "N_atom_levels": 2,
                "omega_c": 1.0,
                "omega_a": 1.0,
                "g": 0.1
            },
            "rabi_model": {
                "N_photons": 10,
                "omega_c": 1.0,
                "omega_a": 1.0,
                "g": 0.1
            },
            "transmon_qubit": {
                "N_levels": 5,
                "E_C": 0.2,
                "E_J": 10.0,
                "d": 0.0
            }
        }
        
        return defaults.get(system_type, {})