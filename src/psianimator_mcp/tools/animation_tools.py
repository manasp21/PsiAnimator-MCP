"""
Animation Tools for PsiAnimator-MCP

Implements the animate_quantum_process MCP tool for generating Manim animations
of quantum processes including Bloch sphere evolution, Wigner functions, etc.
"""

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import qutip as qt

from ..quantum.state_manager import QuantumStateManager

# Conditional animation imports
try:
    from ..animation.quantum_scene import QuantumScene
    from ..animation.bloch_sphere import BlochSphere3D
    from ..animation.state_tomography import StateTomography
    from ..animation.wigner_function import WignerFunction
    from ..animation.photon_statistics import PhotonStatistics
    from ..animation.quantum_circuit import QuantumCircuit
    from ..animation.energy_levels import EnergyLevelDiagram
    _ANIMATION_IMPORTS_AVAILABLE = True
except ImportError as e:
    # Animation dependencies not available, set placeholders
    QuantumScene = None
    BlochSphere3D = None
    StateTomography = None
    WignerFunction = None
    PhotonStatistics = None
    QuantumCircuit = None
    EnergyLevelDiagram = None
    _ANIMATION_IMPORTS_AVAILABLE = False
from ..server.config import MCPConfig
from ..server.exceptions import (
    AnimationError,
    ValidationError,
    QuantumStateError
)
from .quantum_state_tools import get_state_manager

logger = logging.getLogger(__name__)


async def animate_quantum_process(arguments: Dict[str, Any], config: MCPConfig) -> Dict[str, Any]:
    """
    Generate Manim animation of quantum processes.
    
    Parameters
    ----------
    arguments : dict
        Tool arguments containing:
        - animation_type: str - Type of animation to generate
        - data_source: str - Source of data (state_id, evolution_data, etc.)
        - render_quality: str - Rendering quality ('low', 'medium', 'high', 'production')
        - output_format: str - Output format ('mp4', 'gif', 'webm')
        - frame_rate: int - Animation frame rate
        - duration: float - Animation duration
        - view_config: dict - View and styling configuration
    config : MCPConfig
        Server configuration
        
    Returns
    -------
    dict
        Animation generation results with file paths and metadata
    """
    try:
        # Check if animation functionality is available
        if not _ANIMATION_IMPORTS_AVAILABLE:
            raise AnimationError(
                "Animation functionality not available. "
                "Please install animation dependencies: pip install 'psianimator-mcp[animation]'"
            )
            
        logger.info(f"Generating quantum animation with arguments: {arguments}")
        
        # Validate required arguments
        required_args = ['animation_type', 'data_source']
        for arg in required_args:
            if arg not in arguments:
                raise ValidationError(f"{arg} is required", field=arg)
        
        animation_type = arguments['animation_type']
        data_source = arguments['data_source']
        render_quality = arguments.get('render_quality', config.default_render_quality)
        output_format = arguments.get('output_format', 'mp4')
        frame_rate = arguments.get('frame_rate', config.default_frame_rate)
        duration = arguments.get('duration', 5.0)
        view_config = arguments.get('view_config', {})
        
        # Validate animation type
        valid_types = ['bloch_evolution', 'wigner_dynamics', 'state_tomography', 
                      'circuit_execution', 'energy_levels', 'photon_statistics']
        if animation_type not in valid_types:
            raise ValidationError(
                f"Invalid animation_type: {animation_type}",
                field="animation_type",
                expected_type=f"one of: {', '.join(valid_types)}"
            )
        
        # Validate render quality
        valid_qualities = ['low', 'medium', 'high', 'production']
        if render_quality not in valid_qualities:
            raise ValidationError(
                f"Invalid render_quality: {render_quality}",
                field="render_quality",
                expected_type=f"one of: {', '.join(valid_qualities)}"
            )
        
        # Validate output format
        valid_formats = ['mp4', 'gif', 'webm']
        if output_format not in valid_formats:
            raise ValidationError(
                f"Invalid output_format: {output_format}",
                field="output_format",
                expected_type=f"one of: {', '.join(valid_formats)}"
            )
        
        # Validate frame rate and duration
        if not isinstance(frame_rate, int) or frame_rate < 1 or frame_rate > 120:
            raise ValidationError(
                "frame_rate must be an integer between 1 and 120",
                field="frame_rate"
            )
        
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise ValidationError(
                "duration must be a positive number",
                field="duration"
            )
        
        # Parse data source and prepare animation data
        animation_data = await prepare_animation_data(data_source, animation_type, config)
        
        # Generate animation based on type
        if animation_type == 'bloch_evolution':
            result = await generate_bloch_evolution_animation(
                animation_data, render_quality, output_format, frame_rate, duration, view_config, config
            )
            
        elif animation_type == 'wigner_dynamics':
            result = await generate_wigner_dynamics_animation(
                animation_data, render_quality, output_format, frame_rate, duration, view_config, config
            )
            
        elif animation_type == 'state_tomography':
            result = await generate_state_tomography_animation(
                animation_data, render_quality, output_format, frame_rate, duration, view_config, config
            )
            
        elif animation_type == 'circuit_execution':
            result = await generate_circuit_execution_animation(
                animation_data, render_quality, output_format, frame_rate, duration, view_config, config
            )
            
        elif animation_type == 'energy_levels':
            result = await generate_energy_levels_animation(
                animation_data, render_quality, output_format, frame_rate, duration, view_config, config
            )
            
        elif animation_type == 'photon_statistics':
            result = await generate_photon_statistics_animation(
                animation_data, render_quality, output_format, frame_rate, duration, view_config, config
            )
        
        # Prepare final result
        animation_result = {
            'success': True,
            'animation_type': animation_type,
            'data_source': data_source,
            'render_settings': {
                'quality': render_quality,
                'output_format': output_format,
                'frame_rate': frame_rate,
                'duration': duration
            },
            'view_config': view_config,
            'output_files': result.get('output_files', []),
            'file_info': result.get('file_info', {}),
            'animation_metadata': result.get('metadata', {}),
            'message': f"Successfully generated {animation_type} animation"
        }
        
        logger.info(f"Successfully generated {animation_type} animation")
        return animation_result
        
    except (ValidationError, AnimationError, QuantumStateError) as e:
        logger.error(f"Animation generation failed: {e}")
        return {
            'success': False,
            'error': e.__class__.__name__,
            'message': str(e),
            'details': e.details if hasattr(e, 'details') else {}
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in animate_quantum_process: {e}")
        return {
            'success': False,
            'error': 'UnexpectedError',
            'message': f"An unexpected error occurred: {str(e)}",
            'details': {}
        }


async def prepare_animation_data(data_source: str, animation_type: str, config: MCPConfig) -> Dict[str, Any]:
    """
    Prepare data for animation generation.
    
    Parameters
    ----------
    data_source : str
        Data source specification
    animation_type : str
        Type of animation
    config : MCPConfig
        Server configuration
        
    Returns
    -------
    dict
        Prepared animation data
    """
    try:
        state_manager = get_state_manager(config)
        
        # Parse data source
        if data_source.startswith('state_id:'):
            # Single state reference
            state_id = data_source[9:]  # Remove 'state_id:' prefix
            
            if state_id not in state_manager.list_states():
                raise QuantumStateError(f"State with ID '{state_id}' not found")
            
            quantum_state = state_manager.get_state(state_id)
            state_info = state_manager.get_state_info(state_id)
            
            return {
                'type': 'single_state',
                'state': quantum_state,
                'state_info': state_info,
                'state_id': state_id
            }
            
        elif data_source.startswith('evolution_data:'):
            # Evolution data reference
            evolution_id = data_source[15:]  # Remove 'evolution_data:' prefix
            
            try:
                evolution_result = state_manager.get_evolution_result(evolution_id)
                
                return {
                    'type': 'evolution_data',
                    'evolution_id': evolution_id,
                    'evolution_type': evolution_result['evolution_type'],
                    'initial_state_id': evolution_result['initial_state_id'],
                    'parameters': evolution_result['parameters'],
                    'states': evolution_result['states'],
                    'times': evolution_result['times'],
                    'expectation_values': evolution_result['expectation_values'],
                    'solver_info': evolution_result['solver_info']
                }
                
            except Exception as e:
                raise QuantumStateError(f"Could not retrieve evolution data '{evolution_id}': {str(e)}")
            
        elif data_source.startswith('comparison:'):
            # State comparison data
            state_ids = data_source[11:].split(',')  # Remove 'comparison:' prefix
            
            states_data = {}
            for state_id in state_ids:
                state_id = state_id.strip()
                if state_id in state_manager.list_states():
                    states_data[state_id] = {
                        'state': state_manager.get_state(state_id),
                        'info': state_manager.get_state_info(state_id)
                    }
            
            return {
                'type': 'comparison',
                'states_data': states_data
            }
            
        elif data_source.startswith('circuit:'):
            # Quantum circuit data
            circuit_spec = data_source[8:]  # Remove 'circuit:' prefix
            
            # Parse circuit specification (simplified)
            try:
                circuit_data = json.loads(circuit_spec)
                return {
                    'type': 'circuit',
                    'circuit_data': circuit_data
                }
            except json.JSONDecodeError:
                raise ValidationError(f"Invalid circuit specification: {circuit_spec}")
            
        else:
            raise ValidationError(f"Unknown data source format: {data_source}")
            
    except Exception as e:
        raise AnimationError(
            f"Failed to prepare animation data: {str(e)}",
            animation_type=animation_type
        )


async def generate_bloch_evolution_animation(animation_data: Dict[str, Any], quality: str, output_format: str, 
                                           frame_rate: int, duration: float, view_config: Dict[str, Any], 
                                           config: MCPConfig) -> Dict[str, Any]:
    """Generate Bloch sphere evolution animation."""
    try:
        # Setup output directory
        output_dir = Path(config.output_directory) / "animations" / "bloch_evolution"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temporary scene file
        scene_file = output_dir / f"bloch_scene_{hash(str(animation_data)) % 10000}.py"
        
        # Generate Manim scene code
        scene_code = generate_bloch_scene_code(animation_data, view_config, duration)
        
        with open(scene_file, 'w') as f:
            f.write(scene_code)
        
        # Render animation using Manim CLI
        output_file = await render_manim_animation(
            scene_file, "BlochEvolutionScene", quality, output_format, frame_rate, output_dir
        )
        
        # Cleanup temporary files
        try:
            scene_file.unlink()
        except Exception:
            pass
        
        return {
            'output_files': [str(output_file)],
            'file_info': {
                'size_bytes': output_file.stat().st_size if output_file.exists() else 0,
                'format': output_format,
                'duration_seconds': duration,
                'frame_rate': frame_rate
            },
            'metadata': {
                'animation_type': 'bloch_evolution',
                'render_quality': quality,
                'quantum_state_info': animation_data.get('state_info', {})
            }
        }
        
    except Exception as e:
        raise AnimationError(
            f"Failed to generate Bloch evolution animation: {str(e)}",
            animation_type="bloch_evolution"
        )


async def generate_wigner_dynamics_animation(animation_data: Dict[str, Any], quality: str, output_format: str,
                                           frame_rate: int, duration: float, view_config: Dict[str, Any],
                                           config: MCPConfig) -> Dict[str, Any]:
    """Generate Wigner function dynamics animation."""
    try:
        output_dir = Path(config.output_directory) / "animations" / "wigner_dynamics"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        scene_file = output_dir / f"wigner_scene_{hash(str(animation_data)) % 10000}.py"
        scene_code = generate_wigner_scene_code(animation_data, view_config, duration)
        
        with open(scene_file, 'w') as f:
            f.write(scene_code)
        
        output_file = await render_manim_animation(
            scene_file, "WignerDynamicsScene", quality, output_format, frame_rate, output_dir
        )
        
        try:
            scene_file.unlink()
        except Exception:
            pass
        
        return {
            'output_files': [str(output_file)],
            'file_info': {
                'size_bytes': output_file.stat().st_size if output_file.exists() else 0,
                'format': output_format,
                'duration_seconds': duration,
                'frame_rate': frame_rate
            },
            'metadata': {
                'animation_type': 'wigner_dynamics',
                'render_quality': quality
            }
        }
        
    except Exception as e:
        raise AnimationError(
            f"Failed to generate Wigner dynamics animation: {str(e)}",
            animation_type="wigner_dynamics"
        )


async def generate_state_tomography_animation(animation_data: Dict[str, Any], quality: str, output_format: str,
                                            frame_rate: int, duration: float, view_config: Dict[str, Any],
                                            config: MCPConfig) -> Dict[str, Any]:
    """Generate state tomography animation."""
    try:
        output_dir = Path(config.output_directory) / "animations" / "state_tomography"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        scene_file = output_dir / f"tomography_scene_{hash(str(animation_data)) % 10000}.py"
        scene_code = generate_tomography_scene_code(animation_data, view_config, duration)
        
        with open(scene_file, 'w') as f:
            f.write(scene_code)
        
        output_file = await render_manim_animation(
            scene_file, "StateTomographyScene", quality, output_format, frame_rate, output_dir
        )
        
        try:
            scene_file.unlink()
        except Exception:
            pass
        
        return {
            'output_files': [str(output_file)],
            'file_info': {
                'size_bytes': output_file.stat().st_size if output_file.exists() else 0,
                'format': output_format,
                'duration_seconds': duration,
                'frame_rate': frame_rate
            },
            'metadata': {
                'animation_type': 'state_tomography',
                'render_quality': quality
            }
        }
        
    except Exception as e:
        raise AnimationError(
            f"Failed to generate state tomography animation: {str(e)}",
            animation_type="state_tomography"
        )


async def generate_circuit_execution_animation(animation_data: Dict[str, Any], quality: str, output_format: str,
                                             frame_rate: int, duration: float, view_config: Dict[str, Any],
                                             config: MCPConfig) -> Dict[str, Any]:
    """Generate quantum circuit execution animation."""
    try:
        output_dir = Path(config.output_directory) / "animations" / "circuit_execution"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        scene_file = output_dir / f"circuit_scene_{hash(str(animation_data)) % 10000}.py"
        scene_code = generate_circuit_scene_code(animation_data, view_config, duration)
        
        with open(scene_file, 'w') as f:
            f.write(scene_code)
        
        output_file = await render_manim_animation(
            scene_file, "CircuitExecutionScene", quality, output_format, frame_rate, output_dir
        )
        
        try:
            scene_file.unlink()
        except Exception:
            pass
        
        return {
            'output_files': [str(output_file)],
            'file_info': {
                'size_bytes': output_file.stat().st_size if output_file.exists() else 0,
                'format': output_format,
                'duration_seconds': duration,
                'frame_rate': frame_rate
            },
            'metadata': {
                'animation_type': 'circuit_execution',
                'render_quality': quality
            }
        }
        
    except Exception as e:
        raise AnimationError(
            f"Failed to generate circuit execution animation: {str(e)}",
            animation_type="circuit_execution"
        )


async def generate_energy_levels_animation(animation_data: Dict[str, Any], quality: str, output_format: str,
                                         frame_rate: int, duration: float, view_config: Dict[str, Any],
                                         config: MCPConfig) -> Dict[str, Any]:
    """Generate energy levels animation."""
    try:
        output_dir = Path(config.output_directory) / "animations" / "energy_levels"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        scene_file = output_dir / f"energy_scene_{hash(str(animation_data)) % 10000}.py"
        scene_code = generate_energy_scene_code(animation_data, view_config, duration)
        
        with open(scene_file, 'w') as f:
            f.write(scene_code)
        
        output_file = await render_manim_animation(
            scene_file, "EnergyLevelsScene", quality, output_format, frame_rate, output_dir
        )
        
        try:
            scene_file.unlink()
        except Exception:
            pass
        
        return {
            'output_files': [str(output_file)],
            'file_info': {
                'size_bytes': output_file.stat().st_size if output_file.exists() else 0,
                'format': output_format,
                'duration_seconds': duration,
                'frame_rate': frame_rate
            },
            'metadata': {
                'animation_type': 'energy_levels',
                'render_quality': quality
            }
        }
        
    except Exception as e:
        raise AnimationError(
            f"Failed to generate energy levels animation: {str(e)}",
            animation_type="energy_levels"
        )


async def generate_photon_statistics_animation(animation_data: Dict[str, Any], quality: str, output_format: str,
                                             frame_rate: int, duration: float, view_config: Dict[str, Any],
                                             config: MCPConfig) -> Dict[str, Any]:
    """Generate photon statistics animation."""
    try:
        output_dir = Path(config.output_directory) / "animations" / "photon_statistics"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        scene_file = output_dir / f"photon_scene_{hash(str(animation_data)) % 10000}.py"
        scene_code = generate_photon_scene_code(animation_data, view_config, duration)
        
        with open(scene_file, 'w') as f:
            f.write(scene_code)
        
        output_file = await render_manim_animation(
            scene_file, "PhotonStatisticsScene", quality, output_format, frame_rate, output_dir
        )
        
        try:
            scene_file.unlink()
        except Exception:
            pass
        
        return {
            'output_files': [str(output_file)],
            'file_info': {
                'size_bytes': output_file.stat().st_size if output_file.exists() else 0,
                'format': output_format,
                'duration_seconds': duration,
                'frame_rate': frame_rate
            },
            'metadata': {
                'animation_type': 'photon_statistics',
                'render_quality': quality
            }
        }
        
    except Exception as e:
        raise AnimationError(
            f"Failed to generate photon statistics animation: {str(e)}",
            animation_type="photon_statistics"
        )


def generate_bloch_scene_code(animation_data: Dict[str, Any], view_config: Dict[str, Any], duration: float) -> str:
    """Generate Manim scene code for Bloch sphere animation."""
    return f'''
from manim import *
import numpy as np
import sys
import os

# Add the psianimator_mcp package to path
sys.path.insert(0, os.path.abspath('.'))

from src.psianimator_mcp.animation.bloch_sphere import BlochSphere3D

class BlochEvolutionScene(BlochSphere3D):
    def construct(self):
        # Setup Bloch sphere
        bloch_sphere = self.construct_bloch_sphere()
        self.add(bloch_sphere)
        
        # Create a simple evolution (example)
        from qutip import basis, sigmax
        initial_state = basis(2, 0)  # |0⟩ state
        
        # Simulate evolution under X rotation
        import numpy as np
        times = np.linspace(0, 2*np.pi, 60)
        evolved_states = []
        
        for t in times:
            U = (-1j * sigmax() * t / 2).expm()
            evolved_state = U * initial_state
            evolved_states.append(evolved_state)
        
        # Animate evolution
        self.animate_state_evolution(initial_state, evolved_states)
        
        self.wait(1)
'''


def generate_wigner_scene_code(animation_data: Dict[str, Any], view_config: Dict[str, Any], duration: float) -> str:
    """Generate Manim scene code for Wigner function animation."""
    return f'''
from manim import *
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

from src.psianimator_mcp.animation.wigner_function import WignerFunction

class WignerDynamicsScene(WignerFunction):
    def construct(self):
        # Setup phase space
        axes = self.create_phase_space_axes()
        self.add(axes)
        
        # Create example Wigner evolution
        from qutip import coherent, destroy
        initial_state = coherent(20, 2.0)  # Coherent state
        
        # Simulate free evolution
        import numpy as np
        times = np.linspace(0, 2*np.pi, 30)
        evolved_states = []
        
        for t in times:
            # Simple rotation in phase space
            alpha = 2.0 * np.exp(1j * t)
            evolved_state = coherent(20, alpha)
            evolved_states.append(evolved_state)
        
        # Animate Wigner function evolution
        self.animate_wigner_evolution(initial_state, evolved_states)
        
        self.wait(1)
'''


def generate_tomography_scene_code(animation_data: Dict[str, Any], view_config: Dict[str, Any], duration: float) -> str:
    """Generate Manim scene code for state tomography animation."""
    return f'''
from manim import *
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

from src.psianimator_mcp.animation.state_tomography import StateTomography

class StateTomographyScene(StateTomography):
    def construct(self):
        # Create example density matrix
        from qutip import basis
        state = (basis(2, 0) + basis(2, 1)).unit()
        rho = state * state.dag()
        
        # Create density matrix visualization
        matrix_viz = self.create_density_matrix_visualization(rho)
        self.add(matrix_viz)
        
        # Add Pauli decomposition
        pauli_decomp = self.create_pauli_decomposition(rho, position=DOWN*3)
        self.add(pauli_decomp)
        
        self.wait({duration})
'''


def generate_circuit_scene_code(animation_data: Dict[str, Any], view_config: Dict[str, Any], duration: float) -> str:
    """Generate Manim scene code for quantum circuit animation."""
    return f'''
from manim import *
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

from src.psianimator_mcp.animation.quantum_circuit import QuantumCircuit

class CircuitExecutionScene(QuantumCircuit):
    def construct(self):
        # Create example quantum circuit
        from qutip import basis
        initial_state = basis(4, 0)  # |00⟩ state for 2 qubits
        
        # Define gate sequence
        gate_sequence = [
            {{"name": "H", "qubits": [0]}},
            {{"name": "CNOT", "qubits": [0, 1]}},
            {{"name": "Z", "qubits": [1]}}
        ]
        
        # Animate circuit execution
        self.animate_circuit_execution(initial_state, gate_sequence)
        
        self.wait({duration})
'''


def generate_energy_scene_code(animation_data: Dict[str, Any], view_config: Dict[str, Any], duration: float) -> str:
    """Generate Manim scene code for energy levels animation."""
    return f'''
from manim import *
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

from src.psianimator_mcp.animation.energy_levels import EnergyLevelDiagram

class EnergyLevelsScene(EnergyLevelDiagram):
    def construct(self):
        # Create example energy level system
        energy_levels = [0.0, 1.0, 2.5]  # Three-level system
        
        # Create energy level diagram
        for i, energy in enumerate(energy_levels):
            level = self.create_energy_level(energy, label=f"|{{i}}\\\\rangle")
            self.add(level)
        
        # Add transitions
        transition_01 = self.create_transition_arrow(0.0, 1.0, "allowed")
        transition_12 = self.create_transition_arrow(1.0, 2.5, "allowed")
        self.add(transition_01, transition_12)
        
        # Simulate population dynamics
        time_points = np.linspace(0, 10, 50)
        population_evolution = []
        
        for t in time_points:
            # Example oscillating populations
            p0 = 0.5 * (1 + np.cos(t))
            p1 = 0.3 * (1 + np.sin(t))
            p2 = 1 - p0 - p1
            population_evolution.append([max(0, p0), max(0, p1), max(0, p2)])
        
        # Animate population dynamics
        self.animate_population_dynamics(energy_levels, population_evolution, time_points)
        
        self.wait(1)
'''


def generate_photon_scene_code(animation_data: Dict[str, Any], view_config: Dict[str, Any], duration: float) -> str:
    """Generate Manim scene code for photon statistics animation."""
    return f'''
from manim import *
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

from src.psianimator_mcp.animation.photon_statistics import PhotonStatistics

class PhotonStatisticsScene(PhotonStatistics):
    def construct(self):
        # Create example quantum states
        from qutip import coherent, thermal_dm, fock
        
        states_dict = {{
            "coherent": coherent(15, 2.0),
            "thermal": thermal_dm(15, 3.0),
            "fock": fock(15, 5)
        }}
        
        # Animate photon statistics comparison
        self.animate_photon_statistics_comparison(states_dict, duration={duration})
        
        self.wait(1)
'''


async def render_manim_animation(scene_file: Path, scene_class: str, quality: str, output_format: str, 
                               frame_rate: int, output_dir: Path) -> Path:
    """
    Render Manim animation using subprocess.
    
    Parameters
    ----------
    scene_file : Path
        Path to scene file
    scene_class : str
        Name of scene class to render
    quality : str
        Render quality
    output_format : str
        Output format
    frame_rate : int
        Frame rate
    output_dir : Path
        Output directory
        
    Returns
    -------
    Path
        Path to rendered file
    """
    import subprocess
    
    try:
        # Map quality to Manim quality flags
        quality_flags = {
            'low': '-ql',
            'medium': '-qm', 
            'high': '-qh',
            'production': '-qp'
        }
        
        quality_flag = quality_flags.get(quality, '-qm')
        
        # Construct Manim command
        cmd = [
            'manim',
            quality_flag,
            '--format', output_format,
            '--fps', str(frame_rate),
            '--output_file', f"{scene_class.lower()}_{quality}.{output_format}",
            str(scene_file),
            scene_class
        ]
        
        # Run Manim
        result = subprocess.run(
            cmd,
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            raise AnimationError(
                f"Manim rendering failed: {result.stderr}",
                render_settings={'quality': quality, 'format': output_format}
            )
        
        # Find output file
        output_file = output_dir / f"{scene_class.lower()}_{quality}.{output_format}"
        
        if not output_file.exists():
            # Try to find any generated file
            for file in output_dir.iterdir():
                if file.suffix[1:] == output_format:
                    output_file = file
                    break
        
        if not output_file.exists():
            raise AnimationError(
                "Rendered animation file not found",
                render_settings={'expected_file': str(output_file)}
            )
        
        return output_file
        
    except subprocess.TimeoutExpired:
        raise AnimationError(
            "Manim rendering timed out",
            render_settings={'timeout': 300}
        )
    except Exception as e:
        raise AnimationError(
            f"Failed to render Manim animation: {str(e)}",
            render_settings={'scene_file': str(scene_file), 'scene_class': scene_class}
        )