"""
PsiAnimator-MCP: Quantum Physics Simulation and Animation Server

A Model Context Protocol (MCP) server that integrates QuTip (Quantum Toolbox in Python) 
for quantum physics computations with Manim (Mathematical Animation Engine) for visualization.

Provides comprehensive quantum physics simulation and animation framework accessible 
through the MCP protocol.
"""

__version__ = "0.1.0"
__author__ = "PsiAnimator Development Team"
__email__ = "contact@psianimator.dev"

# Core imports (always available)
from .server import MCPServer

# Optional quantum imports (only if scientific packages available)
try:
    from .quantum import QuantumStateManager, QuantumOperations, QuantumSystems
    from .tools import (
        create_quantum_state,
        evolve_quantum_system, 
        measure_observable,
        quantum_gate_sequence,
        calculate_entanglement
    )
    _QUANTUM_AVAILABLE = True
except ImportError:
    # Scientific dependencies not available
    QuantumStateManager = None
    QuantumOperations = None
    QuantumSystems = None
    create_quantum_state = None
    evolve_quantum_system = None
    measure_observable = None
    quantum_gate_sequence = None
    calculate_entanglement = None
    _QUANTUM_AVAILABLE = False

# Optional animation imports (only if manim is available)
try:
    from .animation import QuantumScene, BlochSphere3D, StateTomography
    from .tools import animate_quantum_process
    _ANIMATION_AVAILABLE = True
except ImportError:
    # Animation dependencies not available
    QuantumScene = None
    BlochSphere3D = None
    StateTomography = None
    animate_quantum_process = None
    _ANIMATION_AVAILABLE = False

# Core exports (always available)
__all__ = [
    "MCPServer",
]

# Add quantum exports if available
if _QUANTUM_AVAILABLE:
    __all__.extend([
        "QuantumStateManager", 
        "QuantumOperations",
        "QuantumSystems",
        "create_quantum_state",
        "evolve_quantum_system",
        "measure_observable", 
        "quantum_gate_sequence",
        "calculate_entanglement",
    ])

# Add animation exports if available
if _ANIMATION_AVAILABLE:
    __all__.extend([
        "QuantumScene",
        "BlochSphere3D", 
        "StateTomography",
        "animate_quantum_process",
    ])

# Add utility functions
__all__.extend(["is_animation_available", "is_quantum_available"])

def is_animation_available() -> bool:
    """Check if animation functionality (manim) is available."""
    return _ANIMATION_AVAILABLE

def is_quantum_available() -> bool:
    """Check if quantum functionality (numpy, qutip) is available."""
    return _QUANTUM_AVAILABLE