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

from .server import MCPServer
from .quantum import QuantumStateManager, QuantumOperations, QuantumSystems
from .animation import QuantumScene, BlochSphere3D, StateTomography
from .tools import (
    create_quantum_state,
    evolve_quantum_system, 
    measure_observable,
    animate_quantum_process,
    quantum_gate_sequence,
    calculate_entanglement
)

__all__ = [
    "MCPServer",
    "QuantumStateManager", 
    "QuantumOperations",
    "QuantumSystems",
    "QuantumScene",
    "BlochSphere3D",
    "StateTomography",
    "create_quantum_state",
    "evolve_quantum_system",
    "measure_observable", 
    "animate_quantum_process",
    "quantum_gate_sequence",
    "calculate_entanglement"
]