"""
MCP tools implementation

Contains all the MCP tool functions that provide the external interface  
for quantum physics computations and animations.
"""

# Core tools (always available)
from .quantum_state_tools import create_quantum_state
from .evolution_tools import evolve_quantum_system
from .measurement_tools import measure_observable  
from .gate_tools import quantum_gate_sequence
from .entanglement_tools import calculate_entanglement

# Optional animation tools
try:
    from .animation_tools import animate_quantum_process
    _ANIMATION_TOOLS_AVAILABLE = True
except ImportError:
    animate_quantum_process = None
    _ANIMATION_TOOLS_AVAILABLE = False

# Core exports
__all__ = [
    "create_quantum_state",
    "evolve_quantum_system",
    "measure_observable",
    "quantum_gate_sequence",
    "calculate_entanglement"
]

# Add animation exports if available
if _ANIMATION_TOOLS_AVAILABLE:
    __all__.append("animate_quantum_process")