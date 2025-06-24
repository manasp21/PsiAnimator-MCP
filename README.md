# PsiAnimator-MCP

**Quantum Physics Simulation and Animation Server**

A Model Context Protocol (MCP) server that integrates [QuTip](https://qutip.org/) (Quantum Toolbox in Python) for quantum physics computations with [Manim](https://www.manim.community/) (Mathematical Animation Engine) for visualization.

## Features

- 🔬 **Comprehensive Quantum Physics**: State management, time evolution, measurements
- 🎬 **Beautiful Animations**: Publication-quality visualizations using Manim  
- 🔌 **MCP Integration**: Easy integration with MCP-compatible clients
- 🧮 **Scientific Computing**: Built on NumPy, SciPy, and QuTip
- 📊 **Interactive Visualizations**: Bloch spheres, Wigner functions, state tomography
- 🎓 **Educational Tools**: Perfect for quantum mechanics education and research

## Quick Start

### Installation

```bash
pip install psianimator-mcp
```

### Development Installation

```bash
git clone https://github.com/psianimator/psianimator-mcp.git
cd psianimator-mcp
pip install -e ".[dev]"
```

### Usage

Start the MCP server:

```bash
psianimator-mcp
```

## MCP Tools

PsiAnimator-MCP provides six core MCP tools:

1. **`create_quantum_state`** - Create pure/mixed quantum states
2. **`evolve_quantum_system`** - Time evolution with various solvers  
3. **`measure_observable`** - Quantum measurements and expectation values
4. **`animate_quantum_process`** - Generate Manim animations
5. **`quantum_gate_sequence`** - Apply gate sequences with visualization
6. **`calculate_entanglement`** - Compute entanglement measures

## Requirements

- Python ≥ 3.9
- QuTip ≥ 4.7.0
- Manim ≥ 0.18.0  
- MCP ≥ 1.0.0

## Documentation

Full documentation is available at [psianimator-mcp.readthedocs.io](https://psianimator-mcp.readthedocs.io)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.