# PsiAnimator-MCP

**Quantum Physics Simulation and Animation Server**

A Model Context Protocol (MCP) server that integrates [QuTip](https://qutip.org/) (Quantum Toolbox in Python) for quantum physics computations with [Manim](https://www.manim.community/) (Mathematical Animation Engine) for visualization.

## Features

- 🔬 **Quantum Physics Engine**: Complete state management, time evolution, and measurement tools
- 🎬 **Manim Animations**: Publication-quality visualizations with quantum-specific scenes
- 🔌 **MCP Integration**: Seamless integration with MCP-compatible clients
- 🧮 **Scientific Computing**: Built on NumPy, SciPy, and QuTip for accuracy
- 📊 **Visualization Types**: Bloch spheres, Wigner functions, state tomography, circuits
- 🎓 **Educational Focus**: Perfect for quantum mechanics education and research

## Installation

### Quick Install

#### Option 1: One-line install (Unix/macOS)
```bash
curl -fsSL https://raw.githubusercontent.com/username/PsiAnimator-MCP/main/scripts/install.sh | bash
```

#### Option 2: PowerShell (Windows)
```powershell
iwr https://raw.githubusercontent.com/username/PsiAnimator-MCP/main/scripts/install.ps1 | iex
```

#### Option 3: pip (when available on PyPI)
```bash
pip install psianimator-mcp
```

#### Option 4: From source
```bash
git clone https://github.com/username/PsiAnimator-MCP.git
cd PsiAnimator-MCP
./scripts/install.sh --from-source
```

### Prerequisites

- Python ≥ 3.10
- Git (for development installation)
- LaTeX (optional, for advanced Manim features)
- FFmpeg (optional, for video generation)

### Manual Installation

For development or custom setups:

```bash
git clone https://github.com/username/PsiAnimator-MCP.git
cd PsiAnimator-MCP
pip install -e ".[dev]"
```

### Dependencies

Core dependencies are automatically installed:
- QuTip ≥ 4.7.0 (quantum physics)
- Manim ≥ 0.18.0 (animations)
- MCP ≥ 1.0.0 (protocol)
- NumPy, SciPy, matplotlib (scientific computing)

### Post-Installation Setup

After installation, run the setup command:
```bash
psianimator-mcp setup
```

This will:
- Create configuration directory (`~/.config/psianimator-mcp/`)
- Copy example configuration file
- Provide Claude Desktop integration instructions

## Claude Desktop Integration

### Automatic Configuration

Generate Claude Desktop configuration:
```bash
psianimator-mcp claude-config
```

### Manual Configuration

Add to your Claude Desktop configuration file:

**Windows:** `%USERPROFILE%\AppData\Roaming\Claude\claude_desktop_config.json`
**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Linux:** `~/.config/claude-desktop/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "psianimator-mcp": {
      "command": "python3",
      "args": ["-m", "psianimator_mcp.cli", "serve"],
      "env": {
        "PSIANIMATOR_CONFIG": "~/.config/psianimator-mcp/config.json"
      }
    }
  }
}
```

**Note:** Restart Claude Desktop after configuration changes.

## Quick Start

### 1. Start the Server

**Default (serves via MCP protocol):**
```bash
psianimator-mcp
```

**Stdio transport explicitly:**
```bash
psianimator-mcp serve --transport stdio
```

**WebSocket transport:**
```bash
psianimator-mcp serve --transport websocket --port 3000
```

### 2. Test Installation

```bash
psianimator-mcp test
```

### 3. Basic Usage Example

```python
import asyncio
from psianimator_mcp.tools.quantum_state_tools import create_quantum_state
from psianimator_mcp.tools.measurement_tools import measure_observable
from psianimator_mcp.server.config import MCPConfig

async def basic_example():
    config = MCPConfig()
    
    # Create a qubit in |0⟩ state
    result = await create_quantum_state({
        'state_type': 'pure',
        'system_dims': [2],
        'parameters': {'state_indices': [0]},
        'basis': 'computational'
    }, config)
    
    state_id = result['state_id']
    
    # Measure ⟨σz⟩
    measurement = await measure_observable({
        'state_id': state_id,
        'observable': 'sigmaz',
        'measurement_type': 'expectation'
    }, config)
    
    print(f"⟨σz⟩ = {measurement['measurement_results']['expectation_value']}")

asyncio.run(basic_example())
```

## MCP Tools

### 1. `create_quantum_state`
Create quantum states of various types:
- **Pure states**: |ψ⟩ (ket vectors)
- **Mixed states**: ρ (density matrices)
- **Coherent states**: |α⟩ (harmonic oscillator)
- **Squeezed states**: reduced uncertainty
- **Thermal states**: finite temperature
- **Fock states**: definite photon number

### 2. `evolve_quantum_system`
Time evolution with multiple methods:
- **Unitary**: Schrödinger equation (closed systems)
- **Master equation**: Lindblad form (open systems)
- **Monte Carlo**: Quantum trajectories
- **Stochastic**: Continuous measurement

### 3. `measure_observable`
Quantum measurements and analysis:
- **Expectation values**: ⟨O⟩
- **Variances**: Δ²O
- **Probability distributions**: P(outcome)
- **Correlation functions**: ⟨A⟩⟨B⟩

### 4. `animate_quantum_process`
Generate Manim animations:
- **Bloch sphere evolution**: Qubit dynamics
- **Wigner functions**: Phase space representation
- **State tomography**: Density matrix visualization
- **Circuit execution**: Gate sequence animation
- **Energy levels**: Population dynamics

### 5. `quantum_gate_sequence`
Apply quantum gates with visualization:
- **Single-qubit gates**: Pauli, Hadamard, rotations
- **Two-qubit gates**: CNOT, CZ, SWAP
- **Parameterized gates**: RX, RY, RZ with custom angles
- **Circuit visualization**: Step-by-step animation

### 6. `calculate_entanglement`
Compute entanglement measures:
- **Von Neumann entropy**: S(ρ) = -Tr(ρ log ρ)
- **Concurrence**: Two-qubit entanglement measure
- **Negativity**: Partial transpose criterion
- **Mutual information**: I(A:B)

## Configuration

Configure via environment variables or `MCPConfig`:

```python
from psianimator_mcp.server.config import MCPConfig

config = MCPConfig(
    quantum_precision=1e-12,
    max_hilbert_dimension=1024,
    animation_cache_size=100,
    output_directory="./output",
    render_backend="cairo"
)
```

### Environment Variables

Configure PsiAnimator-MCP via environment variables:

**Server Configuration:**
- `PSIANIMATOR_CONFIG` - Path to configuration file
- `PSIANIMATOR_TRANSPORT` - Transport protocol (stdio/websocket)
- `PSIANIMATOR_HOST` - Host for WebSocket transport
- `PSIANIMATOR_PORT` - Port for WebSocket transport

**Quantum Settings:**
- `PSIANIMATOR_QUANTUM_PRECISION` - Quantum computation precision
- `PSIANIMATOR_MAX_HILBERT_DIM` - Maximum Hilbert space dimension
- `PSIANIMATOR_OUTPUT_DIR` - Output directory for animations

Example:
```bash
export PSIANIMATOR_TRANSPORT=websocket
export PSIANIMATOR_PORT=3001
psianimator-mcp
```

## CLI Commands

PsiAnimator-MCP provides several CLI commands:

```bash
psianimator-mcp                    # Start server (default: stdio)
psianimator-mcp serve              # Start server with options
psianimator-mcp config             # Show current configuration
psianimator-mcp setup              # Run post-installation setup
psianimator-mcp test               # Test installation
psianimator-mcp claude-config      # Generate Claude Desktop config
psianimator-mcp examples           # Show usage examples
psianimator-mcp version            # Show version
psianimator-mcp --help             # Show help
```

### Command Examples

**Start with custom config:**
```bash
psianimator-mcp serve --config /path/to/config.json
```

**WebSocket mode:**
```bash
psianimator-mcp serve --transport websocket --host 0.0.0.0 --port 8080
```

**Verbose logging:**
```bash
psianimator-mcp serve -vvv
```

## Examples

Comprehensive examples are provided in the `examples/` directory:

- `basic_usage.py` - Core functionality walkthrough
- Bell state creation and entanglement analysis
- Harmonic oscillator coherent state evolution
- Multi-qubit quantum circuits

Run examples:
```bash
python examples/basic_usage.py
```

## Development

### Setup Development Environment

```bash
git clone https://github.com/username/PsiAnimator-MCP.git
cd PsiAnimator-MCP
pip install -e ".[dev]"
pre-commit install
```

### Run Tests

```bash
pytest tests/
```

### Code Quality

```bash
black src/ tests/
isort src/ tests/
mypy src/
```

## Architecture

```
PsiAnimator-MCP/
├── src/psianimator_mcp/
│   ├── server/          # MCP server implementation
│   ├── quantum/         # Quantum physics engine
│   ├── animation/       # Manim visualization components
│   └── tools/           # MCP tool implementations
├── tests/               # Comprehensive test suite
├── examples/            # Usage examples
└── docs/               # Documentation
```

## Limitations

- Animation rendering requires sufficient system resources
- Large Hilbert spaces (>1024 dimensions) may impact performance
- Some advanced quantum error correction features are not yet implemented

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development guidelines
- Code standards
- Testing requirements
- Pull request process

## Support

- **Documentation**: See `docs/API_REFERENCE.md`
- **Examples**: Check `examples/` directory
- **Issues**: Report bugs via GitHub issues