# Changelog

All notable changes to PsiAnimator-MCP will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial implementation of PsiAnimator-MCP server
- Complete quantum physics simulation engine using QuTip
- Manim integration for quantum visualizations
- Six core MCP tools for quantum operations
- Cross-platform installation scripts
- Comprehensive configuration system
- CLI with multiple commands and environment variable support
- GitHub Actions CI/CD pipeline
- Automated testing across Python 3.9-3.12
- Security scanning and code quality checks

### Features

#### MCP Tools
- `create_quantum_state` - Create pure/mixed quantum states
- `evolve_quantum_system` - Time evolution with multiple solvers
- `measure_observable` - Quantum measurements and expectation values
- `animate_quantum_process` - Generate Manim animations
- `quantum_gate_sequence` - Apply gate sequences with visualization
- `calculate_entanglement` - Compute entanglement measures

#### Quantum Physics Engine
- Support for pure and mixed quantum states
- Multiple time evolution methods (unitary, master equation, Monte Carlo, stochastic)
- Comprehensive measurement capabilities
- Entanglement analysis tools
- State validation and normalization

#### Animation Framework
- Bloch sphere 3D visualizations
- Wigner function dynamics
- State tomography visualizations
- Quantum circuit animations
- Energy level diagrams
- Photon statistics plots

#### Installation & Distribution
- Cross-platform installation scripts (Unix/macOS/Windows)
- Post-installation configuration setup
- Claude Desktop integration helpers
- PyPI distribution readiness
- Multiple installation methods

#### CLI & Configuration
- Rich CLI with multiple commands
- Environment variable support
- Configuration file management
- Built-in testing and validation
- Claude Desktop configuration generation

## [0.1.0] - 2024-XX-XX

### Added
- Initial release of PsiAnimator-MCP
- Complete MCP server implementation
- Quantum physics simulation capabilities
- Manim animation integration
- Production-ready deployment configuration

---

## Release Notes

### Version 0.1.0

This is the initial release of PsiAnimator-MCP, a comprehensive quantum physics simulation and animation server built on the Model Context Protocol (MCP). 

**Key Features:**
- Full quantum mechanics simulation using QuTip
- Beautiful animations with Manim
- Easy integration with Claude Desktop
- Cross-platform installation
- Production-ready deployment

**System Requirements:**
- Python ≥ 3.9
- Optional: LaTeX for advanced Manim features
- Optional: FFmpeg for video generation

**Installation:**
```bash
pip install psianimator-mcp
```

For development installation and examples, see the [README](README.md).

---

*Note: This changelog is automatically updated by our CI/CD pipeline during releases.*