# Contributing to PsiAnimator-MCP

Thank you for your interest in contributing to PsiAnimator-MCP! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/psianimator/psianimator-mcp.git
   cd psianimator-mcp
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Code Standards

### Code Style
- Use **Black** for code formatting
- Use **isort** for import sorting
- Follow **PEP 8** style guidelines
- Use **type hints** throughout the codebase

### Documentation
- Use **NumPy-style docstrings** with LaTeX math notation
- Include parameter descriptions and return values
- Add usage examples for complex functions
- Mathematical formulas should use proper LaTeX notation

### Testing
- Write **comprehensive unit tests** for all new functionality
- Include **integration tests** for MCP tools
- Use **pytest** with async support
- Validate quantum physics calculations against analytical results

## Quantum Physics Standards

### Accuracy Requirements
- **Proper normalization** enforcement for quantum states
- **Hermiticity checks** for observables and Hamiltonians  
- **Unitarity verification** for time evolution operators
- **Trace preservation** for quantum channels
- **Correct tensor products** for composite systems

### Numerical Precision
- Use `quantum_precision = 1e-12` for numerical tolerances
- Handle **numerical instabilities** in quantum calculations
- Provide **error bounds** for approximate calculations

## MCP Integration

### Tool Implementation
- Follow **MCP protocol specifications** exactly
- Provide **comprehensive input validation**
- Return **structured error responses** with details
- Include **usage examples** in docstrings

### Error Handling
- Use **custom exception hierarchy** for quantum-specific errors
- Provide **detailed error messages** with context
- Include **suggestions for fixing** common errors

## Animation Guidelines

### Manim Integration
- Use **QuantumScene** base class for quantum visualizations
- Ensure **publication-quality** output
- Optimize for **real-time parameter updates**
- Support **multiple output formats** (MP4, GIF, WebM)

### Visualization Standards
- Use **consistent color schemes** for quantum elements
- Provide **accessibility options** (colorblind-friendly palettes)
- Include **proper scaling** for quantum amplitudes
- Add **informative labels** and legends

## Submission Process

### Before Submitting
1. **Run all tests**: `pytest tests/`
2. **Check code style**: `black --check src/ tests/`
3. **Type checking**: `mypy src/`
4. **Run pre-commit**: `pre-commit run --all-files`

### Pull Request Guidelines
- **Clear description** of changes and motivation
- **Reference related issues** if applicable
- **Include tests** for new functionality
- **Update documentation** as needed
- **Add examples** for new features

### Commit Messages
- Use **clear, descriptive** commit messages
- Follow **conventional commits** format when possible
- Reference **issue numbers** in commit messages

## Code Review Process

### Review Criteria
- **Correctness** of quantum physics implementations
- **Code quality** and maintainability
- **Test coverage** and quality
- **Documentation** completeness
- **Performance** considerations

### Quantum Physics Review
- **Mathematical accuracy** of quantum mechanical calculations
- **Physical interpretation** of results
- **Proper handling** of quantum mechanical principles
- **Validation** against known analytical results

## Areas for Contribution

### High Priority
- **Additional quantum systems** (cavity QED, quantum optics)
- **Advanced entanglement measures** (quantum discord, etc.)
- **Optimization algorithms** for quantum control
- **Educational content** generation

### Medium Priority  
- **Performance optimizations** for large Hilbert spaces
- **Additional animation types** for quantum processes
- **Integration** with other quantum software
- **Documentation** improvements and tutorials

### Research Features
- **Quantum error correction** visualization
- **Multi-qubit gate** optimization
- **Quantum machine learning** algorithms
- **Open quantum systems** advanced models

## Getting Help

- **Issues**: Report bugs and request features via GitHub issues
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Documentation**: Check API reference and examples
- **Code Review**: Ask for help in pull request comments

## Recognition

Contributors will be acknowledged in:
- **README.md** contributors section
- **Release notes** for significant contributions
- **Academic publications** using PsiAnimator-MCP (when appropriate)

## License

By contributing to PsiAnimator-MCP, you agree that your contributions will be licensed under the MIT License.

Thank you for helping make quantum physics visualization and simulation more accessible!