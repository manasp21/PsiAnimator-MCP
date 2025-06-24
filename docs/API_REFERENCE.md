# PsiAnimator-MCP API Reference

## MCP Tools

PsiAnimator-MCP provides six core MCP tools for quantum physics simulations and animations.

### 1. create_quantum_state

Creates quantum states of various types.

**Input Schema:**
```json
{
  "state_type": "pure|mixed|coherent|squeezed|thermal|fock",
  "system_dims": [2, 2, ...],
  "parameters": {},
  "basis": "computational|fock|spin|position",
  "state_id": "optional_custom_id"
}
```

**Examples:**

Create a qubit in |0⟩ state:
```json
{
  "state_type": "pure",
  "system_dims": [2],
  "parameters": {"state_indices": [0]},
  "basis": "computational"
}
```

Create a coherent state:
```json
{
  "state_type": "coherent", 
  "system_dims": [20],
  "parameters": {"alpha": 2.0}
}
```

Create a Bell state:
```json
{
  "state_type": "pure",
  "system_dims": [2, 2],
  "parameters": {"coefficients": [0.707, 0, 0, 0.707]},
  "basis": "computational"
}
```

### 2. evolve_quantum_system

Performs time evolution of quantum systems.

**Input Schema:**
```json
{
  "state_id": "state_identifier",
  "hamiltonian": "hamiltonian_specification",
  "evolution_type": "unitary|master|monte_carlo|stochastic",
  "time_span": [t_start, t_end, n_steps],
  "collapse_operators": ["operator_specs"],
  "solver_options": {}
}
```

**Examples:**

Unitary evolution under Pauli-X:
```json
{
  "state_id": "my_qubit",
  "hamiltonian": "sigmax",
  "evolution_type": "unitary",
  "time_span": [0, 3.14159, 50]
}
```

Master equation with damping:
```json
{
  "state_id": "my_qubit",
  "hamiltonian": "sigmaz",
  "evolution_type": "master", 
  "time_span": [0, 10, 100],
  "collapse_operators": ["sigmam"]
}
```

### 3. measure_observable

Performs quantum measurements and calculates expectation values.

**Input Schema:**
```json
{
  "state_id": "state_identifier",
  "observable": "observable_specification",
  "measurement_type": "expectation|variance|probability|correlation",
  "measurement_basis": "optional_basis",
  "n_measurements": 1
}
```

**Examples:**

Expectation value measurement:
```json
{
  "state_id": "my_qubit",
  "observable": "sigmaz",
  "measurement_type": "expectation"
}
```

Probability distribution:
```json
{
  "state_id": "my_qubit", 
  "observable": "sigmaz",
  "measurement_type": "probability"
}
```

Photon number statistics:
```json
{
  "state_id": "coherent_state",
  "observable": "num",
  "measurement_type": "variance"
}
```

### 4. animate_quantum_process

Generates Manim animations of quantum processes.

**Input Schema:**
```json
{
  "animation_type": "bloch_evolution|wigner_dynamics|state_tomography|circuit_execution|energy_levels|photon_statistics",
  "data_source": "data_source_specification", 
  "render_quality": "low|medium|high|production",
  "output_format": "mp4|gif|webm",
  "frame_rate": 30,
  "duration": 5.0,
  "view_config": {}
}
```

**Examples:**

Bloch sphere evolution:
```json
{
  "animation_type": "bloch_evolution",
  "data_source": "state_id:my_qubit",
  "render_quality": "medium",
  "output_format": "mp4",
  "duration": 5.0
}
```

Wigner function dynamics:
```json
{
  "animation_type": "wigner_dynamics",
  "data_source": "state_id:coherent_state",
  "render_quality": "high",
  "output_format": "gif"
}
```

### 5. quantum_gate_sequence

Applies sequences of quantum gates with visualization.

**Input Schema:**
```json
{
  "state_id": "state_identifier",
  "gates": [
    {
      "name": "gate_name",
      "qubits": [0, 1],
      "parameters": {}
    }
  ],
  "animate_steps": false,
  "show_intermediate_states": true
}
```

**Examples:**

Bell state preparation:
```json
{
  "state_id": "two_qubit_state",
  "gates": [
    {"name": "H", "qubits": [0]},
    {"name": "CNOT", "qubits": [0, 1]}
  ],
  "show_intermediate_states": true
}
```

Parameterized rotation:
```json
{
  "state_id": "my_qubit",
  "gates": [
    {
      "name": "RY", 
      "qubits": [0],
      "parameters": {"angle": 1.5708}
    }
  ]
}
```

### 6. calculate_entanglement

Computes entanglement measures and correlations.

**Input Schema:**
```json
{
  "state_id": "state_identifier",
  "measure_type": "von_neumann|linear_entropy|concurrence|negativity|mutual_information",
  "subsystem_partition": [[0], [1, 2]],
  "visualize_correlations": false,
  "detect_entanglement": true
}
```

**Examples:**

Concurrence for two-qubit state:
```json
{
  "state_id": "bell_state",
  "measure_type": "concurrence"
}
```

Von Neumann entropy with custom partition:
```json
{
  "state_id": "three_qubit_state",
  "measure_type": "von_neumann", 
  "subsystem_partition": [[0], [1, 2]],
  "visualize_correlations": true
}
```

## Quantum State Types

### Pure States
- **Computational basis**: States like |0⟩, |1⟩, |01⟩
- **Superposition**: Custom linear combinations
- **Spin states**: Arbitrary points on Bloch sphere

### Mixed States  
- **Random mixed**: Random density matrices with specified purity
- **Statistical mixtures**: Weighted combinations of pure states

### Coherent States
- **Harmonic oscillator**: Minimum uncertainty states |α⟩
- **Parameters**: Complex amplitude α

### Squeezed States
- **Squeezed coherent**: Reduced uncertainty in one quadrature
- **Parameters**: Displacement α, squeezing r, angle θ

### Thermal States
- **Harmonic oscillator**: Thermal equilibrium states
- **Parameters**: Average photon number n̄

### Fock States
- **Number states**: Definite photon number |n⟩
- **Parameters**: Photon number n

## Observable Types

### Qubit Observables
- `sigmax`, `sigmay`, `sigmaz` - Pauli operators
- `sigmap`, `sigmam` - Raising/lowering operators

### Harmonic Oscillator Observables  
- `num` - Number operator
- `create`, `destroy` - Ladder operators
- `position`, `momentum` - Quadrature operators

### Custom Observables
- Matrix specification in JSON format
- Operator expressions like `sigmax + sigmay`

## Evolution Types

### Unitary Evolution
- **Schrödinger equation**: U(t) = exp(-iHt)
- **Closed quantum systems**
- **Preserves purity**

### Master Equation
- **Lindblad form**: Open quantum systems
- **Includes decoherence and dissipation**
- **Requires collapse operators**

### Monte Carlo
- **Quantum trajectories**: Stochastic unraveling
- **Individual trajectory evolution**
- **Statistical averaging**

### Stochastic
- **Stochastic Schrödinger equation**
- **Continuous measurement models**
- **Noise-driven evolution**

## Error Handling

All tools return structured error responses:

```json
{
  "success": false,
  "error": "ErrorType",
  "message": "Human-readable error message",
  "details": {
    "field": "problematic_field",
    "expected_type": "expected_type",
    "received_value": "actual_value"
  }
}
```

Common error types:
- `ValidationError` - Invalid input parameters
- `QuantumStateError` - Quantum state issues
- `QuantumOperationError` - Operation failures
- `AnimationError` - Animation generation problems
- `DimensionError` - Hilbert space dimension issues

## Configuration

Server configuration via `MCPConfig`:

```python
config = MCPConfig(
    quantum_precision=1e-12,
    max_hilbert_dimension=1024,
    animation_cache_size=100,
    parallel_workers=4,
    render_backend="cairo",
    output_directory="./output"
)
```

Environment variables:
- `PSIANIMATOR_QUANTUM_PRECISION`
- `PSIANIMATOR_MAX_HILBERT_DIM` 
- `PSIANIMATOR_OUTPUT_DIR`
- `PSIANIMATOR_RENDER_QUALITY`