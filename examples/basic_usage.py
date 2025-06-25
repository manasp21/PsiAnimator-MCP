"""
Basic usage examples for PsiAnimator-MCP

Demonstrates core functionality including state creation, evolution, 
measurement, and animation generation.
"""

import asyncio
import json
from pathlib import Path

from psianimator_mcp.server.config import MCPConfig
from psianimator_mcp.tools.quantum_state_tools import create_quantum_state
from psianimator_mcp.tools.evolution_tools import evolve_quantum_system
from psianimator_mcp.tools.measurement_tools import measure_observable
from psianimator_mcp.tools.gate_tools import quantum_gate_sequence
from psianimator_mcp.tools.entanglement_tools import calculate_entanglement

# Animation functionality (optional)
try:
    from psianimator_mcp.tools.animation_tools import animate_quantum_process
    _ANIMATION_AVAILABLE = True
except ImportError:
    animate_quantum_process = None
    _ANIMATION_AVAILABLE = False


async def basic_qubit_example():
    """Example: Basic qubit operations."""
    print("\\n=== Basic Qubit Example ===")
    
    config = MCPConfig()
    
    # 1. Create a qubit in |0⟩ state
    print("Creating qubit in |0⟩ state...")
    create_result = await create_quantum_state({
        'state_type': 'pure',
        'system_dims': [2],
        'parameters': {'state_indices': [0]},
        'basis': 'computational'
    }, config)
    
    if create_result['success']:
        state_id = create_result['state_id']
        print(f"Created state with ID: {state_id}")
        
        # 2. Measure initial state
        print("\\nMeasuring ⟨σz⟩ for |0⟩ state...")
        measurement_result = await measure_observable({
            'state_id': state_id,
            'observable': 'sigmaz',
            'measurement_type': 'expectation'
        }, config)
        
        if measurement_result['success']:
            expectation = measurement_result['measurement_results']['expectation_value']
            print(f"⟨σz⟩ = {expectation:.6f}")
        
        # 3. Apply Hadamard gate
        print("\\nApplying Hadamard gate...")
        gate_result = await quantum_gate_sequence({
            'state_id': state_id,
            'gates': [{'name': 'H', 'qubits': [0]}],
            'show_intermediate_states': True
        }, config)
        
        if gate_result['success']:
            final_state_id = gate_result['gate_application_results']['intermediate_state_ids'][-1]
            print(f"State after Hadamard: {final_state_id}")
            
            # 4. Measure superposition state
            print("\\nMeasuring ⟨σz⟩ for |+⟩ state...")
            measurement_result = await measure_observable({
                'state_id': final_state_id,
                'observable': 'sigmaz',
                'measurement_type': 'expectation'
            }, config)
            
            if measurement_result['success']:
                expectation = measurement_result['measurement_results']['expectation_value']
                print(f"⟨σz⟩ = {expectation:.6f}")
            
            # 5. Measure probability distribution
            print("\\nMeasuring probability distribution...")
            prob_result = await measure_observable({
                'state_id': final_state_id,
                'observable': 'sigmaz',
                'measurement_type': 'probability'
            }, config)
            
            if prob_result['success']:
                prob_dist = prob_result['measurement_results']['distribution_statistics']
                print(f"Probabilities: P(+1) = {prob_dist['probabilities'][1]:.3f}, P(-1) = {prob_dist['probabilities'][0]:.3f}")


async def bell_state_example():
    """Example: Bell state creation and entanglement analysis."""
    print("\\n=== Bell State Example ===")
    
    config = MCPConfig()
    
    # 1. Create two-qubit system in |00⟩
    print("Creating two-qubit system in |00⟩...")
    create_result = await create_quantum_state({
        'state_type': 'pure',
        'system_dims': [2, 2],
        'parameters': {'coefficients': [1, 0, 0, 0]},
        'basis': 'computational'
    }, config)
    
    if create_result['success']:
        state_id = create_result['state_id']
        print(f"Created state with ID: {state_id}")
        
        # 2. Create Bell state using gates
        print("\\nCreating Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2...")
        gate_result = await quantum_gate_sequence({
            'state_id': state_id,
            'gates': [
                {'name': 'H', 'qubits': [0]},      # Hadamard on qubit 0
                {'name': 'CNOT', 'qubits': [0, 1]}  # CNOT: control=0, target=1
            ],
            'show_intermediate_states': True
        }, config)
        
        if gate_result['success']:
            bell_state_id = gate_result['gate_application_results']['intermediate_state_ids'][-1]
            print(f"Bell state created: {bell_state_id}")
            
            # 3. Analyze entanglement
            print("\\nAnalyzing entanglement...")
            entangle_result = await calculate_entanglement({
                'state_id': bell_state_id,
                'measure_type': 'concurrence',
                'visualize_correlations': True
            }, config)
            
            if entangle_result['success']:
                analysis = entangle_result['entanglement_analysis']
                concurrence = analysis['entanglement_values']['concurrence']
                eof = analysis['entanglement_values']['entanglement_of_formation']
                
                print(f"Concurrence: {concurrence:.6f}")
                print(f"Entanglement of Formation: {eof:.6f}")
                print(f"Entanglement Level: {analysis['interpretation']['entanglement_level']}")


async def harmonic_oscillator_example():
    """Example: Harmonic oscillator coherent state evolution."""
    print("\\n=== Harmonic Oscillator Example ===")
    
    config = MCPConfig()
    
    # 1. Create coherent state
    print("Creating coherent state α = 2.0...")
    create_result = await create_quantum_state({
        'state_type': 'coherent',
        'system_dims': [20],  # 20 Fock states
        'parameters': {'alpha': 2.0}
    }, config)
    
    if create_result['success']:
        state_id = create_result['state_id']
        print(f"Created coherent state: {state_id}")
        
        # 2. Measure photon number
        print("\\nMeasuring photon number...")
        measurement_result = await measure_observable({
            'state_id': state_id,
            'observable': 'num',
            'measurement_type': 'expectation'
        }, config)
        
        if measurement_result['success']:
            mean_n = measurement_result['measurement_results']['expectation_value']
            print(f"⟨n⟩ = {mean_n:.6f}")
        
        # 3. Evolve under free evolution
        print("\\nEvolving under harmonic oscillator Hamiltonian...")
        evolution_result = await evolve_quantum_system({
            'state_id': state_id,
            'hamiltonian': 'harmonic_oscillator',
            'evolution_type': 'unitary',
            'time_span': [0, 6.28, 20],  # One full oscillation
            'solver_options': {'store_evolved_states': True}
        }, config)
        
        if evolution_result['success']:
            print(f"Evolution completed with {evolution_result['n_time_steps']} time steps")
            print(f"Stored {len(evolution_result['stored_state_ids'])} intermediate states")


async def quantum_circuit_example():
    """Example: Quantum circuit with measurements."""
    print("\\n=== Quantum Circuit Example ===")
    
    config = MCPConfig()
    
    # 1. Create 3-qubit GHZ state
    print("Creating 3-qubit GHZ state...")
    create_result = await create_quantum_state({
        'state_type': 'pure',
        'system_dims': [2, 2, 2],
        'parameters': {'coefficients': [1, 0, 0, 0, 0, 0, 0, 0]},  # |000⟩
        'basis': 'computational'
    }, config)
    
    if create_result['success']:
        state_id = create_result['state_id']
        
        # 2. Apply GHZ preparation circuit
        print("\\nApplying GHZ preparation circuit...")
        gate_result = await quantum_gate_sequence({
            'state_id': state_id,
            'gates': [
                {'name': 'H', 'qubits': [0]},        # Hadamard on qubit 0
                {'name': 'CNOT', 'qubits': [0, 1]},  # CNOT: 0 → 1
                {'name': 'CNOT', 'qubits': [0, 2]}   # CNOT: 0 → 2
            ],
            'show_intermediate_states': True,
            'gate_visualization': 'circuit'
        }, config)
        
        if gate_result['success']:
            ghz_state_id = gate_result['gate_application_results']['intermediate_state_ids'][-1]
            print(f"GHZ state created: {ghz_state_id}")
            
            # 3. Analyze multipartite entanglement
            print("\\nAnalyzing multipartite entanglement...")
            entangle_result = await calculate_entanglement({
                'state_id': ghz_state_id,
                'measure_type': 'von_neumann',
                'subsystem_partition': [[0], [1, 2]]  # Bipartite cut
            }, config)
            
            if entangle_result['success']:
                analysis = entangle_result['entanglement_analysis']
                entropy = analysis['entanglement_values']['entanglement_entropy']
                print(f"Entanglement entropy: {entropy:.6f}")


async def animation_example():
    """Example: Animation generation (requires animation dependencies)."""
    print("\\n=== Animation Example ===")
    
    if not _ANIMATION_AVAILABLE:
        print("⚠️ Animation functionality not available!")
        print("Install animation dependencies: pip install 'psianimator-mcp[animation]'")
        return
    
    config = MCPConfig()
    
    # 1. Create qubit state for Bloch sphere animation
    print("Creating qubit for Bloch sphere animation...")
    create_result = await create_quantum_state({
        'state_type': 'pure',
        'system_dims': [2],
        'parameters': {'state_indices': [0]},
        'basis': 'computational'
    }, config)
    
    if create_result['success']:
        state_id = create_result['state_id']
        
        # 2. Generate Bloch sphere evolution animation
        print("\\nGenerating Bloch sphere animation...")
        animation_result = await animate_quantum_process({
            'animation_type': 'bloch_evolution',
            'data_source': f'state_id:{state_id}',
            'render_quality': 'medium',
            'output_format': 'mp4',
            'frame_rate': 30,
            'duration': 5.0,
            'view_config': {'show_trajectory': True}
        }, config)
        
        if animation_result['success']:
            output_files = animation_result['output_files']
            print(f"Animation saved to: {output_files}")
        else:
            print(f"Animation failed: {animation_result['message']}")


async def main():
    """Run all examples."""
    print("PsiAnimator-MCP Examples")
    print("=" * 50)
    
    try:
        await basic_qubit_example()
        await bell_state_example() 
        await harmonic_oscillator_example()
        await quantum_circuit_example()
        
        # Run animation example if available
        await animation_example()
        
        print("\\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\\nExample failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())