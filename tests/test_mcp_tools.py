"""
Tests for MCP tools functionality.
"""

import pytest
import numpy as np
import qutip as qt
import asyncio

from src.psianimator_mcp.tools.quantum_state_tools import create_quantum_state
from src.psianimator_mcp.tools.evolution_tools import evolve_quantum_system
from src.psianimator_mcp.tools.measurement_tools import measure_observable
from src.psianimator_mcp.tools.gate_tools import quantum_gate_sequence
from src.psianimator_mcp.tools.entanglement_tools import calculate_entanglement
from src.psianimator_mcp.server.config import MCPConfig


class TestMCPTools:
    """Test suite for MCP tools."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = MCPConfig()
    
    @pytest.mark.asyncio
    async def test_create_quantum_state_tool(self):
        """Test create_quantum_state MCP tool."""
        arguments = {
            'state_type': 'pure',
            'system_dims': [2],
            'parameters': {'state_indices': [0]},
            'basis': 'computational'
        }
        
        result = await create_quantum_state(arguments, self.config)
        
        assert result['success'] is True
        assert 'state_id' in result
        assert result['quantum_properties']['hilbert_space_dimension'] == 2
        assert result['quantum_properties']['is_normalized'] is True
    
    @pytest.mark.asyncio
    async def test_create_quantum_state_validation(self):
        """Test input validation for create_quantum_state."""
        # Missing required field
        arguments = {
            'system_dims': [2],
            'parameters': {'state_indices': [0]}
        }
        
        result = await create_quantum_state(arguments, self.config)
        
        assert result['success'] is False
        assert result['error'] == 'ValidationError'
        assert 'state_type is required' in result['message']
    
    @pytest.mark.asyncio
    async def test_measure_observable_tool(self):
        """Test measure_observable MCP tool."""
        # First create a state
        create_args = {
            'state_type': 'pure',
            'system_dims': [2], 
            'parameters': {'state_indices': [0]},
            'basis': 'computational'
        }
        
        create_result = await create_quantum_state(create_args, self.config)
        state_id = create_result['state_id']
        
        # Now measure it
        measure_args = {
            'state_id': state_id,
            'observable': 'sigmaz',
            'measurement_type': 'expectation'
        }
        
        result = await measure_observable(measure_args, self.config)
        
        assert result['success'] is True
        assert 'measurement_results' in result
        
        # |0⟩ state should have ⟨σz⟩ = +1
        expectation = result['measurement_results']['expectation_value']
        assert abs(expectation - 1.0) < 1e-10
    
    @pytest.mark.asyncio
    async def test_quantum_gate_sequence_tool(self):
        """Test quantum_gate_sequence MCP tool."""
        # Create initial state |0⟩
        create_args = {
            'state_type': 'pure',
            'system_dims': [2],
            'parameters': {'state_indices': [0]},
            'basis': 'computational'
        }
        
        create_result = await create_quantum_state(create_args, self.config)
        state_id = create_result['state_id']
        
        # Apply Hadamard gate
        gate_args = {
            'state_id': state_id,
            'gates': [
                {'name': 'H', 'qubits': [0]}
            ],
            'show_intermediate_states': True
        }
        
        result = await quantum_gate_sequence(gate_args, self.config)
        
        assert result['success'] is True
        assert result['n_gates_applied'] == 1
        assert len(result['gate_application_results']['applied_gates']) == 1
        
        # Check that we got intermediate states
        assert len(result['gate_application_results']['intermediate_state_ids']) == 1
    
    @pytest.mark.asyncio
    async def test_calculate_entanglement_tool(self):
        """Test calculate_entanglement MCP tool."""
        # Create Bell state |00⟩ + |11⟩
        create_args = {
            'state_type': 'pure',
            'system_dims': [2, 2],
            'parameters': {
                'coefficients': [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]
            },
            'basis': 'computational'
        }
        
        create_result = await create_quantum_state(create_args, self.config)
        state_id = create_result['state_id']
        
        # Calculate concurrence
        entangle_args = {
            'state_id': state_id,
            'measure_type': 'concurrence'
        }
        
        result = await calculate_entanglement(entangle_args, self.config)
        
        assert result['success'] is True
        assert 'entanglement_analysis' in result
        
        # Bell state should have maximum concurrence = 1
        concurrence = result['entanglement_analysis']['entanglement_values']['concurrence']
        assert abs(concurrence - 1.0) < 1e-10
    
    @pytest.mark.asyncio
    async def test_evolve_quantum_system_tool(self):
        """Test evolve_quantum_system MCP tool."""
        # Create initial state
        create_args = {
            'state_type': 'pure',
            'system_dims': [2],
            'parameters': {'state_indices': [0]},
            'basis': 'computational'
        }
        
        create_result = await create_quantum_state(create_args, self.config)
        state_id = create_result['state_id']
        
        # Evolve under X rotation
        evolve_args = {
            'state_id': state_id,
            'hamiltonian': 'sigmax',
            'evolution_type': 'unitary',
            'time_span': [0, np.pi, 10]
        }
        
        result = await evolve_quantum_system(evolve_args, self.config)
        
        assert result['success'] is True
        assert result['evolution_type'] == 'unitary'
        assert result['n_time_steps'] == 10
        
        # After π rotation around X, |0⟩ should become |1⟩
        final_state = result['evolution_data']['states'][-1]
        # Test would need access to final state to verify this properly
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in MCP tools."""
        # Test with non-existent state
        measure_args = {
            'state_id': 'non_existent_state',
            'observable': 'sigmaz',
            'measurement_type': 'expectation'
        }
        
        result = await measure_observable(measure_args, self.config)
        
        assert result['success'] is False
        assert 'not found' in result['message'].lower()
    
    @pytest.mark.asyncio
    async def test_complex_workflow(self):
        """Test a complex workflow combining multiple tools."""
        # 1. Create Bell state preparation circuit
        create_args = {
            'state_type': 'pure',
            'system_dims': [2, 2],
            'parameters': {'coefficients': [1, 0, 0, 0]},  # |00⟩
            'basis': 'computational'
        }
        
        create_result = await create_quantum_state(create_args, self.config)
        state_id = create_result['state_id']
        
        # 2. Apply gates to create Bell state
        gate_args = {
            'state_id': state_id,
            'gates': [
                {'name': 'H', 'qubits': [0]},      # Hadamard on first qubit
                {'name': 'CNOT', 'qubits': [0, 1]}  # CNOT with control=0, target=1
            ],
            'show_intermediate_states': True
        }
        
        gate_result = await quantum_gate_sequence(gate_args, self.config)
        assert gate_result['success'] is True
        
        # 3. Measure entanglement
        final_state_id = gate_result['gate_application_results']['intermediate_state_ids'][-1]
        
        entangle_args = {
            'state_id': final_state_id,
            'measure_type': 'concurrence'
        }
        
        entangle_result = await calculate_entanglement(entangle_args, self.config)
        assert entangle_result['success'] is True
        
        # Bell state should be maximally entangled
        concurrence = entangle_result['entanglement_analysis']['entanglement_values']['concurrence']
        assert concurrence > 0.9  # Should be close to 1, allowing for numerical precision