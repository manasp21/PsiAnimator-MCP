"""
Tests for QuantumStateManager functionality.
"""

import pytest
import numpy as np
import qutip as qt

from psianimator_mcp.quantum.state_manager import QuantumStateManager
from psianimator_mcp.server.exceptions import QuantumStateError, ValidationError


class TestQuantumStateManager:
    """Test suite for QuantumStateManager."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.state_manager = QuantumStateManager(max_dimension=64)
    
    def test_create_pure_state(self):
        """Test creation of pure quantum states."""
        # Test computational basis state
        state_id = self.state_manager.create_state(
            state_type="pure",
            system_dims=[2],
            parameters={"state_indices": [0]},
            basis="computational"
        )
        
        state = self.state_manager.get_state(state_id)
        expected = qt.basis(2, 0)
        
        assert state.type == 'ket'
        assert abs(state.overlap(expected) - 1.0) < 1e-10
    
    def test_create_coherent_state(self):
        """Test creation of coherent states."""
        alpha = 1.5 + 0.5j
        state_id = self.state_manager.create_state(
            state_type="coherent",
            system_dims=[10],
            parameters={"alpha": alpha}
        )
        
        state = self.state_manager.get_state(state_id)
        expected = qt.coherent(10, alpha)
        
        assert state.type == 'ket'
        assert abs(state.overlap(expected) - 1.0) < 1e-10
    
    def test_create_thermal_state(self):
        """Test creation of thermal states."""
        n_avg = 2.0
        state_id = self.state_manager.create_state(
            state_type="thermal",
            system_dims=[10],
            parameters={"n_avg": n_avg}
        )
        
        state = self.state_manager.get_state(state_id)
        expected = qt.thermal_dm(10, n_avg)
        
        assert state.type == 'oper'
        assert abs(state.tr() - 1.0) < 1e-10
        
        # Check mean photon number
        n_op = qt.num(10)
        mean_n = qt.expect(n_op, state)
        assert abs(mean_n - n_avg) < 1e-10
    
    def test_state_validation(self):
        """Test state validation and normalization."""
        # Test invalid dimension
        with pytest.raises(ValidationError):
            self.state_manager.create_state(
                state_type="pure",
                system_dims=[1],  # Invalid dimension
                parameters={"state_indices": [0]}
            )
        
        # Test dimension too large
        with pytest.raises(Exception):
            QuantumStateManager(max_dimension=4).create_state(
                state_type="pure",
                system_dims=[10],  # Exceeds max_dimension
                parameters={"state_indices": [0]}
            )
    
    def test_tensor_product(self):
        """Test tensor product of states."""
        # Create two qubit states
        state_id_1 = self.state_manager.create_state(
            state_type="pure",
            system_dims=[2],
            parameters={"state_indices": [0]}
        )
        
        state_id_2 = self.state_manager.create_state(
            state_type="pure", 
            system_dims=[2],
            parameters={"state_indices": [1]}
        )
        
        # Create tensor product
        product_id = self.state_manager.tensor_product([state_id_1, state_id_2])
        product_state = self.state_manager.get_state(product_id)
        
        # Verify dimensions and properties
        assert product_state.dims == [[2, 2], [1, 1]]
        assert product_state.shape == (4, 1)
        
        # Verify it's the |01⟩ state
        expected = qt.tensor(qt.basis(2, 0), qt.basis(2, 1))
        assert abs(product_state.overlap(expected) - 1.0) < 1e-10
    
    def test_state_info(self):
        """Test state information retrieval."""
        state_id = self.state_manager.create_state(
            state_type="coherent",
            system_dims=[10],
            parameters={"alpha": 2.0}
        )
        
        info = self.state_manager.get_state_info(state_id)
        
        assert info['state_type'] == 'coherent'
        assert info['hilbert_dim'] == 10
        assert info['system_dims'] == [10]
        assert 'mean_photon_number' in info
        assert abs(info['mean_photon_number'] - 4.0) < 1e-10
    
    def test_state_management(self):
        """Test state storage and deletion."""
        # Create state
        state_id = self.state_manager.create_state(
            state_type="pure",
            system_dims=[2],
            parameters={"state_indices": [0]}
        )
        
        # Verify it exists
        assert state_id in self.state_manager.list_states()
        
        # Delete state
        self.state_manager.delete_state(state_id)
        assert state_id not in self.state_manager.list_states()
        
        # Test clear all
        state_id_1 = self.state_manager.create_state("pure", [2], {"state_indices": [0]})
        state_id_2 = self.state_manager.create_state("pure", [2], {"state_indices": [1]})
        
        assert len(self.state_manager.list_states()) == 2
        
        self.state_manager.clear_all_states()
        assert len(self.state_manager.list_states()) == 0