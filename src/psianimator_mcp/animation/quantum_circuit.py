"""
Quantum Circuit Visualization for PsiAnimator-MCP

Provides visualization of quantum circuits with gate sequences, state tracking,
and measurement outcomes.
"""

from __future__ import annotations

import numpy as np
import qutip as qt
from typing import Dict, List, Optional, Union, Tuple, Any
import logging

from manim import *
from manim.constants import *

from .quantum_scene import QuantumScene
from ..quantum.validation import validate_quantum_state
from ..server.exceptions import AnimationError, ValidationError

logger = logging.getLogger(__name__)


class QuantumCircuit(QuantumScene):
    """
    Quantum circuit visualization.
    
    Provides visualization of quantum circuits with gate sequences,
    intermediate state visualization, and measurement outcomes.
    """
    
    def __init__(self, **kwargs):
        """Initialize quantum circuit scene."""
        super().__init__(**kwargs)
        
        # Circuit layout settings
        self.qubit_spacing = 1.0
        self.gate_spacing = 1.5
        self.wire_length = 8.0
        self.gate_size = 0.4
        
        # Colors for circuit elements
        self.circuit_colors = {
            'wire': WHITE,
            'qubit_label': WHITE,
            'single_gate': BLUE,
            'two_gate': RED,
            'measurement': GREEN,
            'classical_wire': ORANGE,
            'state_vector': YELLOW
        }
        
        logger.debug("QuantumCircuit initialized")
    
    def create_qubit_wire(self, qubit_index: int, position: np.ndarray = ORIGIN) -> VGroup:
        """
        Create a qubit wire with label.
        
        Parameters
        ----------
        qubit_index : int
            Index of the qubit (0, 1, 2, ...)
        position : np.ndarray, optional
            Starting position of the wire
            
        Returns
        -------
        VGroup
            Qubit wire visualization
        """
        wire_group = VGroup()
        
        # Wire line
        wire = Line(
            start=position,
            end=position + RIGHT * self.wire_length,
            color=self.circuit_colors['wire'],
            stroke_width=2
        )
        wire_group.add(wire)
        
        # Qubit label
        qubit_label = MathTex(f"|q_{{{qubit_index}}}\\rangle").next_to(wire.get_start(), LEFT, buff=0.3)
        wire_group.add(qubit_label)
        
        return wire_group
    
    def create_quantum_gate(self, gate_name: str, position: np.ndarray, qubits: List[int]) -> VGroup:
        """
        Create a quantum gate visualization.
        
        Parameters
        ----------
        gate_name : str
            Name of the gate (X, Y, Z, H, CNOT, etc.)
        position : np.ndarray
            Position for the gate
        qubits : list of int
            List of qubit indices the gate acts on
            
        Returns
        -------
        VGroup
            Gate visualization
        """
        gate_group = VGroup()
        
        if len(qubits) == 1:
            # Single-qubit gate
            gate_box = Square(
                side_length=self.gate_size,
                fill_color=self.circuit_colors['single_gate'],
                fill_opacity=0.8,
                stroke_color=WHITE,
                stroke_width=2
            ).move_to(position)
            
            gate_text = Text(gate_name, font_size=16, color=WHITE).move_to(position)
            gate_group.add(gate_box, gate_text)
            
        elif len(qubits) == 2:
            # Two-qubit gate (simplified representation)
            if gate_name.upper() in ['CNOT', 'CX']:
                # Control qubit (filled circle)
                control = Dot(radius=0.1, color=self.circuit_colors['two_gate']).move_to(position)
                
                # Target qubit (circle with plus)
                target_pos = position + DOWN * self.qubit_spacing * (qubits[1] - qubits[0])
                target_circle = Circle(radius=0.2, color=self.circuit_colors['two_gate'], stroke_width=2).move_to(target_pos)
                target_plus = Cross(stroke_width=3, color=self.circuit_colors['two_gate']).scale(0.3).move_to(target_pos)
                
                # Connection line
                connection = Line(position, target_pos, color=self.circuit_colors['two_gate'], stroke_width=2)
                
                gate_group.add(control, target_circle, target_plus, connection)
            else:
                # Generic two-qubit gate box
                gate_height = abs(qubits[1] - qubits[0]) * self.qubit_spacing + self.gate_size
                gate_box = Rectangle(
                    width=self.gate_size,
                    height=gate_height,
                    fill_color=self.circuit_colors['two_gate'],
                    fill_opacity=0.8,
                    stroke_color=WHITE,
                    stroke_width=2
                ).move_to(position + DOWN * (qubits[1] - qubits[0]) * self.qubit_spacing / 2)
                
                gate_text = Text(gate_name, font_size=14, color=WHITE).move_to(gate_box.get_center())
                gate_group.add(gate_box, gate_text)
        
        return gate_group
    
    def create_measurement(self, position: np.ndarray, classical_bit: Optional[int] = None) -> VGroup:
        """
        Create measurement symbol.
        
        Parameters
        ----------
        position : np.ndarray
            Position for measurement symbol
        classical_bit : int, optional
            Classical bit index for measurement outcome
            
        Returns
        -------
        VGroup
            Measurement visualization
        """
        meas_group = VGroup()
        
        # Measurement box
        meas_box = Rectangle(
            width=self.gate_size,
            height=self.gate_size,
            fill_color=self.circuit_colors['measurement'],
            fill_opacity=0.8,
            stroke_color=WHITE,
            stroke_width=2
        ).move_to(position)
        
        # Measurement symbol (semicircle with arrow)
        arc = Arc(radius=0.15, start_angle=0, angle=PI, color=BLACK, stroke_width=2).move_to(position)
        arrow = Arrow(
            start=position,
            end=position + UP * 0.1 + RIGHT * 0.05,
            color=BLACK,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.3
        ).scale(0.7)
        
        meas_group.add(meas_box, arc, arrow)
        
        # Classical wire if specified
        if classical_bit is not None:
            classical_wire = Line(
                start=position + RIGHT * self.gate_size / 2,
                end=position + RIGHT * self.gate_size / 2 + RIGHT * 1.0,
                color=self.circuit_colors['classical_wire'],
                stroke_width=3
            )
            
            # Double line for classical wire
            classical_wire2 = Line(
                start=position + RIGHT * self.gate_size / 2 + UP * 0.05,
                end=position + RIGHT * self.gate_size / 2 + RIGHT * 1.0 + UP * 0.05,
                color=self.circuit_colors['classical_wire'],
                stroke_width=3
            )
            
            bit_label = MathTex(f"c_{{{classical_bit}}}").next_to(classical_wire.get_end(), RIGHT, buff=0.2)
            
            meas_group.add(classical_wire, classical_wire2, bit_label)
        
        return meas_group
    
    def animate_circuit_execution(self,
                                initial_state: qt.Qobj,
                                gate_sequence: List[Dict[str, Any]],
                                show_intermediate_states: bool = True) -> None:
        """
        Animate quantum circuit execution with state evolution.
        
        Parameters
        ----------
        initial_state : qt.Qobj
            Initial quantum state
        gate_sequence : list of dict
            List of gate specifications
        show_intermediate_states : bool, optional
            Whether to show intermediate quantum states
        """
        try:
            n_qubits = int(np.log2(initial_state.shape[0]))
            
            # Create circuit layout
            circuit_group = VGroup()
            
            # Create qubit wires
            qubit_wires = []
            for i in range(n_qubits):
                wire_pos = UP * (n_qubits - 1 - i) * self.qubit_spacing
                wire = self.create_qubit_wire(i, wire_pos)
                qubit_wires.append(wire)
                circuit_group.add(wire)
            
            self.add(circuit_group)
            
            # Add initial state visualization
            if show_intermediate_states:
                initial_state_viz = self.create_quantum_state_vector(
                    initial_state,
                    position=LEFT * 5
                )
                self.add(initial_state_viz)
                current_state_viz = initial_state_viz
            
            # Current quantum state
            current_state = initial_state
            
            # Execute gates one by one
            for gate_idx, gate_spec in enumerate(gate_sequence):
                gate_name = gate_spec.get('name', 'U')
                qubits = gate_spec.get('qubits', [0])
                
                # Position for this gate
                gate_x = LEFT * 3 + RIGHT * gate_idx * self.gate_spacing
                gate_y = UP * (n_qubits - 1 - qubits[0]) * self.qubit_spacing
                gate_pos = gate_x + gate_y
                
                # Create and animate gate appearance
                gate_viz = self.create_quantum_gate(gate_name, gate_pos, qubits)
                
                self.play(FadeIn(gate_viz), run_time=0.5)
                
                # Update quantum state (simplified - would need proper gate implementation)
                # For now, just show the concept
                if show_intermediate_states:
                    # Create new state visualization
                    new_state_viz = self.create_quantum_state_vector(
                        current_state,  # In reality, apply the gate here
                        position=LEFT * 5
                    )
                    
                    self.play(
                        Transform(current_state_viz, new_state_viz),
                        run_time=1.0
                    )
                
                self.wait(0.5)
            
            self.wait(2)
            
        except Exception as e:
            raise AnimationError(
                f"Failed to animate circuit execution: {str(e)}",
                animation_type="circuit_execution"
            )