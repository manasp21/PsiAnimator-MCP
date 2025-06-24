"""
Bloch Sphere 3D Visualization for PsiAnimator-MCP

Provides interactive Bloch sphere visualization for qubit states with
real-time state evolution, measurement visualization, and quantum gates.
"""

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


class BlochSphere3D(QuantumScene):
    """
    3D Bloch sphere visualization for qubit states.
    
    Provides visualization of single qubit states on the Bloch sphere
    with support for state evolution, measurement visualization, and
    quantum gate operations.
    """
    
    def __init__(self, **kwargs):
        """Initialize Bloch sphere scene."""
        super().__init__(**kwargs)
        
        # Bloch sphere settings
        self.sphere_radius = 2.0
        self.sphere_resolution = 24
        self.show_axes = True
        self.show_equator = True
        self.show_meridians = True
        
        # Animation settings
        self.state_vector_length = self.sphere_radius * 0.95
        self.measurement_cone_angle = 0.3
        
        # Colors for Bloch sphere elements
        self.bloch_colors = {
            'sphere': BLUE_E,
            'x_axis': RED,
            'y_axis': GREEN, 
            'z_axis': BLUE,
            'equator': WHITE,
            'meridians': GRAY,
            'state_vector': YELLOW,
            'measurement_cone': ORANGE,
            'trajectory': PURPLE
        }
        
        logger.debug("BlochSphere3D initialized")
    
    def construct_bloch_sphere(self) -> VGroup:
        """
        Construct the basic Bloch sphere with axes and grid lines.
        
        Returns
        -------
        VGroup
            Complete Bloch sphere visualization
        """
        sphere_group = VGroup()
        
        # Main sphere surface
        sphere = Sphere(
            radius=self.sphere_radius,
            resolution=(self.sphere_resolution, self.sphere_resolution),
            fill_color=self.bloch_colors['sphere'],
            fill_opacity=0.1,
            stroke_color=self.bloch_colors['sphere'],
            stroke_width=0.5
        )
        sphere_group.add(sphere)
        
        if self.show_axes:
            # X, Y, Z axes
            x_axis = Arrow3D(
                start=[-self.sphere_radius * 1.2, 0, 0],
                end=[self.sphere_radius * 1.2, 0, 0],
                color=self.bloch_colors['x_axis'],
                thickness=0.02
            )
            y_axis = Arrow3D(
                start=[0, -self.sphere_radius * 1.2, 0],
                end=[0, self.sphere_radius * 1.2, 0],
                color=self.bloch_colors['y_axis'],
                thickness=0.02
            )
            z_axis = Arrow3D(
                start=[0, 0, -self.sphere_radius * 1.2],
                end=[0, 0, self.sphere_radius * 1.2],
                color=self.bloch_colors['z_axis'],
                thickness=0.02
            )
            
            sphere_group.add(x_axis, y_axis, z_axis)
            
            # Axis labels
            x_label = Text("|+⟩", font_size=24).move_to([self.sphere_radius * 1.4, 0, 0])
            x_neg_label = Text("|-⟩", font_size=24).move_to([-self.sphere_radius * 1.4, 0, 0])
            y_label = Text("|+i⟩", font_size=24).move_to([0, self.sphere_radius * 1.4, 0])
            y_neg_label = Text("|-i⟩", font_size=24).move_to([0, -self.sphere_radius * 1.4, 0])
            z_label = Text("|0⟩", font_size=24).move_to([0, 0, self.sphere_radius * 1.4])
            z_neg_label = Text("|1⟩", font_size=24).move_to([0, 0, -self.sphere_radius * 1.4])
            
            sphere_group.add(x_label, x_neg_label, y_label, y_neg_label, z_label, z_neg_label)
        
        if self.show_equator:
            # Equatorial circle (XY plane)
            equator = Circle(
                radius=self.sphere_radius,
                color=self.bloch_colors['equator'],
                stroke_width=2
            ).rotate(PI/2, axis=RIGHT)
            sphere_group.add(equator)
        
        if self.show_meridians:
            # Meridian circles
            meridian_xz = Circle(
                radius=self.sphere_radius,
                color=self.bloch_colors['meridians'],
                stroke_width=1,
                stroke_opacity=0.5
            ).rotate(PI/2, axis=UP)
            
            meridian_yz = Circle(
                radius=self.sphere_radius,
                color=self.bloch_colors['meridians'],
                stroke_width=1,
                stroke_opacity=0.5
            ).rotate(PI/2, axis=RIGHT).rotate(PI/2, axis=OUT)
            
            sphere_group.add(meridian_xz, meridian_yz)
        
        return sphere_group
    
    def qubit_to_bloch_vector(self, state: qt.Qobj) -> np.ndarray:
        """
        Convert qubit state to Bloch vector coordinates.
        
        Parameters
        ----------
        state : qt.Qobj
            Qubit state (ket or density matrix)
            
        Returns
        -------
        np.ndarray
            Bloch vector [x, y, z] coordinates
            
        Raises
        ------
        ValidationError
            If state is not a valid qubit
        """
        try:
            state = validate_quantum_state(state)
            
            # Check if it's a qubit (dimension 2)
            if state.shape[0] != 2:
                raise ValidationError(f"Expected qubit (dimension 2), got dimension {state.shape[0]}")
            
            if state.type == 'ket':
                # Convert ket to density matrix
                rho = state * state.dag()
            elif state.type == 'oper':
                rho = state
            else:
                raise ValidationError(f"Invalid state type for Bloch sphere: {state.type}")
            
            # Calculate Bloch vector components using Pauli matrices
            sigma_x = qt.sigmax()
            sigma_y = qt.sigmay()
            sigma_z = qt.sigmaz()
            
            x = np.real((sigma_x * rho).tr())
            y = np.real((sigma_y * rho).tr())
            z = np.real((sigma_z * rho).tr())
            
            return np.array([x, y, z])
            
        except Exception as e:
            raise ValidationError(f"Failed to convert state to Bloch vector: {str(e)}")
    
    def create_state_vector(self, bloch_vector: np.ndarray) -> VGroup:
        """
        Create visual representation of quantum state on Bloch sphere.
        
        Parameters
        ----------
        bloch_vector : np.ndarray
            Bloch vector [x, y, z] coordinates
            
        Returns
        -------
        VGroup
            State vector visualization
        """
        state_group = VGroup()
        
        # Normalize vector to sphere surface
        magnitude = np.linalg.norm(bloch_vector)
        if magnitude > 1e-10:
            normalized_vector = bloch_vector / magnitude * self.state_vector_length
        else:
            normalized_vector = np.array([0, 0, self.state_vector_length])
        
        # State vector arrow
        state_arrow = Arrow3D(
            start=[0, 0, 0],
            end=normalized_vector,
            color=self.bloch_colors['state_vector'],
            thickness=0.03
        )
        state_group.add(state_arrow)
        
        # State point on sphere surface
        state_point = Sphere(
            radius=0.08,
            color=self.bloch_colors['state_vector']
        ).move_to(normalized_vector)
        state_group.add(state_point)
        
        # Purity indicator (for mixed states)
        if magnitude < 0.99:  # Mixed state
            purity_sphere = Sphere(
                radius=magnitude * self.sphere_radius,
                fill_opacity=0.05,
                stroke_opacity=0.3,
                stroke_color=self.bloch_colors['state_vector']
            )
            state_group.add(purity_sphere)
        
        return state_group
    
    def animate_state_evolution(self,
                              initial_state: qt.Qobj,
                              time_evolution: List[qt.Qobj],
                              show_trajectory: bool = True,
                              trajectory_fade: bool = True) -> None:
        """
        Animate quantum state evolution on the Bloch sphere.
        
        Parameters
        ----------
        initial_state : qt.Qobj
            Initial qubit state
        time_evolution : list of qt.Qobj
            List of states at different time points
        show_trajectory : bool, optional
            Whether to show trajectory path
        trajectory_fade : bool, optional
            Whether trajectory should fade over time
        """
        try:
            # Setup Bloch sphere
            bloch_sphere = self.construct_bloch_sphere()
            self.add(bloch_sphere)
            
            # Create initial state vector
            initial_bloch = self.qubit_to_bloch_vector(initial_state)
            state_vector = self.create_state_vector(initial_bloch)
            self.add(state_vector)
            
            # Trajectory points for path visualization
            trajectory_points = []
            trajectory_lines = VGroup()
            
            # Animation function
            def update_state_vector(mob, alpha):
                # Get current state
                idx = int(alpha * (len(time_evolution) - 1))
                current_state = time_evolution[idx]
                current_bloch = self.qubit_to_bloch_vector(current_state)
                
                # Update state vector
                new_state_vector = self.create_state_vector(current_bloch)
                mob.become(new_state_vector)
                
                # Add to trajectory
                if show_trajectory:
                    current_point = current_bloch / np.linalg.norm(current_bloch) * self.sphere_radius
                    trajectory_points.append(current_point.copy())
                    
                    if len(trajectory_points) > 1:
                        # Create line segment for trajectory
                        line = Line3D(
                            start=trajectory_points[-2],
                            end=trajectory_points[-1],
                            color=self.bloch_colors['trajectory'],
                            stroke_width=3
                        )
                        
                        if trajectory_fade:
                            # Fade older trajectory segments
                            opacity = max(0.1, 1.0 - len(trajectory_points) / 50)
                            line.set_stroke(opacity=opacity)
                        
                        trajectory_lines.add(line)
                        
                        # Add trajectory to scene if not already present
                        if trajectory_lines not in self.mobjects:
                            self.add(trajectory_lines)
            
            # Run evolution animation
            self.play(
                UpdateFromAlphaFunc(state_vector, update_state_vector),
                run_time=5.0,
                rate_func=linear
            )
            
        except Exception as e:
            raise AnimationError(
                f"Failed to animate Bloch sphere evolution: {str(e)}",
                animation_type="bloch_evolution",
                render_settings={"states_count": len(time_evolution)}
            )
    
    def animate_measurement(self,
                          state: qt.Qobj,
                          measurement_axis: str = "z",
                          show_probability_cones: bool = True) -> None:
        """
        Animate quantum measurement on the Bloch sphere.
        
        Parameters
        ----------
        state : qt.Qobj
            Quantum state to measure
        measurement_axis : str, optional
            Measurement axis ('x', 'y', or 'z')
        show_probability_cones : bool, optional
            Whether to show measurement probability cones
        """
        try:
            # Setup Bloch sphere
            bloch_sphere = self.construct_bloch_sphere()
            self.add(bloch_sphere)
            
            # Create state vector
            bloch_vector = self.qubit_to_bloch_vector(state)
            state_vector = self.create_state_vector(bloch_vector)
            self.add(state_vector)
            
            # Calculate measurement probabilities
            if measurement_axis == "z":
                prob_0 = abs(state[0, 0])**2  # |0⟩ outcome
                prob_1 = abs(state[1, 0])**2  # |1⟩ outcome
                axis_vector = np.array([0, 0, 1])
            elif measurement_axis == "x":
                # Transform to X basis
                H = qt.hadamard_transform()
                state_x = H * state
                prob_0 = abs(state_x[0, 0])**2  # |+⟩ outcome
                prob_1 = abs(state_x[1, 0])**2  # |-⟩ outcome
                axis_vector = np.array([1, 0, 0])
            elif measurement_axis == "y":
                # Transform to Y basis
                basis_change = qt.Qobj([[1, -1j], [1, 1j]]) / np.sqrt(2)
                state_y = basis_change * state
                prob_0 = abs(state_y[0, 0])**2  # |+i⟩ outcome
                prob_1 = abs(state_y[1, 0])**2  # |-i⟩ outcome
                axis_vector = np.array([0, 1, 0])
            else:
                raise ValidationError(f"Unknown measurement axis: {measurement_axis}")
            
            if show_probability_cones:
                # Create probability cones
                cone_0 = Cone(
                    base_radius=self.sphere_radius * np.sin(self.measurement_cone_angle),
                    height=self.sphere_radius * np.cos(self.measurement_cone_angle),
                    direction=axis_vector,
                    fill_color=GREEN,
                    fill_opacity=prob_0 * 0.3,
                    stroke_width=0
                ).move_to(axis_vector * self.sphere_radius * 0.5)
                
                cone_1 = Cone(
                    base_radius=self.sphere_radius * np.sin(self.measurement_cone_angle),
                    height=self.sphere_radius * np.cos(self.measurement_cone_angle),
                    direction=-axis_vector,
                    fill_color=RED,
                    fill_opacity=prob_1 * 0.3,
                    stroke_width=0
                ).move_to(-axis_vector * self.sphere_radius * 0.5)
                
                # Show measurement cones
                self.play(
                    FadeIn(cone_0),
                    FadeIn(cone_1),
                    run_time=1.0
                )
                
                # Display probabilities
                prob_text_0 = Text(f"P(0) = {prob_0:.3f}", font_size=20).to_corner(UL)
                prob_text_1 = Text(f"P(1) = {prob_1:.3f}", font_size=20).next_to(prob_text_0, DOWN)
                
                self.play(
                    Write(prob_text_0),
                    Write(prob_text_1),
                    run_time=1.0
                )
                
                self.wait(2)
                
                # Collapse to measurement outcome (higher probability)
                if prob_0 > prob_1:
                    final_vector = axis_vector * self.state_vector_length
                    outcome_text = Text("Outcome: |0⟩", font_size=24).to_corner(UR)
                else:
                    final_vector = -axis_vector * self.state_vector_length
                    outcome_text = Text("Outcome: |1⟩", font_size=24).to_corner(UR)
                
                final_state_vector = self.create_state_vector(final_vector / self.state_vector_length * self.sphere_radius)
                
                self.play(
                    Transform(state_vector, final_state_vector),
                    Write(outcome_text),
                    FadeOut(cone_0),
                    FadeOut(cone_1),
                    run_time=2.0
                )
                
        except Exception as e:
            raise AnimationError(
                f"Failed to animate measurement: {str(e)}",
                animation_type="measurement",
                render_settings={"measurement_axis": measurement_axis}
            )
    
    def animate_quantum_gate(self,
                           initial_state: qt.Qobj,
                           gate: Union[qt.Qobj, str],
                           gate_time: float = 2.0,
                           show_gate_name: bool = True) -> None:
        """
        Animate quantum gate operation on the Bloch sphere.
        
        Parameters
        ----------
        initial_state : qt.Qobj
            Initial qubit state
        gate : qt.Qobj or str
            Quantum gate to apply (operator or name)
        gate_time : float, optional
            Duration of gate animation
        show_gate_name : bool, optional
            Whether to display gate name
        """
        try:
            # Parse gate
            if isinstance(gate, str):
                gate_name = gate.upper()
                if gate_name == "X":
                    gate_op = qt.sigmax()
                elif gate_name == "Y":
                    gate_op = qt.sigmay()
                elif gate_name == "Z":
                    gate_op = qt.sigmaz()
                elif gate_name == "H":
                    gate_op = qt.hadamard_transform()
                elif gate_name == "S":
                    gate_op = qt.phasegate(np.pi/2)
                elif gate_name == "T":
                    gate_op = qt.phasegate(np.pi/4)
                else:
                    raise ValidationError(f"Unknown gate: {gate_name}")
            else:
                gate_op = gate
                gate_name = "CUSTOM"
            
            # Calculate final state
            final_state = gate_op * initial_state
            
            # Setup Bloch sphere
            bloch_sphere = self.construct_bloch_sphere()
            self.add(bloch_sphere)
            
            # Create initial state vector
            initial_bloch = self.qubit_to_bloch_vector(initial_state)
            state_vector = self.create_state_vector(initial_bloch)
            self.add(state_vector)
            
            # Show gate name
            if show_gate_name:
                gate_label = Text(f"Gate: {gate_name}", font_size=24).to_corner(UL)
                self.play(Write(gate_label), run_time=0.5)
            
            # Animate gate operation
            final_bloch = self.qubit_to_bloch_vector(final_state)
            final_state_vector = self.create_state_vector(final_bloch)
            
            self.play(
                Transform(state_vector, final_state_vector),
                run_time=gate_time,
                rate_func=smooth
            )
            
            self.wait(1)
            
        except Exception as e:
            raise AnimationError(
                f"Failed to animate quantum gate: {str(e)}",
                animation_type="quantum_gate",
                render_settings={"gate": str(gate)}
            )