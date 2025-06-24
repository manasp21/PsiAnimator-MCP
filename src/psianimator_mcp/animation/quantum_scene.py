"""
Quantum Scene Base Class for PsiAnimator-MCP

Provides specialized Manim scene class with quantum-specific utilities,
automatic scaling, and common quantum visualization tools.
"""

import numpy as np
import qutip as qt
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import logging

from manim import *
from manim.constants import *

from ..quantum.validation import validate_quantum_state, ensure_qobj
from ..server.exceptions import AnimationError, ValidationError

logger = logging.getLogger(__name__)


class QuantumScene(Scene):
    """
    Base scene class for quantum physics animations.
    
    Extends Manim's Scene with quantum-specific utilities including
    automatic scaling for quantum amplitudes, complex number visualization,
    and common quantum physics display elements.
    """
    
    def __init__(self, **kwargs):
        """Initialize quantum scene with quantum-specific settings."""
        super().__init__(**kwargs)
        
        # Quantum-specific settings
        self.quantum_precision = 1e-10
        self.amplitude_scale = 2.0  # Scale factor for probability amplitudes
        self.phase_scale = 1.0      # Scale factor for phase visualization
        self.auto_normalize = True  # Automatically normalize quantum states
        
        # Color schemes for quantum visualizations
        self.quantum_colors = {
            'real_positive': BLUE,
            'real_negative': RED, 
            'imaginary_positive': GREEN,
            'imaginary_negative': ORANGE,
            'probability': YELLOW,
            'phase': PURPLE,
            'entangled': PINK,
            'ground_state': BLUE_B,
            'excited_state': RED_B
        }
        
        # Common quantum objects storage
        self.quantum_states: Dict[str, qt.Qobj] = {}
        self.quantum_operators: Dict[str, qt.Qobj] = {}
        self.animation_data: Dict[str, Any] = {}
        
        logger.debug("QuantumScene initialized")
    
    def setup_quantum_coordinates(self, 
                                center: np.ndarray = ORIGIN,
                                scale: float = 1.0) -> None:
        """
        Setup coordinate system optimized for quantum visualizations.
        
        Parameters
        ----------
        center : np.ndarray, optional
            Center point for quantum coordinate system
        scale : float, optional
            Overall scale factor for quantum elements
        """
        self.quantum_center = center
        self.quantum_scale = scale
        
        # Setup axes for quantum number displays if needed
        self.quantum_axes = Axes(
            x_range=[-1.2, 1.2, 0.4],
            y_range=[-1.2, 1.2, 0.4],
            axis_config={"color": GRAY, "stroke_width": 1}
        ).scale(scale).move_to(center)
    
    def add_quantum_state(self, state_id: str, state: qt.Qobj) -> None:
        """
        Add a quantum state to the scene for visualization.
        
        Parameters
        ----------
        state_id : str
            Identifier for the state
        state : qt.Qobj
            Quantum state to add
        """
        try:
            validated_state = validate_quantum_state(state)
            self.quantum_states[state_id] = validated_state
            logger.debug(f"Added quantum state '{state_id}' to scene")
        except Exception as e:
            raise AnimationError(
                f"Failed to add quantum state to scene: {str(e)}",
                animation_type="quantum_scene",
                render_settings={"state_id": state_id}
            )
    
    def add_quantum_operator(self, op_id: str, operator: qt.Qobj) -> None:
        """
        Add a quantum operator to the scene for visualization.
        
        Parameters
        ----------
        op_id : str
            Identifier for the operator
        operator : qt.Qobj
            Quantum operator to add
        """
        try:
            self.quantum_operators[op_id] = ensure_qobj(operator)
            logger.debug(f"Added quantum operator '{op_id}' to scene")
        except Exception as e:
            raise AnimationError(
                f"Failed to add quantum operator to scene: {str(e)}",
                animation_type="quantum_scene",
                render_settings={"operator_id": op_id}
            )
    
    def create_complex_number_display(self, 
                                    z: complex,
                                    position: np.ndarray = ORIGIN,
                                    show_magnitude: bool = True,
                                    show_phase: bool = True,
                                    show_components: bool = True) -> VGroup:
        """
        Create visual representation of a complex number.
        
        Parameters
        ----------
        z : complex
            Complex number to visualize
        position : np.ndarray, optional
            Position for the display
        show_magnitude : bool, optional
            Whether to show magnitude
        show_phase : bool, optional
            Whether to show phase
        show_components : bool, optional
            Whether to show real and imaginary components
            
        Returns
        -------
        VGroup
            Manim group containing complex number visualization
        """
        group = VGroup()
        
        # Real and imaginary components
        if show_components:
            real_part = np.real(z)
            imag_part = np.imag(z)
            
            # Real component bar
            if abs(real_part) > self.quantum_precision:
                real_color = self.quantum_colors['real_positive'] if real_part >= 0 else self.quantum_colors['real_negative']
                real_bar = Rectangle(
                    width=abs(real_part) * self.amplitude_scale,
                    height=0.1,
                    fill_color=real_color,
                    fill_opacity=0.7,
                    stroke_width=1
                ).move_to(position + RIGHT * real_part * self.amplitude_scale / 2)
                group.add(real_bar)
            
            # Imaginary component bar
            if abs(imag_part) > self.quantum_precision:
                imag_color = self.quantum_colors['imaginary_positive'] if imag_part >= 0 else self.quantum_colors['imaginary_negative']
                imag_bar = Rectangle(
                    width=0.1,
                    height=abs(imag_part) * self.amplitude_scale,
                    fill_color=imag_color,
                    fill_opacity=0.7,
                    stroke_width=1
                ).move_to(position + UP * imag_part * self.amplitude_scale / 2)
                group.add(imag_bar)
        
        # Magnitude circle
        if show_magnitude:
            magnitude = abs(z)
            if magnitude > self.quantum_precision:
                mag_circle = Circle(
                    radius=magnitude * self.amplitude_scale,
                    color=self.quantum_colors['probability'],
                    stroke_width=2,
                    fill_opacity=0.1
                ).move_to(position)
                group.add(mag_circle)
        
        # Phase arrow
        if show_phase and abs(z) > self.quantum_precision:
            phase = np.angle(z)
            phase_arrow = Arrow(
                start=position,
                end=position + abs(z) * self.amplitude_scale * np.array([np.cos(phase), np.sin(phase), 0]),
                color=self.quantum_colors['phase'],
                stroke_width=3,
                max_tip_length_to_length_ratio=0.1
            )
            group.add(phase_arrow)
        
        return group.move_to(position)
    
    def create_probability_distribution(self,
                                      probabilities: List[float],
                                      labels: Optional[List[str]] = None,
                                      position: np.ndarray = ORIGIN,
                                      bar_width: float = 0.5,
                                      max_height: float = 2.0) -> VGroup:
        """
        Create bar chart for probability distribution.
        
        Parameters
        ----------
        probabilities : list of float
            Probability values to display
        labels : list of str, optional
            Labels for each probability
        position : np.ndarray, optional
            Base position for the chart
        bar_width : float, optional
            Width of each bar
        max_height : float, optional
            Maximum height for scaling
            
        Returns
        -------
        VGroup
            Manim group containing probability distribution
        """
        group = VGroup()
        
        if not probabilities:
            return group
        
        max_prob = max(probabilities) if probabilities else 1.0
        scale_factor = max_height / max_prob if max_prob > 0 else 1.0
        
        for i, prob in enumerate(probabilities):
            if prob > self.quantum_precision:
                # Create probability bar
                bar_height = prob * scale_factor
                bar = Rectangle(
                    width=bar_width,
                    height=bar_height,
                    fill_color=self.quantum_colors['probability'],
                    fill_opacity=0.7,
                    stroke_color=WHITE,
                    stroke_width=1
                )
                
                # Position bar
                x_pos = position[0] + i * (bar_width + 0.1) - (len(probabilities) - 1) * (bar_width + 0.1) / 2
                bar.move_to([x_pos, position[1] + bar_height / 2, position[2]])
                group.add(bar)
                
                # Add probability value label
                prob_label = Text(f"{prob:.3f}", font_size=16).next_to(bar, UP, buff=0.1)
                group.add(prob_label)
                
                # Add custom label if provided
                if labels and i < len(labels):
                    custom_label = Text(labels[i], font_size=14).next_to(bar, DOWN, buff=0.1)
                    group.add(custom_label)
        
        return group
    
    def create_quantum_state_vector(self,
                                  state: qt.Qobj,
                                  position: np.ndarray = ORIGIN,
                                  show_amplitudes: bool = True,
                                  show_phases: bool = True,
                                  max_components: int = 8) -> VGroup:
        """
        Create visualization of quantum state vector.
        
        Parameters
        ----------
        state : qt.Qobj
            Quantum state to visualize
        position : np.ndarray, optional
            Base position for visualization
        show_amplitudes : bool, optional
            Whether to show amplitude magnitudes
        show_phases : bool, optional
            Whether to show phases
        max_components : int, optional
            Maximum number of components to display
            
        Returns
        -------
        VGroup
            Manim group containing state vector visualization
        """
        group = VGroup()
        
        try:
            state = validate_quantum_state(state)
            
            if state.type != 'ket':
                raise ValidationError("State vector visualization requires ket state")
            
            # Get state amplitudes
            amplitudes = state.full().flatten()
            n_components = min(len(amplitudes), max_components)
            
            for i in range(n_components):
                amp = amplitudes[i]
                
                if abs(amp) > self.quantum_precision:
                    # Position for this component
                    comp_pos = position + DOWN * i * 0.8
                    
                    # Create complex number display for this amplitude
                    amp_display = self.create_complex_number_display(
                        amp,
                        comp_pos,
                        show_magnitude=show_amplitudes,
                        show_phase=show_phases,
                        show_components=True
                    )
                    group.add(amp_display)
                    
                    # Add basis state label
                    basis_label = MathTex(f"|{i}\\rangle").next_to(amp_display, LEFT, buff=0.3)
                    group.add(basis_label)
            
            # Add state vector label
            if state.dims and len(state.dims[0]) == 1:
                dim = state.dims[0][0]
                state_label = MathTex(f"|\\psi\\rangle \\in \\mathbb{{C}}^{{{dim}}}").next_to(group, UP, buff=0.5)
                group.add(state_label)
            
            return group
            
        except Exception as e:
            raise AnimationError(
                f"Failed to create state vector visualization: {str(e)}",
                animation_type="state_vector",
                render_settings={"state_type": state.type if hasattr(state, 'type') else 'unknown'}
            )
    
    def create_operator_matrix(self,
                             operator: qt.Qobj,
                             position: np.ndarray = ORIGIN,
                             show_real: bool = True,
                             show_imag: bool = True,
                             max_size: int = 6) -> VGroup:
        """
        Create matrix visualization of quantum operator.
        
        Parameters
        ----------
        operator : qt.Qobj
            Quantum operator to visualize
        position : np.ndarray, optional
            Position for matrix display
        show_real : bool, optional
            Whether to show real parts
        show_imag : bool, optional
            Whether to show imaginary parts
        max_size : int, optional
            Maximum matrix size to display
            
        Returns
        -------
        VGroup
            Manim group containing matrix visualization
        """
        group = VGroup()
        
        try:
            op_matrix = operator.full()
            rows, cols = op_matrix.shape
            
            if rows > max_size or cols > max_size:
                # Show truncated version
                op_matrix = op_matrix[:max_size, :max_size]
                rows, cols = op_matrix.shape
            
            # Create matrix elements
            element_size = 0.6
            for i in range(rows):
                for j in range(cols):
                    element = op_matrix[i, j]
                    
                    if abs(element) > self.quantum_precision:
                        # Position for this matrix element
                        elem_pos = position + RIGHT * j * element_size + DOWN * i * element_size
                        
                        # Create complex number display
                        elem_display = self.create_complex_number_display(
                            element,
                            elem_pos,
                            show_magnitude=False,
                            show_phase=False,
                            show_components=True
                        ).scale(0.3)
                        
                        group.add(elem_display)
            
            # Add matrix brackets
            left_bracket = MathTex("\\begin{pmatrix}").scale(2).move_to(position + LEFT * (cols * element_size / 2 + 0.3))
            right_bracket = MathTex("\\end{pmatrix}").scale(2).move_to(position + RIGHT * (cols * element_size / 2 + 0.3))
            
            group.add(left_bracket, right_bracket)
            
            return group
            
        except Exception as e:
            raise AnimationError(
                f"Failed to create operator matrix visualization: {str(e)}",
                animation_type="operator_matrix",
                render_settings={"operator_shape": operator.shape if hasattr(operator, 'shape') else 'unknown'}
            )
    
    def animate_state_evolution(self,
                              initial_state: qt.Qobj,
                              time_evolution_data: List[qt.Qobj],
                              duration: float = 5.0,
                              update_rate: int = 30) -> None:
        """
        Animate quantum state evolution over time.
        
        Parameters
        ----------
        initial_state : qt.Qobj
            Initial quantum state
        time_evolution_data : list of qt.Qobj
            States at different time points
        duration : float, optional
            Animation duration in seconds
        update_rate : int, optional
            Updates per second
        """
        try:
            # Create initial state visualization
            state_viz = self.create_quantum_state_vector(initial_state)
            self.add(state_viz)
            
            # Animate evolution
            def update_state(mob, alpha):
                # Interpolate between states
                idx = int(alpha * (len(time_evolution_data) - 1))
                current_state = time_evolution_data[idx]
                
                # Update visualization
                new_viz = self.create_quantum_state_vector(current_state)
                mob.become(new_viz)
            
            self.play(
                UpdateFromAlphaFunc(state_viz, update_state),
                run_time=duration,
                rate_func=linear
            )
            
        except Exception as e:
            raise AnimationError(
                f"Failed to animate state evolution: {str(e)}",
                animation_type="state_evolution",
                render_settings={"duration": duration, "states_count": len(time_evolution_data)}
            )
    
    def get_quantum_color_scheme(self, scheme_name: str = "default") -> Dict[str, str]:
        """
        Get color scheme for quantum visualizations.
        
        Parameters
        ----------
        scheme_name : str, optional
            Name of color scheme to use
            
        Returns
        -------
        dict
            Color mapping for quantum elements
        """
        schemes = {
            "default": self.quantum_colors,
            "high_contrast": {
                'real_positive': PURE_BLUE,
                'real_negative': PURE_RED,
                'imaginary_positive': PURE_GREEN,
                'imaginary_negative': ORANGE,
                'probability': YELLOW,
                'phase': PURPLE,
                'entangled': PINK,
                'ground_state': BLUE_E,
                'excited_state': RED_E
            },
            "colorblind_friendly": {
                'real_positive': "#0173B2",  # Blue
                'real_negative': "#DE8F05",  # Orange  
                'imaginary_positive': "#029E73",  # Green
                'imaginary_negative': "#CC78BC",  # Pink
                'probability': "#FBF338",  # Yellow
                'phase': "#8B0063",  # Purple
                'entangled': "#FB8500",  # Orange-red
                'ground_state': "#219EBC",  # Light blue
                'excited_state': "#FB5607"  # Red-orange
            }
        }
        
        return schemes.get(scheme_name, self.quantum_colors)