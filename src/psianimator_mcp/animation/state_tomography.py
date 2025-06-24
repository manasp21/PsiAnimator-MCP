"""
State Tomography Visualization for PsiAnimator-MCP

Provides visualization of quantum state tomography including density matrix
representation, Pauli decomposition, and tomographic reconstruction.
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


class StateTomography(QuantumScene):
    """
    Quantum state tomography visualization.
    
    Provides comprehensive visualization of quantum state tomography including
    density matrix visualization, Pauli decomposition, measurement statistics,
    and tomographic reconstruction process.
    """
    
    def __init__(self, **kwargs):
        """Initialize state tomography scene."""
        super().__init__(**kwargs)
        
        # Tomography settings
        self.matrix_scale = 0.3
        self.measurement_scale = 1.5
        self.show_real_parts = True
        self.show_imaginary_parts = True
        self.show_pauli_decomposition = True
        
        # Colors for tomography elements
        self.tomo_colors = {
            'density_matrix': BLUE,
            'real_part': BLUE,
            'imaginary_part': RED,
            'diagonal': GREEN,
            'off_diagonal': ORANGE,
            'pauli_x': RED,
            'pauli_y': GREEN,
            'pauli_z': BLUE,
            'identity': WHITE,
            'measurement_stats': YELLOW,
            'reconstruction': PURPLE
        }
        
        logger.debug("StateTomography initialized")
    
    def create_density_matrix_visualization(self,
                                          density_matrix: qt.Qobj,
                                          position: np.ndarray = ORIGIN,
                                          show_values: bool = True,
                                          show_colormap: bool = True) -> VGroup:
        """
        Create visual representation of density matrix.
        
        Parameters
        ----------
        density_matrix : qt.Qobj
            Density matrix to visualize
        position : np.ndarray, optional
            Position for the visualization
        show_values : bool, optional
            Whether to show numerical values
        show_colormap : bool, optional
            Whether to use color mapping for matrix elements
            
        Returns
        -------
        VGroup
            Density matrix visualization
        """
        group = VGroup()
        
        try:
            rho = validate_quantum_state(density_matrix)
            if rho.type != 'oper':
                raise ValidationError("Expected density matrix (operator type)")
            
            matrix_data = rho.full()
            dim = matrix_data.shape[0]
            
            # Create matrix grid
            cell_size = self.matrix_scale
            
            for i in range(dim):
                for j in range(dim):
                    element = matrix_data[i, j]
                    
                    # Cell position
                    cell_pos = position + RIGHT * j * cell_size + DOWN * i * cell_size
                    
                    # Determine color based on element properties
                    if show_colormap:
                        if i == j:  # Diagonal elements (populations)
                            color = self.tomo_colors['diagonal']
                            opacity = min(1.0, abs(element) * 2)
                        else:  # Off-diagonal elements (coherences)
                            color = self.tomo_colors['off_diagonal']
                            opacity = min(0.8, abs(element) * 4)
                    else:
                        color = self.tomo_colors['density_matrix']
                        opacity = 0.7
                    
                    # Create cell
                    cell = Square(
                        side_length=cell_size * 0.9,
                        fill_color=color,
                        fill_opacity=opacity,
                        stroke_color=WHITE,
                        stroke_width=1
                    ).move_to(cell_pos)
                    
                    group.add(cell)
                    
                    # Add numerical values
                    if show_values and abs(element) > 1e-3:
                        if abs(np.imag(element)) < 1e-10:
                            # Real number
                            value_text = Text(
                                f"{np.real(element):.2f}",
                                font_size=12,
                                color=BLACK if opacity > 0.5 else WHITE
                            ).move_to(cell_pos)
                        else:
                            # Complex number
                            real_part = np.real(element)
                            imag_part = np.imag(element)
                            if abs(real_part) > 1e-10:
                                value_text = Text(
                                    f"{real_part:.2f}\\n{imag_part:+.2f}i",
                                    font_size=10,
                                    color=BLACK if opacity > 0.5 else WHITE
                                ).move_to(cell_pos)
                            else:
                                value_text = Text(
                                    f"{imag_part:.2f}i",
                                    font_size=12,
                                    color=BLACK if opacity > 0.5 else WHITE
                                ).move_to(cell_pos)
                        
                        group.add(value_text)
            
            # Add matrix brackets and label
            bracket_height = dim * cell_size
            left_bracket = Text("⎡", font_size=bracket_height * 20).move_to(
                position + LEFT * cell_size * 0.7 + DOWN * (dim - 1) * cell_size / 2
            )
            right_bracket = Text("⎤", font_size=bracket_height * 20).move_to(
                position + RIGHT * (dim - 0.3) * cell_size + DOWN * (dim - 1) * cell_size / 2
            )
            
            matrix_label = MathTex("\\rho").next_to(left_bracket, LEFT, buff=0.3)
            
            group.add(left_bracket, right_bracket, matrix_label)
            
            return group
            
        except Exception as e:
            raise AnimationError(
                f"Failed to create density matrix visualization: {str(e)}",
                animation_type="density_matrix",
                render_settings={"matrix_dim": density_matrix.shape[0] if hasattr(density_matrix, 'shape') else 'unknown'}
            )
    
    def create_pauli_decomposition(self,
                                 density_matrix: qt.Qobj,
                                 position: np.ndarray = ORIGIN) -> VGroup:
        """
        Create Pauli decomposition visualization for qubit density matrix.
        
        Parameters
        ----------
        density_matrix : qt.Qobj
            Qubit density matrix
        position : np.ndarray, optional
            Position for the visualization
            
        Returns
        -------
        VGroup
            Pauli decomposition visualization
        """
        group = VGroup()
        
        try:
            rho = validate_quantum_state(density_matrix)
            if rho.shape[0] != 2:
                raise ValidationError("Pauli decomposition requires qubit (2x2) density matrix")
            
            # Pauli matrices
            pauli_I = qt.qeye(2)
            pauli_X = qt.sigmax()
            pauli_Y = qt.sigmay()
            pauli_Z = qt.sigmaz()
            
            # Calculate Pauli coefficients
            coeff_I = (pauli_I * rho).tr() / 2
            coeff_X = (pauli_X * rho).tr()
            coeff_Y = (pauli_Y * rho).tr()
            coeff_Z = (pauli_Z * rho).tr()
            
            coefficients = [
                (coeff_I, "I", self.tomo_colors['identity']),
                (coeff_X, "σ_x", self.tomo_colors['pauli_x']),
                (coeff_Y, "σ_y", self.tomo_colors['pauli_y']),
                (coeff_Z, "σ_z", self.tomo_colors['pauli_z'])
            ]
            
            # Create decomposition equation
            equation_parts = []
            
            # ρ = label
            rho_label = MathTex("\\rho", "=").move_to(position + LEFT * 3)
            group.add(rho_label)
            
            x_offset = -1.5
            for i, (coeff, pauli_name, color) in enumerate(coefficients):
                if abs(coeff) > 1e-10:  # Only show non-zero terms
                    # Coefficient
                    coeff_text = MathTex(f"{np.real(coeff):.3f}").set_color(color)
                    
                    # Pauli matrix symbol
                    pauli_symbol = MathTex(pauli_name).set_color(color)
                    
                    # Position elements
                    term_group = VGroup(coeff_text, pauli_symbol).arrange(RIGHT, buff=0.1)
                    term_group.move_to(position + RIGHT * x_offset)
                    
                    group.add(term_group)
                    
                    # Add plus sign (except for first term)
                    if x_offset > -1.5:
                        plus_sign = MathTex("+").move_to(position + RIGHT * (x_offset - 0.3))
                        group.add(plus_sign)
                    
                    x_offset += 1.0
            
            # Add Bloch vector representation
            bloch_label = MathTex("\\vec{r}", "=", f"({np.real(coeff_X):.3f},", f"{np.real(coeff_Y):.3f},", f"{np.real(coeff_Z):.3f})").move_to(position + DOWN * 1.5)
            bloch_label[2].set_color(self.tomo_colors['pauli_x'])
            bloch_label[3].set_color(self.tomo_colors['pauli_y'])
            bloch_label[4].set_color(self.tomo_colors['pauli_z'])
            group.add(bloch_label)
            
            return group
            
        except Exception as e:
            raise AnimationError(
                f"Failed to create Pauli decomposition: {str(e)}",
                animation_type="pauli_decomposition"
            )
    
    def create_measurement_statistics(self,
                                    measurement_results: Dict[str, List[float]],
                                    position: np.ndarray = ORIGIN) -> VGroup:
        """
        Create visualization of measurement statistics for tomography.
        
        Parameters
        ----------
        measurement_results : dict
            Dictionary of measurement results {basis: [probabilities]}
        position : np.ndarray, optional
            Position for the visualization
            
        Returns
        -------
        VGroup
            Measurement statistics visualization
        """
        group = VGroup()
        
        try:
            bases = list(measurement_results.keys())
            n_bases = len(bases)
            
            if n_bases == 0:
                return group
            
            # Create measurement basis charts
            chart_width = 1.5
            chart_spacing = 2.0
            
            for i, basis in enumerate(bases):
                probabilities = measurement_results[basis]
                
                # Chart position
                chart_pos = position + RIGHT * (i - (n_bases - 1) / 2) * chart_spacing
                
                # Create probability bars
                bar_chart = self.create_probability_distribution(
                    probabilities,
                    labels=[f"|{j}⟩" for j in range(len(probabilities))],
                    position=chart_pos,
                    bar_width=0.3,
                    max_height=self.measurement_scale
                )
                
                group.add(bar_chart)
                
                # Add basis label
                basis_label = Text(f"{basis} basis", font_size=16).next_to(bar_chart, UP, buff=0.3)
                group.add(basis_label)
            
            # Add title
            title = Text("Measurement Statistics", font_size=20).next_to(group, UP, buff=0.5)
            group.add(title)
            
            return group
            
        except Exception as e:
            raise AnimationError(
                f"Failed to create measurement statistics: {str(e)}",
                animation_type="measurement_statistics"
            )
    
    def animate_tomographic_reconstruction(self,
                                         true_state: qt.Qobj,
                                         measurement_data: Dict[str, List[float]],
                                         reconstruction_steps: List[qt.Qobj]) -> None:
        """
        Animate the tomographic reconstruction process.
        
        Parameters
        ----------
        true_state : qt.Qobj
            True quantum state
        measurement_data : dict
            Measurement results used for reconstruction
        reconstruction_steps : list of qt.Qobj
            Intermediate reconstruction states
        """
        try:
            # Setup scene with true state
            true_matrix_viz = self.create_density_matrix_visualization(
                true_state,
                position=LEFT * 4 + UP * 2
            )
            true_label = Text("True State", font_size=16).next_to(true_matrix_viz, UP)
            
            self.add(true_matrix_viz, true_label)
            
            # Add measurement statistics
            meas_stats = self.create_measurement_statistics(
                measurement_data,
                position=DOWN * 2
            )
            self.add(meas_stats)
            
            # Create reconstructed state visualization (starts empty)
            recon_matrix_viz = self.create_density_matrix_visualization(
                reconstruction_steps[0] if reconstruction_steps else qt.qeye(true_state.shape[0]) / true_state.shape[0],
                position=RIGHT * 4 + UP * 2
            )
            recon_label = Text("Reconstructed State", font_size=16).next_to(recon_matrix_viz, UP)
            
            self.add(recon_matrix_viz, recon_label)
            
            # Add fidelity tracker
            fidelity_text = Text("Fidelity: 0.000", font_size=14).to_corner(UR)
            self.add(fidelity_text)
            
            # Animate reconstruction process
            for i, recon_state in enumerate(reconstruction_steps):
                # Calculate fidelity with true state
                fidelity = qt.fidelity(true_state, recon_state)
                
                # Update reconstructed matrix
                new_recon_viz = self.create_density_matrix_visualization(
                    recon_state,
                    position=RIGHT * 4 + UP * 2
                )
                
                # Update fidelity text
                new_fidelity_text = Text(f"Fidelity: {fidelity:.3f}", font_size=14).to_corner(UR)
                
                # Animate changes
                self.play(
                    Transform(recon_matrix_viz, new_recon_viz),
                    Transform(fidelity_text, new_fidelity_text),
                    run_time=0.5
                )
                
                # Brief pause between steps
                if i < len(reconstruction_steps) - 1:
                    self.wait(0.2)
            
            # Final pause to show converged result
            self.wait(2)
            
            # Add Pauli decomposition for final state
            if true_state.shape[0] == 2:  # Qubit case
                true_pauli = self.create_pauli_decomposition(true_state, position=LEFT * 4 + DOWN * 0.5)
                recon_pauli = self.create_pauli_decomposition(reconstruction_steps[-1], position=RIGHT * 4 + DOWN * 0.5)
                
                self.play(
                    FadeIn(true_pauli),
                    FadeIn(recon_pauli),
                    run_time=1.0
                )
                
                self.wait(2)
            
        except Exception as e:
            raise AnimationError(
                f"Failed to animate tomographic reconstruction: {str(e)}",
                animation_type="tomographic_reconstruction"
            )
    
    def create_process_tomography_visualization(self,
                                              process_matrix: qt.Qobj,
                                              position: np.ndarray = ORIGIN) -> VGroup:
        """
        Create visualization for quantum process tomography.
        
        Parameters
        ----------
        process_matrix : qt.Qobj
            Process matrix (chi matrix) to visualize
        position : np.ndarray, optional
            Position for the visualization
            
        Returns
        -------
        VGroup
            Process tomography visualization
        """
        group = VGroup()
        
        try:
            chi_matrix = process_matrix.full()
            dim = chi_matrix.shape[0]
            
            # Create process matrix visualization (similar to density matrix but different coloring)
            cell_size = self.matrix_scale * 0.8
            
            for i in range(dim):
                for j in range(dim):
                    element = chi_matrix[i, j]
                    
                    # Cell position
                    cell_pos = position + RIGHT * j * cell_size + DOWN * i * cell_size
                    
                    # Color based on magnitude and phase
                    magnitude = abs(element)
                    if magnitude > 1e-10:
                        phase = np.angle(element)
                        # Use phase to determine color hue
                        hue = (phase + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
                        color = Color(hue=hue, saturation=1.0, luminance=0.5)
                        opacity = min(1.0, magnitude * 2)
                    else:
                        color = BLACK
                        opacity = 0.1
                    
                    # Create cell
                    cell = Square(
                        side_length=cell_size * 0.9,
                        fill_color=color,
                        fill_opacity=opacity,
                        stroke_color=WHITE,
                        stroke_width=0.5
                    ).move_to(cell_pos)
                    
                    group.add(cell)
            
            # Add matrix label
            matrix_label = MathTex("\\chi").next_to(
                group, LEFT, buff=0.3
            ).scale(1.5)
            group.add(matrix_label)
            
            # Add colorbar legend for phase
            colorbar = VGroup()
            n_colors = 20
            for k in range(n_colors):
                hue = k / n_colors
                bar_color = Color(hue=hue, saturation=1.0, luminance=0.5)
                color_rect = Rectangle(
                    width=0.1,
                    height=0.2,
                    fill_color=bar_color,
                    fill_opacity=1.0,
                    stroke_width=0
                ).move_to(position + RIGHT * (dim * cell_size + 1) + DOWN * k * 0.2)
                colorbar.add(color_rect)
            
            # Phase labels
            phase_labels = VGroup(
                Text("π", font_size=12).next_to(colorbar[0], RIGHT),
                Text("0", font_size=12).next_to(colorbar[n_colors//2], RIGHT),
                Text("-π", font_size=12).next_to(colorbar[-1], RIGHT)
            )
            colorbar.add(phase_labels)
            
            group.add(colorbar)
            
            return group
            
        except Exception as e:
            raise AnimationError(
                f"Failed to create process tomography visualization: {str(e)}",
                animation_type="process_tomography"
            )