"""
Photon Statistics Visualization for PsiAnimator-MCP

Provides visualization of photon number distributions, correlation functions,
and quantum coherence measures for optical quantum states.
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


class PhotonStatistics(QuantumScene):
    """
    Photon statistics visualization for quantum optical states.
    
    Provides comprehensive visualization of photon number distributions,
    correlation functions, coherence measures, and quantum optical phenomena.
    """
    
    def __init__(self, **kwargs):
        """Initialize photon statistics scene."""
        super().__init__(**kwargs)
        
        # Statistics settings
        self.max_photons = 20
        self.bar_chart_scale = 3.0
        self.correlation_scale = 2.0
        
        # Colors for photon statistics
        self.photon_colors = {
            'number_distribution': BLUE,
            'coherent_state': GREEN,
            'thermal_state': RED,
            'squeezed_state': PURPLE,
            'fock_state': ORANGE,
            'correlation_positive': BLUE_B,
            'correlation_negative': RED_B,
            'g2_classical': GRAY,
            'g2_quantum': YELLOW,
            'mandel_q': PINK
        }
        
        logger.debug("PhotonStatistics initialized")
    
    def calculate_photon_distribution(self, state: qt.Qobj) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate photon number distribution for a quantum state.
        
        Parameters
        ----------
        state : qt.Qobj
            Quantum state (ket or density matrix)
            
        Returns
        -------
        tuple
            (photon_numbers, probabilities) arrays
        """
        try:
            state = validate_quantum_state(state)
            
            if state.type == 'ket':
                rho = state * state.dag()
            else:
                rho = state
            
            # Get state dimension
            dim = rho.shape[0]
            max_n = min(dim - 1, self.max_photons)
            
            # Calculate probabilities for each photon number
            photon_numbers = np.arange(max_n + 1)
            probabilities = np.zeros(max_n + 1)
            
            for n in photon_numbers:
                if n < dim:
                    probabilities[n] = np.real(rho[n, n])
            
            return photon_numbers, probabilities
            
        except Exception as e:
            raise AnimationError(
                f"Failed to calculate photon distribution: {str(e)}",
                animation_type="photon_distribution"
            )
    
    def create_photon_distribution_chart(self,
                                       photon_numbers: np.ndarray,
                                       probabilities: np.ndarray,
                                       position: np.ndarray = ORIGIN,
                                       state_type: str = "unknown") -> VGroup:
        """
        Create bar chart for photon number distribution.
        
        Parameters
        ----------
        photon_numbers : np.ndarray
            Array of photon numbers
        probabilities : np.ndarray
            Corresponding probabilities
        position : np.ndarray, optional
            Position for the chart
        state_type : str, optional
            Type of quantum state for color coding
            
        Returns
        -------
        VGroup
            Photon distribution chart
        """
        chart_group = VGroup()
        
        try:
            # Determine color based on state type
            if state_type in self.photon_colors:
                bar_color = self.photon_colors[state_type]
            else:
                bar_color = self.photon_colors['number_distribution']
            
            # Create axes
            max_prob = np.max(probabilities) if len(probabilities) > 0 else 1.0
            
            axes = Axes(
                x_range=[0, len(photon_numbers), 1],
                y_range=[0, max_prob * 1.1, max_prob / 5],
                x_length=self.bar_chart_scale,
                y_length=self.bar_chart_scale * 0.8,
                axis_config={"color": WHITE, "stroke_width": 2},
                tips=True
            ).move_to(position)
            
            chart_group.add(axes)
            
            # Add axis labels
            x_label = MathTex("n").next_to(axes.x_axis.get_end(), RIGHT)
            y_label = MathTex("P(n)").next_to(axes.y_axis.get_end(), UP)
            chart_group.add(x_label, y_label)
            
            # Create bars
            bar_width = 0.8
            for n, prob in zip(photon_numbers, probabilities):
                if prob > 1e-10:  # Only show significant probabilities
                    bar_height = prob / max_prob * axes.y_length * 0.8
                    
                    bar = Rectangle(
                        width=bar_width * axes.x_length / len(photon_numbers),
                        height=bar_height,
                        fill_color=bar_color,
                        fill_opacity=0.7,
                        stroke_color=WHITE,
                        stroke_width=1
                    )
                    
                    bar_pos = axes.c2p(n + 0.5, prob / 2)
                    bar.move_to(bar_pos)
                    chart_group.add(bar)
                    
                    # Add probability label on top of bar
                    if prob > max_prob * 0.05:  # Only label significant bars
                        prob_label = Text(f"{prob:.3f}", font_size=12).next_to(bar, UP, buff=0.05)
                        chart_group.add(prob_label)
            
            # Add title
            title = Text(f"Photon Number Distribution ({state_type})", font_size=16).next_to(chart_group, UP, buff=0.3)
            chart_group.add(title)
            
            return chart_group
            
        except Exception as e:
            raise AnimationError(
                f"Failed to create photon distribution chart: {str(e)}",
                animation_type="distribution_chart"
            )
    
    def calculate_coherence_measures(self, state: qt.Qobj) -> Dict[str, float]:
        """
        Calculate various coherence measures for the quantum state.
        
        Parameters
        ----------
        state : qt.Qobj
            Quantum state
            
        Returns
        -------
        dict
            Dictionary of coherence measures
        """
        try:
            state = validate_quantum_state(state)
            
            if state.type == 'ket':
                rho = state * state.dag()
            else:
                rho = state
            
            measures = {}
            
            # Mean photon number
            n_op = qt.num(rho.shape[0])
            mean_n = qt.expect(n_op, rho)
            measures['mean_photon_number'] = float(np.real(mean_n))
            
            # Photon number variance
            n_squared = n_op * n_op
            mean_n_squared = qt.expect(n_squared, rho)
            variance_n = float(np.real(mean_n_squared - mean_n**2))
            measures['photon_variance'] = variance_n
            
            # Mandel Q parameter (measure of sub/super-Poissonian statistics)
            if mean_n > 0:
                mandel_q = (variance_n - mean_n) / mean_n
                measures['mandel_q'] = float(np.real(mandel_q))
            else:
                measures['mandel_q'] = 0.0
            
            # Fano factor
            if mean_n > 0:
                measures['fano_factor'] = variance_n / mean_n
            else:
                measures['fano_factor'] = 0.0
            
            # Second-order coherence g^(2)(0)
            if mean_n > 1:
                a = qt.destroy(rho.shape[0])
                a_dag = qt.create(rho.shape[0])
                
                g2_numerator = qt.expect(a_dag * a_dag * a * a, rho)
                g2_denominator = qt.expect(a_dag * a, rho)**2
                
                if abs(g2_denominator) > 1e-10:
                    g2 = g2_numerator / g2_denominator
                    measures['g2_zero'] = float(np.real(g2))
                else:
                    measures['g2_zero'] = 0.0
            else:
                measures['g2_zero'] = 0.0
            
            return measures
            
        except Exception as e:
            raise AnimationError(
                f"Failed to calculate coherence measures: {str(e)}",
                animation_type="coherence_measures"
            )
    
    def create_coherence_dashboard(self,
                                 coherence_measures: Dict[str, float],
                                 position: np.ndarray = ORIGIN) -> VGroup:
        """
        Create dashboard displaying coherence measures.
        
        Parameters
        ----------
        coherence_measures : dict
            Dictionary of coherence measures
        position : np.ndarray, optional
            Position for the dashboard
            
        Returns
        -------
        VGroup
            Coherence measures dashboard
        """
        dashboard_group = VGroup()
        
        try:
            # Dashboard background
            dashboard_bg = RoundedRectangle(
                width=4, height=3,
                corner_radius=0.2,
                fill_color=DARK_GRAY,
                fill_opacity=0.8,
                stroke_color=WHITE,
                stroke_width=2
            ).move_to(position)
            dashboard_group.add(dashboard_bg)
            
            # Title
            title = Text("Coherence Measures", font_size=18, color=WHITE).move_to(position + UP * 1.2)
            dashboard_group.add(title)
            
            # Measures display
            measures_text = []
            y_offset = 0.6
            
            for measure, value in coherence_measures.items():
                if measure == 'mean_photon_number':
                    label = "⟨n⟩"
                    color = WHITE
                elif measure == 'photon_variance':
                    label = "Var(n)"
                    color = WHITE
                elif measure == 'mandel_q':
                    label = "Q_M"
                    # Color code Mandel Q
                    if value < -0.1:
                        color = BLUE  # Sub-Poissonian
                    elif value > 0.1:
                        color = RED   # Super-Poissonian
                    else:
                        color = GREEN # Poissonian
                elif measure == 'fano_factor':
                    label = "F"
                    color = WHITE
                elif measure == 'g2_zero':
                    label = "g⁽²⁾(0)"
                    # Color code g^(2)
                    if value < 0.9:
                        color = BLUE  # Antibunching
                    elif value > 1.1:
                        color = RED   # Bunching
                    else:
                        color = GREEN # Coherent
                else:
                    label = measure
                    color = WHITE
                
                measure_line = Text(
                    f"{label}: {value:.3f}",
                    font_size=14,
                    color=color
                ).move_to(position + UP * y_offset)
                
                measures_text.append(measure_line)
                y_offset -= 0.4
            
            dashboard_group.add(*measures_text)
            
            # Add interpretation indicators
            interpretations = []
            
            # Mandel Q interpretation
            q_value = coherence_measures.get('mandel_q', 0)
            if q_value < -0.1:
                q_interp = Text("Sub-Poissonian", font_size=10, color=BLUE)
            elif q_value > 0.1:
                q_interp = Text("Super-Poissonian", font_size=10, color=RED)
            else:
                q_interp = Text("Poissonian", font_size=10, color=GREEN)
            
            interpretations.append(q_interp)
            
            # g^(2) interpretation
            g2_value = coherence_measures.get('g2_zero', 0)
            if g2_value < 0.9:
                g2_interp = Text("Antibunched", font_size=10, color=BLUE)
            elif g2_value > 1.1:
                g2_interp = Text("Bunched", font_size=10, color=RED)
            else:
                g2_interp = Text("Coherent", font_size=10, color=GREEN)
            
            interpretations.append(g2_interp)
            
            # Position interpretations
            interp_group = VGroup(*interpretations).arrange(DOWN, buff=0.1).move_to(position + DOWN * 1.0)
            dashboard_group.add(interp_group)
            
            return dashboard_group
            
        except Exception as e:
            raise AnimationError(
                f"Failed to create coherence dashboard: {str(e)}",
                animation_type="coherence_dashboard"
            )
    
    def animate_photon_statistics_comparison(self,
                                           states_dict: Dict[str, qt.Qobj],
                                           duration: float = 8.0) -> None:
        """
        Animate comparison of photon statistics for different quantum states.
        
        Parameters
        ----------
        states_dict : dict
            Dictionary of states {state_name: qt.Qobj}
        duration : float, optional
            Total animation duration
        """
        try:
            n_states = len(states_dict)
            if n_states == 0:
                return
            
            # Calculate distributions and measures for all states
            distributions = {}
            coherence_measures = {}
            
            for name, state in states_dict.items():
                photon_nums, probs = self.calculate_photon_distribution(state)
                distributions[name] = (photon_nums, probs)
                coherence_measures[name] = self.calculate_coherence_measures(state)
            
            # Setup layout
            chart_positions = []
            if n_states <= 2:
                chart_positions = [LEFT * 3, RIGHT * 3][:n_states]
            elif n_states <= 4:
                chart_positions = [UL * 2, UR * 2, DL * 2, DR * 2][:n_states]
            else:
                # Arrange in grid
                cols = int(np.ceil(np.sqrt(n_states)))
                rows = int(np.ceil(n_states / cols))
                for i, name in enumerate(states_dict.keys()):
                    row = i // cols
                    col = i % cols
                    pos = UP * (rows/2 - row - 0.5) * 3 + RIGHT * (col - cols/2 + 0.5) * 4
                    chart_positions.append(pos)
            
            # Create charts for each state
            charts = []
            dashboards = []
            
            for i, (name, (photon_nums, probs)) in enumerate(distributions.items()):
                # Create distribution chart
                chart = self.create_photon_distribution_chart(
                    photon_nums, probs,
                    position=chart_positions[i] + UP * 1,
                    state_type=name.lower()
                )
                charts.append(chart)
                
                # Create coherence dashboard
                dashboard = self.create_coherence_dashboard(
                    coherence_measures[name],
                    position=chart_positions[i] + DOWN * 1.5
                )
                dashboards.append(dashboard)
            
            # Animate appearance of charts
            self.play(
                *[FadeIn(chart) for chart in charts],
                run_time=duration * 0.3
            )
            
            self.wait(duration * 0.2)
            
            # Animate appearance of dashboards
            self.play(
                *[FadeIn(dashboard) for dashboard in dashboards],
                run_time=duration * 0.3
            )
            
            self.wait(duration * 0.2)
            
            # Highlight comparisons
            # Compare Mandel Q values
            mandel_values = [coherence_measures[name]['mandel_q'] for name in states_dict.keys()]
            min_mandel_idx = np.argmin(mandel_values)
            max_mandel_idx = np.argmax(mandel_values)
            
            if min_mandel_idx != max_mandel_idx:
                # Highlight most sub-Poissonian
                self.play(
                    dashboards[min_mandel_idx].animate.scale(1.1).set_stroke(color=BLUE, width=3),
                    run_time=1.0
                )
                
                sub_poisson_label = Text("Most Sub-Poissonian", font_size=12, color=BLUE).next_to(
                    dashboards[min_mandel_idx], DOWN
                )
                self.play(Write(sub_poisson_label), run_time=0.5)
                
                self.wait(1.0)
                
                # Reset and highlight most super-Poissonian
                self.play(
                    dashboards[min_mandel_idx].animate.scale(1/1.1).set_stroke(color=WHITE, width=2),
                    FadeOut(sub_poisson_label),
                    run_time=0.5
                )
                
                self.play(
                    dashboards[max_mandel_idx].animate.scale(1.1).set_stroke(color=RED, width=3),
                    run_time=1.0
                )
                
                super_poisson_label = Text("Most Super-Poissonian", font_size=12, color=RED).next_to(
                    dashboards[max_mandel_idx], DOWN
                )
                self.play(Write(super_poisson_label), run_time=0.5)
                
                self.wait(1.0)
                
                # Reset
                self.play(
                    dashboards[max_mandel_idx].animate.scale(1/1.1).set_stroke(color=WHITE, width=2),
                    FadeOut(super_poisson_label),
                    run_time=0.5
                )
            
        except Exception as e:
            raise AnimationError(
                f"Failed to animate photon statistics comparison: {str(e)}",
                animation_type="photon_comparison"
            )