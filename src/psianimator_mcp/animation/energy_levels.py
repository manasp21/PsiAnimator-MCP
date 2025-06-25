"""
Energy Level Diagram Visualization for PsiAnimator-MCP

Provides visualization of atomic energy levels, transitions, population dynamics,
and laser interactions for multi-level quantum systems.
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


class EnergyLevelDiagram(QuantumScene):
    """
    Energy level diagram visualization for atomic systems.
    
    Provides visualization of energy levels, transitions, population dynamics,
    and coherent driving for multi-level atomic systems.
    """
    
    def __init__(self, **kwargs):
        """Initialize energy level diagram scene."""
        super().__init__(**kwargs)
        
        # Diagram settings
        self.level_width = 3.0
        self.level_spacing = 1.2
        self.transition_arrow_scale = 0.8
        self.population_bar_width = 0.3
        
        # Colors for energy level elements
        self.energy_colors = {
            'ground_state': BLUE,
            'excited_state': RED,
            'transition_allowed': GREEN,
            'transition_forbidden': GRAY,
            'laser_field': YELLOW,
            'population': ORANGE,
            'coherence': PURPLE,
            'decay': RED_B
        }
        
        logger.debug("EnergyLevelDiagram initialized")
    
    def create_energy_level(self,
                          energy: float,
                          degeneracy: int = 1,
                          label: str = "",
                          position: np.ndarray = ORIGIN) -> VGroup:
        """
        Create an energy level line with label.
        
        Parameters
        ----------
        energy : float
            Energy of the level (for vertical positioning)
        degeneracy : int, optional
            Degeneracy of the level
        label : str, optional
            Label for the level
        position : np.ndarray, optional
            Base position for the level
            
        Returns
        -------
        VGroup
            Energy level visualization
        """
        level_group = VGroup()
        
        # Determine color based on energy (ground vs excited)
        if energy == 0 or energy == min(energy, 0):
            color = self.energy_colors['ground_state']
        else:
            color = self.energy_colors['excited_state']
        
        # Energy level line
        level_line = Line(
            start=position + LEFT * self.level_width / 2,
            end=position + RIGHT * self.level_width / 2,
            color=color,
            stroke_width=4
        ).shift(UP * energy * self.level_spacing)
        
        level_group.add(level_line)
        
        # Degeneracy indicator (multiple lines for degenerate levels)
        if degeneracy > 1:
            for i in range(1, min(degeneracy, 5)):  # Show up to 4 additional lines
                deg_line = Line(
                    start=position + LEFT * self.level_width / 2,
                    end=position + RIGHT * self.level_width / 2,
                    color=color,
                    stroke_width=2,
                    stroke_opacity=0.6
                ).shift(UP * (energy * self.level_spacing + 0.05 * i))
                level_group.add(deg_line)
        
        # Energy label
        if label:
            energy_label = MathTex(label).next_to(level_line, RIGHT, buff=0.3)
        else:
            energy_label = MathTex(f"E = {energy:.2f}").next_to(level_line, RIGHT, buff=0.3)
        
        level_group.add(energy_label)
        
        # Level index for reference
        level_line.energy = energy
        level_line.degeneracy = degeneracy
        
        return level_group
    
    def create_transition_arrow(self,
                              from_energy: float,
                              to_energy: float,
                              transition_type: str = "allowed",
                              strength: float = 1.0,
                              position: np.ndarray = ORIGIN) -> VGroup:
        """
        Create transition arrow between energy levels.
        
        Parameters
        ----------
        from_energy : float
            Initial energy level
        to_energy : float
            Final energy level
        transition_type : str, optional
            Type of transition ('allowed', 'forbidden', 'laser')
        strength : float, optional
            Transition strength (affects arrow thickness)
        position : np.ndarray, optional
            Base position for the arrow
            
        Returns
        -------
        VGroup
            Transition arrow visualization
        """
        arrow_group = VGroup()
        
        # Determine arrow color and style
        if transition_type == "allowed":
            color = self.energy_colors['transition_allowed']
            stroke_width = max(2, strength * 4)
        elif transition_type == "forbidden":
            color = self.energy_colors['transition_forbidden']
            stroke_width = 1
        elif transition_type == "laser":
            color = self.energy_colors['laser_field']
            stroke_width = 3
        else:
            color = WHITE
            stroke_width = 2
        
        # Calculate arrow positions
        start_pos = position + UP * from_energy * self.level_spacing
        end_pos = position + UP * to_energy * self.level_spacing
        
        # Offset arrows to avoid overlap with level lines
        arrow_offset = LEFT * 0.5
        start_pos += arrow_offset
        end_pos += arrow_offset
        
        # Create arrow
        if from_energy < to_energy:
            # Absorption (upward arrow)
            arrow = Arrow(
                start=start_pos,
                end=end_pos,
                color=color,
                stroke_width=stroke_width,
                max_tip_length_to_length_ratio=0.1
            )
        else:
            # Emission (downward arrow)
            arrow = Arrow(
                start=start_pos,
                end=end_pos,
                color=color,
                stroke_width=stroke_width,
                max_tip_length_to_length_ratio=0.1
            )
        
        arrow_group.add(arrow)
        
        # Add transition frequency label
        freq = abs(to_energy - from_energy)
        if freq > 0.1:  # Only label significant transitions
            freq_label = MathTex(f"\\omega = {freq:.2f}").next_to(arrow, RIGHT, buff=0.2).scale(0.8)
            arrow_group.add(freq_label)
        
        return arrow_group
    
    def create_population_bars(self,
                             energy_levels: List[float],
                             populations: List[float],
                             position: np.ndarray = ORIGIN) -> VGroup:
        """
        Create population bars showing state populations.
        
        Parameters
        ----------
        energy_levels : list of float
            Energy levels
        populations : list of float
            Population of each level
        position : np.ndarray, optional
            Base position for the bars
            
        Returns
        -------
        VGroup
            Population bars visualization
        """
        pop_group = VGroup()
        
        max_pop = max(populations) if populations else 1.0
        
        for i, (energy, pop) in enumerate(zip(energy_levels, populations)):
            if pop > 1e-6:  # Only show significant populations
                # Population bar
                bar_height = pop / max_pop * 0.8
                bar = Rectangle(
                    width=self.population_bar_width,
                    height=bar_height,
                    fill_color=self.energy_colors['population'],
                    fill_opacity=0.7,
                    stroke_color=WHITE,
                    stroke_width=1
                )
                
                # Position bar next to energy level
                bar_pos = position + RIGHT * (self.level_width / 2 + 0.8) + UP * energy * self.level_spacing
                bar.move_to(bar_pos + UP * bar_height / 2)
                
                pop_group.add(bar)
                
                # Population value label
                pop_label = Text(f"{pop:.3f}", font_size=12).next_to(bar, RIGHT, buff=0.1)
                pop_group.add(pop_label)
        
        return pop_group
    
    def animate_population_dynamics(self,
                                  energy_levels: List[float],
                                  population_evolution: List[List[float]],
                                  time_points: List[float],
                                  transitions: Optional[List[Tuple[int, int]]] = None) -> None:
        """
        Animate population dynamics over time.
        
        Parameters
        ----------
        energy_levels : list of float
            Energy levels of the system
        population_evolution : list of list of float
            Population evolution [time_point][level]
        time_points : list of float
            Time points for the evolution
        transitions : list of tuple, optional
            Allowed transitions [(from_level, to_level), ...]
        """
        try:
            n_levels = len(energy_levels)
            
            # Create energy level diagram
            diagram_group = VGroup()
            
            # Create energy levels
            level_viz = []
            for i, energy in enumerate(energy_levels):
                level = self.create_energy_level(
                    energy, 
                    label=f"|{i}\\rangle",
                    position=ORIGIN
                )
                level_viz.append(level)
                diagram_group.add(level)
            
            # Add transitions if specified
            if transitions:
                for from_idx, to_idx in transitions:
                    if 0 <= from_idx < n_levels and 0 <= to_idx < n_levels:
                        transition = self.create_transition_arrow(
                            energy_levels[from_idx],
                            energy_levels[to_idx],
                            transition_type="allowed"
                        )
                        diagram_group.add(transition)
            
            self.add(diagram_group)
            
            # Create initial population bars
            initial_populations = population_evolution[0]
            pop_bars = self.create_population_bars(energy_levels, initial_populations)
            self.add(pop_bars)
            
            # Add time display
            time_text = Text(f"t = {time_points[0]:.2f}", font_size=16).to_corner(UR)
            self.add(time_text)
            
            # Animation function
            def update_populations(alpha):
                # Get current time index
                time_idx = int(alpha * (len(time_points) - 1))
                current_populations = population_evolution[time_idx]
                current_time = time_points[time_idx]
                
                # Update population bars
                new_pop_bars = self.create_population_bars(energy_levels, current_populations)
                pop_bars.become(new_pop_bars)
                
                # Update time display
                new_time_text = Text(f"t = {current_time:.2f}", font_size=16).to_corner(UR)
                time_text.become(new_time_text)
            
            # Run population dynamics animation
            self.play(
                UpdateFromAlphaFunc(lambda m, a: update_populations(a)),
                run_time=6.0,
                rate_func=linear
            )
            
            self.wait(2)
            
        except Exception as e:
            raise AnimationError(
                f"Failed to animate population dynamics: {str(e)}",
                animation_type="population_dynamics"
            )
    
    def create_coherence_visualization(self,
                                     density_matrix: qt.Qobj,
                                     energy_levels: List[float],
                                     position: np.ndarray = ORIGIN) -> VGroup:
        """
        Create visualization of coherences between energy levels.
        
        Parameters
        ----------
        density_matrix : qt.Qobj
            Density matrix of the system
        energy_levels : list of float
            Energy levels
        position : np.ndarray, optional
            Base position for visualization
            
        Returns
        -------
        VGroup
            Coherence visualization
        """
        coherence_group = VGroup()
        
        try:
            rho = validate_quantum_state(density_matrix)
            if rho.type != 'oper':
                raise ValidationError("Expected density matrix for coherence visualization")
            
            rho_matrix = rho.full()
            n_levels = len(energy_levels)
            
            # Visualize off-diagonal elements (coherences)
            for i in range(n_levels):
                for j in range(i + 1, n_levels):
                    coherence = rho_matrix[i, j]
                    
                    if abs(coherence) > 1e-6:  # Only show significant coherences
                        # Create coherence indicator
                        start_energy = energy_levels[i]
                        end_energy = energy_levels[j]
                        
                        # Position between the two levels
                        mid_energy = (start_energy + end_energy) / 2
                        coherence_pos = position + RIGHT * 1.5 + UP * mid_energy * self.level_spacing
                        
                        # Coherence magnitude and phase
                        magnitude = abs(coherence)
                        phase = np.angle(coherence)
                        
                        # Create coherence visualization
                        coherence_circle = Circle(
                            radius=magnitude * 0.3,
                            color=self.energy_colors['coherence'],
                            fill_opacity=0.6,
                            stroke_width=2
                        ).move_to(coherence_pos)
                        
                        # Phase indicator (arrow)
                        phase_arrow = Arrow(
                            start=coherence_pos,
                            end=coherence_pos + magnitude * 0.25 * np.array([np.cos(phase), np.sin(phase), 0]),
                            color=WHITE,
                            stroke_width=2,
                            max_tip_length_to_length_ratio=0.2
                        )
                        
                        coherence_group.add(coherence_circle, phase_arrow)
                        
                        # Label
                        coherence_label = MathTex(f"\\rho_{{{i},{j}}}").next_to(coherence_circle, DOWN, buff=0.1).scale(0.8)
                        coherence_group.add(coherence_label)
            
            return coherence_group
            
        except Exception as e:
            raise AnimationError(
                f"Failed to create coherence visualization: {str(e)}",
                animation_type="coherence_visualization"
            )