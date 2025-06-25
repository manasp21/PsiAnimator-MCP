"""
Wigner Function Visualization for PsiAnimator-MCP

Provides visualization of Wigner quasi-probability distributions in phase space
for harmonic oscillator states and cavity QED systems.
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


class WignerFunction(QuantumScene):
    """
    Wigner function visualization for quantum states in phase space.
    
    Provides real-time visualization of Wigner quasi-probability distributions
    with support for animation of state evolution, squeezing effects, and
    quantum interference patterns.
    """
    
    def __init__(self, **kwargs):
        """Initialize Wigner function scene."""
        super().__init__(**kwargs)
        
        # Phase space settings
        self.x_range = (-4, 4, 0.2)
        self.p_range = (-4, 4, 0.2)
        self.resolution = 64
        self.contour_levels = 20
        
        # Visualization settings
        self.colormap = "RdBu"  # Red-Blue colormap for positive/negative values
        self.show_contours = True
        self.show_negative_regions = True
        self.phase_space_scale = 2.0
        
        # Colors for Wigner function elements
        self.wigner_colors = {
            'positive': BLUE,
            'negative': RED,
            'zero_line': WHITE,
            'axes': GRAY,
            'contour_positive': BLUE_B,
            'contour_negative': RED_B,
            'classical_trajectory': GREEN,
            'quantum_trajectory': YELLOW
        }
        
        logger.debug("WignerFunction initialized")
    
    def calculate_wigner_function(self,
                                state: qt.Qobj,
                                x_range: Optional[Tuple[float, float, float]] = None,
                                p_range: Optional[Tuple[float, float, float]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Wigner function for a quantum state.
        
        Parameters
        ----------
        state : qt.Qobj
            Quantum state (ket or density matrix)
        x_range : tuple, optional
            Position range (min, max, step)
        p_range : tuple, optional
            Momentum range (min, max, step)
            
        Returns
        -------
        tuple
            (X, P, W) arrays for position, momentum, and Wigner function values
        """
        try:
            state = validate_quantum_state(state)
            
            # Use default ranges if not specified
            if x_range is None:
                x_range = self.x_range
            if p_range is None:
                p_range = self.p_range
            
            # Create position and momentum arrays
            x_vals = np.arange(*x_range)
            p_vals = np.arange(*p_range)
            
            # Calculate Wigner function using QuTip
            if state.type == 'ket':
                rho = state * state.dag()
            else:
                rho = state
            
            # Use QuTip's wigner function
            W = qt.wigner(rho, x_vals, p_vals)
            
            # Create meshgrid for plotting
            X, P = np.meshgrid(x_vals, p_vals)
            
            return X, P, W
            
        except Exception as e:
            raise AnimationError(
                f"Failed to calculate Wigner function: {str(e)}",
                animation_type="wigner_calculation"
            )
    
    def create_phase_space_axes(self, position: np.ndarray = ORIGIN) -> VGroup:
        """
        Create phase space coordinate axes.
        
        Parameters
        ----------
        position : np.ndarray, optional
            Center position for the axes
            
        Returns
        -------
        VGroup
            Phase space axes visualization
        """
        axes_group = VGroup()
        
        # Create axes
        axes = Axes(
            x_range=[self.x_range[0], self.x_range[1], 1],
            y_range=[self.p_range[0], self.p_range[1], 1],
            x_length=self.phase_space_scale * 2,
            y_length=self.phase_space_scale * 2,
            axis_config={"color": self.wigner_colors['axes']},
            tips=True
        ).move_to(position)
        
        axes_group.add(axes)
        
        # Add axis labels
        x_label = MathTex("x").next_to(axes.x_axis.get_end(), RIGHT)
        p_label = MathTex("p").next_to(axes.y_axis.get_end(), UP)
        
        axes_group.add(x_label, p_label)
        
        # Add grid lines
        x_grid = VGroup(*[
            Line(
                start=axes.c2p(x, self.p_range[0]),
                end=axes.c2p(x, self.p_range[1]),
                color=self.wigner_colors['axes'],
                stroke_width=0.5,
                stroke_opacity=0.3
            )
            for x in np.arange(self.x_range[0], self.x_range[1] + 1, 1)
        ])
        
        p_grid = VGroup(*[
            Line(
                start=axes.c2p(self.x_range[0], p),
                end=axes.c2p(self.x_range[1], p),
                color=self.wigner_colors['axes'],
                stroke_width=0.5,
                stroke_opacity=0.3
            )
            for p in np.arange(self.p_range[0], self.p_range[1] + 1, 1)
        ])
        
        axes_group.add(x_grid, p_grid)
        
        return axes_group
    
    def create_wigner_surface(self,
                            X: np.ndarray,
                            P: np.ndarray,
                            W: np.ndarray,
                            position: np.ndarray = ORIGIN) -> VGroup:
        """
        Create 3D surface plot of Wigner function.
        
        Parameters
        ----------
        X, P, W : np.ndarray
            Meshgrid arrays for position, momentum, and Wigner values
        position : np.ndarray, optional
            Base position for the surface
            
        Returns
        -------
        VGroup
            Wigner function surface visualization
        """
        surface_group = VGroup()
        
        try:
            # Normalize coordinates to scene scale
            x_norm = (X - X.min()) / (X.max() - X.min()) * self.phase_space_scale * 2 - self.phase_space_scale
            p_norm = (P - P.min()) / (P.max() - P.min()) * self.phase_space_scale * 2 - self.phase_space_scale
            
            # Scale Wigner function values for visualization
            w_scale = 2.0
            W_scaled = W * w_scale
            
            # Create surface using rectangular patches
            for i in range(X.shape[0] - 1):
                for j in range(X.shape[1] - 1):
                    # Get corner values
                    corners = [
                        [x_norm[i, j], p_norm[i, j], W_scaled[i, j]],
                        [x_norm[i+1, j], p_norm[i+1, j], W_scaled[i+1, j]],
                        [x_norm[i+1, j+1], p_norm[i+1, j+1], W_scaled[i+1, j+1]],
                        [x_norm[i, j+1], p_norm[i, j+1], W_scaled[i, j+1]]
                    ]
                    
                    # Average height for color determination
                    avg_height = np.mean([c[2] for c in corners])
                    
                    # Determine color based on sign and magnitude
                    if avg_height > 0:
                        color = self.wigner_colors['positive']
                        opacity = min(0.8, abs(avg_height) / (w_scale * 0.5))
                    else:
                        color = self.wigner_colors['negative']
                        opacity = min(0.8, abs(avg_height) / (w_scale * 0.5))
                    
                    # Create surface patch
                    if abs(avg_height) > 0.01:  # Only show significant values
                        patch = Polygon(
                            *[[c[0], c[1], c[2]] for c in corners],
                            fill_color=color,
                            fill_opacity=opacity,
                            stroke_width=0
                        )
                        surface_group.add(patch)
            
            surface_group.move_to(position)
            return surface_group
            
        except Exception as e:
            raise AnimationError(
                f"Failed to create Wigner surface: {str(e)}",
                animation_type="wigner_surface"
            )
    
    def create_wigner_contours(self,
                             X: np.ndarray,
                             P: np.ndarray,
                             W: np.ndarray,
                             axes: Axes,
                             position: np.ndarray = ORIGIN) -> VGroup:
        """
        Create contour plot of Wigner function.
        
        Parameters
        ----------
        X, P, W : np.ndarray
            Meshgrid arrays for position, momentum, and Wigner values
        axes : Axes
            Manim axes object for coordinate transformation
        position : np.ndarray, optional
            Base position for contours
            
        Returns
        -------
        VGroup
            Wigner function contour visualization
        """
        contour_group = VGroup()
        
        try:
            # Determine contour levels
            w_max = np.max(np.abs(W))
            if w_max > 0:
                positive_levels = np.linspace(0.1 * w_max, w_max, self.contour_levels // 2)
                negative_levels = np.linspace(-w_max, -0.1 * w_max, self.contour_levels // 2)
                levels = np.concatenate([negative_levels, positive_levels])
            else:
                levels = [0]
            
            # Create contour lines (simplified implementation)
            for level in levels:
                contour_points = []
                
                # Find points where W crosses the level
                for i in range(X.shape[0] - 1):
                    for j in range(X.shape[1] - 1):
                        # Check if contour passes through this cell
                        values = [W[i, j], W[i+1, j], W[i+1, j+1], W[i, j+1]]
                        if (min(values) <= level <= max(values)) and (max(values) - min(values) > 1e-10):
                            # Approximate contour point (simplified)
                            x_point = (X[i, j] + X[i+1, j+1]) / 2
                            p_point = (P[i, j] + P[i+1, j+1]) / 2
                            scene_point = axes.c2p(x_point, p_point)
                            contour_points.append(scene_point)
                
                # Create contour line if we have enough points
                if len(contour_points) > 2:
                    color = self.wigner_colors['contour_positive'] if level > 0 else self.wigner_colors['contour_negative']
                    
                    # Connect nearby points to form contour lines
                    for k in range(len(contour_points) - 1):
                        if np.linalg.norm(np.array(contour_points[k+1]) - np.array(contour_points[k])) < 0.5:
                            line = Line(
                                start=contour_points[k],
                                end=contour_points[k+1],
                                color=color,
                                stroke_width=2
                            )
                            contour_group.add(line)
            
            return contour_group
            
        except Exception as e:
            raise AnimationError(
                f"Failed to create Wigner contours: {str(e)}",
                animation_type="wigner_contours"
            )
    
    def animate_wigner_evolution(self,
                               initial_state: qt.Qobj,
                               time_evolution: List[qt.Qobj],
                               show_3d: bool = True,
                               show_contours: bool = True) -> None:
        """
        Animate Wigner function evolution over time.
        
        Parameters
        ----------
        initial_state : qt.Qobj
            Initial quantum state
        time_evolution : list of qt.Qobj
            States at different time points
        show_3d : bool, optional
            Whether to show 3D surface
        show_contours : bool, optional
            Whether to show contour plot
        """
        try:
            # Setup phase space axes
            axes = self.create_phase_space_axes()
            self.add(axes)
            
            # Calculate initial Wigner function
            X, P, W_initial = self.calculate_wigner_function(initial_state)
            
            # Create initial visualization
            if show_3d:
                wigner_surface = self.create_wigner_surface(X, P, W_initial, position=UP * 1)
                self.add(wigner_surface)
            
            if show_contours:
                contours = self.create_wigner_contours(X, P, W_initial, axes.get_axes())
                self.add(contours)
            
            # Add state information
            state_info = VGroup()
            if hasattr(initial_state, 'type'):
                state_type = Text(f"State: {initial_state.type}", font_size=16).to_corner(UL)
                state_info.add(state_type)
            
            # Add Wigner function properties
            w_max = np.max(W_initial)
            w_min = np.min(W_initial)
            
            wigner_stats = VGroup(
                Text(f"W_max: {w_max:.3f}", font_size=14),
                Text(f"W_min: {w_min:.3f}", font_size=14),
                Text(f"Non-classical: {w_min < -1e-10}", font_size=14)
            ).arrange(DOWN, aligned_edge=LEFT).to_corner(UR)
            
            state_info.add(wigner_stats)
            self.add(state_info)
            
            # Animation function
            def update_wigner(alpha):
                # Get current state
                idx = int(alpha * (len(time_evolution) - 1))
                current_state = time_evolution[idx]
                
                # Calculate new Wigner function
                X_new, P_new, W_new = self.calculate_wigner_function(current_state)
                
                # Update visualizations
                if show_3d:
                    new_surface = self.create_wigner_surface(X_new, P_new, W_new, position=UP * 1)
                    wigner_surface.become(new_surface)
                
                if show_contours:
                    new_contours = self.create_wigner_contours(X_new, P_new, W_new, axes.get_axes())
                    contours.become(new_contours)
                
                # Update statistics
                w_max_new = np.max(W_new)
                w_min_new = np.min(W_new)
                
                new_stats = VGroup(
                    Text(f"W_max: {w_max_new:.3f}", font_size=14),
                    Text(f"W_min: {w_min_new:.3f}", font_size=14),
                    Text(f"Non-classical: {w_min_new < -1e-10}", font_size=14)
                ).arrange(DOWN, aligned_edge=LEFT).to_corner(UR)
                
                wigner_stats.become(new_stats)
            
            # Run evolution animation
            self.play(
                UpdateFromAlphaFunc(lambda m, a: update_wigner(a)),
                run_time=6.0,
                rate_func=linear
            )
            
        except Exception as e:
            raise AnimationError(
                f"Failed to animate Wigner evolution: {str(e)}",
                animation_type="wigner_evolution"
            )
    
    def create_phase_space_trajectory(self,
                                    classical_trajectory: List[Tuple[float, float]],
                                    quantum_trajectory: Optional[List[Tuple[float, float]]] = None,
                                    axes: Optional[Axes] = None) -> VGroup:
        """
        Create trajectory visualization in phase space.
        
        Parameters
        ----------
        classical_trajectory : list of tuple
            Classical trajectory points [(x, p), ...]
        quantum_trajectory : list of tuple, optional
            Quantum trajectory points (expectation values)
        axes : Axes, optional
            Axes for coordinate transformation
            
        Returns
        -------
        VGroup
            Trajectory visualization
        """
        trajectory_group = VGroup()
        
        if axes is None:
            axes = self.create_phase_space_axes().get_axes()
        
        # Classical trajectory
        if classical_trajectory:
            classical_points = [axes.c2p(x, p) for x, p in classical_trajectory]
            
            if len(classical_points) > 1:
                classical_path = VMobject()
                classical_path.set_points_as_corners(classical_points)
                classical_path.set_color(self.wigner_colors['classical_trajectory'])
                classical_path.set_stroke_width(3)
                trajectory_group.add(classical_path)
                
                # Add starting point marker
                start_marker = Dot(classical_points[0], color=self.wigner_colors['classical_trajectory'], radius=0.08)
                trajectory_group.add(start_marker)
        
        # Quantum trajectory (expectation values)
        if quantum_trajectory:
            quantum_points = [axes.c2p(x, p) for x, p in quantum_trajectory]
            
            if len(quantum_points) > 1:
                quantum_path = VMobject()
                quantum_path.set_points_as_corners(quantum_points)
                quantum_path.set_color(self.wigner_colors['quantum_trajectory'])
                quantum_path.set_stroke_width(2)
                quantum_path.set_stroke(opacity=0.8)
                trajectory_group.add(quantum_path)
                
                # Add uncertainty ellipses at key points
                for i in range(0, len(quantum_points), len(quantum_points) // 5):
                    ellipse = Ellipse(
                        width=0.2, height=0.2,
                        color=self.wigner_colors['quantum_trajectory'],
                        fill_opacity=0.1
                    ).move_to(quantum_points[i])
                    trajectory_group.add(ellipse)
        
        return trajectory_group