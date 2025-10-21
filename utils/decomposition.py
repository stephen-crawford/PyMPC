"""
Convex decomposition utilities for obstacle representation.

This module provides utilities for decomposing complex obstacles
into convex shapes (ellipsoids, polytopes) for use in MPC constraints.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
try:
    from scipy.spatial import ConvexHull
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class ConvexDecomposition:
    """
    Convex decomposition utilities for obstacle representation.
    
    This class provides methods for decomposing complex obstacles
    into convex shapes suitable for MPC constraints.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the decomposition utilities.
        
        Args:
            **kwargs: Additional parameters
        """
        self.parameters = kwargs
    
    def decompose_to_ellipsoids(self, 
                               obstacle_points: np.ndarray,
                               max_ellipsoids: int = 5,
                               tolerance: float = 0.1) -> List[Dict[str, Any]]:
        """
        Decompose a set of obstacle points into ellipsoids.
        
        Args:
            obstacle_points: Array of obstacle points [N, 2]
            max_ellipsoids: Maximum number of ellipsoids to use
            tolerance: Tolerance for decomposition accuracy
            
        Returns:
            List of ellipsoid dictionaries with keys:
            - 'center': center position [x, y]
            - 'shape': shape matrix (2x2)
            - 'rotation': rotation angle
        """
        if len(obstacle_points) < 3:
            raise ValueError("At least 3 points required for decomposition")
        
        # Compute convex hull
        hull = ConvexHull(obstacle_points)
        hull_points = obstacle_points[hull.vertices]
        
        # Decompose into ellipsoids
        ellipsoids = []
        remaining_points = hull_points.copy()
        
        while len(remaining_points) > 0 and len(ellipsoids) < max_ellipsoids:
            # Find best fitting ellipsoid for remaining points
            ellipsoid = self._fit_ellipsoid(remaining_points)
            ellipsoids.append(ellipsoid)
            
            # Remove points covered by this ellipsoid
            remaining_points = self._remove_covered_points(
                remaining_points, ellipsoid, tolerance
            )
        
        return ellipsoids
    
    def _fit_ellipsoid(self, points: np.ndarray) -> Dict[str, Any]:
        """
        Fit an ellipsoid to a set of points.
        
        Args:
            points: Array of points [N, 2]
            
        Returns:
            Ellipsoid dictionary
        """
        # Compute center
        center = np.mean(points, axis=0)
        
        # Center the points
        centered_points = points - center
        
        # Compute covariance matrix
        cov_matrix = np.cov(centered_points.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue magnitude
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Compute rotation angle
        rotation = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        
        # Compute shape matrix
        # Scale by eigenvalues with some padding
        scale_factor = 2.0  # 2-sigma coverage
        shape_matrix = np.diag(eigenvalues * scale_factor)
        
        # Rotate shape matrix
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        shape_matrix = R @ shape_matrix @ R.T
        
        return {
            'center': center,
            'shape': shape_matrix,
            'rotation': rotation
        }
    
    def _remove_covered_points(self, 
                              points: np.ndarray, 
                              ellipsoid: Dict[str, Any],
                              tolerance: float) -> np.ndarray:
        """
        Remove points covered by an ellipsoid.
        
        Args:
            points: Array of points [N, 2]
            ellipsoid: Ellipsoid dictionary
            tolerance: Tolerance for coverage
            
        Returns:
            Remaining points not covered by ellipsoid
        """
        center = ellipsoid['center']
        shape = ellipsoid['shape']
        
        # Compute inverse shape matrix
        try:
            shape_inv = np.linalg.inv(shape)
        except np.linalg.LinAlgError:
            # If singular, use identity
            shape_inv = np.eye(2)
        
        # Check which points are covered
        covered_mask = []
        for point in points:
            diff = point - center
            distance_sq = diff.T @ shape_inv @ diff
            covered_mask.append(distance_sq <= (1.0 + tolerance) ** 2)
        
        covered_mask = np.array(covered_mask)
        return points[~covered_mask]
    
    def decompose_to_polytopes(self, 
                              obstacle_points: np.ndarray,
                              max_vertices: int = 8) -> List[Dict[str, Any]]:
        """
        Decompose a set of obstacle points into polytopes.
        
        Args:
            obstacle_points: Array of obstacle points [N, 2]
            max_vertices: Maximum number of vertices per polytope
            
        Returns:
            List of polytope dictionaries with keys:
            - 'vertices': array of vertices [N, 2]
            - 'center': center position [x, y]
        """
        if len(obstacle_points) < 3:
            raise ValueError("At least 3 points required for decomposition")
        
        # Compute convex hull
        hull = ConvexHull(obstacle_points)
        hull_points = obstacle_points[hull.vertices]
        
        # If hull has few vertices, return as single polytope
        if len(hull_points) <= max_vertices:
            return [{
                'vertices': hull_points,
                'center': np.mean(hull_points, axis=0)
            }]
        
        # Decompose into smaller polytopes
        polytopes = []
        remaining_points = hull_points.copy()
        
        while len(remaining_points) > 0:
            # Select subset of points for this polytope
            n_vertices = min(max_vertices, len(remaining_points))
            selected_indices = np.linspace(0, len(remaining_points)-1, n_vertices, dtype=int)
            selected_points = remaining_points[selected_indices]
            
            # Create polytope
            polytope = {
                'vertices': selected_points,
                'center': np.mean(selected_points, axis=0)
            }
            polytopes.append(polytope)
            
            # Remove used points
            remaining_points = np.delete(remaining_points, selected_indices, axis=0)
        
        return polytopes
    
    def ellipsoid_to_constraint(self, 
                               ellipsoid: Dict[str, Any],
                               safety_margin: float = 0.5) -> Dict[str, Any]:
        """
        Convert an ellipsoid to a constraint dictionary.
        
        Args:
            ellipsoid: Ellipsoid dictionary
            safety_margin: Additional safety margin
            
        Returns:
            Constraint dictionary for use in MPC
        """
        center = ellipsoid['center']
        shape = ellipsoid['shape']
        
        # Add safety margin
        shape_with_margin = shape / (1 + safety_margin) ** 2
        
        return {
            'center': center,
            'shape': shape_with_margin,
            'rotation': ellipsoid.get('rotation', 0.0)
        }
    
    def polytope_to_constraints(self, 
                               polytope: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert a polytope to linear constraints.
        
        Args:
            polytope: Polytope dictionary
            
        Returns:
            List of linear constraint dictionaries
        """
        vertices = polytope['vertices']
        n_vertices = len(vertices)
        
        constraints = []
        
        # Create linear constraints for each edge
        for i in range(n_vertices):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n_vertices]
            
            # Compute edge normal (pointing outward)
            edge = v2 - v1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / np.linalg.norm(normal)
            
            # Create constraint: normal^T * (x - v1) >= 0
            A = normal.reshape(1, 2)
            B = np.zeros((1, 1))  # No control dependence
            c = np.dot(normal, v1)
            
            constraints.append({
                'A': A,
                'B': B,
                'c': c,
                'type': 'inequality'
            })
        
        return constraints
    
    def visualize_decomposition(self, 
                               obstacle_points: np.ndarray,
                               ellipsoids: List[Dict[str, Any]],
                               polytopes: Optional[List[Dict[str, Any]]] = None,
                               save_path: Optional[str] = None) -> None:
        """
        Visualize the decomposition results.
        
        Args:
            obstacle_points: Original obstacle points
            ellipsoids: Decomposed ellipsoids
            polytopes: Decomposed polytopes (optional)
            save_path: Path to save the plot (optional)
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot original points
        ax.scatter(obstacle_points[:, 0], obstacle_points[:, 1], 
                  c='red', s=50, alpha=0.7, label='Original Points')
        
        # Plot ellipsoids
        for i, ellipsoid in enumerate(ellipsoids):
            self._plot_ellipsoid(ax, ellipsoid, color=f'C{i}', 
                               label=f'Ellipsoid {i+1}')
        
        # Plot polytopes if provided
        if polytopes is not None:
            for i, polytope in enumerate(polytopes):
                self._plot_polytope(ax, polytope, color=f'C{i+len(ellipsoids)}',
                                  label=f'Polytope {i+1}')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Convex Decomposition')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_ellipsoid(self, ax, ellipsoid: Dict[str, Any], 
                       color: str = 'blue', label: str = 'Ellipsoid') -> None:
        """Plot an ellipsoid on the given axes."""
        center = ellipsoid['center']
        shape = ellipsoid['shape']
        rotation = ellipsoid.get('rotation', 0.0)
        
        # Generate points on unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        unit_circle = np.array([np.cos(theta), np.sin(theta)])
        
        # Transform to ellipsoid
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        
        # Scale by eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(shape)
        scale_matrix = np.diag(np.sqrt(eigenvalues))
        
        transformed_points = R @ scale_matrix @ unit_circle + center[:, np.newaxis]
        
        ax.plot(transformed_points[0], transformed_points[1], 
               color=color, linewidth=2, label=label)
    
    def _plot_polytope(self, ax, polytope: Dict[str, Any], 
                      color: str = 'green', label: str = 'Polytope') -> None:
        """Plot a polytope on the given axes."""
        vertices = polytope['vertices']
        
        # Close the polygon
        closed_vertices = np.vstack([vertices, vertices[0]])
        
        ax.plot(closed_vertices[:, 0], closed_vertices[:, 1], 
               color=color, linewidth=2, label=label)
