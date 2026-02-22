"""
Trajectory management for Visual Odometry and VSLAM.

This module provides classes for managing and visualizing camera trajectories.
"""

import numpy as np
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .pose_estimation import (
    compose_transformation,
    transformation_to_matrix,
    matrix_to_transformation,
)


class Trajectory:
    """
    Camera trajectory manager.

    Stores and manages the sequence of camera poses for Visual Odometry.
    Each pose represents the camera position and orientation in world coordinates.

    Attributes:
        poses: List of 4x4 transformation matrices
        positions: List of camera positions (x, y, z)
    """

    def __init__(self):
        """Initialize empty trajectory with identity pose."""
        self.poses: List[np.ndarray] = [np.eye(4, dtype=np.float64)]
        self.positions: List[np.ndarray] = [np.zeros(3, dtype=np.float64)]

    def add_relative_pose(self, R: np.ndarray, t: np.ndarray) -> None:
        """
        Add a new pose relative to the last pose.

        Args:
            R: Rotation matrix (3, 3) from previous to current frame
            t: Translation vector (3, 1) or (3,) from previous to current frame
        """
        # Get last pose
        last_pose = self.poses[-1]
        R_last, t_last = matrix_to_transformation(last_pose)

        # Compose transformation
        R_new, t_new = compose_transformation(R_last, t_last, R, t)

        # Create and store new pose
        new_pose = transformation_to_matrix(R_new, t_new)
        self.poses.append(new_pose)
        self.positions.append(t_new.ravel())

    def get_current_pose(self) -> np.ndarray:
        """
        Get the current (latest) pose.

        Returns:
            Current pose as 4x4 transformation matrix
        """
        return self.poses[-1]

    def get_current_position(self) -> np.ndarray:
        """
        Get the current (latest) camera position.

        Returns:
            Current position as (3,) array
        """
        return self.positions[-1]

    def get_positions_array(self) -> np.ndarray:
        """
        Get all positions as numpy array.

        Returns:
            Array of shape (N, 3) containing all camera positions
        """
        return np.array(self.positions)

    def get_trajectory_length(self) -> float:
        """
        Calculate total trajectory length.

        Returns:
            Total distance traveled by the camera
        """
        positions = self.get_positions_array()
        if len(positions) < 2:
            return 0.0

        diffs = np.diff(positions, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return np.sum(distances)

    def __len__(self) -> int:
        """Return number of poses in trajectory."""
        return len(self.poses)

    def plot_2d(
        self,
        ax: Optional[plt.Axes] = None,
        show_orientation: bool = False,
        title: str = "Camera Trajectory (Top View)",
    ) -> plt.Axes:
        """
        Plot 2D trajectory (top view: X-Z plane).

        Args:
            ax: Matplotlib axes to plot on. If None, creates new figure.
            show_orientation: If True, draw camera orientation arrows.
            title: Plot title.

        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        positions = self.get_positions_array()

        # Plot trajectory path
        ax.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=2, label='Trajectory')

        # Plot start and end points
        ax.plot(positions[0, 0], positions[0, 2], 'go', markersize=10, label='Start')
        ax.plot(positions[-1, 0], positions[-1, 2], 'ro', markersize=10, label='End')

        # Plot orientation arrows if requested
        if show_orientation:
            for i in range(0, len(self.poses), max(1, len(self.poses) // 20)):
                R, t = matrix_to_transformation(self.poses[i])
                # Z-axis of camera (pointing forward)
                forward = R[:, 2] * 0.5  # Scale for visibility
                ax.arrow(
                    t[0], t[2],
                    forward[0], forward[2],
                    head_width=0.1, head_length=0.1,
                    fc='red', ec='red', alpha=0.5
                )

        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Z (meters)')
        ax.set_title(title)
        ax.axis('equal')
        ax.grid(True)
        ax.legend()

        return ax

    def plot_3d(
        self,
        ax: Optional[Axes3D] = None,
        show_orientation: bool = False,
        title: str = "Camera Trajectory (3D)",
    ) -> Axes3D:
        """
        Plot 3D trajectory.

        Args:
            ax: Matplotlib 3D axes to plot on. If None, creates new figure.
            show_orientation: If True, draw camera orientation axes.
            title: Plot title.

        Returns:
            Matplotlib 3D axes object
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

        positions = self.get_positions_array()

        # Plot trajectory path
        ax.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            'b-', linewidth=2, label='Trajectory'
        )

        # Plot start and end points
        ax.scatter(
            positions[0, 0], positions[0, 1], positions[0, 2],
            c='green', marker='o', s=100, label='Start'
        )
        ax.scatter(
            positions[-1, 0], positions[-1, 1], positions[-1, 2],
            c='red', marker='o', s=100, label='End'
        )

        # Plot orientation axes if requested
        if show_orientation:
            for i in range(0, len(self.poses), max(1, len(self.poses) // 20)):
                R, t = matrix_to_transformation(self.poses[i])
                scale = 0.3  # Scale for visibility

                # Draw coordinate axes
                colors = ['r', 'g', 'b']
                for j, color in enumerate(colors):
                    axis = R[:, j] * scale
                    ax.plot(
                        [t[0], t[0] + axis[0]],
                        [t[1], t[1] + axis[1]],
                        [t[2], t[2] + axis[2]],
                        color=color, alpha=0.5, linewidth=1
                    )

        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title(title)
        ax.legend()

        # Set equal aspect ratio
        positions = self.get_positions_array()
        max_range = np.array([
            positions[:, 0].max() - positions[:, 0].min(),
            positions[:, 1].max() - positions[:, 1].min(),
            positions[:, 2].max() - positions[:, 2].min()
        ]).max() / 2.0

        mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        return ax

    def save_to_file(self, filepath: str) -> None:
        """
        Save trajectory to file.

        Args:
            filepath: Path to output file
        """
        data = {
            'poses': np.array(self.poses),
            'positions': np.array(self.positions),
        }
        np.savez(filepath, **data)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'Trajectory':
        """
        Load trajectory from file.

        Args:
            filepath: Path to trajectory file

        Returns:
            Loaded Trajectory object
        """
        data = np.load(filepath)
        trajectory = cls()
        trajectory.poses = list(data['poses'])
        trajectory.positions = list(data['positions'])
        return trajectory
