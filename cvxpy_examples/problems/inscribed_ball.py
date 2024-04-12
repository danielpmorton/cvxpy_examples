"""Detemining the maximum radius inscribed ball in a polyhedron

As an example, this is commonly used in dexterous manipulation (i.e. the Ferrari Canny metric), which defines the
stability of a grasp to disturbances wrenches (forces and torques)
"""

from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import cvxpy as cp

from cvxpy_examples.utils.cp_problem import OptimizationProblem


class InscribedBallProblem(OptimizationProblem):
    """Determine the largest ball inside a polyhedron defined by Ax <= b

    This is parameterized for re-solves with different ball center points, holding the polyhedron fixed.
    A different parameterization may allow for updating the polyhedron repeatedly, for instance.

    Args:
        A (np.ndarray): Array defining hyperplane normals, shape (n_planes, dim)
        b (npt.ArrayLike): Array defining hyperplane offset, shape (n_planes)
        center (Optional[npt.ArrayLike]): If fixing the center of the ball, include its position here.
            Defaults to None (determine the optimal center of the ball as well)
        verbose (bool, optional): Whether to print info about the optimization problem after it is solved.
            Defaults to True
    """

    def __init__(
        self,
        A: np.ndarray,
        b: npt.ArrayLike,
        center: Optional[npt.ArrayLike] = None,
        verbose: bool = False,
    ):
        n_hyperplanes, dim = A.shape
        b = np.ravel(b)
        assert len(b) == n_hyperplanes
        self.variable_center = center is None
        if self.variable_center:
            center = cp.Variable(dim)
        else:
            center = np.ravel(center)
            assert len(center) == dim
            center = cp.Parameter(dim, value=center)

        self.center = center
        self.r = cp.Variable()
        self.A = A
        self.b = b
        # Construct the problem
        super().__init__(verbose)

    @property
    def objective(self) -> Union[cp.Maximize, cp.Minimize]:
        return cp.Maximize(self.r)

    @property
    def constraints(self) -> list[cp.Expression]:
        return [
            self.A @ self.center + self.r * cp.norm(self.A, axis=1) <= self.b,
            self.r >= 0,
        ]


    @property
    def optimal_center(self) -> np.ndarray:
        """Optimal center of the ball, shape (dim,)"""
        # Note: this is valid if center is a Variable or Parameter
        if self.center.value is None:
            raise ValueError("Cannot return the center, problem has not been solved")
        return self.center.value

    @property
    def optimal_radius(self) -> float:
        """Optimal radius of the ball"""
        if self.r.value is None:
            raise ValueError("Cannot return the radius, problem has not been solved")
        return self.r.value

    def update_center(self, new_center: npt.ArrayLike) -> None:
        """Update the center of the ball (if the center is parameterized, and not a variable)

        Args:
            new_center (npt.ArrayLike): New center, shape (dim,)
        """
        if self.variable_center:
            raise ValueError("Center is not a Parameter")
        new_center = np.ravel(new_center)
        if new_center.shape != self.center.shape:
            raise ValueError("Shape mismatch, cannot update the parameter")
        self.center.value = new_center

