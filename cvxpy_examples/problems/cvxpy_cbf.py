"""Control Barrier Functions (CBFs) in CVXPY (and some Jax)

This code is DPP-parameterized for efficient re-solves, and uses Jax for automatic differentiation
of the barrier function

NOTE: This is intended more as a tutorial/intro to CBFs rather than high-performance code.
If you want really high-performance CBFs, use CBFpy.
https://github.com/danielpmorton/cbfpy

Or, if you are interested in CBFs for manipulator control, check out OSCBF
https://github.com/StanfordASL/oscbf
"""

from functools import partial
from typing import Union, Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np
import cvxpy as cp
import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike


class CBF(ABC):
    """Control Barrier Function (CBF) class, using CVXPY and Jax

    Args:
        n (int): State dimension
        m (int): Control dimension
        solver (Optional[str]): CVXPY solver to use. Defaults to None (use default solver)
    """

    def __init__(self, n: int, m: int, solver: Optional[str] = None):
        assert (
            isinstance(n, int) and n > 0
        ), "State dimension must be a positive integer"
        assert (
            isinstance(m, int) and m > 0
        ), "Control dimension must be a positive integer"
        assert (
            isinstance(solver, str) or solver is None
        ), "Solver must be a string or None"

        self.n = n
        self.m = m
        self.solver = solver

        # Test if the barrier function(s) are provided and evaluate its dimension
        test_barrier = self.h(jnp.zeros(self.n))
        if test_barrier.ndim != 1:
            raise ValueError("Barrier function must output a 1D array")
        self.num_barr = test_barrier.shape[0]

        # Take the jacobian of the barrier function(s)
        self._dh_dz = jax.jacobian(self.h)

        # Construct CVXPY optimization variables, parameters, and problem
        self.u = cp.Variable(self.m)  # Control
        self.u_des = cp.Parameter(self.m)  # Desired control
        self.Lfh_eval = cp.Parameter(self.num_barr)
        self.Lgh_eval = cp.Parameter((self.num_barr, self.m))
        self.h_eval = cp.Parameter(self.num_barr)
        self.prob = cp.Problem(self.objective, self.constraints)

    @abstractmethod
    def f(self, z: ArrayLike) -> ArrayLike:
        """The uncontrolled dynamics function. Possibly nonlinear, and locally Lipschitz

        i.e. the function f, such that z_dot = f(z) + g(z) u

        Args:
            z (ArrayLike): The state, shape (n,)

        Returns:
            ArrayLike: Propagated dynamics, shape (n,)
        """
        raise NotImplementedError(
            "The uncontrolled dynamics function f must be provided"
        )

    @abstractmethod
    def g(self, z: ArrayLike) -> ArrayLike:
        """The control affine dynamics function. Locally Lipschitz.

        i.e. the function g, such that z_dot = f(z) + g(z) u

        Args:
            z (ArrayLike): The state, shape (n,)

        Returns:
            ArrayLike: Control matrix, shape (n, m)
        """
        raise NotImplementedError("The control dynamics function g must be provided")

    @abstractmethod
    def h(self, z: ArrayLike) -> ArrayLike:
        """Barrier function(s). Output must be a 1D array, with length dictated by how many barriers are used

        Args:
            z (ArrayLike): State, shape (n,)

        Returns:
            ArrayLike: Barrier function(s), shape (num_barr,)
        """
        raise NotImplementedError("The barrier function h must be provided")

    @abstractmethod
    def alpha(self, h_z: ArrayLike) -> ArrayLike:
        """An extended class Kappa_inf function, dictating the "gain"/"aggressiveness" of the barrier function

        For reference, a class Kappa function is a monotonically increasing function which passes through the origin.
        A simple example is alpha(h_z) = h_z

        Args:
            h_z (ArrayLike): Evaluation of the barrier function(s) at the current state

        Returns:
            ArrayLike: alpha(h(z)), shape (num_barr,)
        """
        raise NotImplementedError("The alpha function must be provided")

    def dh_dz(self, z: ArrayLike) -> ArrayLike:
        """Jacobian of the barrier function(s) wrt the state

        Args:
            z (ArrayLike): State, shape (n,)

        Returns:
            ArrayLike: dh/dz, shape (num_barr, n)
        """
        return jax.jacfwd(self.h)(z)

    def Lfh(self, z: ArrayLike) -> ArrayLike:
        """Lie derivative of the barrier function(s) wrt the autonomous dynamics f(z)

        Args:
            z (ArrayLike): State, shape (n,)

        Returns:
            ArrayLike: Lfh, shape (num_barr,)
        """
        # Note: this can be computed faster using JVPs (see CBFpy for more details)
        return self.dh_dz(z) @ self.f(z)

    def Lgh(self, z: ArrayLike) -> ArrayLike:
        """Lie derivative of the barrier function(s) wrt the control dynamics g(z)u

        Args:
            z (ArrayLike): State, shape (n,)

        Returns:
            ArrayLike: Lgh, shape (num_barr, m)
        """
        # Note: this can be computed faster using JVPs (see CBFpy for more details)
        return self.dh_dz(z) @ self.g(z)

    @property
    def objective(self) -> Union[cp.Minimize, cp.Maximize]:
        """CBF optimization objective

        Defaults to the standard min-norm objective: minimize ||u - u_des||_{2}^{2}

        Returns:
            Union[cp.Minimize, cp.Maximize]: CVXPY objective expression
        """
        return cp.Minimize(cp.sum_squares(self.u - self.u_des))

    @property
    def constraints(self) -> list[cp.Expression]:
        """CBF constraints

        These enforce the barrier(s) and ensure that h_dot >= -alpha(h(z))

        Returns:
            list[cp.Expression]: Constraints, length = num_barr
        """
        return [self.Lfh_eval + self.Lgh_eval @ self.u >= -self.alpha(self.h_eval)]

    @property
    def optimal_control(self) -> np.ndarray:
        """Optimal control input, balancing the desired input with the CBF safety filter

        Raises:
            ValueError: If the problem has not yet been solved

        Returns:
            np.ndarray: Control, shape (m,)
        """
        if self.u.value is None:
            raise ValueError("Cannot return the control, problem has not been solved")
        return self.u.value

    # Note: the class is not static, but we can mark it as such since we are using only
    # the pure functions (f, g, h, dh/dz) and not the updated class variables
    @partial(jax.jit, static_argnums=(0,))
    def _jit_update_params(self, z: ArrayLike) -> Tuple[Array, Array, Array]:
        """JIT-compiled version of update_params for faster re-solving

        Args:
            z (ArrayLike): State, shape (n,)
            u_des (ArrayLike): Desired control, shape (m,)

        Returns:
            Tuple[Array, Array, Array]:
                Array: h(z)
                Array: Lfh(z)
                Array: Lgh(z)
        """
        f = self.f(z)
        g = self.g(z)
        dh_dz = self.dh_dz(z)
        h = self.h(z)
        lfh = dh_dz @ f
        lgh = dh_dz @ g
        return h, lfh, lgh

    def update_params(self, z: ArrayLike, u_des: ArrayLike) -> None:
        """Update the problem parameters before a re-solve

        Args:
            z (ArrayLike): State, shape (n,)
            u_des (ArrayLike): Desired control, shape (m,)
        """
        h, lfh, lgh = self._jit_update_params(z)
        self.u_des.value = u_des
        self.h_eval.value = np.asarray(h)
        self.Lfh_eval.value = np.asarray(lfh)
        self.Lgh_eval.value = np.asarray(lgh)

    def solve(self) -> None:
        """Solves the optimization problem"""
        self.prob.solve(solver=self.solver)
        if self.prob.status != cp.OPTIMAL:
            raise RuntimeError(
                "Could not find a solution\n" + f"Problem status: {self.prob.status}"
            )
