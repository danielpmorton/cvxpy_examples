"""General class structure for an optimization problem using CVXPY"""

from typing import Union, Optional

import cvxpy as cp

from cvxpy_examples.utils.errors import OptimizationError
from cvxpy_examples.utils.cp_utils import print_problem_info


class OptimizationProblem:
    """General class for defining an optimization problem with CVXPY

    Args:
        verbose (bool, optional): Whether to print info about the optimization problem after it is solved.
            Defaults to True
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.prob = cp.Problem(self.objective, self.constraints)

    @property
    def objective(self) -> Union[cp.Minimize, cp.Maximize]:
        """Optimization objective"""
        return cp.Minimize(0)

    @property
    def constraints(self) -> list[cp.Expression]:
        """Optimization constraints"""
        return []

    def solve(self, solver: Optional[str] = None) -> None:
        """Solves the optimization problem

        Args:
            solver (Optional[str]): CVXPY solver to use. Defaults to None (use default solver)
        """
        self.prob.solve(solver=solver)
        if self.prob.status != cp.OPTIMAL:
            raise OptimizationError("Could not find a solution", self.prob.status)
        if self.verbose:
            print_problem_info(self.prob)
