"""Custom exceptions"""

from typing import Optional


class OptimizationError(Exception):
    """Custom exception type for handling cvxpy solver errors

    Args:
        message (str): Info to display about the error
        status (str, optional): CVXPY solver status. Defaults to None.
    """

    def __init__(self, message: str, status: Optional[str] = None):
        self.status = status
        if status is not None:
            message += f"\nProblem status: {status}"
        super().__init__(message)
