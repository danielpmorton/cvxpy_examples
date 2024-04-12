"""Assorted helper functions related to math operations / linear algebra"""

import numpy as np
import numpy.typing as npt


def normalize(vec: npt.ArrayLike) -> np.ndarray:
    """Normalizes a vector to have magnitude 1

    If normalizing an array of vectors, each vector will have magnitude 1

    Args:
        vec (npt.ArrayLike): Input vector or array. Shape (dim,) or (n_vectors, dim)

    Returns:
        np.ndarray: Unit vector(s), shape (dim,) or (n_vectors, dim) (same shape as the input)
    """
    vec = np.atleast_1d(vec)
    # Single vector, shape (dim,)
    if np.ndim(vec) == 1:
        norm = np.linalg.norm(vec)
        if abs(norm) < 1e-12:
            raise ZeroDivisionError("Cannot normalize the vector, it has norm 0")
        return vec / norm
    # Array of vectors, shape (n, dim) or something more complex (take norm over last axis)
    else:
        norms = np.linalg.norm(vec, axis=-1)
        if np.any(np.abs(norms) <= 1e-12):
            raise ZeroDivisionError(
                "Cannot normalize the array, at least 1 vector has norm 0"
            )
        return vec / norms[..., np.newaxis]

