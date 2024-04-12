"""Geometry and convex hull-related tools"""

import itertools

import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull


def hull_to_matrices(hull: ConvexHull) -> tuple[np.ndarray, np.ndarray]:
    """Determine the matrices A, b from a polyhedron convex hull such that Ax <= b

    Args:
        hull (ConvexHull): Convex hull defining a polyhedron

    Returns:
        tuple[np.ndarray, np.ndarray]:
            np.ndarray: A matrix, s.t. Ax <= b. Shape (m, n)
            np.ndarray: b array, s.t. Ax <= b. Shape (m,)
    """
    eqs = hull.equations
    return eqs[:, :-1], -1 * eqs[:, -1]


def polyhedron_hull(A: np.ndarray, b: np.ndarray) -> ConvexHull:
    """Convex hull of a polyhedron defined by Ax <= b

    Note: This can be inefficient to solve for high dimensions

    Args:
        A (np.ndarray): Array defining hyperplane normals, shape (n_planes, 3)
        b (np.ndarray): Array defining hyperplane offset, shape (n_planes)

    Returns:
        ConvexHull: Convex hull of the vertices of the polyhedron
    """
    n, dim = A.shape
    # Determine all possible intersections of hyperplanes and look for the corresponding vertex
    combos = itertools.combinations(range(n), dim)
    verts = []
    eps = 1e-12  # Tolerance
    for combo in combos:
        Ai = A[combo, :]
        bi = b[[combo]].reshape(-1)
        try:
            v = np.linalg.solve(Ai, bi)
        except np.linalg.LinAlgError:
            # This can occur if the Ai matrix is singular, in which case there is no vertex solution
            continue
        # Validate that this vertex solution is correct and  inside / on the boundary of the polyhedron
        is_vertex = np.linalg.norm(Ai @ v - bi) <= eps
        if not is_vertex:
            continue
        is_in_polyhedron = np.all(A @ v - b <= eps * np.ones_like(b))
        if is_in_polyhedron:
            verts.append(v)
    # Not all intersections of planes will be on the hull of the polyhedron, so use ConvexHull to manage this
    return ConvexHull(verts)


def intersect_polyhedra(
    A: np.ndarray, b: np.ndarray, C: np.ndarray, d: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the intersection of two polyhedra defined by Ax <= b and Cx <= d

    Args:
        A (np.ndarray): A, such that Ax <= b. Shape (m, n)
        b (np.ndarray): b, such that Ax <= b. Shape (m,)
        C (np.ndarray): C, such that Cx <= d. Shape (p, n)
        d (np.ndarray): d, such that Cx <= d. Shape (p,)

    Returns:
        tuple[np.ndarray, np.ndarray]: Intersection of the two polyhedra
            np.ndarray: F, such that Fx <= g. Shape (m + p, n)
            np.ndarray: g, such that Fx <= g. Shape (m + p)
    """
    return np.row_stack([A, C]), np.concatenate([b, d])


def cube_points(
    center: npt.ArrayLike = (0, 0, 0),
    sidelength: float = 1,
    rotation: np.ndarray = np.eye(3),
) -> np.ndarray:
    """Construct a set of eight points defining the corners of a cube

    Args:
        center (npt.ArrayLike, optional): Center of the cube. Defaults to (0, 0, 0).
        sidelength (float, optional): Length of the cube's sides. Defaults to 1.
        rotation (np.ndarray, optional): Rotation matrix defining orientation of the points. Defaults to np.eye(3).

    Returns:
        np.ndarray: Points, shape (8, 3)
    """
    return (
        np.asarray(center)
        + np.array(
            [
                [0.5, 0.5, 0.5],
                [0.5, 0.5, -0.5],
                [0.5, -0.5, 0.5],
                [0.5, -0.5, -0.5],
                [-0.5, 0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, -0.5, -0.5],
            ]
        )
        @ rotation.T
        * sidelength
    )
