"""Helper functions for plotting in Matplotlib (especially, geometric objects)"""

from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull


def gca_3d() -> plt.Axes:
    """Gets the current matplotlib 3D axes, if one exists. If not, create a new figure/axis

    Returns:
        plt.Axes: 3D axes
    """
    if len(plt.get_fignums()) == 0:  # No existing figures
        _, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    else:
        ax = plt.gca()
        if ax.name != "3d":  # An axis exists, but it is not 3D
            _, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    return ax


def gca_2d() -> plt.Axes:
    """Gets the current matplotlib 2D axes, if one exists. If not, create a new figure/axis

    Returns:
        plt.Axes: 2D axes
    """
    if len(plt.get_fignums()) == 0:  # No existing figures
        _, ax = plt.subplots(1, 1)
    else:
        ax = plt.gca()
        if ax.name != "rectilinear":  # An axis exists, but it is not 2D
            _, ax = plt.subplots(1, 1)
    return ax



def plot_circle(
    center: npt.ArrayLike,
    radius: float,
    n: int = 50,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """Plots a circle

    Args:
        center (npt.ArrayLike): Center of the circle, shape (2,)
        radius (float): Radius of the circle
        n (int, optional): Number of points to use for plotting. Defaults to 50.
        ax (Optional[plt.Axes]): Existing axes to plot on. Defaults to None (Create new plotting axes)
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The plot
    """
    if ax is None:
        ax = gca_2d()
    ax.set_aspect("equal")
    thetas = np.linspace(0, 2 * np.pi, n)
    x = center[0] + radius * np.cos(thetas)
    y = center[1] + radius * np.sin(thetas)
    ax.plot(x, y)
    if show:
        plt.show()
    return ax


def plot_sphere(
    center: npt.ArrayLike,
    radius: float,
    n: int = 10,
    color: Union[str, npt.ArrayLike] = (1, 0, 0),
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """Plots a sphere

    Args:
        center (npt.ArrayLike): Center of the sphere, shape (3,)
        radius (float): Radius of the sphere
        n (int, optional): Number of angular discretizations for plotting. Defaults to 10.
        color (Union[str, npt.ArrayLike], optional). Color of the sphere. Defaults to (1, 0, 0) (red)
        ax (Optional[plt.Axes]): Existing axes to plot on. Defaults to None (Create new plotting axes)
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The plot
    """
    if ax is None:
        ax = gca_3d()
    ax.set_aspect("equal")

    elevation_angles = np.linspace(0, np.pi, n)
    azimuth_angles = np.linspace(0, 2 * np.pi, n)

    sin_elevations = np.sin(elevation_angles)
    cos_elevations = np.cos(elevation_angles)

    sin_azimuths = np.sin(azimuth_angles)
    cos_azimuths = np.cos(azimuth_angles)

    X = center[0] + radius * np.outer(sin_elevations, sin_azimuths)
    Y = center[1] + radius * np.outer(sin_elevations, cos_azimuths)
    Z = center[2] + radius * np.outer(cos_elevations, np.ones_like(azimuth_angles))

    ax.plot_surface(X, Y, Z, color=color)
    if show:
        plt.show()
    return ax



def plot_3d_hull(
    hull: ConvexHull,
    ax: Optional[plt.Axes] = None,
    centered: bool = True,
    show: bool = True,
) -> plt.Axes:
    """Plots a 3D convex hull

    Args:
        hull (ConvexHull): Convex hull to plot
        ax (Optional[plt.Axes]): Existing axes to plot on. Defaults to None (Create new plotting axes)
        centered (bool, optional): Whether to center the plot on the hull. Defaults to True
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The plot
    """
    if ax is None:
        ax = gca_3d()
    faces = hull.points[hull.simplices]  # Shape (n_simplices, 3, 3)
    ax.add_collection3d(
        Poly3DCollection(
            faces,
            facecolors="cyan",  # TODO include alpha channel in rgba
            linewidths=0.75,
            edgecolors=(0, 0, 0, 0.3),
            alpha=0.2,
        )
    )
    # Center the plot on the hull
    if centered:
        lims = np.column_stack(
            [np.min(hull.points, axis=0), np.max(hull.points, axis=0)]
        )
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
        ax.set_zlim(*lims[2])
    ax.set_aspect("equal")
    if show:
        plt.show()
    return ax


def plot_2d_hull(
    hull: ConvexHull,
    ax: Optional[plt.Axes] = None,
    color: str = "k",
    show: bool = True,
    **plt_kwargs
) -> plt.Axes:
    """Plots a 2D convex hull

    Args:
        hull (ConvexHull): Convex hull to plot
        ax (Optional[plt.Axes]): Existing axes to plot on. Defaults to None (Create new plotting axes)
        color (str, optional): Matplotlib line color. Defaults to "k" (black)
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The plot
    """
    if ax is None:
        ax = gca_2d()
    ax.set_aspect("equal")
    edges = hull.points[hull.simplices]  # Shape (n_simplices, 2, 2)
    ax.plot(*edges.T, color=color, **plt_kwargs)
    if show:
        plt.show()
    return ax
