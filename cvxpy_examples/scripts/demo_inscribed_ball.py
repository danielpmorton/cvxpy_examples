"""Script to demonstrate the inscribed ball problem"""

import time

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from cvxpy_examples.utils.math_utils import normalize
from cvxpy_examples.utils.plotting_utils import plot_2d_hull, plot_3d_hull, plot_circle, plot_sphere
from cvxpy_examples.utils.geometry_utils import polyhedron_hull, cube_points
from cvxpy_examples.problems.inscribed_ball import InscribedBallProblem


def test_3d():
    A = normalize(cube_points())
    b = np.ones(A.shape[0])
    prob = InscribedBallProblem(A, b, center=(0, 0, 0))
    prob.solve()
    center = prob.optimal_center
    radius = prob.optimal_radius
    hull = polyhedron_hull(A, b)
    ax = plot_3d_hull(hull, show=False)
    plot_sphere(center, radius, ax=ax, show=True)


def test_2d():
    n = 8
    thetas = np.linspace(0, 2 * ((n - 1) / n) * np.pi, n)
    x = np.cos(thetas)
    y = np.sin(thetas)

    np.random.seed(3)
    m = 8 # Number of sides of the polyhedron
    n = 2 # 2D example
    A = normalize(np.random.normal(0, 1, (m, n)))
    # A = np.column_stack([x, y])
    b = np.ones(A.shape[0])
    hull = polyhedron_hull(A, b)
    prob = InscribedBallProblem(A, b, center=(0, 0))
    prob.solve()
    center = prob.optimal_center
    radius = prob.optimal_radius
    ax = plot_2d_hull(hull, show=False)
    plot_circle(center, radius, ax=ax)

def test_dpp_solve_time():
    np.random.seed(0)
    num_trials = 100
    # Create a large polyhedron
    m = 100 # Number of sides of the polyhedron
    n = 30 # Dimension of the problem
    A = normalize(np.random.normal(np.zeros(n), np.ones(n), (m, n)))
    b = np.ones(m)

    standard_times = []
    dpp_times = []
    # Initialize the DPP problem
    dpp_prob = InscribedBallProblem(A, b, center=np.zeros(n), verbose=False)
    for _ in range(num_trials):
        # Sample a center point
        center = np.random.uniform(-0.5, 0.5, size=n)
        # Solve with the standard construction method
        start_time = time.time()
        radius = cp.Variable()
        objective = cp.Maximize(radius)
        constraints = [A @ center + radius * cp.norm(A, axis=1) <= b, radius >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        end_time = time.time()
        assert prob.status == cp.OPTIMAL
        standard_times.append(end_time - start_time)

        # Solve with the parameterized class
        start_time = time.time()
        dpp_prob.update_center(center)
        dpp_prob.solve()
        end_time = time.time()
        dpp_times.append(end_time - start_time)
    
    fig, ax = plt.subplots()
    x = np.arange(num_trials)
    ax.scatter(x, standard_times, label="Non-DPP")
    ax.scatter(x, dpp_times, label="DPP")
    ax.legend()
    plt.show()    

def main():
    test_2d()
    test_3d()
    test_dpp_solve_time()

if __name__ == "__main__":
    main()
