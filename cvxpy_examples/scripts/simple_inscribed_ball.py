"""A basic construction of the "maximum inscribed ball in a polyhedron" problem"""

import numpy as np
import cvxpy as cp


# Construct a simple polyhedron example: a 2D regular polygon centered at the origin
# Intuitively, we should have the optimal result be centered at (0, 0) and have a radius of 1
# based on the construction of the polyhedron (regular polygon)

m = 5 # Number of sides of the polyhedron
n = 2 # 2D example
thetas = np.linspace(0, 2 * np.pi, m)
A = np.column_stack([np.cos(thetas), np.sin(thetas)]) # Plane normals
b = np.ones(m) # Plane offsets from the origin

# Construct our variables
center = cp.Variable(n) # 2D variable
radius = cp.Variable() # Scalar variable

# Form the objective function and constraints
# i.e. maximize the radius of the ball, subject to the ball being contained in the polyhedron
objective = cp.Maximize(radius)
constraints = [A @ center + radius * cp.norm(A, axis=1) <= b, radius >= 0]

# Construct and solve the problem
prob = cp.Problem(objective, constraints)
# ECOS is a fast solver for simple problems like this, but there are many others (see cvxpy docs)
prob.solve(solver=cp.ECOS)

print("Optimal radius: ", radius.value)
print("Optimal center: ", center.value)
print("Solve time: ", prob.solver_stats.solve_time)
