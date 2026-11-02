"""Small, readable Python examples for MATH 4512.

This file is deliberately written without project infrastructure.  Read it
alongside python_primer.ipynb, then adapt its patterns for Investigation 1.
"""

import numpy as np
import matplotlib.pyplot as plt


# A number is stored in a variable with a descriptive name.
number_of_subintervals = 8
interval_length = 1.0
mesh_width = interval_length / number_of_subintervals

print("mesh width =", mesh_width)


# NumPy arrays store a collection of numbers.  The grid includes both endpoints.
grid_points = np.linspace(0.0, interval_length, number_of_subintervals + 1)
function_values = np.sin(np.pi * grid_points)

print("grid points:", grid_points)
print("function values:", function_values)


# An explicit loop is often the clearest way to build a calculation.
sum_of_squares = 0.0
for value in function_values:
    sum_of_squares = sum_of_squares + value**2

discrete_l2_norm = np.sqrt(mesh_width * sum_of_squares)
print("discrete L2 norm =", discrete_l2_norm)


def make_dirichlet_laplacian(number_of_subintervals, interval_length):
    """Build the centered-difference matrix for -u'' with u(0)=u(L)=0."""
    mesh_width = interval_length / number_of_subintervals
    number_of_unknowns = number_of_subintervals - 1
    matrix = np.zeros((number_of_unknowns, number_of_unknowns))

    for row in range(number_of_unknowns):
        matrix[row, row] = 2.0
        if row > 0:
            matrix[row, row - 1] = -1.0
        if row < number_of_unknowns - 1:
            matrix[row, row + 1] = -1.0

    matrix = matrix / mesh_width**2
    return matrix


matrix = make_dirichlet_laplacian(4, 1.0)
print("\nDiscrete Dirichlet Laplacian:")
print(matrix)


# Matrix-vector multiplication represents applying the discrete operator.
interior_values = np.array([0.1, 0.15, 0.1])
operator_values = matrix @ interior_values
print("\nA_h u =", operator_values)


# A plot makes the samples visible.  The 'o' marker emphasizes grid values.
plt.plot(grid_points, function_values, "o-", label="samples of sin(pi x)")
plt.xlabel("x")
plt.ylabel("value")
plt.title("A function represented by samples")
plt.grid()
plt.legend()
plt.show()
