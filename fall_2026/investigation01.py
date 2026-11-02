"""Investigation 1: a worked D-D finite-difference example.

Run this file before editing it. The D-D calculation is complete. For Part E,
derive and add the N-N, D-N, and N-D cases in the marked section below.
"""

import numpy as np
import matplotlib.pyplot as plt

from common import dirichlet_laplacian, output_folder, save_figure


def cell_centered_laplacian(number_of_cells, left_boundary, right_boundary):
    """Build the N-N, D-N, or N-D matrix after you derive its boundary rows."""
    h = 1.0 / number_of_cells
    x = (np.arange(number_of_cells) + 0.5) * h
    A = np.zeros((number_of_cells, number_of_cells))

    # This loop builds the usual -u'' stencil in the interior.
    for row in range(number_of_cells):
        A[row, row] = 2.0
        if row > 0:
            A[row, row - 1] = -1.0
        if row < number_of_cells - 1:
            A[row, row + 1] = -1.0

    # TODO: Replace the first and last rows using your ghost-value derivation.
    # Use left_boundary and right_boundary, each equal to "D" or "N".
    # Then return x, A / h**2, h.
    raise NotImplementedError("Add the boundary rows for N-N, D-N, and N-D")


# -----------------------------------------------------------------------------
# Complete D-D example: -u'' = lambda u on (0,1), u(0)=u(1)=0.
# -----------------------------------------------------------------------------

number_of_subintervals = 32
x, A, h = dirichlet_laplacian(number_of_subintervals)

# A is small, so the ordinary dense symmetric eigensolver is enough here.
eigenvalues, eigenvectors = np.linalg.eigh(A.toarray())
number_of_modes = 6
eigenvalues = eigenvalues[:number_of_modes]
eigenvectors = eigenvectors[:, :number_of_modes]

folder = output_folder("investigation01")

# Plot the first three computed modes beside sqrt(2) sin(k pi x).
fine_x = np.linspace(0.0, 1.0, 400)
figure, axes = plt.subplots(1, 3, figsize=(10, 3))
for column in range(3):
    mode = column + 1
    exact_at_nodes = np.sqrt(2.0) * np.sin(mode * np.pi * x)
    computed = eigenvectors[:, column]

    # Normalize using h sum_j v_j^2, then make the signs agree for plotting.
    computed = computed / np.sqrt(h * np.sum(computed**2))
    if np.dot(computed, exact_at_nodes) < 0:
        computed = -computed

    exact = np.sqrt(2.0) * np.sin(mode * np.pi * fine_x)
    axes[column].plot(fine_x, exact, label="continuous")
    axes[column].plot(x, computed, "o", label="computed")
    axes[column].set_title(f"mode {mode}")
    axes[column].set_xlabel("x")
    axes[column].grid()
axes[0].set_ylabel("normalized mode")
axes[0].legend()
figure.suptitle("D-D computed modes and sine eigenfunctions")
save_figure(figure, folder / "dd_mode_comparison.png")

# Compare six matrix eigenvalues with the continuous values (k pi)^2.
modes = np.arange(1, number_of_modes + 1)
exact_eigenvalues = (modes * np.pi)**2
figure, axis = plt.subplots(figsize=(5, 3))
axis.plot(modes, eigenvalues, "o", label="computed")
axis.plot(modes, exact_eigenvalues, "x", label="continuous")
axis.set_xlabel("mode")
axis.set_ylabel("eigenvalue")
axis.set_title("D-D computed and continuous eigenvalues")
axis.grid()
axis.legend()
save_figure(figure, folder / "dd_eigenvalue_comparison.png")

# Repeat the first two D-D eigenvalue calculations on three grids.
rows = []
for N in [8, 16, 32]:
    _, refined_A, refined_h = dirichlet_laplacian(N)
    refined_eigenvalues, _ = np.linalg.eigh(refined_A.toarray())
    for mode in [1, 2]:
        exact_value = (mode * np.pi)**2
        computed_value = refined_eigenvalues[mode - 1]
        error = abs(computed_value - exact_value)
        rows.append([N, refined_h, mode, computed_value, error])

table = np.array(rows)
np.savetxt(folder / "dd_refinement.csv", table, delimiter=",",
           header="N,h,mode,computed_eigenvalue,absolute_error", comments="")

figure, axis = plt.subplots(figsize=(5, 3))
for mode in [1, 2]:
    selected_rows = table[table[:, 2] == mode]
    axis.loglog(selected_rows[:, 1], selected_rows[:, 4], "o-",
                label=f"mode {mode}")
axis.set_xlabel("h")
axis.set_ylabel("absolute eigenvalue error")
axis.set_title("D-D refinement study")
axis.grid()
axis.legend()
save_figure(figure, folder / "dd_refinement.png")

# Check that A v is very close to lambda v for the computed eigenpairs.
residual = A @ eigenvectors - eigenvectors * eigenvalues
print("Largest residual:", np.max(np.abs(residual)))
print("D-D files written to:", folder)
