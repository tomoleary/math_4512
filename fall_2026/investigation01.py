"""Investigation 1 starter: a complete D-D baseline plus boundary-condition TODOs.

Run this file once before editing it. It constructs the Dirichlet--Dirichlet
(D-D) discrete Laplacian, computes its modes, overlays them with sine modes,
and makes D-D eigenvalue/refinement plots. For Part E, reuse the plotting
helpers after deriving and implementing the N-N, D-N, and N-D boundary rows.
"""

import numpy as np
from matplotlib import pyplot as plt

from common import (dirichlet_laplacian, eigenpair_residuals, output_folder,
                    save_figure, smallest_eigenpairs)


def normalize_discrete(vector, mesh_width):
    """Scale a nonzero grid vector so h sum_j v_j^2 equals one."""
    return vector / np.sqrt(mesh_width * np.sum(vector**2))


def align_sign(vector, reference):
    """Choose the sign of a numerical eigenvector to agree with a reference."""
    if np.dot(vector, reference) < 0:
        return -vector
    return vector


def dd_eigenvalue(mode, length=1.0):
    """Continuous D-D eigenvalue for mode 1, 2, ... ."""
    return (mode * np.pi / length)**2


def dd_mode(points, mode, length=1.0):
    """L2(0,L)-normalized continuous D-D eigenfunction."""
    return np.sqrt(2.0 / length) * np.sin(mode * np.pi * points / length)


def plot_mode_comparison(points, vectors, mesh_width, exact_mode, title, path,
                         mode_indices=None):
    """Plot up to three discrete modes against supplied continuous modes.

    ``exact_mode(points, mode)`` returns the continuous mode for the requested
    index. By default the indices are 1, 2, ...; pass ``(0, 1, 2)`` for the
    first three N-N modes. Reuse this helper after implementing a new matrix.
    """
    count = min(3, vectors.shape[1])
    if mode_indices is None:
        mode_indices = range(1, count + 1)
    if len(mode_indices) < count:
        raise ValueError("Need one mode index for each displayed vector")
    smooth_points = np.linspace(0.0, 1.0, 400)
    figure, axes = plt.subplots(1, count, figsize=(3.5 * count, 3.1),
                                constrained_layout=True)
    if count == 1:
        axes = [axes]
    for index, axis in enumerate(axes):
        mode = mode_indices[index]
        reference_at_nodes = exact_mode(points, mode)
        numerical = normalize_discrete(vectors[:, index], mesh_width)
        numerical = align_sign(numerical, reference_at_nodes)
        axis.plot(smooth_points, exact_mode(smooth_points, mode), "-",
                  label="continuous")
        axis.plot(points, numerical, "o", label="computed")
        axis.set(xlabel="x", ylabel="normalized mode", title=f"Mode {mode}")
        axis.grid()
        axis.legend()
    figure.suptitle(title)
    save_figure(figure, path)


def plot_eigenvalue_comparison(values, exact_values, title, path, mode_indices=None):
    """Plot computed and continuous eigenvalues against shared mode indices."""
    if mode_indices is None:
        mode_indices = np.arange(1, len(values) + 1)
    if len(mode_indices) != len(values):
        raise ValueError("Need one mode index for each eigenvalue")
    figure, axis = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    axis.plot(mode_indices, values, "o", label="computed")
    axis.plot(mode_indices, exact_values, "x", label="continuous")
    axis.set(xlabel="mode", ylabel="eigenvalue", title=title)
    axis.grid()
    axis.legend()
    save_figure(figure, path)


def cell_centered_laplacian(n, left="N", right="N", length=1.0):
    """TODO: implement the N-N, D-N, or N-D cell-centered matrix.

    Use ``left`` and ``right`` equal to ``"D"`` or ``"N"``. First make the
    cell-center coordinates x_j=(j-1/2)h and the usual three-point matrix.
    Then use your ghost-value derivation to replace the first and last rows.
    Return ``(points, matrix, mesh_width)`` with a symmetric matrix.
    """
    raise NotImplementedError("Derive and implement the cell-centered boundary rows")


def run_dd_baseline(number_of_subintervals=32, length=1.0):
    """Run the complete D-D computation supplied with Investigation 1."""
    points, matrix, mesh_width = dirichlet_laplacian(number_of_subintervals, length)
    mode_count = min(6, matrix.shape[0])
    values, vectors = smallest_eigenpairs(matrix, mode_count)
    exact_values = np.array([dd_eigenvalue(mode, length)
                             for mode in range(1, mode_count + 1)])
    folder = output_folder("investigation01")

    plot_mode_comparison(points, vectors[:, :3], mesh_width,
                         lambda x, mode: dd_mode(x, mode, length),
                         "D-D computed modes and sine eigenfunctions",
                         folder / "dd_mode_comparison.png")
    plot_eigenvalue_comparison(values, exact_values,
                               "D-D computed and continuous eigenvalues",
                               folder / "dd_eigenvalue_comparison.png")

    grid_sizes = np.array([8, 16, 32])
    rows = []
    for n in grid_sizes:
        _, refined_matrix, h = dirichlet_laplacian(n, length)
        refined_values, _ = smallest_eigenpairs(refined_matrix, 2)
        for mode in (1, 2):
            error = abs(refined_values[mode - 1] - dd_eigenvalue(mode, length))
            rows.append((n, h, mode, refined_values[mode - 1], error))
    table = np.asarray(rows)
    np.savetxt(folder / "dd_refinement.csv", table, delimiter=",",
               header="N,h,mode,computed_eigenvalue,absolute_error", comments="")

    figure, axis = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    for mode in (1, 2):
        selected = table[table[:, 2] == mode]
        axis.loglog(selected[:, 1], selected[:, 4], "o-", label=f"mode {mode}")
    axis.set(xlabel="h", ylabel="absolute eigenvalue error",
             title="D-D refinement study")
    axis.grid()
    axis.legend()
    save_figure(figure, folder / "dd_refinement.png")

    residuals = eigenpair_residuals(matrix, values, vectors)
    print("Maximum relative eigenpair residual:", residuals.max())
    print("D-D files written to:", folder)
    return points, matrix, values, vectors


if __name__ == "__main__":
    run_dd_baseline()
