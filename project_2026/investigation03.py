"""Investigation 3 baseline: Dirichlet eigenmodes on an L-shaped domain.

Student work: validate on a square, investigate one geometry parameter,
refine the grid, and explain eigenvalue/eigenspace comparisons.
"""

import numpy as np
from matplotlib import pyplot as plt
from common import domain_mask, masked_laplacian, smallest_eigenpairs, eigenpair_residuals, output_folder, save_figure


def run(n=20, shape="lshape", parameter=0.2):
    x, y, mask = domain_mask(n, shape, parameter)
    matrix = masked_laplacian(mask, 1 / n)
    values, vectors = smallest_eigenpairs(matrix, 6)
    folder = output_folder("investigation03")
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    for index, axis in enumerate(axes.flat):
        field = np.full(mask.shape, np.nan)
        field[mask] = vectors[:, index]
        limit = np.nanmax(np.abs(field))
        plot = axis.pcolormesh(x, y, field, shading="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        axis.set(xlabel="x", ylabel="y", title=f"Mode {index + 1}: {values[index]:.3f}", aspect="equal")
        fig.colorbar(plot, ax=axis)
    save_figure(fig, folder / "baseline_modes.png")
    np.savetxt(folder / "baseline_eigenvalues.csv", np.c_[np.arange(1, 7), values],
               delimiter=",", header="mode,eigenvalue", comments="")
    print("Maximum relative eigenpair residual:", eigenpair_residuals(matrix, values, vectors).max())
    print("Outputs:", folder)
    # Extend here: square validation, geometry sweep, mesh refinement.
    return values, vectors


if __name__ == "__main__":
    run()
