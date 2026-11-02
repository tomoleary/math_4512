"""Investigation 1 baseline: construct one matrix and inspect its computed modes.

Required analytical work: D-D derivation and convergence, then N-N and mixed
boundary conditions, compatibility, and invertibility. Part E requires code
for mode overlays, eigenvalue comparisons, refinement, and residual checks.
"""

import numpy as np
from matplotlib import pyplot as plt
from common import dirichlet_laplacian, smallest_eigenpairs, eigenpair_residuals, output_folder, save_figure


def cell_centered_laplacian(n, left="N", right="N", length=1.0):
    """Implement the ghost-value scheme from Investigation 1 C-D for Part E.

    Return (x, matrix, h), with n cell centers and an n-by-n symmetric matrix.
    Use D or N for each endpoint; derive the boundary rows before coding them.
    This is a different grid from the nodal D-D baseline.
    For these small matrices, np.linalg.eigh on a dense array also handles
    the Neumann zero mode without inverting a singular matrix.
    """
    raise NotImplementedError("Derive the cell-centered boundary rows from the handout")


def run(n=12):
    x, matrix, h = dirichlet_laplacian(n)
    values, vectors = smallest_eigenpairs(matrix, 3)
    folder = output_folder("investigation01")
    np.savetxt(folder / "baseline_eigenvalues.csv", np.column_stack((np.arange(1, 4), values)),
               delimiter=",", header="mode,eigenvalue", comments="")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    for index, axis in enumerate(axes):
        axis.plot(np.r_[0, x, 1], np.r_[0, vectors[:, index], 0], "o-")
        axis.set(xlabel="x", ylabel="Euclidean-normalized eigenvector", title=f"Mode {index + 1}")
    save_figure(fig, folder / "baseline_modes.png")
    print("Maximum relative eigenpair residual:", eigenpair_residuals(matrix, values, vectors).max())
    print("Outputs:", folder)
    # Required Part E: add continuous overlays, spectra, refinement, and boundary variants.
    return values, vectors


if __name__ == "__main__":
    run()
