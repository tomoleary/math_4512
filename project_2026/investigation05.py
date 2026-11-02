"""Investigation 5 adaptation scaffold, not an automatically approved project.

Demonstration: -Delta u + gamma*u**3 = f on the unit square with zero boundary
data. The manufactured exact solution supplies a check before a model change.
Students must propose and obtain approval for their own question/deliverables.
"""

import numpy as np
from matplotlib import pyplot as plt
from common import domain_mask, masked_laplacian, output_folder, save_figure
from nonlinear import newton


def nonlinear_residual(state, matrix, gamma, forcing):
    return matrix @ state + gamma * state**3 - forcing


def run(n=10, gamma=2.0):
    if gamma < 0:
        raise ValueError("This example uses gamma>=0")
    x, y, mask = domain_mask(n, "square")
    matrix = masked_laplacian(mask, 1 / n)
    exact = np.sin(np.pi * x[mask]) * np.sin(np.pi * y[mask])
    forcing = 2 * np.pi**2 * exact + gamma * exact**3
    residual = lambda state: nonlinear_residual(state, matrix, gamma, forcing)
    state, history = newton(residual, np.zeros_like(exact))
    folder = output_folder("investigation05")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for axis, values, title in zip(axes, (state, state - exact), ("Computed solution", "Error at grid points")):
        field = values.reshape(mask.shape)
        plot = axis.pcolormesh(x, y, field, shading="auto")
        axis.set(xlabel="x", ylabel="y", title=title, aspect="equal")
        fig.colorbar(plot, ax=axis)
    save_figure(fig, folder / "baseline_manufactured_problem.png")
    np.savetxt(folder / "newton_history.csv", history, delimiter=",",
               header="iteration,residual_inf,step_length", comments="")
    print("Final residual:", np.linalg.norm(residual(state), ord=np.inf))
    print("Manufactured-solution maximum error:", np.max(np.abs(state - exact)))
    print("Outputs:", folder)
    # Replace only after approval: model/data, residual, and validation plan.
    return state, exact, history


if __name__ == "__main__":
    run()
