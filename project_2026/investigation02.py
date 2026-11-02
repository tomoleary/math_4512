"""Investigation 2 baseline: numerical ODE reference, without modal formulas.

Student work includes deriving and implementing the switched-forcing response,
checking energies, and comparing continuous and discrete dynamics.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from common import dirichlet_laplacian, output_folder, save_figure


def unforced_solution(values, vectors, initial, time, equation):
    """Implement your derived unforced modal solution (zero wave velocity)."""
    raise NotImplementedError("Derive and implement the modal solution from the handout")


def reference_solution(matrix, initial, times, equation):
    """Independent numerical reference in physical, not modal, coordinates."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0 or times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Use increasing nonnegative output times")
    if equation == "heat":
        rhs = lambda t, state: -matrix @ state
        state0 = initial
    elif equation == "wave":
        size = len(initial)
        rhs = lambda t, state: np.r_[state[size:], -matrix @ state[:size]]
        state0 = np.r_[initial, np.zeros_like(initial)]
    else:
        raise ValueError("Use heat or wave")
    if times[-1] == 0:
        return np.asarray(initial)[None, :].copy()
    solution = solve_ivp(rhs, (0, times[-1]), state0, t_eval=times,
                         method="DOP853", rtol=1e-10, atol=1e-12)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[:len(initial)].T


def switched_modal_response(values, time, switch_time, equation):
    """Implement the zero-initial-data response to a unit step in each mode."""
    raise NotImplementedError("Derive the scalar transformed equation before implementing this function")


def run(n=16):
    x, matrix, h = dirichlet_laplacian(n)
    initial = np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)
    folder = output_folder("investigation02")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for equation, times, axis in zip(("heat", "wave"), ((0, 0.02, 0.08), (0, 0.2, 0.4)), axes):
        profiles = reference_solution(matrix, initial, times, equation)
        for t, profile in zip(times, profiles):
            axis.plot(np.r_[0, x, 1], np.r_[0, profile, 0], label=f"t={t}")
        axis.set(xlabel="x", ylabel="u", title=f"Unforced {equation}: ODE reference")
        axis.legend()
        np.savetxt(folder / f"{equation}_baseline.csv", np.column_stack((x, *profiles)),
                   delimiter=",", header="x," + ",".join(f"u_at_{t}" for t in times), comments="")
    save_figure(fig, folder / "unforced_profiles.png")
    print("Outputs:", folder)
    # Implement both modal functions, then compare your result with this reference.


if __name__ == "__main__":
    run()
