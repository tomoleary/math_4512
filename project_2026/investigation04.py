"""Investigation 4 baseline: one unforced nonlinear heat calculation.

Student work: residual derivation, linear-limit and manufactured checks,
time/space/tolerance studies, and an original conductivity experiment.
"""

import numpy as np
from matplotlib import pyplot as plt
from nonlinear import solve_heat, manufactured_source, manufactured_temperature
from common import output_folder, save_figure


def run(n=24, beta=2.0, dt=0.005):
    result = solve_heat(n=n, beta=beta, dt=dt)
    folder = output_folder("investigation04")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for index in (0, len(result.times) // 2, len(result.times) - 1):
        axes[0].plot(np.r_[0, result.x, 1], np.r_[0, result.profiles[index], 0],
                     label=f"t={result.times[index]:.3f}")
    axes[0].set(xlabel="x", ylabel="temperature", title=f"Nonlinear heat, beta={beta}")
    axes[0].legend()
    axes[1].plot(result.times, result.energy)
    axes[1].set(xlabel="time", ylabel="discrete squared-temperature energy")
    save_figure(fig, folder / "baseline_heat.png")
    np.savetxt(folder / "newton_history.csv", result.history, delimiter=",",
               header="time_step,time,iteration,residual_inf,step_length", comments="")
    np.savetxt(folder / "final_profile.csv", np.c_[result.x, result.profiles[-1]],
               delimiter=",", header="x,temperature", comments="")
    print("Final nonlinear residual:", result.history[-1, 3])
    print("Effective time step:", result.dt)
    print("Outputs:", folder)
    # After implementing manufactured_source from your derivation:
    # source = lambda x, t: manufactured_source(x, t, beta)
    # verified = solve_heat(n=n, beta=beta, dt=dt, source=source)
    return result


if __name__ == "__main__":
    run()
