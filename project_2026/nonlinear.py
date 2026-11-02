"""Small-grid nonlinear solvers supplied as teaching infrastructure."""

from dataclasses import dataclass
import numpy as np


def numerical_jacobian(residual, state):
    base = residual(state)
    jacobian = np.empty((state.size, state.size))
    for column in range(state.size):
        perturbed = state.copy()
        step = np.sqrt(np.finfo(float).eps) * (1 + abs(state[column]))
        perturbed[column] += step
        jacobian[:, column] = (residual(perturbed) - base) / step
    return jacobian


def newton(residual, initial, tolerance=1e-10, max_iterations=30):
    """History columns: iteration, residual infinity norm, accepted step length."""
    if tolerance <= 0 or max_iterations < 0:
        raise ValueError("Need tolerance>0 and max_iterations>=0")
    state = np.asarray(initial, dtype=float).copy()
    history = []
    for iteration in range(max_iterations + 1):
        value = residual(state)
        norm = float(np.linalg.norm(value, ord=np.inf))
        if not np.isfinite(norm):
            raise RuntimeError("Non-finite Newton residual")
        if norm <= tolerance:
            history.append((iteration, norm, 0.0))
            return state, np.array(history)
        if iteration == max_iterations:
            raise RuntimeError(f"Newton failed to converge: residual={norm:.3e}")
        correction = np.linalg.solve(numerical_jacobian(residual, state), -value)
        alpha = 1.0
        for _ in range(25):
            candidate = state + alpha * correction
            candidate_norm = np.linalg.norm(residual(candidate))
            if np.isfinite(candidate_norm) and candidate_norm <= (1 - 1e-4 * alpha) * np.linalg.norm(value):
                history.append((iteration, norm, alpha))
                state = candidate
                break
            alpha /= 2
        else:
            raise RuntimeError("Newton line search failed; reduce the time step or revise the guess")


def diffusion(interior, h, beta):
    values = np.concatenate(([0.0], interior, [0.0]))
    conductivity = 1 + 0.5 * beta * (values[:-1]**2 + values[1:]**2)
    flux = conductivity * np.diff(values) / h
    return np.diff(flux) / h


def manufactured_temperature(x, t):
    return np.exp(-t) * np.sin(np.pi * x)


def manufactured_source(x, t, beta):
    """Student task: substitute manufactured_temperature into the PDE."""
    raise NotImplementedError("Derive the manufactured source in Investigation 4 before implementing it")


@dataclass
class HeatResult:
    x: np.ndarray
    times: np.ndarray
    profiles: np.ndarray
    energy: np.ndarray
    history: np.ndarray
    dt: float


def solve_heat(n=24, dt=0.005, final_time=0.1, beta=2.0,
               amplitude=1.0, source=None, tolerance=1e-10):
    if (not isinstance(n, (int, np.integer)) or n < 2 or dt <= 0 or final_time <= 0
            or beta < 0 or tolerance <= 0):
        raise ValueError("Need integer n>=2, dt>0, final_time>0, beta>=0, tolerance>0")
    steps = int(np.ceil(final_time / dt))
    dt = final_time / steps
    h = 1 / n
    x = np.arange(1, n) * h
    state = amplitude * np.sin(np.pi * x)
    profiles, energies, history = [state.copy()], [0.5 * h * state @ state], []
    for step in range(1, steps + 1):
        t = step * dt
        previous = state.copy()
        forcing = np.zeros_like(x) if source is None else np.asarray(source(x, t))

        def residual(candidate):
            return candidate - previous - dt * (diffusion(candidate, h, beta) + forcing)

        state, records = newton(residual, previous, tolerance=tolerance)
        history.extend((step, t, iteration, norm, alpha) for iteration, norm, alpha in records)
        profiles.append(state.copy())
        energies.append(0.5 * h * state @ state)
    return HeatResult(x, np.arange(steps + 1) * dt, np.array(profiles),
                      np.array(energies), np.array(history), dt)
