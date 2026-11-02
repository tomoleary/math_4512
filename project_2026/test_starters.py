"""Implementation checks for the public infrastructure, not project solutions."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from common import (dirichlet_laplacian, smallest_eigenpairs, eigenpair_residuals,
                    graph_laplacian, path_weights, cycle_weights, cluster_weights,
                    domain_mask, masked_laplacian)
from nonlinear import (newton, diffusion, solve_heat, manufactured_source,
                       manufactured_temperature, numerical_jacobian)


ROOT = Path(__file__).resolve().parent


class OperatorTests(unittest.TestCase):
    def test_dirichlet_matrix_and_energy(self):
        x, matrix, h = dirichlet_laplacian(4)
        expected = 16 * np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
        np.testing.assert_allclose(matrix.toarray(), expected)
        state = np.array([0.3, -0.7, 1.2])
        differences = np.diff(np.r_[0, state, 0])
        self.assertAlmostEqual(state @ matrix @ state, differences @ differences / h**2)
        _, single, _ = dirichlet_laplacian(2)
        np.testing.assert_allclose(single.toarray(), [[8]])

    def test_small_symmetric_eigensolver(self):
        from scipy.sparse import csr_matrix
        matrix = csr_matrix([[4.0, 0.0, 0.0], [0.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
        values, vectors = smallest_eigenpairs(matrix, 3)
        np.testing.assert_allclose(values, [1, 3, 4])
        np.testing.assert_allclose(vectors.T @ vectors, np.eye(3), atol=1e-13)
        self.assertLess(eigenpair_residuals(matrix, values, vectors).max(), 1e-12)

    def test_graph_energy_and_boundary_pinning(self):
        weights = path_weights(5)
        matrix = graph_laplacian(weights)
        state = np.array([0.2, 1.0, -0.4, 0.3, 0.7])
        energy = 0.5 * np.sum(weights * (state[:, None] - state[None, :])**2)
        self.assertAlmostEqual(state @ matrix @ state, energy)
        _, dirichlet, h = dirichlet_laplacian(4)
        np.testing.assert_allclose(matrix[1:-1, 1:-1] / h**2, dirichlet.toarray())
        np.testing.assert_allclose(matrix.sum(axis=1), 0)

    def test_cycle_and_disconnected_graph(self):
        np.testing.assert_allclose(np.linalg.eigvalsh(graph_laplacian(cycle_weights(4))),
                                   [0, 2, 2, 4], atol=1e-13)
        disconnected = np.linalg.eigvalsh(graph_laplacian(cluster_weights(3, 0)))
        connected = np.linalg.eigvalsh(graph_laplacian(cluster_weights(3, 0.1)))
        self.assertEqual(np.count_nonzero(np.abs(disconnected) < 1e-10), 2)
        self.assertEqual(np.count_nonzero(np.abs(connected) < 1e-10), 1)
        with self.assertRaises(ValueError):
            graph_laplacian([[0, -1], [-1, 0]])

    def test_square_assembly_and_boundary_diagonal(self):
        _, one_d, _ = dirichlet_laplacian(5)
        matrix = masked_laplacian(np.ones((4, 4), dtype=bool), 0.2)
        expected = np.kron(one_d.toarray(), np.eye(4)) + np.kron(np.eye(4), one_d.toarray())
        np.testing.assert_allclose(matrix.toarray(), expected)
        np.testing.assert_allclose(masked_laplacian([[True]], 0.5).toarray(), [[16]])

    def test_geometry_principal_submatrix_and_sparse_eigensolver(self):
        _, _, full = domain_mask(12, "square")
        x, y, hole = domain_mask(12, "hole", 0.2)
        whole = masked_laplacian(full, 1 / 12)
        punctured = masked_laplacian(hole, 1 / 12)
        kept = np.flatnonzero(hole.ravel())
        np.testing.assert_allclose(punctured.toarray(), whole[kept][:, kept].toarray())
        values, vectors = smallest_eigenpairs(punctured, 4)
        np.testing.assert_allclose(punctured.toarray(), punctured.toarray().T)
        self.assertGreater(values.min(), 0)
        self.assertLess(eigenpair_residuals(punctured, values, vectors).max(), 1e-8)


class EvolutionTests(unittest.TestCase):
    def test_ode_reference(self):
        from investigation02 import reference_solution, unforced_solution, switched_modal_response
        from scipy.linalg import expm
        x, matrix, _ = dirichlet_laplacian(8)
        values, vectors = np.linalg.eigh(matrix.toarray())
        initial = np.sin(np.pi * x)
        for equation in ("heat", "wave"):
            profiles = reference_solution(matrix, initial, [0, 0.05], equation)
            np.testing.assert_allclose(profiles[0], initial, atol=1e-13)
        reference = reference_solution(matrix, initial, [0, 0.05], "heat")[-1]
        np.testing.assert_allclose(reference, expm(-0.05 * matrix.toarray()) @ initial, atol=1e-9)
        with self.assertRaises(NotImplementedError):
            unforced_solution(values, vectors, initial, 0, "heat")
        with self.assertRaises(NotImplementedError):
            switched_modal_response(values, 1, 0.2, "heat")

    def test_linear_heat_limit(self):
        result = solve_heat(n=16, dt=0.01, beta=0)
        x, matrix, _ = dirichlet_laplacian(16)
        expected = np.sin(np.pi * x)
        step_matrix = np.eye(len(x)) + result.dt * matrix.toarray()
        for _ in result.times[1:]:
            expected = np.linalg.solve(step_matrix, expected)
        np.testing.assert_allclose(result.profiles[-1], expected, atol=1e-10)

    def test_heat_energy_and_residuals(self):
        result = solve_heat(n=16, beta=2)
        self.assertTrue(np.all(np.diff(result.energy) <= 1e-10))
        final_residuals = result.history[result.history[:, 4] == 0, 3]
        self.assertLessEqual(final_residuals.max(), 1e-10)

    def test_polynomial_spatial_consistency(self):
        # A different profile tests the flux implementation, not the assigned source.
        from numpy.polynomial import Polynomial
        profile = Polynomial([0, 0, 1, -2, 1])
        exact_operator = ((1 + 2 * profile**2) * profile.deriv()).deriv()
        errors = []
        for n in (12, 24, 48):
            x = np.arange(1, n) / n
            errors.append(np.max(np.abs(diffusion(profile(x), 1 / n, 2) - exact_operator(x))))
        ratios = np.array(errors[:-1]) / errors[1:]
        self.assertTrue(np.all((ratios > 3) & (ratios < 5)), ratios)
        with self.assertRaises(NotImplementedError):
            manufactured_source(np.array([0.5]), 0.1, 2)

    def test_jacobian_and_failure_reporting(self):
        matrix = np.array([[3.0, -1.0], [-1.0, 2.0]])
        rhs = np.array([1.0, 4.0])
        residual = lambda state: matrix @ state - rhs
        jacobian = numerical_jacobian(residual, np.array([0.3, -0.7]))
        np.testing.assert_allclose(jacobian, matrix, atol=1e-7)
        state, history = newton(residual, np.zeros(2))
        np.testing.assert_allclose(state, np.linalg.solve(matrix, rhs), atol=1e-10)
        with self.assertRaises(RuntimeError):
            newton(residual, np.zeros(2), max_iterations=0)

    def test_semilinear_manufactured_example(self):
        from investigation05 import nonlinear_residual
        errors = []
        for n in (6, 12):
            x, y, mask = domain_mask(n)
            matrix = masked_laplacian(mask, 1 / n)
            exact = np.sin(np.pi * x[mask]) * np.sin(np.pi * y[mask])
            forcing = 2 * np.pi**2 * exact + 2 * exact**3
            residual = lambda state: nonlinear_residual(state, matrix, 2, forcing)
            state, _ = newton(residual, np.zeros_like(exact))
            self.assertLess(np.linalg.norm(residual(state), ord=np.inf), 1e-10)
            errors.append(np.max(np.abs(state - exact)))
        self.assertGreater(errors[0] / errors[1], 3.5)


class DeliveryTests(unittest.TestCase):
    def test_notebooks_match_scripts_and_are_unexecuted(self):
        for number in range(1, 6):
            stem = f"investigation{number:02d}"
            notebook = json.loads((ROOT / f"{stem}.ipynb").read_text())
            self.assertEqual(notebook["nbformat"], 4)
            baseline = next(cell for cell in notebook["cells"] if cell["id"] == "baseline")
            self.assertEqual("".join(baseline["source"]), (ROOT / f"{stem}.py").read_text())
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    self.assertEqual(cell["outputs"], [])
                    self.assertIsNone(cell["execution_count"])
                    compile("".join(cell["source"]), str(ROOT / f"{stem}.ipynb"), "exec")

    def test_all_five_scripts_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = dict(os.environ, MPLBACKEND="Agg", MATH4512_OUTPUT_DIR=temporary)
            for number in range(1, 6):
                stem = f"investigation{number:02d}"
                completed = subprocess.run([sys.executable, str(ROOT / f"{stem}.py")],
                                           cwd=ROOT, env=environment, capture_output=True, text=True, timeout=90)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertTrue(list((Path(temporary) / stem).glob("*.png")))
                self.assertTrue(list((Path(temporary) / stem).glob("*.csv")))


if __name__ == "__main__":
    unittest.main()
