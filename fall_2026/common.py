"""Shared numerical infrastructure, not project solutions."""

import os
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh


def dirichlet_laplacian(n, length=1.0):
    """n subintervals, n-1 interior unknowns, homogeneous Dirichlet data."""
    if not isinstance(n, (int, np.integer)) or n < 2 or length <= 0:
        raise ValueError("Need integer n>=2 and length>0")
    h = length / n
    x = np.arange(1, n) * h
    matrix = diags([-np.ones(n - 2), 2 * np.ones(n - 1), -np.ones(n - 2)],
                   [-1, 0, 1], shape=(n - 1, n - 1), format="csr") / h**2
    return x, matrix, h


def smallest_eigenpairs(matrix, count=6):
    """Use only for symmetric positive-definite matrices, not graph zero modes."""
    size = matrix.shape[0]
    if matrix.shape != (size, size) or count < 1 or count > size:
        raise ValueError("Need square matrix and 1<=count<=matrix size")
    if size <= 64 or count == size:
        values, vectors = np.linalg.eigh(matrix.toarray())
        return values[:count], vectors[:, :count]
    values, vectors = eigsh(matrix, k=count, sigma=0.0, which="LM",
                            tol=1e-11, v0=np.random.default_rng(4512).normal(size=size))
    order = np.argsort(values)
    return values[order], vectors[:, order]


def eigenpair_residuals(matrix, values, vectors):
    defects = matrix @ vectors - vectors * values[None, :]
    scales = np.maximum(1.0, np.abs(values)) * np.linalg.norm(vectors, axis=0)
    return np.linalg.norm(defects, axis=0) / scales


def graph_laplacian(weights):
    weights = np.asarray(weights, dtype=float)
    if (weights.ndim != 2 or weights.shape[0] != weights.shape[1]
            or not np.all(np.isfinite(weights)) or np.any(weights < 0)
            or not np.allclose(weights, weights.T) or np.any(np.diag(weights) != 0)):
        raise ValueError("Use symmetric nonnegative weights with zero diagonal")
    return np.diag(weights.sum(axis=1)) - weights


def path_weights(size):
    if size < 2:
        raise ValueError("Need at least two vertices")
    return np.diag(np.ones(size - 1), 1) + np.diag(np.ones(size - 1), -1)


def cycle_weights(size):
    if size < 3:
        raise ValueError("Need at least three vertices")
    weights = path_weights(size)
    weights[0, -1] = weights[-1, 0] = 1.0
    return weights


def cluster_weights(size=5, bridge=0.1):
    if size < 2 or bridge < 0:
        raise ValueError("Need cluster size>=2 and bridge>=0")
    weights = np.zeros((2 * size, 2 * size))
    for start in (0, size):
        weights[start:start + size, start:start + size] = np.ones((size, size)) - np.eye(size)
    weights[size - 1, size] = weights[size, size - 1] = bridge
    return weights


def domain_mask(n, shape="square", parameter=0.2):
    """Interior nodes only; square, L-shaped, square hole, or narrow-neck domain."""
    if n < 4 or not isinstance(n, (int, np.integer)):
        raise ValueError("Need integer n>=4")
    axis = np.arange(1, n) / n
    x, y = np.meshgrid(axis, axis, indexing="xy")
    if shape == "square":
        mask = np.ones(x.shape, dtype=bool)
    elif shape == "lshape":
        mask = (x < 0.5) | (y < 0.5)
    elif shape == "hole":
        if not 0 < parameter < 0.5:
            raise ValueError("Hole half-width must be in (0, .5)")
        mask = (np.abs(x - 0.5) > parameter) | (np.abs(y - 0.5) > parameter)
    elif shape == "neck":
        if not 0 < parameter < 1:
            raise ValueError("Neck width must be in (0, 1)")
        mask = (x < 0.4) | (x > 0.6) | (np.abs(y - 0.5) < parameter / 2)
    else:
        raise ValueError("Unknown domain shape")
    return x, y, mask


def masked_laplacian(mask, h):
    """Five-point Dirichlet operator; missing neighbors are zero, not removed edges."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any() or h <= 0:
        raise ValueError("Need a nonempty 2D mask and h>0")
    indices = np.full(mask.shape, -1, dtype=int)
    indices[mask] = np.arange(mask.sum())
    rows, columns, entries = [], [], []
    for row, col in zip(*np.nonzero(mask)):
        current = indices[row, col]
        rows.append(current)
        columns.append(current)
        entries.append(4.0 / h**2)
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            rr, cc = row + dr, col + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]:
                rows.append(current)
                columns.append(indices[rr, cc])
                entries.append(-1.0 / h**2)
    count = int(mask.sum())
    return coo_matrix((entries, (rows, columns)), shape=(count, count)).tocsr()


def output_folder(module):
    folder = Path(os.environ.get("MATH4512_OUTPUT_DIR", "outputs")) / module
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_figure(figure, path):
    from matplotlib import pyplot as plt
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
