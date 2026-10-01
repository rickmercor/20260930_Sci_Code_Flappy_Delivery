"""
With every component known, the fluctuation is the sum of p_j phi_j over the whole spectrum and its sample values follow by combining the basis fields. Because the eigenstates form a complete orthonormal set for the energetic inner product on the discrete mean-free space, this sum is the exact discrete solution of the cell problem, and the step verifies that by evaluating the equilibrium residual div(C (E + grad u)) at the sample points relative to the largest sample value of the load, or in absolute terms when the load vanishes as it does for a uniform medium. The effective conductivity is defined through the average flux, C_eff E = average of C (E + grad u); for the exact solution this agrees with the energy definition, and because the eigenstates diagonalise the cell operator the two definitions also agree for any expansion truncated to a subset of eigenstates. Inserting the expansion into the flux average gives the spectral form C_eff = average of C plus the sum over j of beta_j beta_j / (lambda_j - 1), a rank-one correction per eigenstate whose weight is the outer product of the spectral vector divided by the shifted eigenvalue. Every term of the correction is negative semi-definite because each eigenvalue lies below one, so the effective conductivity never exceeds the arithmetic mean. The step returns the flux-average column for the given loading, the full spectral tensor, the discrepancy between the two for that loading, and the energetic norm of the fluctuation, which by orthonormality is the root of the sum of the squared components.

Returns
-------
dict with float array fluctuation of shape (n, n), float array effective_column of shape (2,) holding the average flux for the given loading, float array effective_tensor of shape (2, 2) from the spectral sum, native float mean_conductivity, native float consistency_residual between the tensor applied to E and the flux average, native float equilibrium_residual, and native float energy_norm.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reconstruct_solution_and_effective_tensor(
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    components: np.ndarray,
    spectral_vectors: np.ndarray,
    conductivity: np.ndarray,
    cell_lengths: tuple,
    macroscopic_gradient: np.ndarray,
) -> dict:
    """Rebuild the fluctuation from its eigenstate expansion and form the effective conductivity.

    Raises
    ------
    ValueError
        If the array shapes are mutually inconsistent, if any eigenvalue is not strictly below one, if macroscopic_gradient does not have two components, or if a cell length is not strictly positive.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _frequency_multipliers(n_pixels, cell_lengths):
    """Spectral derivative multipliers 2 pi i k_d / L_d on the odd grid, one array per axis."""
    integer_frequencies = np.fft.fftfreq(n_pixels, d=1.0 / n_pixels)
    k_1, k_2 = np.meshgrid(integer_frequencies, integer_frequencies, indexing="ij")
    return (2j * np.pi * k_1 / float(cell_lengths[0]), 2j * np.pi * k_2 / float(cell_lengths[1]))


def _spectral_gradient(field, multipliers):
    """Exact derivative of the trigonometric interpolant along both axes."""
    transform = np.fft.fft2(field, axes=(-2, -1))
    return tuple(np.real(np.fft.ifft2(m * transform, axes=(-2, -1))) for m in multipliers)


def _spectral_divergence(component_1, component_2, multipliers):
    """Spectral divergence of a vector field; its mean vanishes by construction."""
    transform = (multipliers[0] * np.fft.fft2(component_1, axes=(-2, -1))
                 + multipliers[1] * np.fft.fft2(component_2, axes=(-2, -1)))
    return np.real(np.fft.ifft2(transform, axes=(-2, -1)))


def _oracle_reconstruct_solution_and_effective_tensor(
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    components: np.ndarray,
    spectral_vectors: np.ndarray,
    conductivity: np.ndarray,
    cell_lengths: tuple,
    macroscopic_gradient: np.ndarray,
) -> dict:
    """Reference implementation."""
    basis = np.asarray(basis, dtype=float)
    eigenvalues = np.asarray(eigenvalues, dtype=float).ravel()
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    components = np.asarray(components, dtype=float).ravel()
    spectral_vectors = np.asarray(spectral_vectors, dtype=float)
    conductivity = np.asarray(conductivity, dtype=float)
    gradient = np.asarray(macroscopic_gradient, dtype=float).ravel()
    if basis.ndim != 3 or basis.shape[1] != basis.shape[2] or conductivity.shape != basis.shape[1:]:
        raise ValueError("basis must have shape (N - 1, n, n) matching the conductivity grid")
    n_modes = basis.shape[0]
    if (eigenvalues.shape != (n_modes,) or eigenvectors.shape != (n_modes, n_modes)
            or components.shape != (n_modes,) or spectral_vectors.shape != (n_modes, 2)):
        raise ValueError("spectral arrays must match the number of basis fields")
    if np.any(eigenvalues >= 1.0):
        raise ValueError("every eigenvalue must be strictly below one")
    if gradient.shape != (2,):
        raise ValueError("macroscopic_gradient must have two components")
    lengths = tuple(float(v) for v in cell_lengths)
    if len(lengths) != 2 or min(lengths) <= 0.0:
        raise ValueError("cell_lengths must be two strictly positive values")
    n_pixels = basis.shape[1]
    multipliers = _frequency_multipliers(n_pixels, lengths)
    fluctuation = np.tensordot(eigenvectors @ components, basis, axes=(0, 0))
    gradient_1, gradient_2 = _spectral_gradient(fluctuation, multipliers)
    flux_1 = conductivity * (gradient[0] + gradient_1)
    flux_2 = conductivity * (gradient[1] + gradient_2)
    effective_column = np.array([flux_1.mean(), flux_2.mean()])
    load_field = _spectral_divergence(conductivity * gradient[0], conductivity * gradient[1], multipliers)
    residual_field = _spectral_divergence(flux_1, flux_2, multipliers)
    load_scale = float(np.abs(load_field).max())
    mean_conductivity = float(conductivity.mean())
    effective_tensor = (mean_conductivity * np.eye(2)
                        + (spectral_vectors.T * (1.0 / (eigenvalues - 1.0))) @ spectral_vectors)
    return {
        "fluctuation": fluctuation,
        "effective_column": effective_column,
        "effective_tensor": effective_tensor,
        "mean_conductivity": mean_conductivity,
        "consistency_residual": float(np.abs(effective_tensor @ gradient - effective_column).max()),
        "equilibrium_residual": float(np.abs(residual_field).max() / (load_scale if load_scale > 0.0 else 1.0)),
        "energy_norm": float(np.sqrt(np.sum(components ** 2))),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
import scipy.linalg as sla
N, L = 15, (1.0, 1.0)
DISCS = [(0.27, 0.31, 0.24, 3), (0.76, 0.72, 0.21, 3), (0.78, 0.22, 0.155, 2), (0.24, 0.79, 0.135, 2)]
COND = np.array([1.0, 3.0, 10.0])
E = np.array([1.0, 0.0])
def prepare():
    ax = np.arange(N) / N
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    phase = np.ones((N, N), dtype=int)
    for cx, cy, r, ph in DISCS:
        dx = np.abs(X - cx); dx = np.minimum(dx, 1 - dx)
        dy = np.abs(Y - cy); dy = np.minimum(dy, 1 - dy)
        phase[np.sqrt(dx * dx + dy * dy) < r] = ph
    C = COND[phase - 1]
    k = np.fft.fftfreq(N, d=1.0 / N)
    K1, K2 = np.meshgrid(k, k, indexing="ij")
    G = (2j * np.pi * K1, 2j * np.pi * K2)
    modes = []
    for k1 in k.astype(int):
        for k2 in k.astype(int):
            if k1 > 0 or (k1 == 0 and k2 > 0):
                a = 2 * np.pi * (k1 * X + k2 * Y)
                modes += [np.sqrt(2 / N ** 2) * np.cos(a), np.sqrt(2 / N ** 2) * np.sin(a)]
    Q = np.stack(modes)
    def grad(V):
        vh = np.fft.fft2(V)
        return np.real(np.fft.ifft2(G[0] * vh)), np.real(np.fft.ifft2(G[1] * vh))
    def div(f1, f2):
        return np.real(np.fft.ifft2(G[0] * np.fft.fft2(f1) + G[1] * np.fft.fft2(f2)))
    def apply(V, c):
        g1, g2 = grad(V)
        return -div(c * g1, c * g2)
    flat = Q.reshape(len(Q), -1)
    A = flat @ apply(Q, C).reshape(len(Q), -1).T / N ** 2
    A0 = flat @ apply(Q, np.full((N, N), 5.5)).reshape(len(Q), -1).T / N ** 2
    w, V = sla.eigh(0.5 * (A + A.T), 0.5 * (A0 + A0.T))
    lam = 1 - w
    o = np.argsort(lam); lam, V = lam[o], V[:, o]
    b = flat @ div(C * E[0], C * E[1]).ravel() / N ** 2
    p = (V.T @ b) / (1 - lam)
    PS = np.tensordot(V.T, Q, axes=(1, 0))
    g1, g2 = grad(PS)
    beta = np.stack([(C * g1).mean(axis=(1, 2)), (C * g2).mean(axis=(1, 2))], axis=1)
    u_direct = np.tensordot(np.linalg.solve(0.5 * (A + A.T), b), Q, axes=(0, 0))
    return Q, lam, V, p, beta, C, u_direct
Q, LAM, V, P, BETA, C, U_DIRECT = prepare()
def summarize(out):
    T = out["effective_tensor"]
    return (
        out["fluctuation"].shape,
        round(float(np.abs(out["fluctuation"] - U_DIRECT).max()), 9),
        tuple(round(float(v), 9) for v in out["effective_column"]),
        round(float(T[0, 0]), 9), round(float(T[1, 1]), 9), round(float(T[0, 1]), 9), round(float(abs(T[0, 1] - T[1, 0])), 12),
        round(float(out["mean_conductivity"]), 12), int(out["consistency_residual"] < 1e-10),
        int(out["equilibrium_residual"] < 1e-9), round(float(out["energy_norm"]), 9),
        int(T[0, 0] < out["mean_conductivity"]), int(T[0, 0] > 1.0 / np.mean(1.0 / C)),
    )
""",
            "call": "summarize(reconstruct_solution_and_effective_tensor(Q, LAM, V, P, BETA, C, L, E))",
            "gold_call": "summarize(_oracle_reconstruct_solution_and_effective_tensor(Q, LAM, V, P, BETA, C, L, E))",
        },
        {
            "setup": """import numpy as np
N, L = 9, (1.0, 1.0)
ax = np.arange(N) / N
X, Y = np.meshgrid(ax, ax, indexing="ij")
modes = []
lap = []
k = np.fft.fftfreq(N, d=1.0 / N).astype(int)
for k1 in k:
    for k2 in k:
        if k1 > 0 or (k1 == 0 and k2 > 0):
            a = 2 * np.pi * (k1 * X + k2 * Y)
            modes += [np.sqrt(2 / N ** 2) * np.cos(a), np.sqrt(2 / N ** 2) * np.sin(a)]
            lap += [(2 * np.pi) ** 2 * (k1 ** 2 + k2 ** 2)] * 2
Q = np.stack(modes)
LAP = np.array(lap)
V = np.diag(np.sqrt(N ** 2 / (2.0 * LAP)))
LAM = np.zeros(len(LAP))
P = np.zeros(len(LAP))
BETA = np.zeros((len(LAP), 2))
def summarize(out):
    return (
        round(float(np.abs(out["fluctuation"]).max()), 12),
        tuple(round(float(v), 12) for v in out["effective_column"]),
        round(float(np.abs(out["effective_tensor"] - 2.0 * np.eye(2)).max()), 12),
        int(out["equilibrium_residual"] < 1e-12), round(float(out["energy_norm"]), 12),
    )
""",
            "call": "summarize(reconstruct_solution_and_effective_tensor(Q, LAM, V, P, BETA, np.full((N, N), 2.0), L, np.array([0.7, -0.4])))",
            "gold_call": "summarize(_oracle_reconstruct_solution_and_effective_tensor(Q, LAM, V, P, BETA, np.full((N, N), 2.0), L, np.array([0.7, -0.4])))",
        },
        {
            "setup": """import numpy as np
def run(fn):
    try:
        fn(np.ones((8, 3, 3)), np.full(8, 1.5), np.eye(8), np.zeros(8), np.zeros((8, 2)), np.ones((3, 3)), (1.0, 1.0), np.array([1.0, 0.0]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(reconstruct_solution_and_effective_tensor)",
            "gold_call": "run(_oracle_reconstruct_solution_and_effective_tensor)",
        },
    ]
