"""
Under a uniform macroscopic gradient E the fluctuation u of the potential solves A u = b with load b = div(C E), the divergence of the flux that the macroscopic gradient alone would produce. Preconditioning gives (I - S) u = A_0^{-1} b, and expanding u on the eigenstates, which are orthonormal for the energetic inner product, yields the components p_j = (A_0^{-1} b, phi_j) / (1 - lambda_j). The numerator is simply the pixel average of b phi_j, and integrating by parts once more turns it into -E . beta_j with beta_j the average of C grad phi_j, a vector attached to each eigenstate that measures how strongly it couples to a uniform loading; the same vectors govern the spectral form of the effective conductivity in the next step. Both routes are computed here and the largest discrepancy between them is returned, a check that the load, the eigenvectors, the normalisation and the spectral derivative are mutually consistent. Eigenstates confined to a single phase have beta_j equal to zero, since the average of C grad phi_j reduces to C_p times the average of grad phi_j over the cell, which vanishes by periodicity, so only eigenstates with non-trivial interface behaviour receive a component. The dominant eigenstate, the one with the largest component magnitude, is identified along with its eigenvalue.

Returns
-------
dict with float array load of shape (N - 1,) holding the basis coefficients of the pixel-averaged load, float array components of shape (N - 1,) holding p_j, float array spectral_vectors of shape (N - 1, 2) holding beta_j, native float identity_residual, native int dominant_index, native float dominant_eigenvalue and native float dominant_magnitude.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def project_cell_load_onto_eigenstates(
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    conductivity: np.ndarray,
    cell_lengths: tuple,
    macroscopic_gradient: np.ndarray,
) -> dict:
    """Expand the cell load on the eigenstates and form the spectral coupling vectors.

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


def _oracle_project_cell_load_onto_eigenstates(
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    conductivity: np.ndarray,
    cell_lengths: tuple,
    macroscopic_gradient: np.ndarray,
) -> dict:
    """Reference implementation."""
    basis = np.asarray(basis, dtype=float)
    eigenvalues = np.asarray(eigenvalues, dtype=float).ravel()
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    conductivity = np.asarray(conductivity, dtype=float)
    gradient = np.asarray(macroscopic_gradient, dtype=float).ravel()
    if basis.ndim != 3 or basis.shape[1] != basis.shape[2] or conductivity.shape != basis.shape[1:]:
        raise ValueError("basis must have shape (N - 1, n, n) matching the conductivity grid")
    n_modes = basis.shape[0]
    if eigenvalues.shape != (n_modes,) or eigenvectors.shape != (n_modes, n_modes):
        raise ValueError("eigenvalues and eigenvectors must match the number of basis fields")
    if np.any(eigenvalues >= 1.0):
        raise ValueError("every eigenvalue must be strictly below one")
    if gradient.shape != (2,):
        raise ValueError("macroscopic_gradient must have two components")
    lengths = tuple(float(v) for v in cell_lengths)
    if len(lengths) != 2 or min(lengths) <= 0.0:
        raise ValueError("cell_lengths must be two strictly positive values")
    n_pixels = basis.shape[1]
    n_points = n_pixels * n_pixels
    multipliers = _frequency_multipliers(n_pixels, lengths)
    load_field = _spectral_divergence(conductivity * gradient[0], conductivity * gradient[1], multipliers)
    load = basis.reshape(n_modes, n_points) @ load_field.ravel() / n_points
    projected = eigenvectors.T @ load
    components = projected / (1.0 - eigenvalues)
    eigenstates = np.tensordot(eigenvectors.T, basis, axes=(1, 0))
    gradient_1, gradient_2 = _spectral_gradient(eigenstates, multipliers)
    spectral_vectors = np.stack([
        (conductivity * gradient_1).mean(axis=(1, 2)),
        (conductivity * gradient_2).mean(axis=(1, 2)),
    ], axis=1)
    dominant = int(np.argmax(np.abs(components)))
    return {
        "load": load,
        "components": components,
        "spectral_vectors": spectral_vectors,
        "identity_residual": float(np.abs(projected + spectral_vectors @ gradient).max()),
        "dominant_index": dominant,
        "dominant_eigenvalue": float(eigenvalues[dominant]),
        "dominant_magnitude": float(abs(components[dominant])),
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
    def apply(V, c):
        vh = np.fft.fft2(V)
        g1 = np.real(np.fft.ifft2(G[0] * vh)); g2 = np.real(np.fft.ifft2(G[1] * vh))
        return -np.real(np.fft.ifft2(G[0] * np.fft.fft2(c * g1) + G[1] * np.fft.fft2(c * g2)))
    flat = Q.reshape(len(Q), -1)
    A = flat @ apply(Q, C).reshape(len(Q), -1).T / N ** 2
    A0 = flat @ apply(Q, np.full((N, N), 5.5)).reshape(len(Q), -1).T / N ** 2
    w, V = sla.eigh(0.5 * (A + A.T), 0.5 * (A0 + A0.T))
    lam = 1 - w
    o = np.argsort(lam)
    return Q, lam[o], V[:, o], C
Q, LAM, V, C = prepare()
def summarize(out):
    p = out["components"]; beta = out["spectral_vectors"]
    return (
        p.shape, beta.shape, out["load"].shape,
        round(float(np.sum(p ** 2)), 9), round(float(np.abs(p).max()), 9),
        round(float(out["dominant_magnitude"]), 9), round(float(out["dominant_eigenvalue"]), 9),
        round(float(np.sum(beta ** 2)), 9), int(out["identity_residual"] < 1e-10),
        round(float(np.sum(out["load"] ** 2)), 9),
    )
""",
            "call": "summarize(project_cell_load_onto_eigenstates(Q, LAM, V, C, L, np.array([1.0, 0.0])))",
            "gold_call": "summarize(_oracle_project_cell_load_onto_eigenstates(Q, LAM, V, C, L, np.array([1.0, 0.0])))",
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
def summarize(out):
    return (
        round(float(np.abs(out["components"]).max()), 12),
        round(float(np.abs(out["spectral_vectors"]).max()), 12),
        round(float(np.abs(out["load"]).max()), 12),
        int(out["identity_residual"] < 1e-12),
    )
""",
            "call": "summarize(project_cell_load_onto_eigenstates(Q, LAM, V, np.full((N, N), 2.0), L, np.array([0.3, -1.2])))",
            "gold_call": "summarize(_oracle_project_cell_load_onto_eigenstates(Q, LAM, V, np.full((N, N), 2.0), L, np.array([0.3, -1.2])))",
        },
        {
            "setup": """import numpy as np
def run(fn):
    try:
        fn(np.ones((8, 3, 3)), np.zeros(8), np.eye(8), np.ones((3, 3)), (1.0, 1.0), np.array([1.0, 0.0, 0.0]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(project_cell_load_onto_eigenstates)",
            "gold_call": "run(_oracle_project_cell_load_onto_eigenstates)",
        },
    ]
