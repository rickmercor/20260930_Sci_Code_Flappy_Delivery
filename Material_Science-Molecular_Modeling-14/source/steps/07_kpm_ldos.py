"""
Evaluate the momentum local density of states at the base momentum through the source's kernel polynomial approximation of the delta function at polynomial order P and scaling s: Chebyshev moments of the scaled Hamiltonian on the zero-reciprocal-vector unit vectors of each layer, damped by the source's kernel coefficients, reconstructed with the source's normalization, and averaged over layers and sublattices as the source defines the momentum LDoS.

The source approximates spectral densities with a damped Chebyshev expansion; the exact damping coefficients, the reconstruction prefactor, and which diagonal entries enter the momentum LDoS all follow the source.

Returns
-------
return (nE,) float64: momentum LDoS at the requested energies
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def kpm_ldos(Hri, dofs, Elist, P, s):
    """Hri: (2n, 2n, 2) stacked Hamiltonian; dofs: (n, 5) degrees of freedom;
    Elist: (nE,) energies; P: polynomial order; s: spectral scaling with
    s * spectrum inside (-1, 1). Returns (nE,) float64: the momentum LDoS at
    each energy, computed with the source's damped Chebyshev approximation
    on the zero-reciprocal-vector entries and averaged as the source
    prescribes. Raises ValueError on an invalid order."""
    return np.zeros(np.asarray(Elist).shape[0])

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 7: Jackson-damped Chebyshev momentum LDoS."""

import numpy as np


def _jackson_g(P):
    n = np.arange(P)
    return (2.0 - (n == 0)) * ((P - n + 1) * np.cos(np.pi * n / (P + 1)) +
                               np.sin(np.pi * n / (P + 1)) / np.tan(np.pi / (P + 1))) / (P + 1)


def _oracle_kpm_ldos(Hri, dofs, Elist, P, s):
    if isinstance(P, bool) or not isinstance(P, (int, np.integer)) or P < 2:
        raise ValueError("invalid polynomial order")
    Hri = np.asarray(Hri, dtype=np.float64)
    H = Hri[:, :, 0] + 1j * Hri[:, :, 1]
    dofs = np.asarray(dofs, dtype=np.int64)
    Elist = np.asarray(Elist, dtype=np.float64)
    dim = H.shape[0]
    Hs = s * H
    g = _jackson_g(int(P))
    idx = []
    for r in range(dofs.shape[0]):
        if np.all(dofs[r, 1:5] == 0):
            idx.extend([2 * r, 2 * r + 1])
    V = np.zeros((dim, len(idx)), dtype=np.complex128)
    for c, i in enumerate(idx):
        V[i, c] = 1.0
    mu = np.zeros((int(P), len(idx)))
    Tm2 = V.copy()
    Tm1 = Hs @ V
    for n in range(int(P)):
        if n == 0:
            Tn = Tm2
        elif n == 1:
            Tn = Tm1
        else:
            Tn = 2.0 * (Hs @ Tm1) - Tm2
            Tm2 = Tm1
            Tm1 = Tn
        for c, i in enumerate(idx):
            mu[n, c] = float(Tn[i, c].real)
    out = np.empty(Elist.shape[0])
    for ei, E in enumerate(Elist):
        sE = s * float(E)
        acc = np.zeros(len(idx))
        for n in range(int(P)):
            acc += g[n] * mu[n, :] * np.cos(n * np.arccos(np.clip(sE, -1.0, 1.0)))
        vals = acc / (np.pi * np.sqrt(max(1.0 - sE * sE, 1e-15)))
        out[ei] = float(np.sum(vals)) / 6.0
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\ngeom=layer_geometry(_n.array([-0.06,0.0,0.11]))\nq=geom[1,8:10]+_n.array([0.03,0.02])\ndofs=wl_dof(q, geom, 1.15, 5.0, 3)\nH=assemble_hamiltonian(q, dofs, geom)', "call": "kpm_ldos(H, dofs, _n.array([-0.5,0.0,0.5]), 64, 0.25)", "gold_call": "_oracle_kpm_ldos(H, dofs, _n.array([-0.5,0.0,0.5]), 64, 0.25)", "tol": 1e-08},
        {"setup": 'import numpy as _n\ngeom=layer_geometry(_n.array([-0.09,0.0,0.07]))\nq=geom[1,10:12]+_n.array([0.03,0.02])\ndofs=wl_dof(q, geom, 1.05, 4.6, 3)\nH=assemble_hamiltonian(q, dofs, geom)', "call": "kpm_ldos(H, dofs, _n.array([-0.5,0.0,0.5]), 48, 0.25)", "gold_call": "_oracle_kpm_ldos(H, dofs, _n.array([-0.5,0.0,0.5]), 48, 0.25)", "tol": 1e-08},
        {"setup": 'import numpy as _n\ngeom=layer_geometry(_n.array([-0.05,0.0,0.09]))\nq=geom[1,8:10]+_n.array([0.03,0.02])\ndofs=wl_dof(q, geom, 0.9, 4.8, 3)\nH=assemble_hamiltonian(q, dofs, geom)', "call": "kpm_ldos(H, dofs, _n.array([0.0,0.3]), 64, 0.3)", "gold_call": "_oracle_kpm_ldos(H, dofs, _n.array([0.0,0.3]), 64, 0.3)", "tol": 1e-08},
    ]
