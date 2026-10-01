"""
Assemble the Hermitian reciprocal Hamiltonian on the truncated degrees of freedom: intralayer blocks on the diagonal at the shifted momenta, and interlayer couplings only between the layer pairs the source admits, subject to the source's matching condition on the third layer's reciprocal vector, with the source's normalization and sublattice phase factors, the declared interlayer coupling h(xi) = 0.045 exp(-0.1 |xi|^2) evaluated at the momentum the source prescribes, and tapered by the source's bump with tau = 3.0, delta = 0.6. The last axis stacks real and imaginary parts.

Which layer pairs couple, which reciprocal index must match across a coupling, at which combined momentum the interlayer function is evaluated, and which normalization and phases multiply it, are all fixed by the source's reciprocal-space representation.

Returns
-------
return (2n, 2n, 2) float64: reciprocal Hamiltonian, real and imaginary parts stacked
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def assemble_hamiltonian(q, dofs, geom):
    """q: (2,) base momentum; dofs: (n, 5) int64 degrees of freedom from the
    truncation step; geom: (3, 14) packed geometry. Returns (2n, 2n, 2)
    float64: the Hermitian reciprocal Hamiltonian with two sublattice rows
    per degree of freedom in order, assembled exactly as the source
    specifies, real part in [..., 0] and imaginary part in [..., 1]."""
    return np.zeros((2 * np.asarray(dofs).shape[0], 2 * np.asarray(dofs).shape[0], 2))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 6: assembly of the truncated reciprocal Hamiltonian."""

import numpy as np

_W0_INTER = 0.045
_ETA_INTER = 0.1
_TAU_TR = 3.0
_DELTA_TR = 0.6


def _hhat_inter(xi):
    xi = np.asarray(xi, dtype=np.float64)
    return _W0_INTER * np.exp(-_ETA_INTER * float(xi @ xi))


def _oracle_assemble_hamiltonian(q, dofs, geom):
    q = np.asarray(q, dtype=np.float64)
    dofs = np.asarray(dofs, dtype=np.int64)
    geom = np.asarray(geom, dtype=np.float64)
    nd = dofs.shape[0]
    dim = 2 * nd
    H = np.zeros((dim, dim), dtype=np.complex128)
    detB = abs(np.linalg.det(geom[0, 4:8].reshape(2, 2)))
    cstar2 = detB
    cache = []
    for r in range(nd):
        j = int(dofs[r, 0])
        others = [t for t in range(3) if t != j]
        k, l = others[0], others[1]
        Gk = geom[k, 4:8].reshape(2, 2) @ dofs[r, 1:3].astype(np.float64)
        Gl = geom[l, 4:8].reshape(2, 2) @ dofs[r, 3:5].astype(np.float64)
        cache.append((j, k, l, Gk, Gl))
    for r in range(nd):
        j, k, l, Gk, Gl = cache[r]
        blk = _oracle_intralayer_block(q + Gk + Gl, geom[j, 0:4], geom[j, 12:14], _TAU_TR)
        H[2 * r:2 * r + 2, 2 * r:2 * r + 2] = blk[:, :, 0] + 1j * blk[:, :, 1]
    for r in range(nd):
        j, kj, lj, Gkj, Glj = cache[r]
        for r2 in range(nd):
            j2, kj2, lj2, Gkj2, Glj2 = cache[r2]
            if abs(j - j2) != 1:
                continue
            shared = [t for t in range(3) if t != j and t != j2][0]
            Gj_sh = Gkj if kj == shared else Glj
            Gj2_sh = Gkj2 if kj2 == shared else Glj2
            if np.linalg.norm(Gj_sh - Gj2_sh) > 1e-9:
                continue
            Gj2_at_j = Gkj if kj == j2 else Glj
            Gj_at_j2 = Gkj2 if kj2 == j else Glj2
            xi = q + Gj_at_j2 + Gj2_at_j + Gj_sh
            amp = _hhat_inter(xi) * float(_oracle_bump_gtau(np.array([np.linalg.norm(xi)]), _TAU_TR, _DELTA_TR)[0]) * cstar2
            taus_j = [np.zeros(2), geom[j, 12:14]]
            taus_j2 = [np.zeros(2), geom[j2, 12:14]]
            for al in range(2):
                for be in range(2):
                    phase = np.exp(1j * (Gj_at_j2 @ taus_j[al] - Gj2_at_j @ taus_j2[be]))
                    H[2 * r + al, 2 * r2 + be] += amp * phase
    H = (H + H.conj().T) / 2.0
    out = np.empty((dim, dim, 2))
    out[:, :, 0] = H.real
    out[:, :, 1] = H.imag
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\ngeom=layer_geometry(_n.array([-0.06,0.0,0.11]))\nq=geom[1,8:10]+_n.array([0.03,0.02])\ndofs=wl_dof(q, geom, 1.15, 5.0, 3)', "call": "assemble_hamiltonian(q, dofs, geom)", "gold_call": "_oracle_assemble_hamiltonian(q, dofs, geom)", "tol": 1e-08},
        {"setup": 'import numpy as _n\ngeom=layer_geometry(_n.array([-0.09,0.0,0.07]))\nq=geom[1,10:12]+_n.array([0.03,0.02])\ndofs=wl_dof(q, geom, 1.05, 4.6, 3)', "call": "assemble_hamiltonian(q, dofs, geom)", "gold_call": "_oracle_assemble_hamiltonian(q, dofs, geom)", "tol": 1e-08},
        {"setup": 'import numpy as _n\ngeom=layer_geometry(_n.array([-0.05,0.0,0.09]))\nq=geom[1,8:10]+_n.array([0.03,0.02])\ndofs=wl_dof(q, geom, 0.9, 4.8, 3)', "call": "assemble_hamiltonian(q, dofs, geom)", "gold_call": "_oracle_assemble_hamiltonian(q, dofs, geom)", "tol": 1e-08},
    ]
