"""
Evaluate the 2x2 intralayer Bloch block of one layer at an effective momentum, from the declared hopping functions: amplitude t1 = -1 on the Bravais steps (0,0), (-1,0), (-1,1) from sublattice A to B; t2 = 0.15 on the steps (1,0), (0,1), (1,-1) and their negatives within each sublattice; t3 = 0.08 on the A-to-B steps (0,1), (-2,1), (0,-1); t4 = 0.05 on the A-to-B steps (2,-1), (-2,0), (1,1); each with its Hermitian partner. The Bloch phase convention and the real-space truncation of the hop sum at radius tau follow the source. The last axis stacks real and imaginary parts.

The intralayer part of the momentum space Hamiltonian is a truncated Bloch sum; both the phase convention (what multiplies the momentum in the exponent) and the hard truncation on the hop vectors are the source's.

Returns
-------
return (2, 2, 2) float64: intralayer Bloch block, real and imaginary parts stacked
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def intralayer_block(qeff, Aj, tauB, tau):
    """qeff: (2,) effective momentum; Aj: layer Bravais matrix, flattened
    (4,) row-major or (2, 2); tauB: (2,) B-sublattice offset; tau: real-space
    truncation radius. Returns (2, 2, 2) float64: the Hermitian intralayer
    Bloch block from the declared hopping shells, phases and truncation
    exactly as the source specifies, real part in [..., 0] and imaginary part
    in [..., 1]."""
    return np.zeros((2, 2, 2))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 5: truncated intralayer Bloch block."""

import numpy as np

_T1, _T2, _T3, _T4 = -1.0, 0.15, 0.08, 0.05


def _hop_list(Aj, tauB):
    a1 = Aj[:, 0]
    a2 = Aj[:, 1]
    hops = []
    for (n1, n2) in ((0, 0), (-1, 0), (-1, 1)):
        R = n1 * a1 + n2 * a2
        hops.append((R, 0, 1, _T1))
        hops.append((-R, 1, 0, _T1))
    for (n1, n2) in ((1, 0), (0, 1), (1, -1)):
        R = n1 * a1 + n2 * a2
        for al in (0, 1):
            hops.append((R, al, al, _T2))
            hops.append((-R, al, al, _T2))
    for (n1, n2) in ((0, 1), (-2, 1), (0, -1)):
        R = n1 * a1 + n2 * a2
        hops.append((R, 0, 1, _T3))
        hops.append((-R, 1, 0, _T3))
    for (n1, n2) in ((2, -1), (-2, 0), (1, 1)):
        R = n1 * a1 + n2 * a2
        hops.append((R, 0, 1, _T4))
        hops.append((-R, 1, 0, _T4))
    return hops


def _oracle_intralayer_block(qeff, Aj, tauB, tau):
    qeff = np.asarray(qeff, dtype=np.float64)
    Aj = np.asarray(Aj, dtype=np.float64).reshape(2, 2)
    tauB = np.asarray(tauB, dtype=np.float64)
    taus = [np.zeros(2), tauB]
    H = np.zeros((2, 2), dtype=np.complex128)
    for (R, al, be, t) in _hop_list(Aj, tauB):
        if np.linalg.norm(R) > tau:
            continue
        H[al, be] += t * np.exp(-1j * (qeff @ (R + taus[al] - taus[be])))
    H = (H + H.conj().T) / 2.0
    out = np.empty((2, 2, 2))
    out[:, :, 0] = H.real
    out[:, :, 1] = H.imag
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\ngeom=layer_geometry(_n.array([-0.06,0.0,0.11]))\nq=geom[1,8:10]+_n.array([0.03,0.02])', "call": "intralayer_block(q, geom[0,0:4], geom[0,12:14], 3.0)", "gold_call": "_oracle_intralayer_block(q, geom[0,0:4], geom[0,12:14], 3.0)", "tol": 1e-09},
        {"setup": 'import numpy as _n\ngeom=layer_geometry(_n.array([-0.09,0.0,0.07]))\nq=geom[1,10:12]+_n.array([0.03,0.02])\nqe=q+geom[0,4:8].reshape(2,2)@_n.array([1.0,-1.0])', "call": "intralayer_block(qe, geom[2,0:4], geom[2,12:14], 3.0)", "gold_call": "_oracle_intralayer_block(qe, geom[2,0:4], geom[2,12:14], 3.0)", "tol": 1e-09},
        {"setup": 'import numpy as _n\ngeom=layer_geometry(_n.array([-0.05,0.0,0.09]))\nq=geom[1,8:10]+_n.array([0.03,0.02])', "call": "intralayer_block(q, geom[1,0:4], geom[1,12:14], 5.5)", "gold_call": "_oracle_intralayer_block(q, geom[1,0:4], geom[1,12:14], 5.5)", "tol": 1e-09},
    ]
