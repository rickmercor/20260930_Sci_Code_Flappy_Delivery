"""
Project screened interactions into vertical electron-hole pairs.

Use the real-space direct interaction matrix element and the Fourier

convention W(r,r') proportional to exp(+i(q+G).r) W_GG'(q)

exp(-i(q+G').r'). Thus the row-to-column pair transfer is q=k-k'.

No further thickness vertex weights are applied at this stage: their effects

are already contained in the effective screened interaction supplied here.

Returns
-------
complex direct-interaction matrix of shape (K, K).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def direct_kernel(
    vectors: "np.ndarray",
    k_points: "np.ndarray",
    g_vectors: "np.ndarray",
    centres: "np.ndarray",
    interactions: "np.ndarray",
    pair_indices: "np.ndarray",
) -> "np.ndarray":
    """Contract W into the direct term for one valence and one conduction band.

    Parameters
    ----------
    vectors : "np.ndarray"
        Finite complex array (K, 2, 2), K >= 1, indexed by k, orbital, band;
        valence is band 0 and conduction band 1. General finite coefficients
        are supported; physical inputs are orthonormal Bloch eigenvectors.
    k_points : "np.ndarray"
        Finite real array (K, 2) of Cartesian momenta in inverse angstroms.
    g_vectors : "np.ndarray"
        Finite real array (G, 2), G >= 1, in inverse angstroms.
    centres : "np.ndarray"
        Finite real array (2, 2) of in-plane orbital centres in angstroms.
    interactions : "np.ndarray"
        Finite complex array (T, G, G), T >= 1, of W matrices in eV, in the
        Fourier convention in the module description. It already includes
        the selected zero-mode convention and all thickness effects.
    pair_indices : "np.ndarray"
        Integer array (K, K), with entries in [0, T), mapping ordered pair
        (i,j) to the W matrix for the unreduced transfer k_i-k_j. Boolean
        arrays are invalid. Consistency with the listed momenta is the
        caller's responsibility.

    Returns
    -------
    direct : "np.ndarray"
        Complex array (K, K) in eV. Each element is the contraction of the
        conduction matrix element of exp(+i(q+G).r), W_GG'(q), and the
        valence matrix element of exp(-i(q+G').r), with a 1/K mesh factor.
        No spin multiplier or exchange term is included. Mutual transfer
        symmetries are not enforced; physical consistent inputs yield a
        Hermitian direct kernel, covariant under independent band phases.

    Raises
    ------
    ValueError
        If a shape, finite/real input requirement, or integer index bound
        above is violated.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_direct_kernel(
    vectors: "np.ndarray",
    k_points: "np.ndarray",
    g_vectors: "np.ndarray",
    centres: "np.ndarray",
    interactions: "np.ndarray",
    pair_indices: "np.ndarray",
) -> "np.ndarray":
    """Contract the real-space direct matrix element using the stated signs."""
    u = _finite_array(vectors, complex, "vectors")
    k = _finite_array(k_points, float, "k_points")
    gs = _finite_array(g_vectors, float, "g_vectors")
    tau = _finite_array(centres, float, "centres")
    ws = _finite_array(interactions, complex, "interactions")
    ids = _finite_array(pair_indices, float, "pair_indices")
    if (
        u.ndim != 3
        or u.shape[1:] != (2, 2)
        or len(u) < 1
        or k.shape != (len(u), 2)
        or tau.shape != (2, 2)
    ):
        raise ValueError("invalid band or geometry shape")
    if gs.ndim != 2 or gs.shape[1] != 2 or len(gs) < 1:
        raise ValueError("g_vectors must have shape (G, 2)")
    if ws.ndim != 3 or len(ws) < 1 or ws.shape[1:] != (len(gs), len(gs)):
        raise ValueError("interactions must have shape (T, G, G)")
    original_ids = np.asarray(pair_indices)
    if (
        ids.shape != (len(u), len(u))
        or original_ids.dtype.kind not in "iu"
        or np.any(ids < 0)
        or np.any(ids >= len(ws))
    ):
        raise ValueError(
            "pair_indices must be integer indices into interactions"
        )
    ids = original_ids.astype(np.intp, copy=False)
    transfers = k[:, None, :] - k[None, :, :]
    phase = np.exp(
        1j * np.einsum("ijgd,ad->ijga", transfers[:, :, None, :] + gs, tau)
    )
    cproduct = u[:, None, :, 1].conj() * u[None, :, :, 1]
    vproduct = u[:, None, :, 0].conj() * u[None, :, :, 0]
    cv = np.einsum("ija,ijga->ijg", cproduct, phase)
    vv = np.einsum("ija,ijga->ijg", vproduct, phase)
    return np.einsum("ijg,ijgh,ijh->ij", cv, ws[ids], vv.conj()) / len(u)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """
    Single reciprocal component, complex local fields and a zero
    interaction.
    """
    return [
        {
            "setup": """import numpy as np

u = np.broadcast_to(np.eye(2), (2, 2, 2)).copy().astype(complex)
k = np.array([[0.0, 0.0], [0.4, -0.2]])
g = np.zeros((1, 2))
t = np.array([[0.0, 0.0], [0.8, 0.5]])
w = np.array([[[1.0]], [[2.0]], [[2.0]]], dtype=complex)
ids = np.array([[0, 1], [2, 0]])
""",
            "call": (
                (
                    "direct_kernel(u.copy(), k.copy(), g.copy(), "
                    "t.copy(), w.copy(), ids.copy())"
                )
            ),
            "gold_call": (
                (
                    "_oracle_direct_kernel(u.copy(), k.copy(), "
                    "g.copy(), t.copy(), w.copy(), ids.copy())"
                )
            ),
        },
        {
            "setup": """import numpy as np

u = np.array(
    [
        [[1.0, 1.0j], [1.0j, 1.0]],
        [[1.0, -1.0], [1.0, 1.0]],
        [[1.0, -1.0j], [-1.0j, 1.0]],
    ]
) / np.sqrt(2)
k = np.array([[0.0, 0.0], [0.4, -0.2], [-0.1, 0.3]])
g = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
t = np.array([[0.2, -0.3], [0.8, 0.5]])
b = np.array([[0.2, 0.3j], [0.6, -0.1], [-0.2j, 0.4]])
w = np.stack([b @ b.conj().T, np.eye(3), 2 * np.eye(3)])
ids = np.array([[0, 1, 2], [1, 0, 2], [2, 2, 0]])
""",
            "call": (
                (
                    "direct_kernel(u.copy(), k.copy(), g.copy(), "
                    "t.copy(), w.copy(), ids.copy())"
                )
            ),
            "gold_call": (
                (
                    "_oracle_direct_kernel(u.copy(), k.copy(), "
                    "g.copy(), t.copy(), w.copy(), ids.copy())"
                )
            ),
        },
        {
            "setup": """import numpy as np

u = np.eye(2, dtype=complex)[None]
k = np.array([[0.2, -0.4]])
g = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
t = np.array([[0.0, 0.0], [0.7, 0.6]])
w = np.zeros((1, 3, 3), dtype=complex)
ids = np.zeros((1, 1), dtype=int)
""",
            "call": (
                (
                    "direct_kernel(u.copy(), k.copy(), g.copy(), "
                    "t.copy(), w.copy(), ids.copy())"
                )
            ),
            "gold_call": (
                (
                    "_oracle_direct_kernel(u.copy(), k.copy(), "
                    "g.copy(), t.copy(), w.copy(), ids.copy())"
                )
            ),
        },
        {
            "setup": """import numpy as np


def rejected(fn):
    try:
        fn(
            np.eye(2)[None],
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            np.zeros((2, 2)),
            np.ones((1, 1, 1)),
            np.ones((1, 1), dtype=int),
        )
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "rejected(direct_kernel)",
            "gold_call": "rejected(_oracle_direct_kernel)",
        },
    ]
