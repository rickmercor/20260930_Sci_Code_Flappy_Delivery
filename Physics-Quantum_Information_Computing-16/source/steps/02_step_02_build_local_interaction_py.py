"""
Step 2: Construct the PSD two-qutrit interaction h and its shift epsilon.

The clock operators sigma = diag(1, w, w^2) and tau|j> = |j+1 mod 3> satisfy sigma tau = w tau sigma. The two-site interaction is h = A + A^dagger + epsilon I, with epsilon equal to the negative of the lowest eigenvalue of A + A^dagger, so that h is positive semidefinite with a three-dimensional kernel spanned by the deformed product ground states.

Returns
-------
# tuple, (h, epsilon) with h a (9,9) complex Hermitian ndarray and epsilon a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================
def build_local_interaction(
    r: float,
    s: float,
) -> tuple[np.ndarray, float]:
    """Construct the 9x9 Hermitian PSD local interaction h and its shift epsilon.

    Parameters
    ----------
    r, s : float
        Positive finite deformation parameters.

    Returns
    -------
    result : tuple[np.ndarray, float]
        (h, epsilon): h complex Hermitian of shape (9, 9),
        epsilon a native float.

    Raises
    ------
    ValueError
        If r or s is non-finite or <= 0.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================
import numpy as np

def _oracle_build_local_interaction(
    r: float,
    s: float,
) -> tuple[np.ndarray, float]:
    import numpy as np

    f, g1, g2 = _oracle_deformation_coefficients(r, s)

    w = np.exp(2j * np.pi / 3.0)
    sigma = np.diag([1.0, w, w**2]).astype(complex)

    tau = np.zeros((3, 3), dtype=complex)
    for j in range(3):
        tau[(j + 1) % 3, j] = 1.0

    I3 = np.eye(3, dtype=complex)

    A = -(
        np.kron(sigma, sigma.conj().T)
        + 0.5 * f * (np.kron(tau, I3) + np.kron(I3, tau))
        + g1 * np.kron(tau, tau)
        + g2 * np.kron(tau, tau.conj().T)
    )

    B = A + A.conj().T
    epsilon = -float(np.linalg.eigvalsh(B)[0])
    h = B + epsilon * np.eye(9, dtype=complex)

    return 0.5 * (h + h.conj().T), epsilon

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nr=0.347; s=0.783\npack=lambda t: np.concatenate([np.concatenate([np.array([np.ndim(x),*np.shape(x)],dtype=complex), np.asarray(x,dtype=complex).ravel()]) for x in t])",
         "call": "pack(build_local_interaction(r,s))",
         "gold_call": "pack(_oracle_build_local_interaction(r,s))"},
        # boundary: undeformed classical Potts, h = 2 - sigma sigma^dag - h.c.
        {"setup": "import numpy as np\nr=1.0; s=1.0\npack=lambda t: np.concatenate([np.concatenate([np.array([np.ndim(x),*np.shape(x)],dtype=complex), np.asarray(x,dtype=complex).ravel()]) for x in t])",
         "call": "pack(build_local_interaction(r,s))",
         "gold_call": "pack(_oracle_build_local_interaction(r,s))"},
        # edge: strongly deformed
        {"setup": "import numpy as np\nr=0.15; s=0.783\npack=lambda t: np.concatenate([np.concatenate([np.array([np.ndim(x),*np.shape(x)],dtype=complex), np.asarray(x,dtype=complex).ravel()]) for x in t])",
         "call": "pack(build_local_interaction(r,s))",
         "gold_call": "pack(_oracle_build_local_interaction(r,s))"},
    ]
