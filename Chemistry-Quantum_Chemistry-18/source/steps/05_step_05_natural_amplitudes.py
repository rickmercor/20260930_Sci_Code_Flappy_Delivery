"""
Step 05: Natural amplitudes of one angular momentum channel. Natural amplitudes of one angular momentum channel of the pair function Psi, normalized with respect to the whole function.

A normalized singlet spatial function has the spectral representation Psi(r1_vec, r2_vec) = sum over n of
lambda_n phi_n(r1_vec) phi_n(r2_vec), with orthonormal real natural orbitals phi_n and real natural amplitudes lambda_n
satisfying sum over n of lambda_n^2 = 1. For a spherically symmetric Psi the natural orbitals carry a definite angular
momentum, so the amplitudes separate into independent channels labelled by l, and every orbital of a channel shares the
amplitude of its shell with the other 2l + 1 members.

The amplitudes returned here are those of the whole function rather than of the channel alone, so that squaring them and
summing over every channel with the degeneracy gives one. Channels of higher l contribute amplitudes that fall off
slowly, which is what makes the natural-orbital expansion of a correlated pair function converge so poorly.

Returns
-------
numpy.ndarray of shape (n_keep,): signed natural amplitudes of the l channel ordered by decreasing magnitude
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def natural_amplitudes(omega: float, c: float, l: int, n_keep: int) -> "np.ndarray":
    '''Leading natural amplitudes of angular momentum l, ordered by decreasing magnitude.

    Parameters
    ----------
    omega : float
        Gaussian exponent parameter, omega > 0 (atomic units).
    c : float
        Coefficient of r12^2 in p(s) = 1 + s/2 + c s^2, c >= 0.
    l : int
        Angular momentum of the channel, 0 <= l <= 12.
    n_keep : int
        Number of amplitudes returned, 1 <= n_keep <= 80.

    Returns
    -------
    result : np.ndarray
        Float array of shape (n_keep,) with the n_keep amplitudes of largest magnitude of the l channel, signed and
        ordered by decreasing magnitude, normalized so that the amplitudes of all channels, each counted 2l + 1 times,
        have squares summing to one. Each amplitude is accurate to 1e-10 absolute.

    Raises
    ------
    ValueError
        If omega is not positive, c is negative, or l or n_keep lies outside its range.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _radial_grid(omega):
    """Gauss-Legendre nodes and weights on [0, R] with the Gaussian envelope negligible at R."""
    x, w = np.polynomial.legendre.leggauss(1000)
    r_max = np.sqrt(100.0 / omega)
    return 0.5 * r_max * (x + 1.0), 0.5 * r_max * w


def _channel_spectrum(omega, c, l):
    """Signed eigenvalues and radial eigenvectors of the l-channel kernel, ordered by decreasing magnitude."""
    r, w = _radial_grid(omega)
    norm = _oracle_pair_normalization(omega, c)
    envelope = np.exp(-0.5 * omega * r * r)
    kernel = (4.0 * np.pi / (2 * l + 1)) * norm * _oracle_pair_partial_wave(r[:, None], r[None, :], l, c)
    kernel = kernel * envelope[:, None] * envelope[None, :]
    scale = np.sqrt(w) * r
    values, vectors = np.linalg.eigh(scale[:, None] * kernel * scale[None, :])
    order = np.argsort(-np.abs(values))
    return values[order], vectors[:, order], r, w, scale


def _oracle_natural_amplitudes(omega: float, c: float, l: int, n_keep: int) -> "np.ndarray":
    """Reference implementation."""
    omega = float(omega)
    c = float(c)
    if not omega > 0.0:
        raise ValueError("omega must be positive")
    if c < 0.0:
        raise ValueError("c must be non-negative")
    if not 0 <= int(l) <= 12 or not 1 <= int(n_keep) <= 80:
        raise ValueError("l or n_keep out of range")
    values = _channel_spectrum(omega, c, int(l))[0]
    return values[: int(n_keep)].copy()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: s channel of the exact omega = 1/10 ground state ---
        {
            "setup": "import numpy as np\n",
            "call": "natural_amplitudes(0.1, 0.05, 0, 12)",
            "gold_call": "_oracle_natural_amplitudes(0.1, 0.05, 0, 12)",
            "tol": 1e-10,
        },
        # --- Normal: p channel, which carries the second-largest amplitude ---
        {
            "setup": "import numpy as np\n",
            "call": "natural_amplitudes(0.1, 0.05, 1, 6)",
            "gold_call": "_oracle_natural_amplitudes(0.1, 0.05, 1, 6)",
            "tol": 1e-10,
        },
        # --- Boundary: a single amplitude of a high channel at omega = 1/2 ---
        {
            "setup": "import numpy as np\n",
            "call": "natural_amplitudes(0.5, 0.0, 7, 1)",
            "gold_call": "_oracle_natural_amplitudes(0.5, 0.0, 7, 1)",
            "tol": 1e-10,
        },
        # --- Edge: many weak s amplitudes in a tight trap ---
        {
            "setup": "import numpy as np\n",
            "call": "natural_amplitudes(3.0, 0.1, 0, 40)",
            "gold_call": "_oracle_natural_amplitudes(3.0, 0.1, 0, 40)",
            "tol": 1e-10,
        },
        # --- Error: a negative angular momentum must raise ValueError ---
        {
            "setup": "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(0.1, 0.05, -2, 5)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    return 0\n",
            "call": "_probe(natural_amplitudes)",
            "gold_call": "_probe(_oracle_natural_amplitudes)",
        },
    ]
