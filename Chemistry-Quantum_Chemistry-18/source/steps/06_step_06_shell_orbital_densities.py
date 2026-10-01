"""
Step 06: Shell densities of natural orbitals at a given radius. Natural amplitudes of angular momentum l paired with the summed squares of their natural orbitals at distance r from the centre.

The natural orbitals of angular momentum l are a radial function times a spherical harmonic, with the radial functions
normalized as the integral of R_k^2 r^2 dr = 1, and all 2l + 1 members of a shell share one amplitude. By the addition
theorem the sum of their squares at a point does not depend on direction, so a shell contributes a single isotropic
density to any spherically averaged quantity. That shell sum is what enters the natural-orbital expressions for the
one-electron density and for the on-top density.

The radius may be any non-negative value, including the trap centre and points far outside the region where the
quadrature that produces the radial functions has its nodes.

Returns
-------
numpy.ndarray of shape (n_keep, 2): l-channel amplitudes and shell sums (2l+1) R_k(r)^2 / (4 pi) at radius r
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def shell_orbital_densities(omega: float, c: float, l: int, n_keep: int, r: float) -> "np.ndarray":
    '''Pairs (lambda_k, (2l + 1) R_k(r)^2 / (4 pi)) for the n_keep l-channel natural orbitals of largest amplitude magnitude.

    Parameters
    ----------
    omega : float
        Gaussian exponent parameter, omega > 0 (atomic units).
    c : float
        Coefficient of r12^2 in p(s) = 1 + s/2 + c s^2, c >= 0.
    l : int
        Angular momentum of the channel, 0 <= l <= 12.
    n_keep : int
        Number of radial functions, 1 <= n_keep <= 80.
    r : float
        Distance from the trap centre in bohr, 0 <= r <= 5 omega^(-1/2).

    Returns
    -------
    result : np.ndarray
        Float array of shape (n_keep, 2); column 0 holds the signed amplitudes ordered by decreasing magnitude and column 1
        the shell sums (2l + 1) R_k(r)^2 / (4 pi) in bohr^-3 of the same orbitals. Column 0 is accurate to 1e-10 absolute
        and column 1 to 1e-6 relative plus 1e-12 absolute for k <= 15.

    Raises
    ------
    ValueError
        If r is negative.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_shell_orbital_densities(omega: float, c: float, l: int, n_keep: int, r: float) -> "np.ndarray":
    """Reference implementation."""
    r = float(r)
    if r < 0.0:
        raise ValueError("r must be non-negative")
    l = int(l)
    n_keep = int(n_keep)
    amplitudes = _oracle_natural_amplitudes(omega, c, l, n_keep)
    values, vectors, grid, weights, scale = _channel_spectrum(float(omega), float(c), l)
    norm = _oracle_pair_normalization(omega, c)
    radial = vectors[:, :n_keep] / scale[:, None]
    row = (4.0 * np.pi / (2 * l + 1)) * norm * _oracle_pair_partial_wave(np.full_like(grid, r), grid, l, c)
    row = row * np.exp(-0.5 * float(omega) * (r * r + grid * grid))
    at_r = np.einsum("i,ik->k", row * grid * grid * weights, radial) / values[:n_keep]
    out = np.empty((n_keep, 2))
    out[:, 0] = amplitudes
    out[:, 1] = (2 * l + 1) * at_r ** 2 / (4.0 * np.pi)
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: p shells of the exact omega = 1/10 ground state near the density maximum ---
        {
            "setup": "import numpy as np\ndef parts(table):\n    t = np.asarray(table, dtype=float).reshape(-1, 2)\n    return np.concatenate([t[:, 0] * 1e4, np.log(t[:, 1])])\n",
            "call": "parts(shell_orbital_densities(0.1, 0.05, 1, 6, 3.96))",
            "gold_call": "parts(_oracle_shell_orbital_densities(0.1, 0.05, 1, 6, 3.96))",
            "tol": 1e-6,
        },
        # --- Normal: s orbitals at the centre, the only channel that survives there ---
        {
            "setup": "import numpy as np\ndef parts(table):\n    t = np.asarray(table, dtype=float).reshape(-1, 2)\n    return np.concatenate([t[:, 0] * 1e4, np.log(t[:, 1])])\n",
            "call": "parts(shell_orbital_densities(0.1, 0.05, 0, 10, 0.0))",
            "gold_call": "parts(_oracle_shell_orbital_densities(0.1, 0.05, 0, 10, 0.0))",
            "tol": 1e-6,
        },
        # --- Boundary: a single high-l shell at omega = 1/2 ---
        {
            "setup": "import numpy as np\ndef parts(table):\n    t = np.asarray(table, dtype=float).reshape(-1, 2)\n    return np.concatenate([t[:, 0] * 1e4, np.log(t[:, 1])])\n",
            "call": "parts(shell_orbital_densities(0.5, 0.0, 6, 1, 2.0))",
            "gold_call": "parts(_oracle_shell_orbital_densities(0.5, 0.0, 6, 1, 2.0))",
            "tol": 1e-6,
        },
        # --- Edge: d shells of a tight trap in the density tail ---
        {
            "setup": "import numpy as np\ndef parts(table):\n    t = np.asarray(table, dtype=float).reshape(-1, 2)\n    return np.concatenate([t[:, 0] * 1e4, np.log(t[:, 1])])\n",
            "call": "parts(shell_orbital_densities(3.0, 0.1, 2, 8, 1.4))",
            "gold_call": "parts(_oracle_shell_orbital_densities(3.0, 0.1, 2, 8, 1.4))",
            "tol": 1e-6,
        },
        # --- Error: a negative radius must raise ValueError ---
        {
            "setup": "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(0.1, 0.05, 1, 6, -0.5)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    return 0\n",
            "call": "_probe(shell_orbital_densities)",
            "gold_call": "_probe(_oracle_shell_orbital_densities)",
        },
    ]
