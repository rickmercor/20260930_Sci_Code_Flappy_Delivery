"""
Return the array [h, V] of shape (2, K, K) for the open PPP chain with K = len(eps_site) sites: h_{i,i+1} = h_{i+1,i} = -t (1 + delta (-1)^i) for i = 0..K-2 (all other off-diagonal elements zero), h_ii = eps_i - sum_{j != i} V_ij (every site carries a +1 background charge), and the Ohno interaction V_ij = U / sqrt(1 + (U |i - j| / kappa)^2), so V_ii = U. The two-electron integrals of the model are of density-density form, (pq|rs) = delta_pq delta_rs V_pr in the site basis (chemists' notation). Raise ValueError if eps_site is not a 1-D array with at least two entries, if t, U or kappa is not positive, or if |delta| >= 1.

Pariser-Parr-Pople chains are the standard pi-electron models of conjugated polyenes: nearest-neighbour hopping with bond alternation, an on-site Hubbard repulsion and a screened long-range Coulomb tail, with the neutral-background term making the half-filled chain the neutral molecule.

Returns
-------
float array of shape (2, K, K): [h, V].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ppp_hamiltonian(t: float, delta: float, U: float, kappa: float, eps_site: "np.ndarray") -> "np.ndarray":
    '''One- and two-body matrices of the open PPP chain in the site basis.

    Parameters
    ----------
    t : float
        Hopping scale, t > 0.
    delta : float
        Bond alternation, |delta| < 1.
    U : float
        On-site repulsion, U > 0.
    kappa : float
        Ohno screening parameter, kappa > 0.
    eps_site : np.ndarray
        Site energies eps_i, 1-D array of length K >= 2.

    Returns
    -------
    hV : np.ndarray
        Array of shape (2, K, K): hV[0] the one-body matrix h, hV[1] the site-basis interaction matrix V.

    Raises
    ------
    ValueError
        If eps_site is not a 1-D array with at least two entries, or t, U or kappa is not positive, or |delta| >= 1.
    '''
    return hV

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_ppp_hamiltonian(t: float, delta: float, U: float, kappa: float, eps_site: "np.ndarray") -> "np.ndarray":
    """Open PPP chain: h_ii = eps_i - sum_{j!=i} V_ij (neutral +1 background per site),
    h_{i,i+1} = -t(1 + delta(-1)^i), Ohno V_ij = U / sqrt(1 + (U|i-j|/kappa)^2), V_ii = U.
    Returns array (2, K, K): [h, V]."""
    eps_site = np.asarray(eps_site, dtype=float)
    if eps_site.ndim != 1 or eps_site.size < 2:
        raise ValueError("eps_site must be a 1-D array with at least 2 sites")
    t, delta, U, kappa = float(t), float(delta), float(U), float(kappa)
    if t <= 0.0 or U <= 0.0 or kappa <= 0.0 or not (-1.0 < delta < 1.0):
        raise ValueError("need t > 0, U > 0, kappa > 0, |delta| < 1")
    K = eps_site.size
    idx = np.arange(K)
    V = U / np.sqrt(1.0 + (U * np.abs(idx[:, None] - idx[None, :]) / kappa) ** 2)
    h = np.zeros((K, K))
    for i in range(K - 1):
        h[i, i + 1] = h[i + 1, i] = -t * (1.0 + delta * (-1.0) ** i)
    h[idx, idx] = eps_site - (V.sum(axis=1) - np.diag(V))
    return np.stack([h, V])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of step test cases (normal, boundary and edge inputs, plus one invalid input)."""
    return [
        {
            "setup": "import numpy as np\nt, delta, U, kappa = 1.0, 0.15, 6.0, 4.0\neps_site = -4.0 + np.array([0.0, 0.3, -0.2, 0.1, -0.3, 0.2, 0.15, -0.25])",
            "call": "ppp_hamiltonian(t, delta, U, kappa, eps_site)",
            "gold_call": "_oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nt, delta, U, kappa = 1.0, 0.1, 4.0, 3.0\neps_site = -3.5 + np.array([0.0, 0.2, -0.1, 0.3, -0.2, 0.1])",
            "call": "ppp_hamiltonian(t, delta, U, kappa, eps_site)",
            "gold_call": "_oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nt, delta, U, kappa = 0.8, 0.0, 5.0, 2.5\neps_site = np.array([-3.0, -3.2, -2.75, -3.15])",
            "call": "ppp_hamiltonian(t, delta, U, kappa, eps_site)",
            "gold_call": "_oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)",
            "tol": 1e-12,
        },
        {  # invalid input: the documented ValueError contract
            "setup": "import numpy as np\ndef run_model():\n    try:\n        ppp_hamiltonian(1.0, 1.5, 6.0, 4.0, np.array([-4.0, -3.7, -4.2, -3.9]))\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_ppp_hamiltonian(1.0, 1.5, 6.0, 4.0, np.array([-4.0, -3.7, -4.2, -3.9]))\n        return 0\n    except ValueError:\n        return 1",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
