"""
Return the one-electron matrix h and the site-site interaction matrix V of an open Pariser-Parr-Pople chain of K sites, stacked as an array of shape (2, K, K). With sites numbered i = 0..K-1: the nearest-neighbour hopping is h_{i,i+1} = h_{i+1,i} = -t (1 + delta (-1)^i); the interaction is the Ohno form V_ij = U / sqrt(1 + (U |i - j| / kappa)^2) for i != j and V_ii = U; the diagonal of h is h_ii = eps_i - sum_{j != i} V_ij, i.e. the site energy shifted by the interaction with a +1 neutralising background on every site. All other elements of h are zero. Raise ValueError if eps_site is not one-dimensional with at least two entries, or if t <= 0, U <= 0, kappa <= 0 or |delta| >= 1.

The PPP model is the standard pi-electron model of conjugated chains: a tight-binding chain with alternating bonds, an on-site Hubbard repulsion and a long-range Ohno-interpolated Coulomb tail, with a neutralising background so that the neutral chain has one electron per site.

Returns
-------
numpy.ndarray of float64 with shape (2, K, K): [h, V].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ppp_hamiltonian(t: float, delta: float, U: float, kappa: float, eps_site: "np.ndarray") -> "np.ndarray":
    """Return the one-electron matrix h and the site-site interaction matrix V of an open Pariser-Parr-Pople chain of K sites, stacked as an array of shape (2, K, K). With sites numbered i = 0..K-1: the nearest-neighbour hopping is h_{i,i+1} = h_{i+1,i} = -t (1 + delta (-1)^i); the interaction is the Ohno form V_ij = U / sqrt(1 + (U |i - j| / kappa)^2) for i != j and V_ii = U; the diagonal of h is h_ii = eps_i - sum_{j != i} V_ij, i.e. the site energy shifted by the interaction with a +1 neutralising background on every site. All other elements of h are zero. Raise ValueError if eps_site is not one-dimensional with at least two entries, or if t <= 0, U <= 0, kappa <= 0 or |delta| >= 1.

    Parameters
    ----------
    t : float
        Nearest-neighbour hopping (positive).
    delta : float
        Bond alternation, |delta| < 1.
    U : float
        On-site repulsion (positive).
    kappa : float
        Ohno screening length (positive).
    eps_site : numpy.ndarray
        Site energies, length K >= 2.

    Returns
    -------
    hV : numpy.ndarray
        Array (2, K, K): hV[0] = h, hV[1] = V.

    Raises
    ------
    ValueError
        If eps_site is not 1-D with at least two sites, or t, U, kappa are not positive, or |delta| >= 1.
    """
    return hV

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_ppp_hamiltonian(t: float, delta: float, U: float, kappa: float, eps_site: "np.ndarray") -> "np.ndarray":
    """Open PPP chain of K sites: h_ii = eps_i - sum_{j != i} V_ij (a +1 neutralising background on every site),
    h_{i,i+1} = h_{i+1,i} = -t (1 + delta (-1)^i) (0-based i), Ohno V_ij = U / sqrt(1 + (U |i - j| / kappa)^2),
    V_ii = U.  Returns array (2, K, K) = [h, V]."""
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
    """Return the list of test case specifications."""
    return [
        {
            "setup": "eps_site = np.array([0.4, -0.3, 0.2, -0.1, 0.3, -0.5, 0.1, -0.2]) - 1.5\nt, delta, U, kappa, n_pairs = 1.0, 0.15, 4.0, 1.5, 4\n",
            "call": "ppp_hamiltonian(t, delta, U, kappa, eps_site)",
            "gold_call": "_oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)",
        },
        {
            "setup": "eps_site = np.array([0.2, -0.4, 0.1, 0.3, -0.2, -0.1]) - 1.0\nt, delta, U, kappa, n_pairs = 1.0, 0.1, 3.0, 2.0, 3\n",
            "call": "ppp_hamiltonian(t, delta, U, kappa, eps_site)",
            "gold_call": "_oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)",
        },
        {
            "setup": "eps_site = np.array([0.3, -0.1, 0.2, -0.4]) - 0.8\nt, delta, U, kappa, n_pairs = 1.0, 0.2, 2.5, 1.0, 2\n",
            "call": "ppp_hamiltonian(t, delta, U, kappa, eps_site)",
            "gold_call": "_oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)",
        },
        {
            "setup": "eps_site = np.array([-0.5, -0.9])\nt, delta, U, kappa, n_pairs = 1.0, 0.15, 4.0, 1.5, 1\n",
            "call": "ppp_hamiltonian(t, delta, U, kappa, eps_site)",
            "gold_call": "_oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)",
        },
        {
            "setup": "eps_site = np.array([0.4, -0.3, 0.2, -0.1, 0.3, -0.5, 0.1, -0.2]) - 1.5\nt, delta, U, kappa, n_pairs = 1.0, 0.15, 4.0, 1.5, 4\nU = 0.0\ndef run_model():\n    try:\n        ppp_hamiltonian(t, delta, U, kappa, eps_site)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
