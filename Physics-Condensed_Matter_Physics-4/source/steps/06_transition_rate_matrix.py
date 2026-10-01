"""
Build the phonon-assisted transition-rate matrix between moire exciton states for given state dephasings, replacing strict energy conservation by a broadened kernel.

A strict delta function on a discrete momentum grid either misses or overcounts transitions, and the
Lorentzian of the collisional-broadening expansion spreads weight far from resonance, which inflates
relaxation. The kernel used here is a normalised Gaussian of the energy mismatch whose width is the sum of
the dephasings of the initial and final states: K(x; w) = exp(-(x/w)^2) / (w sqrt(pi)), w = Gamma_i + Gamma_f.
For emission the mismatch is E_f - E_i + hbar Omega, for absorption E_f - E_i - hbar Omega, with the
acoustic energy of the specific grid pair, umklapp vector and layer, or the optical energy of the layer.
The rate from state i to state f is the sum over all channels of weight times kernel.

Returns
-------
rates : np.ndarray -- (S, S) float array in 1/ps, S = N_k * n_bands, column i holding the rates out of state i.
"""

import numpy as np
from scipy.linalg import expm

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def transition_rate_matrix(energies: "np.ndarray", weights: tuple, gamma: "np.ndarray") -> "np.ndarray":
    """Return R with R[f, i] the phonon-assisted rate from state i into state f.

    Parameters
    ----------
    energies : np.ndarray
        (N_k, n_bands) mini-band energies, meV; state s = k * n_bands + n.
    weights : tuple
        (w_ac, omega, w_op, e_op) in the layout returned by scattering_weights.
    gamma : np.ndarray
        Dephasing of every state, meV, shape (N_k * n_bands,) or (N_k, n_bands), positive.

    Returns
    -------
    rates : np.ndarray
        (S, S) float array in 1/ps, S = N_k * n_bands, column i holding the rates out of state i.
    """
    return rates

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _broadening(x, width):
    return np.exp(-(x / width) ** 2) / (width * np.sqrt(np.pi))


def _oracle_transition_rate_matrix(energies: "np.ndarray", weights: tuple, gamma: "np.ndarray") -> "np.ndarray":
    w_ac, om, w_op, e_op = weights
    nk_tot, nb = energies.shape
    e = energies.reshape(-1)
    gam = np.asarray(gamma, dtype=float).reshape(-1)
    om_s = np.repeat(np.repeat(om, nb, axis=0), nb, axis=1)
    de = e[None, :] - e[:, None]
    width = gam[:, None] + gam[None, :]
    rate = np.sum(w_ac[..., 0] * _broadening(de[:, :, None, None] + om_s, width[:, :, None, None]), axis=(2, 3))
    rate += np.sum(w_ac[..., 1] * _broadening(de[:, :, None, None] - om_s, width[:, :, None, None]), axis=(2, 3))
    rate += np.sum(w_op[..., 0] * _broadening(de[:, :, None] + e_op, width[:, :, None]), axis=2)
    rate += np.sum(w_op[..., 1] * _broadening(de[:, :, None] - e_op, width[:, :, None]), axis=2)
    return rate.T.copy()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = """import numpy as np
def make(nk, nb, seed, spread):
    rng = np.random.default_rng(seed)
    s = nk * nb
    e = np.sort(rng.uniform(0.0, spread, size=(nk, nb)), axis=1)
    w_ac = rng.uniform(0.0, 2.0, size=(s, s, 19, 2, 2))
    om = rng.uniform(0.0, 6.0, size=(nk, nk, 19, 2))
    w_op = rng.uniform(0.0, 5.0, size=(s, s, 2, 2))
    return e, (w_ac, om, w_op, np.array([36.6, 30.8]))
"""
    return [
        {"setup": base + "e, w = make(3, 2, 1, 80.0)\ng = np.random.default_rng(9).uniform(0.5, 3.0, size=6)",
         "call": "transition_rate_matrix(e, w, g)",
         "gold_call": "_oracle_transition_rate_matrix(e, w, g)"},
        # energy spacing matching the optical phonons makes the optical kernels resonant
        {"setup": base + "e, w = make(2, 2, 4, 1.0)\ne = e + np.array([[0.0, 36.6], [30.8, 67.4]])\ng = np.full((2, 2), 0.8)",
         "call": "transition_rate_matrix(e, w, g)",
         "gold_call": "_oracle_transition_rate_matrix(e, w, g)"},
        # very narrow dephasing: rates concentrate on near-resonant pairs
        {"setup": base + "e, w = make(2, 3, 7, 20.0)\ng = np.full(6, 0.05)",
         "call": "transition_rate_matrix(e, w, g)",
         "gold_call": "_oracle_transition_rate_matrix(e, w, g)"},
    ]
