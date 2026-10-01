"""
Return the six moments mu_0 ... mu_5 of the source's approximate moment closure for the half-filled Hubbard model on the Bethe lattice, given the interaction U, the half-bandwidth D and the nearest-neighbour bond amplitude. This is the source's self-consistent moment map, in which the second and fourth moments depend on the bond amplitude through the scaled Bethe hopping t = D/2 and the coordination-scaled t^2 z_NN; it is not the exact equation-of-motion moment sequence of the Hubbard Hamiltonian. The odd moments vanish at particle-hole symmetry.

The Gauss-Christoffel rule at pole order N is fixed by the first 2N moments. In the source's closure the second and fourth moments carry the interaction and the hopping-weighted bond amplitude, and the form chosen for the fourth moment decides whether a third, central pole can exist at all.

Returns
-------
numpy.ndarray, Array of shape (6,) holding mu_0 ... mu_5 (float64).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def half_filled_moments(U: float, half_bandwidth: float, bond_amplitude: float) -> "np.ndarray":
    """Return the six moments mu_0 ... mu_5 of the source's approximate moment closure for the half-filled Hubbard model on the Bethe lattice, given the interaction U, the half-bandwidth D and the nearest-neighbour bond amplitude. This is the source's self-consistent moment map, in which the second and fourth moments depend on the bond amplitude through the scaled Bethe hopping t = D/2 and the coordination-scaled t^2 z_NN; it is not the exact equation-of-motion moment sequence of the Hubbard Hamiltonian. The odd moments vanish at particle-hole symmetry.

    Parameters
    ----------
    U : float
        Nonnegative on-site interaction.
    half_bandwidth : float
        Positive half-bandwidth D of the semielliptic band.
    bond_amplitude : float
        Finite nearest-neighbour bond amplitude <c+ c> of one spin.

    Returns
    -------
    moments : numpy.ndarray
        Array of shape (6,) holding mu_0 ... mu_5 (float64).

    Raises
    ------
    ValueError
        If U is negative or not finite, half_bandwidth is not positive and finite, or bond_amplitude is not finite.
    """
    return moments

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_half_filled_moments(U: float, half_bandwidth: float, bond_amplitude: float) -> "np.ndarray":
    """Eq (30a-e): the source's approximate moment closure mu0..mu5 for the half-filled Bethe
    Hubbard model (its self-consistent pseudo-moment map, not the exact EOM moments, whose
    second moment would be U^2/4 + D^2/4 with no bond dependence).

    X = t^2 z_NN <c+c> with the scaled Bethe hopping t = D/2, t^2 z_NN = D^2/4.
    mu1 = mu3 = mu5 = 0 by particle-hole symmetry.  mu2 = U^2/4 + X.  mu4 keeps only the
    leading connected correction, mu4 = mu2^2 + <eps^2>_bath X with <eps^2>_bath = D^2/4.
    """
    if not (np.isfinite(U) and U >= 0.0):
        raise ValueError("U must be nonnegative and finite")
    if not (np.isfinite(half_bandwidth) and half_bandwidth > 0.0):
        raise ValueError("half_bandwidth must be positive and finite")
    if not np.isfinite(bond_amplitude):
        raise ValueError("bond_amplitude must be finite")
    D = float(half_bandwidth)
    t2z = D * D / 4.0                 # t^2 z_NN for the Bethe lattice with half-bandwidth D
    eps2 = D * D / 4.0                # second moment of the semielliptic band
    X = t2z * float(bond_amplitude)
    mu2 = U * U / 4.0 + X
    mu4 = mu2 * mu2 + eps2 * X        # (30e): leading connected correction only
    return np.array([1.0, 0.0, mu2, 0.0, mu4, 0.0], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "U, half_bandwidth, bond_amplitude = 1.0, 1.0, 0.13509491152311703\n",
            "call": "half_filled_moments(U, half_bandwidth, bond_amplitude)",
            "gold_call": "_oracle_half_filled_moments(U, half_bandwidth, bond_amplitude)",
        },
        {
            "setup": "U, half_bandwidth, bond_amplitude = 0.0, 1.0, 0.0\n",
            "call": "half_filled_moments(U, half_bandwidth, bond_amplitude)",
            "gold_call": "_oracle_half_filled_moments(U, half_bandwidth, bond_amplitude)",
        },
        {
            "setup": "U, half_bandwidth, bond_amplitude = 2.5, 0.8, 0.02\n",
            "call": "half_filled_moments(U, half_bandwidth, bond_amplitude)",
            "gold_call": "_oracle_half_filled_moments(U, half_bandwidth, bond_amplitude)",
        },
        {
            "setup": "U, half_bandwidth, bond_amplitude = -1.0, 1.0, 0.1\ndef run_model():\n    try:\n        half_filled_moments(U, half_bandwidth, bond_amplitude)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_half_filled_moments(U, half_bandwidth, bond_amplitude)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
