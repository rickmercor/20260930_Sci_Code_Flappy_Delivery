"""
Find the bath reorganization energy at which the largest coherence impact on the ensemble acceptor population inside a delay window falls to a prescribed level.

The strength of the coupling between the electronic excitation and the protein vibrations, measured by the
reorganization energy, controls how quickly the environment destroys the phase relation between donor and acceptor
and how fast the exciton populations relax. A weakly coupled pair keeps a large coherence impact on the acceptor
signal for longer, and a strongly coupled pair loses it within the first vibrational periods. For an ensemble with a
given static disorder and a given delay window, the maximum coherence impact therefore falls as the reorganization
energy grows, and the reorganization energy at which it crosses a chosen threshold separates a regime in which site
coherence can still change the measured acceptor population by at least that amount from a regime in which it
provably cannot.

The crossing is located by a bracketing root search, and every evaluation of the maximum requires the full disorder
average of the hierarchical dynamics.

Returns
-------
float, the reorganization energy E_R* in cm^-1 at which the window maximum Q equals the target
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def critical_reorganization_energy(dimer: "np.ndarray", site_sigma_cm: float, cutoff_cm: float,
                                   temperature_K: float, n_matsubara: int, depth: int, n_nodes: int,
                                   window_ps: "np.ndarray", target: float, reorg_bracket: "np.ndarray") -> float:
    '''Reorganization energy at which the maximum coherence impact over the delay window equals a target.

    Parameters
    ----------
    dimer : np.ndarray
        Shape (4,), [J, mean_gap, gamma_rec, kappa], as in ensemble_readout_operator.
    site_sigma_cm : float
        Standard deviation of each site energy (cm^-1), as in ensemble_readout_operator.
    cutoff_cm : float
        Drude-Lorentz cutoff gamma_c (cm^-1) of both baths.
    temperature_K : float
        Bath temperature (K).
    n_matsubara : int
        Matsubara terms per bath, as in heom_readout_operator.
    depth : int
        Hierarchy truncation, as in heom_readout_operator.
    n_nodes : int
        Gauss-Hermite nodes of the disorder average, as in ensemble_readout_operator.
    window_ps : np.ndarray
        Shape (2,), the delay window [t_a, t_b] in ps, as in post_overlap_supremum.
    target : float
        Threshold value of the maximum coherence impact Q, 0 < target < 1.
    reorg_bracket : np.ndarray
        Shape (2,), [E_lo, E_hi] with 0 <= E_lo < E_hi, reorganization energies in cm^-1 at which Q - target has
        opposite signs.

    Returns
    -------
    reorg_star : float
        The reorganization energy E_R* (cm^-1) inside the bracket at which Q(E_R) from post_overlap_supremum, with the
        bath [E_R, cutoff_cm, temperature_K], equals target, located to 1e-6 cm^-1.

    Raises
    ------
    ValueError
        If the bracket is not ordered and non-negative, target is outside (0, 1), or Q - target has the same sign at
        both ends of the bracket.
    '''
    return reorg_star

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def _oracle_critical_reorganization_energy(dimer: "np.ndarray", site_sigma_cm: float, cutoff_cm: float,
                                           temperature_K: float, n_matsubara: int, depth: int, n_nodes: int,
                                           window_ps: "np.ndarray", target: float,
                                           reorg_bracket: "np.ndarray") -> float:
    """Reference implementation."""
    import numpy as np
    from scipy.optimize import brentq
    e_lo, e_hi = [float(v) for v in np.asarray(reorg_bracket, dtype=float).reshape(2)]
    if not 0.0 <= e_lo < e_hi:
        raise ValueError("the bracket must satisfy 0 <= E_lo < E_hi")
    if not 0.0 < target < 1.0:
        raise ValueError("target must lie strictly between 0 and 1")

    def _excess(reorg):
        bath = np.array([reorg, cutoff_cm, temperature_K])
        return _oracle_post_overlap_supremum(dimer, site_sigma_cm, bath, n_matsubara, depth, n_nodes,
                                             window_ps)[0] - target

    if _excess(e_lo) * _excess(e_hi) > 0.0:
        raise ValueError("Q - target does not change sign inside the bracket")
    return float(brentq(_excess, e_lo, e_hi, xtol=1e-8, rtol=1e-14))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: disordered reference ensemble on a shallow hierarchy and a short window ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([85.0, 140.0, 0.1, 1.0])\n"
                     "window = np.array([0.15, 0.3])\n"
                     "bracket = np.array([10.0, 60.0])\n",
            "call": "critical_reorganization_energy(dimer.copy(), 70.0, 60.0, 295.0, 1, 2, 5, window.copy(), 0.1, bracket.copy())",
            "gold_call": "_oracle_critical_reorganization_energy(dimer, 70.0, 60.0, 295.0, 1, 2, 5, window, 0.1, bracket)",
            "tol": 1e-6,
        },
        # --- Boundary: homogeneous sample (no disorder, one node) with the Drude term only ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([85.0, 140.0, 0.1, 1.0])\n"
                     "window = np.array([0.15, 0.3])\n"
                     "bracket = np.array([5.0, 80.0])\n",
            "call": "critical_reorganization_energy(dimer.copy(), 0.0, 60.0, 295.0, 0, 3, 1, window.copy(), 0.12, bracket.copy())",
            "gold_call": "_oracle_critical_reorganization_energy(dimer, 0.0, 60.0, 295.0, 0, 3, 1, window, 0.12, bracket)",
            "tol": 1e-6,
        },
        # --- Edge: slow bath at low temperature with a strongly detuned pair and a high threshold ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([110.0, 260.0, 0.0, 0.8])\n"
                     "window = np.array([0.1, 0.25])\n"
                     "bracket = np.array([2.0, 60.0])\n",
            "call": "critical_reorganization_energy(dimer.copy(), 30.0, 35.0, 180.0, 1, 2, 4, window.copy(), 0.26, bracket.copy())",
            "gold_call": "_oracle_critical_reorganization_energy(dimer, 30.0, 35.0, 180.0, 1, 2, 4, window, 0.26, bracket)",
            "tol": 1e-6,
        },
        # --- Invalid: a bracket without a sign change must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([85.0, 140.0, 0.1, 1.0])\n"
                     "def run(fn):\n"
                     "    try:\n"
                     "        fn(dimer, 70.0, 60.0, 295.0, 0, 1, 3, np.array([0.15, 0.3]), 0.9, np.array([10.0, 20.0]))\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n",
            "call": "run(critical_reorganization_energy)",
            "gold_call": "run(_oracle_critical_reorganization_energy)",
        },
    ]
