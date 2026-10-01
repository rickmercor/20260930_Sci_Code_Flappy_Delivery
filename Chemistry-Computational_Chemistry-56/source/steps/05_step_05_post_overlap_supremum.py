"""
Find the largest coherence impact on the ensemble acceptor population inside a window of pump-probe delays, and the delay at which it occurs.

In a pump-probe measurement the signal at delays shorter than the pulse cross-correlation is distorted by coherent
artifacts, so conclusions about the dynamics are drawn from a window of delays that starts after the pulses stop
overlapping. For an inhomogeneous ensemble of donor-acceptor dimers, the largest possible change of the averaged
acceptor population that initial site coherence can cause, compared with the site-dephased preparation, is a
function of delay that rises within the first vibrational and electronic periods and then decays as the bath and
the static disorder wash out the phase information. Its maximum over the window is a single state-independent bound
on how strongly coherence can affect the measured acceptor signal in that window.

The maximum is generally reached between sampled delays, so it has to be located on the continuous time axis.

Returns
-------
numpy.ndarray of shape (2,): [Q, t_star], the maximum coherence impact over the delay window and its delay in ps
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def post_overlap_supremum(dimer: "np.ndarray", site_sigma_cm: float, bath: "np.ndarray", n_matsubara: int,
                          depth: int, n_nodes: int, window_ps: "np.ndarray") -> "np.ndarray":
    '''Maximum over a delay window of the coherence impact on the ensemble-averaged acceptor population.

    Parameters
    ----------
    dimer : np.ndarray
        Shape (4,), [J, mean_gap, gamma_rec, kappa], as in ensemble_readout_operator.
    site_sigma_cm : float
        Standard deviation of each site energy (cm^-1), as in ensemble_readout_operator.
    bath : np.ndarray
        Shape (3,), [E_R, gamma_c, T], as in heom_readout_operator.
    n_matsubara : int
        Matsubara terms per bath, as in heom_readout_operator.
    depth : int
        Hierarchy truncation, as in heom_readout_operator.
    n_nodes : int
        Gauss-Hermite nodes of the disorder average, as in ensemble_readout_operator.
    window_ps : np.ndarray
        Shape (2,), [t_a, t_b] with 0 <= t_a < t_b, the closed delay window in ps.

    Returns
    -------
    supremum : np.ndarray
        Shape (2,), [Q, t_star]. C(t) is the coherence impact C of coherence_diagnostics for the ensemble acceptor
        readout (readout_site = 1) at delay t, Q is the maximum of C(t) over t_a <= t <= t_b, and t_star (ps) is the
        delay at which it is reached, located to 1e-7 ps on the continuous time axis (the earliest such delay if the
        maximum is reached more than once).

    Raises
    ------
    ValueError
        If the window does not satisfy 0 <= t_a < t_b, and in the cases listed for ensemble_readout_operator.
    '''
    return supremum

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import minimize_scalar


def _hermite_abs_maximum(t0, t1, row0, row1):
    """Maximum of |m(t)| on [t0, t1] for the cubic Hermite interpolant of Re m and Im m from values and derivatives."""
    import numpy as np
    from scipy.optimize import minimize_scalar
    h = t1 - t0

    def _value(t):
        s = (t - t0) / h
        h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
        re = h00 * row0[2] + h10 * h * row0[6] + h01 * row1[2] + h11 * h * row1[6]
        im = h00 * row0[3] + h10 * h * row0[7] + h01 * row1[3] + h11 * h * row1[7]
        return np.hypot(re, im)

    res = minimize_scalar(lambda t: -_value(t), bounds=(t0, t1), method="bounded", options={"xatol": 1e-13})
    candidates = [(_value(t0), t0), (-res.fun, res.x), (_value(t1), t1)]
    best = max(c[0] for c in candidates)
    return min((c for c in candidates if c[0] >= best - 1e-15), key=lambda c: c[1])


def _oracle_post_overlap_supremum(dimer: "np.ndarray", site_sigma_cm: float, bath: "np.ndarray", n_matsubara: int,
                                  depth: int, n_nodes: int, window_ps: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    t_a, t_b = [float(v) for v in np.asarray(window_ps, dtype=float).reshape(2)]
    if not 0.0 <= t_a < t_b:
        raise ValueError("the window must satisfy 0 <= t_a < t_b")
    # coarse scan on a grid of at most 2 fs, then a 0.1 fs grid around the best coarse point
    n_coarse = int(np.ceil((t_b - t_a) / 0.002 - 1e-9))
    coarse = np.linspace(t_a, t_b, n_coarse + 1)
    rows = _oracle_ensemble_readout_operator(dimer, site_sigma_cm, bath, n_matsubara, depth, n_nodes, 1, coarse)
    impact = _oracle_coherence_diagnostics(rows)[:, 0]
    j = int(np.argmax(impact))
    lo, hi = coarse[max(j - 1, 0)], coarse[min(j + 1, n_coarse)]
    fine = np.linspace(lo, hi, int(np.ceil((hi - lo) / 1e-4 - 1e-9)) + 1)
    frows = _oracle_ensemble_readout_operator(dimer, site_sigma_cm, bath, n_matsubara, depth, n_nodes, 1, fine)
    fimpact = _oracle_coherence_diagnostics(frows)[:, 0]
    k = int(np.argmax(fimpact))
    best = (fimpact[k], fine[k])
    for a, b in ((k - 1, k), (k, k + 1)):
        if 0 <= a and b < len(fine):
            cand = _hermite_abs_maximum(fine[a], fine[b], frows[a], frows[b])
            if cand[0] > best[0] + 1e-15 or (abs(cand[0] - best[0]) <= 1e-15 and cand[1] < best[1]):
                best = cand
    return np.array([best[0], best[1]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: disordered reference ensemble, shallow hierarchy, maximum inside the window ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([85.0, 140.0, 0.1, 1.0])\n"
                     "bath = np.array([27.5, 60.0, 295.0])\n"
                     "window = np.array([0.15, 0.4])\n",
            "call": "post_overlap_supremum(dimer.copy(), 70.0, bath.copy(), 1, 2, 6, window.copy())",
            "gold_call": "_oracle_post_overlap_supremum(dimer, 70.0, bath, 1, 2, 6, window)",
            "tol": 1e-6,
        },
        # --- Boundary: window starting after the main lobe, so the maximum sits on the window start ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([85.0, 140.0, 0.1, 1.0])\n"
                     "bath = np.array([27.5, 60.0, 295.0])\n"
                     "window = np.array([0.21, 0.26])\n",
            "call": "post_overlap_supremum(dimer.copy(), 0.0, bath.copy(), 1, 2, 1, window.copy())",
            "gold_call": "_oracle_post_overlap_supremum(dimer, 0.0, bath, 1, 2, 1, window)",
            "tol": 1e-6,
        },
        # --- Edge: resonant pair with broad disorder and weak damping, several lobes compete inside the window ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([100.0, 0.0, 0.05, 0.3])\n"
                     "bath = np.array([12.0, 80.0, 300.0])\n"
                     "window = np.array([0.1, 0.45])\n",
            "call": "post_overlap_supremum(dimer.copy(), 60.0, bath.copy(), 0, 3, 8, window.copy())",
            "gold_call": "_oracle_post_overlap_supremum(dimer, 60.0, bath, 0, 3, 8, window)",
            "tol": 1e-6,
        },
        # --- Edge: large detuning at low temperature with two Matsubara terms and a window from zero delay ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([60.0, 300.0, 0.0, 1.5])\n"
                     "bath = np.array([35.0, 106.0, 150.0])\n"
                     "window = np.array([0.0, 0.3])\n",
            "call": "post_overlap_supremum(dimer.copy(), 25.0, bath.copy(), 2, 2, 5, window.copy())",
            "gold_call": "_oracle_post_overlap_supremum(dimer, 25.0, bath, 2, 2, 5, window)",
            "tol": 1e-6,
        },
        # --- Edge: acceptor above the donor, hot fast bath, disorder wider than the gap, maximum on the first lobe ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([70.0, -60.0, 0.1, 0.4])\n"
                     "bath = np.array([45.0, 150.0, 330.0])\n"
                     "window = np.array([0.01, 0.35])\n",
            "call": "post_overlap_supremum(dimer.copy(), 90.0, bath.copy(), 1, 3, 10, window.copy())",
            "gold_call": "_oracle_post_overlap_supremum(dimer, 90.0, bath, 1, 3, 10, window)",
            "tol": 1e-6,
        },
        # --- Invalid: a reversed window must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([85.0, 140.0, 0.1, 1.0])\n"
                     "bath = np.array([27.5, 60.0, 295.0])\n"
                     "def run(fn):\n"
                     "    try:\n"
                     "        fn(dimer, 70.0, bath, 1, 2, 4, np.array([0.5, 0.2]))\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n",
            "call": "run(post_overlap_supremum)",
            "gold_call": "run(_oracle_post_overlap_supremum)",
        },
    ]
