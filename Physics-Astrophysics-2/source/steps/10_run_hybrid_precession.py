"""
Orchestrator: end-to-end hybrid 2PN precession solution

Chains every earlier step: build the invariants, the 1PN quasi-Keplerian elements, the nutation cubic and its roots, the continuous eccentric anomaly, the hybrid phase argument with its angular anomaly, the nutation angle, the companion angles, and the azimuthal coefficient sets, then accumulate the azimuthal phases of $\mathbf{l}$ and $\mathbf{s}_1$ about the conserved total angular momentum $\mathbf{j}$ from $t = 0$ to the final reduced time. The initial sign of $d\cos\kappa_1/dt$ follows from Eq. (3.12b) of the source analysis: $\mathrm{sign}_0 = \mathrm{sign}[\mathbf{l}\cdot(\mathbf{s}_1\times\mathbf{s}_2)]\times \mathrm{sign}[(\delta_2 - \nu/2)(1 - \lambda)]$.

Returns
-------
np.ndarray of shape (7,): [u(t_final), v_theta(t_final), d, cos kappa1(t_final), cos kappa2(t_final), Delta phi_L, Delta phi_S1]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def run_hybrid_precession(m1: float, m2: float, chi1: float, chi2: float,
                          h: float, l: float, kappa1_0: float,
                          kappa2_0: float, gamma_0: float,
                          triple_sign: float, t_final: float,
                          c: float = 1.0) -> np.ndarray:
    '''Run the full hybrid-precession pipeline and report its diagnostics.

    Parameters
    ----------
    m1, m2 : float
        Component masses with m1 > m2 > 0 (reduced units, m1 + m2 = m).
    chi1, chi2 : float
        Dimensionless spin magnitudes, each in (0, 1].
    h : float
        Reduced orbital energy, must be < 0.
    l : float
        Reduced orbital angular momentum, must be > 0.
    kappa1_0, kappa2_0, gamma_0 : float
        Initial angles between (l, s1), (l, s2) and (s1, s2), in
        radians, each in (0, pi).
    triple_sign : float
        Sign of l . (s1 x s2) at t = 0; must be +1.0 or -1.0.
    t_final : float
        Final reduced time, must be a finite scalar > 0.
    c : float
        Speed of light in reduced units, must be > 0.

    Returns
    -------
    diagnostics : np.ndarray
        Array of shape (7,):
        [0] continuous eccentric anomaly u at t_final, in radians;
        [1] continuous angular anomaly v_theta at t_final, in radians;
        [2] 1PN averaging radius d, in reduced units;
        [3] cos kappa1 at t_final;
        [4] cos kappa2 at t_final;
        [5] accumulated azimuthal phase of the orbital angular momentum,
            Delta phi_L, in radians;
        [6] accumulated azimuthal phase of the primary spin,
            Delta phi_S1, in radians.

    Raises
    ------
    ValueError
        If any configuration parameter is invalid (for example
        m1 <= m2, a spin magnitude outside (0, 1], h >= 0, or
        triple_sign not +1.0 or -1.0) -- such errors propagate from the
        pipeline stages.
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_run_hybrid_precession(m1: float, m2: float, chi1: float,
                                  chi2: float, h: float, l: float,
                                  kappa1_0: float, kappa2_0: float,
                                  gamma_0: float, triple_sign: float,
                                  t_final: float, c: float = 1.0) -> np.ndarray:
    if not (np.isscalar(triple_sign) and float(triple_sign) in (-1.0, 1.0)):
        raise ValueError("triple_sign must be +1.0 or -1.0")
    if not (np.isscalar(t_final) and np.isfinite(float(t_final))
            and float(t_final) > 0.0):
        raise ValueError("t_final must be a finite scalar > 0")
    if not (np.isscalar(h) and np.isfinite(float(h)) and float(h) < 0.0):
        raise ValueError("h must be a finite scalar < 0")
    triple_sign, t_final, h = float(triple_sign), float(t_final), float(h)

    inv = _oracle_binary_invariants(m1, m2, chi1, chi2, l, kappa1_0,
                                    kappa2_0, gamma_0, c)
    nu, mu, delta1, delta2, sig1, sig2, s1, s2, j, lam, big1, big2 = inv
    qk = _oracle_quasi_keplerian_elements(h, l, nu, c)
    a_r, e_r, n_mm, e_t, e_th, d = qk
    roots = _oracle_nutation_cubic_roots(m1, m2, l, s1, s2, lam, big1, big2)
    x_m, x_p, x_3, big_a = roots

    x0 = float(np.cos(float(kappa1_0)))
    sign0 = triple_sign * float(np.sign((delta2 - nu / 2.0) * (1.0 - lam)))

    t_end = np.array([t_final], dtype=float)
    u_end = _oracle_kepler_eccentric_anomaly(t_end, n_mm, e_t)
    v_th_end = _oracle_angular_anomaly(u_end, e_th)

    cosk1 = _oracle_nutation_angle(t_end, x_m, x_p, x_3, big_a, d, n_mm,
                                   e_t, e_th, x0, sign0, c)
    comp = _oracle_companion_angles(cosk1, m1, m2, l, s1, s2, big1, big2)
    cosk2 = comp[0]

    co = _oracle_azimuthal_coefficients(m1, m2, l, s1, s2, j, lam, big1, big2)
    dphi_l = _oracle_azimuthal_phase(t_end, x_m, x_p, x_3, big_a, d, n_mm,
                                     e_t, e_th, x0, sign0, *co[0:5], c)
    dphi_s1 = _oracle_azimuthal_phase(t_end, x_m, x_p, x_3, big_a, d, n_mm,
                                      e_t, e_th, x0, sign0, *co[5:10], c)
    return np.array([u_end[0], v_th_end[0], float(d), float(cosk1[0]),
                     float(cosk2[0]), float(dphi_l[0]),
                     float(dphi_s1[0])], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the task configuration to the final time ---
        {
            "setup": ("import numpy as np\n"
                      "args = (2.0/3.0, 1.0/3.0, 0.9, 0.8, -1.0/256.0, 8.965,\n"
                      "        np.deg2rad(32.0), np.deg2rad(82.0), np.deg2rad(54.0),\n"
                      "        1.0, 5.0e6)\n"),
            "call": "run_hybrid_precession(*args)",
            "gold_call": "_oracle_run_hybrid_precession(*args)",
        },
        # --- Normal: the final-answer component alone ---
        {
            "setup": ("import numpy as np\n"
                      "args = (2.0/3.0, 1.0/3.0, 0.9, 0.8, -1.0/256.0, 8.965,\n"
                      "        np.deg2rad(32.0), np.deg2rad(82.0), np.deg2rad(54.0),\n"
                      "        1.0, 5.0e6)\n"),
            "call": "float(run_hybrid_precession(*args)[6])",
            "gold_call": "float(_oracle_run_hybrid_precession(*args)[6])",
        },
        # --- Normal: negative triple product, shorter horizon ---
        {
            "setup": ("import numpy as np\n"
                      "args = (0.6, 0.4, 0.7, 0.5, -1.0/200.0, 7.5,\n"
                      "        np.deg2rad(50.0), np.deg2rad(100.0), np.deg2rad(70.0),\n"
                      "        -1.0, 1.2e6)\n"),
            "call": "run_hybrid_precession(*args)",
            "gold_call": "_oracle_run_hybrid_precession(*args)",
        },
        # --- Boundary: single orbital period ---
        {
            "setup": ("import numpy as np\n"
                      "args = (2.0/3.0, 1.0/3.0, 0.9, 0.8, -1.0/256.0, 8.965,\n"
                      "        np.deg2rad(32.0), np.deg2rad(82.0), np.deg2rad(54.0),\n"
                      "        1.0, 9231.9)\n"),
            "call": "run_hybrid_precession(*args)",
            "gold_call": "_oracle_run_hybrid_precession(*args)",
        },
        # --- Edge: vanishing secondary spin propagates ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_hybrid_precession(2.0/3.0, 1.0/3.0, 0.9, 0.0, -1.0/256.0, 8.965,
                              np.deg2rad(32.0), np.deg2rad(82.0),
                              np.deg2rad(54.0), 1.0, 5.0e6)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_run_hybrid_precession(2.0/3.0, 1.0/3.0, 0.9, 0.0, -1.0/256.0,
                                      8.965, np.deg2rad(32.0), np.deg2rad(82.0),
                                      np.deg2rad(54.0), 1.0, 5.0e6)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Edge: nonnegative orbital energy propagates ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_hybrid_precession(2.0/3.0, 1.0/3.0, 0.9, 0.8, 0.0, 8.965,
                              np.deg2rad(32.0), np.deg2rad(82.0),
                              np.deg2rad(54.0), 1.0, 5.0e6)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_run_hybrid_precession(2.0/3.0, 1.0/3.0, 0.9, 0.8, 0.0, 8.965,
                                      np.deg2rad(32.0), np.deg2rad(82.0),
                                      np.deg2rad(54.0), 1.0, 5.0e6)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
