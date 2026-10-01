"""
Run the whole audit at the given refinement by calling every earlier step function and using its output, and return the reported diagnostics. Use classical fourth-order Runge-Kutta. Set nominal counts n_exact=26000*refine and n_slow=2600*refine. Evolve the exact system from 0 to t*=5 with round(n_exact*5/65) uniform steps. Then compare at t_j=5+0.15*j for j=1,...,400, advancing both systems between successive comparison times with uniform steps: max(1, round(n_exact*0.15/65)) exact steps and max(1, round(n_slow*0.15/60)) effective steps per interval. Here round uses nearest-integer rounding with ties to even. Use four background-matching sweeps, starting the slow background guess at the exact transition value and computing its slow Hubble rate before each sweep. psi_init is the axion wavefunction at t=0 and defaults to the prescribed instance; the benchmark is axion_reconstruction_audit(1). For the returned density contrast use delta_a_s=delta_rho_a_s/rho_a_s, where rho_a_s=m*abs(psi_s)**2 + 3*(m*abs(psi_s)**2+rho_other_s)*abs(psi_s)**2/(16*m*m_pl**2). Here delta_rho_a_s is the density perturbation returned by slow_perturbation_observables, rho_other_s is the other-component density evaluated at a_s, and m and m_pl are the axion and reduced Planck masses. Evaluate these quantities at the end time.

The audit runs the source's own algorithm end to end at both levels: exact evolution to the transition, matching of the background and then of the perturbation, effective evolution to the end time, and reconstruction of the oscillation at every comparison time. The perturbation matching consumes the background slow mode that the background matching produces, so the two cannot be done in the other order, and the slow curvature perturbation has to be obtained by inverting its own reconstruction rather than carried across unchanged.

Returns
-------
ndarray of shape (9,), float64: the largest and the mean absolute difference between the exactly evolved and the rebuilt wavefunction perturbation over the comparison times, the modulus of the matched slow-mode perturbation at the transition time, the slow-mode trace metric rate there, the modulus of the matched slow-mode background wavefunction there, the slow-mode scale factor at the end time, the axion density contrast of the slow mode at the end time, the slow-mode curvature perturbation there, and the slow-mode expansion rate there.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def axion_reconstruction_audit(
    refine: int, psi_init: complex = 0.312 + 0.0j
) -> "np.ndarray":
    """Run the whole audit at the given refinement by calling every earlier
    step function and using its output, and return the reported
    diagnostics. Use classical fourth-order Runge-Kutta. Set nominal counts
    n_exact=26000*refine and n_slow=2600*refine. Evolve the exact system
    from 0 to t*=5 with round(n_exact*5/65) uniform steps. Then compare at
    t_j=5+0.15*j for j=1,...,400, advancing both systems between successive
    comparison times with uniform steps: max(1, round(n_exact*0.15/65))
    exact steps and max(1, round(n_slow*0.15/60)) effective steps per
    interval. Here round uses nearest-integer rounding with ties to even.
    Use four background-matching sweeps, starting the slow background guess
    at the exact transition value and computing its slow Hubble rate before
    each sweep. psi_init is the axion wavefunction at t=0 and defaults to
    the prescribed instance; the benchmark is
    axion_reconstruction_audit(1). For the returned density contrast use
    delta_a_s=delta_rho_a_s/rho_a_s, where rho_a_s=m*abs(psi_s)**2 +
    3*(m*abs(psi_s)**2+rho_other_s)*abs(psi_s)**2/(16*m*m_pl**2). Here
    delta_rho_a_s is the density perturbation returned by
    slow_perturbation_observables, rho_other_s is the other-component
    density evaluated at a_s, and m and m_pl are the axion and reduced
    Planck masses. Evaluate these quantities at the end time.

    Returns
    -------
    ndarray of shape (9,), float64: the largest and the mean absolute
    difference between the exactly evolved and the rebuilt wavefunction
    perturbation over the comparison times, the modulus of the matched
    slow-mode perturbation at the transition time, the slow-mode trace
    metric rate there, the modulus of the matched slow-mode background
    wavefunction there, the slow-mode scale factor at the end time, the
    axion density contrast of the slow mode at the end time, the slow-mode
    curvature perturbation there, and the slow-mode expansion rate there.

    Raises
    ------
    ValueError
        If refine is not a positive integer.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_axion_reconstruction_audit(
    refine: int, psi_init: complex = 0.312 + 0.0j
) -> "np.ndarray":
    refine = int(refine)
    if refine < 1:
        raise ValueError("refine must be a positive integer")
    # Evaluation configuration: GIVEN in the prompt.
    m = 1.0
    m_pl = 1.0
    rho_m0 = 0.126
    rho_r0 = 0.0432
    k = 0.5
    psi_i = complex(psi_init)
    dpsi_i = 1.0e-3 + 0.0j
    eta_i = 1.0e-3
    a_i = 1.0
    t_i, t_star, t_f = 0.0, 5.0, 65.0
    n_iter = 4
    n_exact = 26000 * refine
    n_slow = 2600 * refine
    n_sample = 400

    def _other(a):
        rho = rho_m0 * a**-3 + rho_r0 * a**-4
        p = rho_r0 * a**-4 / 3.0
        return rho, p

    def _exact_hdot(psi, dpsi, a, eta, h_rate):
        # 00 Einstein equation with the axion density perturbation of Eq (23).
        drho = float(np.real(m * (np.conj(psi) * dpsi + psi * np.conj(dpsi))))
        return (
            (2.0 * m_pl**2 * k**2 / (a**2 * h_rate)) * eta + drho / h_rate
        ) / m_pl**2

    def _exact_etadot(psi, dpsi, t):
        # 0i Einstein equation with the axion velocity perturbation of Eq (25).
        e = np.exp(-2j * m * t)
        du = 0.5j * (
            psi * np.conj(dpsi)
            - np.conj(psi) * dpsi
            + psi * dpsi * e
            - np.conj(psi) * np.conj(dpsi) / e
        )
        return float(np.real(-(m / 2.0) * du)) / m_pl**2

    def _exact_deriv(state, t):
        psi, dpsi = state[0], state[1]
        a, eta = state[2].real, state[4].real
        rho, _ = _other(a)
        h_rate = float(_oracle_exact_hubble(psi, rho, m, m_pl))
        hd = _exact_hdot(psi, dpsi, a, eta, h_rate)
        ed = _exact_etadot(psi, dpsi, t)
        return np.array(
            [
                _oracle_exact_rate(psi, h_rate, m, t),
                _oracle_exact_perturbation_rate(
                    dpsi, psi, h_rate, hd, a, m, k, t
                ),
                complex(a * h_rate, 0.0),
                complex(hd, 0.0),
                complex(ed, 0.0),
            ]
        )

    def _slow_deriv(state, t):
        psi_s, dpsi_s = state[0], state[1]
        a_s, eta_s = state[2].real, state[4].real
        rho, p = _other(a_s)
        h_s = float(_oracle_slow_hubble(psi_s, rho, m, m_pl))
        rates = _oracle_metric_slow_rates(
            psi_s, dpsi_s, eta_s, h_s, a_s, m, m_pl, k
        )
        hd = float(rates[0])
        ed = float(rates[1])
        return np.array(
            [
                _oracle_slow_rate(psi_s, h_s, rho, p, m, m_pl),
                _oracle_slow_perturbation_rate(
                    dpsi_s, psi_s, h_s, hd, a_s, m, m_pl, k
                ),
                complex(a_s * h_s, 0.0),
                complex(hd, 0.0),
                complex(ed, 0.0),
            ]
        )

    def _rk4(deriv, state, t0, t1, n):
        h = (t1 - t0) / n
        t = t0
        for _ in range(n):
            k1 = deriv(state, t)
            k2 = deriv(state + 0.5 * h * k1, t + 0.5 * h)
            k3 = deriv(state + 0.5 * h * k2, t + 0.5 * h)
            k4 = deriv(state + h * k3, t + h)
            state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
        return state

    # Step 2: exact evolution of background and perturbations up to the
    # transition.
    st = np.array(
        [psi_i, dpsi_i, complex(a_i, 0.0), 0.0 + 0.0j, complex(eta_i, 0.0)]
    )
    n_to_star = int(round(n_exact * (t_star - t_i) / (t_f - t_i)))
    st_star = _rk4(_exact_deriv, st, t_i, t_star, n_to_star)

    a_st = st_star[2].real
    rho_st, p_st = _other(a_st)

    # Step 3a: background matching, iterated because the coefficients depend on
    # the
    # unknown through the slow-mode expansion rate.
    psi_s_star = complex(st_star[0])
    for _ in range(n_iter):
        h_guess = float(_oracle_slow_hubble(psi_s_star, rho_st, m, m_pl))
        co = _oracle_matching_coefficients(
            psi_s_star, h_guess, rho_st, p_st, m, m_pl, t_star
        )
        ms = _oracle_solve_matching_system(st_star[0], co)
        psi_s_star = complex(ms[0], ms[1])

    # Step 3b: perturbation matching.  Its coefficients use the background slow
    # mode
    # just found, and it solves for the slow metric rate alongside the
    # perturbation.
    h_s_star = float(_oracle_slow_hubble(psi_s_star, rho_st, m, m_pl))
    h_rate_ex = float(_oracle_exact_hubble(st_star[0], rho_st, m, m_pl))
    hdot_ex = _exact_hdot(
        st_star[0], st_star[1], a_st, st_star[4].real, h_rate_ex
    )
    pm = _oracle_perturbation_matching(
        psi_s_star,
        h_s_star,
        a_st,
        m,
        m_pl,
        k,
        t_star,
        complex(st_star[1]),
        hdot_ex,
    )
    dpsi_s_star = complex(pm[0], pm[1])
    hdot_s_star = float(pm[2])

    # Eq (19) inverted for the slow metric potential at the transition.
    e2 = np.exp(2j * m * t_star)
    eta_s_star = st_star[4].real - float(
        np.real(
            (
                np.conj(psi_s_star) * np.conj(dpsi_s_star) * e2
                + psi_s_star * dpsi_s_star / e2
            )
            / (8.0 * m * m_pl**2)
        )
    )

    # Step 4: EFT evolution to the end time, sampled, with Step 5
    # reconstruction.
    sst = np.array(
        [
            psi_s_star,
            dpsi_s_star,
            complex(a_st, 0.0),
            0.0 + 0.0j,
            complex(eta_s_star, 0.0),
        ]
    )
    est = st_star.copy()
    dt = (t_f - t_star) / n_sample
    n_e = max(1, int(round(n_exact * dt / (t_f - t_i))))
    n_s = max(1, int(round(n_slow * dt / (t_f - t_star))))
    errs = []
    t = t_star
    for _ in range(n_sample):
        est = _rk4(_exact_deriv, est, t, t + dt, n_e)
        sst = _rk4(_slow_deriv, sst, t, t + dt, n_s)
        t += dt
        a_s = sst[2].real
        rho, p = _other(a_s)
        h_s = float(_oracle_slow_hubble(sst[0], rho, m, m_pl))
        rates = _oracle_metric_slow_rates(
            sst[0], sst[1], sst[4].real, h_s, a_s, m, m_pl, k
        )
        rec = _oracle_reconstruct_perturbation(
            sst[1], sst[0], h_s, float(rates[0]), a_s, m, k, t
        )
        errs.append(abs(complex(rec) - complex(est[1])))
    errs = np.asarray(errs, dtype=float)

    a_sf = sst[2].real
    rho_f, p_f = _other(a_sf)
    h_s_f = float(_oracle_slow_hubble(sst[0], rho_f, m, m_pl))
    rates_f = _oracle_metric_slow_rates(
        sst[0], sst[1], sst[4].real, h_s_f, a_sf, m, m_pl, k
    )
    obs_f = _oracle_slow_perturbation_observables(
        sst[0], sst[1], float(rates_f[0]), h_s_f, a_sf, m, k
    )
    amp2_f = float(np.abs(sst[0]) ** 2)
    rho_a_s_f = m * amp2_f + 3.0 * (m * amp2_f + rho_f) * amp2_f / (
        16.0 * m * m_pl**2
    )
    delta_a_f = float(obs_f[0]) / rho_a_s_f

    return np.array(
        [
            float(errs.max()),
            float(errs.mean()),
            float(abs(dpsi_s_star)),
            hdot_s_star,
            float(abs(psi_s_star)),
            a_sf,
            delta_a_f,
            float(sst[4].real),
            h_s_f,
        ],
        dtype=float,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
        {
            "setup": "import numpy as np",
            "call": "axion_reconstruction_audit(1)",
            "gold_call": "_oracle_axion_reconstruction_audit(1)",
        },
        {
            "setup": "import numpy as np",
            "call": "axion_reconstruction_audit(1, 0.25+0.0j)",
            "gold_call": "_oracle_axion_reconstruction_audit(1, 0.25+0.0j)",
        },
        {
            "setup": "import numpy as np",
            "call": "axion_reconstruction_audit(2, 0.312+0.05j)",
            "gold_call": "_oracle_axion_reconstruction_audit(2, 0.312+0.05j)",
        },
        {
            "setup": "import numpy as np\ndef _raises(fn):\n    try:\n        fn(0)\n        return False\n    except ValueError:\n        return True",
            "call": "_raises(axion_reconstruction_audit)",
            "gold_call": "_raises(_oracle_axion_reconstruction_audit)",
        },
    ]
