"""
Compute the exact cross-coalescence rate CCR(t) and log-survival log S_x(t) of red and blue samples under a pulse-then-split demography.

Red and blue lineages are sampled in the present-day demes. \(T_\times\) is the time of the first coalescence between a red and a blue lineage, \(S_\times(t)=\Pr(T_\times>t)\), and \(\mathrm{CCR}(t)=-\frac{d}{dt}\log S_\times(t)\). Red–red and blue–blue coalescences can occur before \(T_\times\) and reduce the colour counts. Between demographic events the survival-weighted distribution over interleaved red and blue count vectors obeys

\[

\frac{d\tilde p^{\times}_t}{dt}=\tilde p^{\times}_t\,Q_\times,\qquad S_\times(t)=\sum\tilde p^{\times}_t,\qquad \mathrm{CCR}(t)=\frac{\sum\tilde p^{\times}_t\,\lambda_\times}{S_\times(t)},

\]

where \(Q_\times\) holds migration of either colour at rates \(\rho_iM_{ij}\) and \(\beta_iM_{ij}\), same-colour coalescence at rates \(\binom{\rho_i}{2}/(2N_i)\) and \(\binom{\beta_i}{2}/(2N_i)\), and removal at the cross-coalescence rate \(\lambda_\times=\sum_i\rho_i\beta_i/(2N_i)\).

The demography is the same pulse-then-split history as for the first-coalescence calculation. Epoch-0 sizes and migration rates apply on \([0,t_{\mathrm{pulse}})\). At \(t_{\mathrm{pulse}}\) a forward-time pulse of proportion \(\alpha\) from the source into the destination deme acts, under which each red and each blue lineage in the destination independently moves (backward in time) to the source with probability \(\alpha\). Epoch-1 sizes and migration rates apply on \([t_{\mathrm{pulse}},t_{\mathrm{split}})\). At \(t_{\mathrm{split}}\) all demes merge into one ancestral deme of size \(N_{\mathrm{anc}}\), in which red and blue lineages keep their colours and same-colour coalescences continue. Returned values are right-continuous (older side) at the event times.

Returns
-------
np.ndarray with shape (2, len(times)), rows [CCR(t), log S_x(t)], right-continuous at the event times
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cross_coalescence_curve(red_counts: "np.ndarray", blue_counts: "np.ndarray", demography: dict, times: "np.ndarray") -> "np.ndarray":
    """Return [CCR(t), log S_x(t)] at the requested times.
 
    Parameters
    ----------
    red_counts : np.ndarray
        One-dimensional array of shape (d,) with the number of red lineages
        sampled in each present-day deme; nonnegative integers (integral
        floats are accepted) with total >= 1.
    blue_counts : np.ndarray
        One-dimensional array of shape (d,) with the number of blue lineages
        sampled in each present-day deme; nonnegative integers (integral
        floats are accepted) with total >= 1.
    demography : dict
        Dictionary with the required keys ``"sizes"`` (2, d),
        ``"migration"`` (2, d, d), ``"t_pulse"``, ``"t_split"``,
        ``"pulse_source"``, ``"pulse_dest"``, ``"alpha"`` and
        ``"ancestral_size"``, with the same meanings and constraints as in
        ``first_coalescence_curve``: positive diploid sizes, nonnegative
        off-diagonal backward-time migration rates (diagonal ignored),
        0 < t_pulse < t_split, distinct integer deme indices for the
        forward-time pulse source and destination, alpha in [0, 1] and a
        positive ancestral size; d >= 2. Extra keys are ignored.
    times : np.ndarray
        Nonempty one-dimensional array of finite times t >= 0, in any order.
 
    Returns
    -------
    curve : np.ndarray
        Float array of shape (2, len(times)); row 0 is CCR(t) in
        cross-coalescences per generation and row 1 is log S_x(t) (natural
        log), right-continuous at the event times and in the order of
        ``times``. Values must agree with the exact solution to relative
        accuracy 1e-9.
 
    Raises
    ------
    ValueError
        If red_counts or blue_counts is not a one-dimensional array of d
        nonnegative integers with total >= 1, if any demography key is missing
        or violates its constraints, or if times is empty, not
        one-dimensional, not finite, or contains a negative value.
    """
    return curve

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
 
def _colour_pulse_kernel(n_red, n_blue, d, source, dest, alpha):
    """Pulse transition matrix on colour states; colours move independently."""
    states = _colour_states(n_red, n_blue, d)
    index = {s: a for a, s in enumerate(states)}
    red_kernels = [_oracle_pulse_occupancy_kernel(r, d, source, dest, alpha) for r in range(n_red + 1)]
    blue_kernels = [_oracle_pulse_occupancy_kernel(b, d, source, dest, alpha) for b in range(n_blue + 1)]
    occupancy = {m: _occupancy_states(m, d) for m in range(max(n_red, n_blue) + 1)}
    kernel = np.zeros((len(states), len(states)), dtype=float)
    for s in states:
        red, blue = s[0::2], s[1::2]
        r, b = sum(red), sum(blue)
        k_red, k_blue = red_kernels[r], blue_kernels[b]
        i_red, i_blue = occupancy[r].index(red), occupancy[b].index(blue)
        for j_red, red_new in enumerate(occupancy[r]):
            if k_red[i_red, j_red] == 0.0:
                continue
            for j_blue, blue_new in enumerate(occupancy[b]):
                weight = k_red[i_red, j_red] * k_blue[i_blue, j_blue]
                if weight == 0.0:
                    continue
                target = tuple(v for pair in zip(red_new, blue_new) for v in pair)
                kernel[index[s], index[target]] += weight
    return kernel
 
 
def _oracle_cross_coalescence_curve(red_counts: "np.ndarray", blue_counts: "np.ndarray", demography: dict, times: "np.ndarray") -> "np.ndarray":
    demo = _check_demography(demography)
    d = demo["d"]
    red = _check_counts(red_counts, d, "red_counts")
    blue = _check_counts(blue_counts, d, "blue_counts")
    n_red, n_blue = sum(red), sum(blue)
    if n_red < 1 or n_blue < 1:
        raise ValueError("at least one red and one blue lineage are required")
    t = _check_times(times)
    t_pulse, t_split = demo["t_pulse"], demo["t_split"]
 
    states = _colour_states(n_red, n_blue, d)
    q0 = _oracle_cross_coalescence_rate_matrix(n_red, n_blue, demo["sizes"][0], demo["migration"][0])
    q1 = _oracle_cross_coalescence_rate_matrix(n_red, n_blue, demo["sizes"][1], demo["migration"][1])
    lam0 = _cross_coalescence_rates(states, demo["sizes"][0])
    lam1 = _cross_coalescence_rates(states, demo["sizes"][1])
    p0 = np.zeros(len(states))
    p0[states.index(tuple(v for pair in zip(red, blue) for v in pair))] = 1.0
    v, log_s_pulse = _propagate(p0, q0, t_pulse)
    kernel = _colour_pulse_kernel(n_red, n_blue, d, demo["source"], demo["dest"], demo["alpha"])
    p_pulse = v @ kernel
    v, log_gain = _propagate(p_pulse, q1, t_split - t_pulse)
    log_s_split = log_s_pulse + log_gain
 
    ancestral_states = _colour_states(n_red, n_blue, 1)
    p_anc = np.zeros(len(ancestral_states))
    for s, w in zip(states, v):
        p_anc[ancestral_states.index((sum(s[0::2]), sum(s[1::2])))] += w
    q_anc = _oracle_cross_coalescence_rate_matrix(n_red, n_blue, np.array([demo["ancestral"]]), np.zeros((1, 1)))
    lam_anc = _cross_coalescence_rates(ancestral_states, np.array([demo["ancestral"]]))
 
    curve = np.zeros((2, t.size), dtype=float)
    for n, time in enumerate(t):
        if time < t_pulse:
            v, log_s = _propagate(p0, q0, time)
            curve[:, n] = (v @ lam0, log_s)
        elif time < t_split:
            v, log_s = _propagate(p_pulse, q1, time - t_pulse)
            curve[:, n] = (v @ lam1, log_s_pulse + log_s)
        else:
            v, log_s = _propagate(p_anc, q_anc, time - t_split)
            curve[:, n] = (v @ lam_anc, log_s_split + log_s)
    return curve

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = """import numpy as np
def make_demo():
    return {
        "sizes": np.array([[4000.0, 1000.0], [4000.0, 2500.0]]),
        "migration": np.array([[[0.0, 1e-4], [2e-4, 0.0]], [[0.0, 1e-4], [2e-4, 0.0]]]),
        "t_pulse": 150.0, "t_split": 1000.0,
        "pulse_source": 1, "pulse_dest": 0, "alpha": 0.25,
        "ancestral_size": 5000.0,
    }
"""
    return [
        {
            "setup": base + """times = np.array([0.0, 75.0, 150.0, 400.0, 999.0, 1000.0, 3000.0, 20000.0])
""",
            "call": "cross_coalescence_curve(np.array([3, 0]), np.array([0, 3]), make_demo(), times.copy())",
            "gold_call": "_oracle_cross_coalescence_curve(np.array([3, 0]), np.array([0, 3]), make_demo(), times.copy())",
            "tol": 1e-8,
        },
        {
            "setup": base + """def demo_b():
    demo = make_demo()
    demo["migration"] = np.zeros((2, 2, 2))
    demo["alpha"] = 0.5
    return demo
times = np.array([100.0, 150.0, 500.0, 1000.0, 1500.0])
""",
            "call": "cross_coalescence_curve(np.array([2, 0]), np.array([0, 2]), demo_b(), times.copy())",
            "gold_call": "_oracle_cross_coalescence_curve(np.array([2, 0]), np.array([0, 2]), demo_b(), times.copy())",
            "tol": 1e-8,
        },
        {
            "setup": base + """def make3():
    return {
        "sizes": np.array([[800.0, 1500.0, 3000.0], [1000.0, 1200.0, 2000.0]]),
        "migration": np.array([[[0.0, 1e-3, 0.0], [5e-4, 0.0, 2e-3], [0.0, 1e-3, 0.0]],
                               [[0.0, 2e-4, 2e-4], [2e-4, 0.0, 2e-4], [2e-4, 2e-4, 0.0]]]),
        "t_pulse": 80.0, "t_split": 600.0,
        "pulse_source": 2, "pulse_dest": 0, "alpha": 0.3,
        "ancestral_size": 6000.0,
    }
times = np.array([700.0, 5.0, 80.0, 250.0, 600.0])
""",
            "call": "cross_coalescence_curve(np.array([1, 1, 0]), np.array([1.0, 0.0, 1.0]), make3(), times.copy())",
            "gold_call": "_oracle_cross_coalescence_curve(np.array([1, 1, 0]), np.array([1.0, 0.0, 1.0]), make3(), times.copy())",
            "tol": 1e-8,
        },
        {
            "setup": base + """def demo_d():
    demo = make_demo()
    demo["alpha"] = 1.0
    return demo
times = np.array([0.0, 150.0, 600.0])
""",
            "call": "cross_coalescence_curve(np.array([2, 1]), np.array([1, 1]), demo_d(), times.copy())",
            "gold_call": "_oracle_cross_coalescence_curve(np.array([2, 1]), np.array([1, 1]), demo_d(), times.copy())",
            "tol": 1e-8,
        },
        {
            "setup": base + """times = np.array([2.0e7, 1.0e6])
""",
            "call": "cross_coalescence_curve(np.array([3, 0]), np.array([0, 3]), make_demo(), times.copy())",
            "gold_call": "_oracle_cross_coalescence_curve(np.array([3, 0]), np.array([0, 3]), make_demo(), times.copy())",
            "tol": 1e-8,
        },
        {
            "setup": base + """def run(fn):
    try:
        fn(np.array([2, 0]), np.array([0, 0]), make_demo(), np.array([10.0]))
        return 0
    except ValueError:
        return 1
""",
            "call": "run(cross_coalescence_curve)",
            "gold_call": "run(_oracle_cross_coalescence_curve)",
        },
        {
            "setup": base + """def run(fn):
    try:
        fn(np.array([1, 0]), np.array([0, 1]), make_demo(), np.array([10.0, -1.0]))
        return 0
    except ValueError:
        return 1
""",
            "call": "run(cross_coalescence_curve)",
            "gold_call": "run(_oracle_cross_coalescence_curve)",
        },
    ]
