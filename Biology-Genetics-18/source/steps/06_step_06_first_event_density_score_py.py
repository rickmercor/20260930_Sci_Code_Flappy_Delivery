"""
Evaluate the density of a first-event time and its score with respect to the pulse proportion.

For a first-event time \(T\), either the first coalescence \(T_k\) of a sample or the first red–blue cross-coalescence \(T_\times\), the density is

\[

f(t;\alpha)=c(t;\alpha)\,S(t;\alpha),

\]

where \(c\) is the hazard (\(\mathrm{ICR}_k\) or \(\mathrm{CCR}\)) and \(S\) the survival function. Local identifiability of a demographic parameter from observed first-event times is measured through the score \(s(t)=\partial_\alpha\log f(t;\alpha)\), which enters the Fisher information \(\mathbb{E}\bigl[s(T)^2\bigr]\).

In the pulse-then-split demography, \(\alpha\) enters only through the pulse at \(t_{\mathrm{pulse}}\). For \(t<t_{\mathrm{pulse}}\) the density therefore does not depend on \(\alpha\) and \(s(t)=0\) exactly. For \(t\ge t_{\mathrm{pulse}}\) the score is the sensitivity of \(\log c(t)+\log S(t)\) to \(\alpha\), holding every other demographic parameter fixed. Both \(f\) and \(s\) are right-continuous at the event times, as are the hazard and survival curves they are built from. At a time where \(f(t)=0\) exactly (for example, before any red and blue lineage can share a deme), the score is defined to be \(0\).

Returns
-------
np.ndarray with shape (2, len(times)), rows [f(t), d log f(t) / d alpha]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def first_event_density_score(kind: str, samples: "np.ndarray", demography: dict, times: "np.ndarray") -> "np.ndarray":
    """Return [f(t), d log f(t) / d alpha] for a first-event time.
 
    Parameters
    ----------
    kind : str
        Exactly ``"icr"`` for the first coalescence time T_k of all sampled
        lineages, or ``"ccr"`` for the first red-blue cross-coalescence time.
    samples : np.ndarray
        For ``"icr"``: shape (d,), lineage counts per present-day deme, as for
        ``first_coalescence_curve``. For ``"ccr"``: shape (2, d); row 0 holds
        the red counts and row 1 the blue counts, as for
        ``cross_coalescence_curve``.
    demography : dict
        Pulse-then-split demography with the keys and constraints documented
        for ``first_coalescence_curve``, except that alpha must satisfy
        0 < alpha < 1 strictly.
    times : np.ndarray
        Nonempty one-dimensional array of finite times t >= 0, in any order.
 
    Returns
    -------
    density_score : np.ndarray
        Float array of shape (2, len(times)). Row 0 is f(t) = c(t) S(t) per
        generation, with relative accuracy 1e-9; row 1 is the score
        d log f(t) / d alpha, with absolute error at most
        1e-8 * max(1, |score|), and 0 wherever f(t) = 0. Both rows are
        right-continuous at the event times.
 
    Raises
    ------
    ValueError
        If kind is not ``"icr"`` or ``"ccr"``, if samples has the wrong shape
        or invalid counts for that kind, if alpha is not strictly between 0
        and 1, or if the demography or times violate the curve contracts.
    """
    return density_score

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
 
def _event_curve(kind, samples, demography, times):
    """Hazard and log-survival curve of the requested first-event time."""
    s = np.asarray(samples, dtype=float)
    if kind == "icr":
        if s.ndim != 1:
            raise ValueError("icr samples must be one-dimensional")
        return _oracle_first_coalescence_curve(s, demography, times)
    if kind == "ccr":
        if s.ndim != 2 or s.shape[0] != 2:
            raise ValueError("ccr samples must have shape (2, d)")
        return _oracle_cross_coalescence_curve(s[0], s[1], demography, times)
    raise ValueError("kind must be 'icr' or 'ccr'")
 
 
def _oracle_first_event_density_score(kind: str, samples: "np.ndarray", demography: dict, times: "np.ndarray") -> "np.ndarray":
    if kind not in ("icr", "ccr"):
        raise ValueError("kind must be 'icr' or 'ccr'")
    demo = _check_demography(demography)
    alpha = demo["alpha"]
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    base = _event_curve(kind, samples, demography, times)
    density = base[0] * np.exp(base[1])
 
    def _log_density(a):
        shifted = dict(demography)
        shifted["alpha"] = a
        c = _event_curve(kind, samples, shifted, times)
        with np.errstate(divide="ignore"):
            return np.log(c[0]) + c[1]
 
    h = min(1e-3, 0.5 * alpha, 0.5 * (1.0 - alpha))
    with np.errstate(invalid="ignore"):
        coarse = (_log_density(alpha + h) - _log_density(alpha - h)) / (2.0 * h)
        fine = (_log_density(alpha + 0.5 * h) - _log_density(alpha - 0.5 * h)) / h
        score = (4.0 * fine - coarse) / 3.0
    score = np.where(base[0] > 0.0, score, 0.0)
    return np.vstack([density, score])

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
def project(fs):
    fs = np.asarray(fs, dtype=float)
    return np.vstack([np.log(np.maximum(fs[0], 1e-300)), fs[1]])
"""
    return [
        {
            "setup": base + """times = np.array([100.0, 150.0, 151.0, 700.0, 1000.0, 5000.0])
""",
            "call": "project(first_event_density_score('icr', np.array([3, 3]), make_demo(), times.copy()))",
            "gold_call": "project(_oracle_first_event_density_score('icr', np.array([3, 3]), make_demo(), times.copy()))",
            "tol": 1e-6,
        },
        {
            "setup": base + """times = np.array([0.0, 149.0, 150.0, 999.0, 1000.0, 20000.0])
""",
            "call": "project(first_event_density_score('ccr', np.array([[3, 0], [0, 3]]), make_demo(), times.copy()))",
            "gold_call": "project(_oracle_first_event_density_score('ccr', np.array([[3, 0], [0, 3]]), make_demo(), times.copy()))",
            "tol": 1e-6,
        },
        {
            "setup": base + """def demo_c():
    demo = make_demo()
    demo["migration"] = np.zeros((2, 2, 2))
    demo["alpha"] = 0.9
    return demo
times = np.array([50.0, 150.0, 400.0, 1200.0])
""",
            "call": "project(first_event_density_score('ccr', np.array([[2, 0], [0, 1]]), demo_c(), times.copy()))",
            "gold_call": "project(_oracle_first_event_density_score('ccr', np.array([[2, 0], [0, 1]]), demo_c(), times.copy()))",
            "tol": 1e-6,
        },
        {
            "setup": base + """def demo_d():
    demo = make_demo()
    demo["alpha"] = 0.02
    return demo
times = np.array([160.0, 900.0, 1100.0])
""",
            "call": "project(first_event_density_score('icr', np.array([2, 0]), demo_d(), times.copy()))",
            "gold_call": "project(_oracle_first_event_density_score('icr', np.array([2, 0]), demo_d(), times.copy()))",
            "tol": 1e-6,
        },
        {
            "setup": base + """def run(fn):
    demo = make_demo()
    demo["alpha"] = 0.0
    try:
        fn('icr', np.array([2, 2]), demo, np.array([200.0]))
        return 0
    except ValueError:
        return 1
""",
            "call": "run(first_event_density_score)",
            "gold_call": "run(_oracle_first_event_density_score)",
        },
        {
            "setup": base + """def run(fn):
    try:
        fn('tmrca', np.array([2, 2]), make_demo(), np.array([200.0]))
        return 0
    except ValueError:
        return 1
""",
            "call": "run(first_event_density_score)",
            "gold_call": "run(_oracle_first_event_density_score)",
        },
    ]
