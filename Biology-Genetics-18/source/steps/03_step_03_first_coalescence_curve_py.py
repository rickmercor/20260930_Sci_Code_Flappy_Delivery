"""
Compute the exact first-coalescence hazard ICR_k(t) and log-survival log S_k(t) for a structured sample under a pulse-then-split demography.

Let \(T_k\) be the time (generations before the present) of the first coalescence among \(k\) sampled lineages, \(S_k(t)=\Pr(T_k>t)\), and \(\mathrm{ICR}_k(t)=-\frac{d}{dt}\log S_k(t)\) the instantaneous first-coalescence rate. Following the lineages backward, the survival-weighted occupancy distribution \(\tilde p_t(x)=\Pr\{X(t)=x,\ T_k>t\}\) obeys, between demographic events,

\[

\frac{d\tilde p_t}{dt}=\tilde p_t\,\bigl(Q_{\mathrm{mig}}-D_{\lambda}\bigr),\qquad S_k(t)=\sum_x\tilde p_t(x),\qquad \mathrm{ICR}_k(t)=\frac{\sum_x\tilde p_t(x)\,\lambda(x)}{S_k(t)},

\]

with \(\lambda(x)=\sum_i\binom{x_i}{2}\frac{1}{2N_i}\). At \(t=0\) the distribution is concentrated on the sampled occupancy vector.

The demography has \(d\) present-day demes and three backward-time intervals. On \([0,t_{\mathrm{pulse}})\) the epoch-0 diploid sizes and migration rates apply. At \(t_{\mathrm{pulse}}\) a forward-time pulse of proportion \(\alpha\) from the source deme into the destination deme acts: backward in time, each lineage in the destination moves to the source with probability \(\alpha\). On \([t_{\mathrm{pulse}},t_{\mathrm{split}})\) the epoch-1 sizes and migration rates apply. At \(t_{\mathrm{split}}\) all \(d\) demes merge (backward in time) into one ancestral deme of diploid size \(N_{\mathrm{anc}}\), which persists for all \(t\ge t_{\mathrm{split}}\) with no migration, so that

\[

\mathrm{ICR}_k(t)=\binom{k}{2}\frac{1}{2N_{\mathrm{anc}}}\qquad (t\ge t_{\mathrm{split}}).

\]

Demographic events change the lineage configuration but never the survival probability, so \(S_k(t)\) is continuous while \(\mathrm{ICR}_k(t)\) can jump at an event. Every returned value is the right-continuous (older-side) value: at \(t=t_{\mathrm{pulse}}\) the pulse has already acted, and at \(t=t_{\mathrm{split}}\) the lineages are already in the ancestral deme.

Returns
-------
np.ndarray with shape (2, len(times)), rows [ICR_k(t), log S_k(t)], right-continuous at the event times
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def first_coalescence_curve(sample_counts: "np.ndarray", demography: dict, times: "np.ndarray") -> "np.ndarray":
    """Return [ICR_k(t), log S_k(t)] at the requested times.
 
    Parameters
    ----------
    sample_counts : np.ndarray
        One-dimensional array of shape (d,) with the number of lineages
        sampled in each present-day deme; entries are nonnegative integers
        (integral floats are accepted) and k = sum(sample_counts) >= 2.
    demography : dict
        Dictionary with exactly these required keys (extra keys are ignored):
        ``"sizes"`` : (2, d) finite, strictly positive diploid sizes, row 0
        for [0, t_pulse) and row 1 for [t_pulse, t_split);
        ``"migration"`` : (2, d, d) finite backward-time migration rates per
        lineage per generation (entry [e, i, j] moves a lineage from deme i
        to deme j in interval e; off-diagonal entries nonnegative, diagonal
        entries ignored);
        ``"t_pulse"``, ``"t_split"`` : finite times with
        0 < t_pulse < t_split;
        ``"pulse_source"``, ``"pulse_dest"`` : distinct integer deme indices
        in [0, d) giving the forward-time source and destination of the pulse;
        ``"alpha"`` : finite pulse proportion in [0, 1];
        ``"ancestral_size"`` : finite, strictly positive diploid size of the
        ancestral deme. Requires d >= 2.
    times : np.ndarray
        Nonempty one-dimensional array of finite times t >= 0, in any order.
 
    Returns
    -------
    curve : np.ndarray
        Float array of shape (2, len(times)); row 0 is ICR_k(t) in
        coalescences per generation and row 1 is log S_k(t) (natural log),
        both right-continuous at the event times, in the order of ``times``.
        Values must agree with the exact solution to relative accuracy 1e-9.
 
    Raises
    ------
    ValueError
        If sample_counts is not a one-dimensional array of d nonnegative
        integers with total k >= 2, if any demography key is missing or
        violates the constraints above, or if times is empty, not
        one-dimensional, not finite, or contains a negative value.
    """
    return curve

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm
 
 
def _check_times(times):
    """Validate a nonempty 1-D array of finite nonnegative times."""
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or t.size < 1 or not np.all(np.isfinite(t)) or np.any(t < 0.0):
        raise ValueError("times must be a nonempty 1-D array of finite nonnegative values")
    return t
 
 
def _check_counts(counts, d, name):
    """Validate a length-d vector of nonnegative integer lineage counts."""
    c = np.asarray(counts, dtype=float)
    if c.shape != (d,) or not np.all(np.isfinite(c)) or np.any(c < 0.0) or np.any(c != np.round(c)):
        raise ValueError(f"{name} must hold {d} nonnegative integers")
    return tuple(int(v) for v in c)
 
 
def _check_demography(demography):
    """Validate the pulse-then-split demography and return its parsed fields."""
    keys = ("sizes", "migration", "t_pulse", "t_split", "pulse_source",
            "pulse_dest", "alpha", "ancestral_size")
    if not isinstance(demography, dict) or any(k not in demography for k in keys):
        raise ValueError("demography must be a dict with the documented keys")
    sizes = np.asarray(demography["sizes"], dtype=float)
    migration = np.asarray(demography["migration"], dtype=float)
    if sizes.ndim != 2 or sizes.shape[0] != 2 or sizes.shape[1] < 2:
        raise ValueError("sizes must have shape (2, d) with d >= 2")
    d = sizes.shape[1]
    if migration.shape != (2, d, d):
        raise ValueError("migration must have shape (2, d, d)")
    for e in range(2):
        _check_sizes_migration(sizes[e], migration[e])
    t_pulse = float(demography["t_pulse"])
    t_split = float(demography["t_split"])
    if not (np.isfinite(t_pulse) and np.isfinite(t_split) and 0.0 < t_pulse < t_split):
        raise ValueError("event times must satisfy 0 < t_pulse < t_split")
    source = _check_count(demography["pulse_source"], "pulse_source", 0)
    dest = _check_count(demography["pulse_dest"], "pulse_dest", 0)
    if source >= d or dest >= d or source == dest:
        raise ValueError("pulse_source and pulse_dest must be distinct indices in [0, d)")
    alpha = demography["alpha"]
    if isinstance(alpha, (bool, np.bool_)) or not np.isfinite(alpha) or not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be a finite number in [0, 1]")
    ancestral = float(demography["ancestral_size"])
    if not np.isfinite(ancestral) or ancestral <= 0.0:
        raise ValueError("ancestral_size must be finite and positive")
    return {"sizes": sizes, "migration": migration, "t_pulse": t_pulse, "t_split": t_split,
            "source": source, "dest": dest, "alpha": float(alpha), "ancestral": ancestral, "d": d}
 
 
def _propagate(p, q, tau):
    """Return (p exp(q tau) normalized to unit mass, log of its total mass).
 
    The interval is split into chunks over which the mass can fall by at most
    a factor exp(-30), so the survival never underflows even for long times.
    """
    v = np.array(p, dtype=float)
    if tau <= 0.0:
        return v, 0.0
    rate = float(np.max(np.abs(np.diag(q))))
    chunk = tau if rate == 0.0 else min(tau, 30.0 / rate)
    n_full = int(tau // chunk)
    remainder = tau - n_full * chunk
    log_mass = 0.0
    if n_full > 0:
        step = expm(q * chunk)
        for _ in range(n_full):
            v = v @ step
            mass = v.sum()
            log_mass += np.log(mass)
            v = v / mass
    if remainder > 0.0:
        v = v @ expm(q * remainder)
        mass = v.sum()
        log_mass += np.log(mass)
        v = v / mass
    return v, log_mass
 
 
def _oracle_first_coalescence_curve(sample_counts: "np.ndarray", demography: dict, times: "np.ndarray") -> "np.ndarray":
    demo = _check_demography(demography)
    d = demo["d"]
    counts = _check_counts(sample_counts, d, "sample_counts")
    k = sum(counts)
    if k < 2:
        raise ValueError("at least two lineages are required")
    t = _check_times(times)
    t_pulse, t_split = demo["t_pulse"], demo["t_split"]
 
    states = _occupancy_states(k, d)
    q0 = _oracle_occupancy_rate_matrix(k, demo["sizes"][0], demo["migration"][0])
    q1 = _oracle_occupancy_rate_matrix(k, demo["sizes"][1], demo["migration"][1])
    lam0 = _occupancy_coalescence_rates(states, demo["sizes"][0])
    lam1 = _occupancy_coalescence_rates(states, demo["sizes"][1])
    p0 = np.zeros(len(states))
    p0[states.index(counts)] = 1.0
    v, log_s_pulse = _propagate(p0, q0, t_pulse)
    kernel = _oracle_pulse_occupancy_kernel(k, d, demo["source"], demo["dest"], demo["alpha"])
    p_pulse = v @ kernel
    _, log_gain = _propagate(p_pulse, q1, t_split - t_pulse)
    log_s_split = log_s_pulse + log_gain
    rate_anc = 0.5 * k * (k - 1) / (2.0 * demo["ancestral"])
 
    curve = np.zeros((2, t.size), dtype=float)
    for n, time in enumerate(t):
        if time < t_pulse:
            v, log_s = _propagate(p0, q0, time)
            curve[:, n] = (v @ lam0, log_s)
        elif time < t_split:
            v, log_s = _propagate(p_pulse, q1, time - t_pulse)
            curve[:, n] = (v @ lam1, log_s_pulse + log_s)
        else:
            curve[:, n] = (rate_anc, log_s_split - rate_anc * (time - t_split))
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
            "setup": base + """times = np.array([0.0, 60.0, 149.5, 150.0, 400.0, 999.0, 1000.0, 2500.0])
""",
            "call": "first_coalescence_curve(np.array([3, 3]), make_demo(), times.copy())",
            "gold_call": "_oracle_first_coalescence_curve(np.array([3, 3]), make_demo(), times.copy())",
            "tol": 1e-8,
        },
        {
            "setup": base + """def demo_b():
    demo = make_demo()
    demo["alpha"] = 0.6
    demo["migration"] = np.array([[[0.0, 5e-3], [1e-3, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
    return demo
times = np.array([900.0, 20.0, 150.0, 1000.0])
""",
            "call": "first_coalescence_curve(np.array([4, 0]), demo_b(), times.copy())",
            "gold_call": "_oracle_first_coalescence_curve(np.array([4, 0]), demo_b(), times.copy())",
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
times = np.array([10.0, 80.0, 300.0, 600.0, 750.0])
""",
            "call": "first_coalescence_curve(np.array([2.0, 1.0, 1.0]), make3(), times.copy())",
            "gold_call": "_oracle_first_coalescence_curve(np.array([2.0, 1.0, 1.0]), make3(), times.copy())",
            "tol": 1e-8,
        },
        {
            "setup": base + """def demo_c():
    demo = make_demo()
    demo["alpha"] = 0.0
    return demo
times = np.array([0.0, 150.0, 1200.0])
""",
            "call": "first_coalescence_curve(np.array([0, 2]), demo_c(), times.copy())",
            "gold_call": "_oracle_first_coalescence_curve(np.array([0, 2]), demo_c(), times.copy())",
            "tol": 1e-8,
        },
        {
            "setup": base + """def run(fn):
    try:
        fn(np.array([1, 0]), make_demo(), np.array([10.0]))
        return 0
    except ValueError:
        return 1
""",
            "call": "run(first_coalescence_curve)",
            "gold_call": "run(_oracle_first_coalescence_curve)",
        },
        {
            "setup": base + """def run(fn):
    demo = make_demo()
    demo["t_pulse"] = 1200.0
    try:
        fn(np.array([2, 2]), demo, np.array([10.0]))
        return 0
    except ValueError:
        return 1
""",
            "call": "run(first_coalescence_curve)",
            "gold_call": "run(_oracle_first_coalescence_curve)",
        },
    ]
