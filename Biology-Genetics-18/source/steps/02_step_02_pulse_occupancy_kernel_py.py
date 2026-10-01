"""
Build the transition matrix that an admixture pulse applies to the occupancy distribution of m lineages.

A pulse is specified forward in time: at time \(t_{\mathrm{pulse}}\) generations before the present, a fraction \(\alpha\) of the individuals in the destination deme are replaced by migrants from the source deme. Followed backward in time, each lineage that is in the destination deme immediately before the event (on the younger side) independently moves to the source deme with probability \(\alpha\) and stays with probability \(1-\alpha\). Lineages in every other deme are unaffected, and no coalescence occurs at the event itself.

On occupancy vectors \(x=(x_1,\ldots,x_d)\), the number \(w\) of lineages that leave the destination is therefore binomially distributed:

\[

K\bigl[x,\;x-w\,e_{\mathrm{dest}}+w\,e_{\mathrm{src}}\bigr]=\binom{x_{\mathrm{dest}}}{w}\alpha^{w}(1-\alpha)^{x_{\mathrm{dest}}-w},\qquad w=0,\ldots,x_{\mathrm{dest}} .

\]

With the row-vector convention used for the survival-weighted distribution, the distribution just after the event (older side) is \(\tilde p\,K\).

Returns
-------
np.ndarray with shape (S, S), the row-stochastic backward-time pulse transition matrix on occupancy states
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pulse_occupancy_kernel(m: int, d: int, source: int, dest: int, alpha: float) -> "np.ndarray":
    """Return the backward-time pulse transition matrix on occupancy states.
 
    Parameters
    ----------
    m : int
        Number of lineages, an integer m >= 0 (``bool`` is invalid).
    d : int
        Number of demes, an integer d >= 2.
    source : int
        Zero-based index of the forward-time source deme, 0 <= source < d.
    dest : int
        Zero-based index of the forward-time destination deme, 0 <= dest < d,
        dest != source.
    alpha : float
        Finite pulse proportion with 0 <= alpha <= 1.
 
    Returns
    -------
    kernel : np.ndarray
        Row-stochastic float array of shape (S, S) on the occupancy states of
        m lineages in d demes, ordered in ascending lexicographic order of
        (x_1, ..., x_d). Entry [x, y] is the probability that configuration x
        on the younger side of the pulse becomes y on the older side.
 
    Raises
    ------
    ValueError
        If m is not an integer >= 0, d is not an integer >= 2, source or dest
        is not an integer index in [0, d), source equals dest, or alpha is not
        a finite number in [0, 1].
    """
    return kernel

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
 
import numpy as np
 
 
def _oracle_pulse_occupancy_kernel(m: int, d: int, source: int, dest: int, alpha: float) -> "np.ndarray":
    m = _check_count(m, "m", 0)
    d = _check_count(d, "d", 2)
    source = _check_count(source, "source", 0)
    dest = _check_count(dest, "dest", 0)
    if source >= d or dest >= d or source == dest:
        raise ValueError("source and dest must be distinct indices in [0, d)")
    if isinstance(alpha, (bool, np.bool_)) or not np.isfinite(alpha) or not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be a finite number in [0, 1]")
    a = float(alpha)
    states = _occupancy_states(m, d)
    index = {s: i for i, s in enumerate(states)}
    kernel = np.zeros((len(states), len(states)), dtype=float)
    for s in states:
        n = s[dest]
        for w in range(n + 1):
            target = list(s)
            target[dest] -= w
            target[source] += w
            kernel[index[s], index[tuple(target)]] += math.comb(n, w) * a**w * (1.0 - a) ** (n - w)
    return kernel

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "",
            "call": "pulse_occupancy_kernel(3, 2, 1, 0, 0.25)",
            "gold_call": "_oracle_pulse_occupancy_kernel(3, 2, 1, 0, 0.25)",
        },
        {
            "setup": "",
            "call": "pulse_occupancy_kernel(3, 3, 0, 2, 0.4)",
            "gold_call": "_oracle_pulse_occupancy_kernel(3, 3, 0, 2, 0.4)",
        },
        {
            "setup": "",
            "call": "pulse_occupancy_kernel(4, 2, 0, 1, 1.0)",
            "gold_call": "_oracle_pulse_occupancy_kernel(4, 2, 0, 1, 1.0)",
        },
        {
            "setup": "",
            "call": "pulse_occupancy_kernel(2, 2, 1, 0, 0.0)",
            "gold_call": "_oracle_pulse_occupancy_kernel(2, 2, 1, 0, 0.0)",
        },
        {
            "setup": "",
            "call": "pulse_occupancy_kernel(0, 3, 2, 1, 0.7)",
            "gold_call": "_oracle_pulse_occupancy_kernel(0, 3, 2, 1, 0.7)",
        },
        {
            "setup": """def run(fn):
    try:
        fn(3, 2, 1, 1, 0.3)
        return 0
    except ValueError:
        return 1
""",
            "call": "run(pulse_occupancy_kernel)",
            "gold_call": "run(_oracle_pulse_occupancy_kernel)",
        },
        {
            "setup": """def run(fn):
    try:
        fn(3, 2, 1, 0, 1.2)
        return 0
    except ValueError:
        return 1
""",
            "call": "run(pulse_occupancy_kernel)",
            "gold_call": "run(_oracle_pulse_occupancy_kernel)",
        },
    ]
