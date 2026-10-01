"""
Compare how much information about an admixture pulse the first cross-coalescence time and the first coalescence time carry.

The same set of sampled genomes can be summarized by two first-event times. One is the first coalescence \(T_k\) among all \(k\) sampled lineages, whose hazard is the first-coalescence rate \(\mathrm{ICR}_k\). The other is the first coalescence \(T_\times\) between a lineage sampled in one population (red) and a lineage sampled in another (blue), whose hazard is the cross-coalescence rate \(\mathrm{CCR}\). Each time has a density \(f=cS\), and the Fisher information \(I=\mathbb{E}\bigl[(\partial_\alpha\log f)^2\bigr]\) measures how precisely one observation constrains the pulse proportion \(\alpha\).

This orchestrator evaluates both informations under one pulse-then-split demography, using the full sample for \(T_k\) and the red/blue partition for \(T_\times\), and returns

\[

R=\frac{I_\times(\alpha)}{I_k(\alpha)} .

\]

A ratio above one means the cross-coalescence time is the more informative single observation about the pulse for this sampling design.

Returns
-------
float, the ratio I_x(alpha) / I_k(alpha) as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pulse_information_ratio(sample_counts: "np.ndarray", red_counts: "np.ndarray", blue_counts: "np.ndarray", demography: dict) -> float:
    """Return I_x(alpha) / I_k(alpha) for one pulse-then-split demography.
 
    Parameters
    ----------
    sample_counts : np.ndarray
        Shape (d,), lineage counts per present-day deme used for the first
        coalescence time T_k; nonnegative integers with total >= 2.
    red_counts : np.ndarray
        Shape (d,), red lineage counts per deme for T_x; nonnegative integers
        with total >= 1.
    blue_counts : np.ndarray
        Shape (d,), blue lineage counts per deme for T_x; nonnegative integers
        with total >= 1.
    demography : dict
        Pulse-then-split demography with the keys and constraints documented
        for ``first_event_density_score`` (0 < alpha < 1 strictly).
 
    Returns
    -------
    ratio : float
        I_x(alpha) / I_k(alpha) as a native Python float, with relative error
        at most 1e-6.
 
    Raises
    ------
    ValueError
        If any input violates the contracts of ``pulse_fisher_information``,
        or if I_k(alpha) is not strictly positive.
    """
    return ratio

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
 
def _oracle_pulse_information_ratio(sample_counts: "np.ndarray", red_counts: "np.ndarray", blue_counts: "np.ndarray", demography: dict) -> float:
    red = np.asarray(red_counts, dtype=float)
    blue = np.asarray(blue_counts, dtype=float)
    if red.ndim != 1 or blue.shape != red.shape:
        raise ValueError("red_counts and blue_counts must be 1-D arrays of equal shape")
    info_k = _oracle_pulse_fisher_information("icr", sample_counts, demography)
    info_x = _oracle_pulse_fisher_information("ccr", np.vstack([red, blue]), demography)
    if not info_k > 0.0:
        raise ValueError("the first-coalescence information must be positive")
    return float(info_x / info_k)

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
            "setup": base,
            "call": "pulse_information_ratio(np.array([2, 2]), np.array([2, 0]), np.array([0, 2]), make_demo())",
            "gold_call": "_oracle_pulse_information_ratio(np.array([2, 2]), np.array([2, 0]), np.array([0, 2]), make_demo())",
            "tol": 1e-6,
        },
        {
            "setup": base + """def demo_b():
    demo = make_demo()
    demo["alpha"] = 0.5
    demo["sizes"] = np.array([[1500.0, 1500.0], [3000.0, 1000.0]])
    return demo
""",
            "call": "pulse_information_ratio(np.array([1, 2]), np.array([1, 0]), np.array([0, 2]), demo_b())",
            "gold_call": "_oracle_pulse_information_ratio(np.array([1, 2]), np.array([1, 0]), np.array([0, 2]), demo_b())",
            "tol": 1e-6,
        },
        {
            "setup": base + """def demo_c():
    demo = make_demo()
    demo["migration"] = np.zeros((2, 2, 2))
    return demo
""",
            "call": "pulse_information_ratio(np.array([1, 1]), np.array([1, 0]), np.array([0, 1]), demo_c())",
            "gold_call": "_oracle_pulse_information_ratio(np.array([1, 1]), np.array([1, 0]), np.array([0, 1]), demo_c())",
            "tol": 1e-6,
        },
        {
            "setup": base + """def run(fn):
    try:
        fn(np.array([2, 2]), np.array([2, 0]), np.array([0, 2, 0]), make_demo())
        return 0
    except ValueError:
        return 1
""",
            "call": "run(pulse_information_ratio)",
            "gold_call": "run(_oracle_pulse_information_ratio)",
        },
    ]
