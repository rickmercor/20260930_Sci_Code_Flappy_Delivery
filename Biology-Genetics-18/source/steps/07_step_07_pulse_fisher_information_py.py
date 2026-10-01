"""
Compute the Fisher information about the pulse proportion carried by one observed first-event time.

For a single observation of a first-event time \(T\) with density \(f(t;\alpha)=c(t;\alpha)\,S(t;\alpha)\), the Fisher information about \(\alpha\) is

\[

I(\alpha)=\mathbb{E}\left[\bigl(\partial_\alpha\log f(T;\alpha)\bigr)^2\right]=\int_0^\infty f(t;\alpha)\,\bigl[\partial_\alpha\log f(t;\alpha)\bigr]^2\,dt .

\]

The integral runs over the whole positive time axis, including the ancestral epoch \(t\ge t_{\mathrm{split}}\), where the survival function decays but has not vanished; truncating it at \(t_{\mathrm{split}}\) omits information. Times before \(t_{\mathrm{pulse}}\) contribute nothing because the density there does not depend on \(\alpha\). The integrand is piecewise smooth, with jumps at \(t_{\mathrm{pulse}}\) and \(t_{\mathrm{split}}\) where the hazard jumps. The information of \(n\) independent observations is \(n\,I(\alpha)\).

Returns
-------
float, the single-observation Fisher information I(alpha) as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pulse_fisher_information(kind: str, samples: "np.ndarray", demography: dict) -> float:
    """Return the Fisher information I(alpha) of one first-event time.
 
    Parameters
    ----------
    kind : str
        Exactly ``"icr"`` (first coalescence time of all sampled lineages) or
        ``"ccr"`` (first red-blue cross-coalescence time).
    samples : np.ndarray
        For ``"icr"``: shape (d,), lineage counts per present-day deme. For
        ``"ccr"``: shape (2, d), with red counts in row 0 and blue counts in
        row 1. Same contracts as ``first_event_density_score``.
    demography : dict
        Pulse-then-split demography with the keys and constraints documented
        for ``first_event_density_score`` (0 < alpha < 1 strictly).
 
    Returns
    -------
    information : float
        I(alpha) = integral_0^infinity f(t) [d log f(t) / d alpha]^2 dt as a
        native Python float, with relative error at most 1e-6.
 
    Raises
    ------
    ValueError
        If kind, samples or demography violate the contracts of
        ``first_event_density_score``.
    """
    return information

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
 
def _gauss_legendre_panels(edges, order):
    """Composite Gauss-Legendre nodes and weights on consecutive panels."""
    x, w = np.polynomial.legendre.leggauss(order)
    nodes, weights = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        nodes.append(0.5 * (b - a) * x + 0.5 * (a + b))
        weights.append(0.5 * (b - a) * w)
    return np.concatenate(nodes), np.concatenate(weights)
 
 
def _oracle_pulse_fisher_information(kind: str, samples: "np.ndarray", demography: dict) -> float:
    if kind not in ("icr", "ccr"):
        raise ValueError("kind must be 'icr' or 'ccr'")
    demo = _check_demography(demography)
    t_pulse, t_split = demo["t_pulse"], demo["t_split"]
    decay = 1.0 / (2.0 * demo["ancestral"])
    middle = np.linspace(t_pulse, t_split, 17)
    tail = t_split + np.concatenate([[0.0], 0.025 * 2.0 ** np.arange(13)]) / decay
    nodes_mid, weights_mid = _gauss_legendre_panels(middle, 16)
    nodes_tail, weights_tail = _gauss_legendre_panels(tail, 16)
    nodes = np.concatenate([nodes_mid, nodes_tail])
    weights = np.concatenate([weights_mid, weights_tail])
    fs = _oracle_first_event_density_score(kind, samples, demography, nodes)
    return float(np.sum(weights * fs[0] * fs[1] ** 2))

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
            "call": "pulse_fisher_information('icr', np.array([2, 2]), make_demo())",
            "gold_call": "_oracle_pulse_fisher_information('icr', np.array([2, 2]), make_demo())",
            "tol": 1e-6,
        },
        {
            "setup": base,
            "call": "pulse_fisher_information('ccr', np.array([[2, 0], [0, 2]]), make_demo())",
            "gold_call": "_oracle_pulse_fisher_information('ccr', np.array([[2, 0], [0, 2]]), make_demo())",
            "tol": 1e-6,
        },
        {
            "setup": base + """def demo_c():
    demo = make_demo()
    demo["migration"] = np.zeros((2, 2, 2))
    demo["alpha"] = 0.6
    demo["ancestral_size"] = 2000.0
    return demo
""",
            "call": "pulse_fisher_information('ccr', np.array([[1, 0], [0, 1]]), demo_c())",
            "gold_call": "_oracle_pulse_fisher_information('ccr', np.array([[1, 0], [0, 1]]), demo_c())",
            "tol": 1e-6,
        },
        {
            "setup": base + """def demo_d():
    demo = make_demo()
    demo["alpha"] = 0.1
    demo["t_pulse"] = 40.0
    demo["t_split"] = 300.0
    return demo
""",
            "call": "pulse_fisher_information('icr', np.array([2, 1]), demo_d())",
            "gold_call": "_oracle_pulse_fisher_information('icr', np.array([2, 1]), demo_d())",
            "tol": 1e-6,
        },
        {
            "setup": base + """def run(fn):
    try:
        fn('ccr', np.array([3, 3]), make_demo())
        return 0
    except ValueError:
        return 1
""",
            "call": "run(pulse_fisher_information)",
            "gold_call": "run(_oracle_pulse_fisher_information)",
        },
    ]
