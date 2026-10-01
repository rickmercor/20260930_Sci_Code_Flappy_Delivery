"""
Compute the mean tension of the chain from the end-to-end distance derivative of its free energy at a fixed breakable-segment length.

Under displacement control, the tension is conjugate to the prescribed end-to-end distance, and at the bottom of the intact basin, the relaxation of the internal coordinate adds nothing to it.

Returns
-------
float: partial derivative of the free energy with respect to the end-to-end distance at fixed segment length, in k_B T per Kuhn length.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_intact_chain_tension(y_bar: float, x_bar: float, n_segments: int) -> float:
    """Return the reduced tension at fixed breakable-segment length.

    The tension, in units of k_B T per Kuhn length, is the partial derivative
    with respect to ``y_bar`` of the free energy returned by
    ``compute_link_free_energy`` at the fixed segment length ``x_bar``.
    Evaluated at the intact-state minimum it is the mean tension of the intact
    chain. The bond energy does not enter. The result must have a relative
    accuracy of 1e-9.

    Parameters
    ----------
    y_bar : float
        End-to-end distance in Kuhn lengths, nonnegative, with
        ``abs(y_bar - x_bar)`` below ``n_segments - 1`` minus 0.01.
    x_bar : float
        Breakable-segment length in Kuhn lengths, positive.
    n_segments : int
        Number of Kuhn segments, at least 2.

    Returns
    -------
    tension : float
        Reduced tension as a native Python float.

    Raises
    ------
    ValueError
        If ``y_bar`` is negative or ``x_bar`` is not positive, if ``n_segments`` is not an
        integer of at least 2, or if ``abs(y_bar - x_bar)`` is not below
        ``n_segments - 1.01``.
    """
    return tension

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_intact_chain_tension(y_bar: float, x_bar: float, n_segments: int) -> float:
    """Reference implementation (Richardson-extrapolated five-point derivative)."""
    import numpy as np

    if not (y_bar >= 0.0 and x_bar > 0.0):
        raise ValueError("y_bar must be nonnegative and x_bar must be positive")
    if isinstance(n_segments, bool) or not isinstance(n_segments, (int, np.integer)) or n_segments < 2:
        raise ValueError("n_segments must be an integer of at least 2")
    if not abs(y_bar - x_bar) < n_segments - 1.01:
        raise ValueError("the rigid segments must be able to close the chain")
    if y_bar == 0.0:
        return 0.0
    # The bond term does not depend on y_bar, so any positive well depth gives the same derivative.
    step = min(1e-3, 0.2 * y_bar)
    offsets = np.array([-2.0, -1.0, 1.0, 2.0])

    def _derivative(h):
        values = np.array([
            _oracle_compute_link_free_energy(np.array([float(x_bar)]), float(y_bar + d * h), n_segments, 1.0)[0]
            for d in offsets
        ])
        return (values[0] - 8.0 * values[1] + 8.0 * values[2] - values[3]) / (12.0 * h)

    fine, coarse = _derivative(step), _derivative(2.0 * step)
    return float((16.0 * fine - coarse) / 15.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    setup = (
        "def _sig(value, digits):\n"
        "    return float('%.*e' % (digits - 1, float(value)))\n"
    )
    status = (
        "def run_model():\n"
        "    try:\n"
        "        compute_intact_chain_tension(30.0, 1.0, 21)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "def run_oracle():\n"
        "    try:\n"
        "        _oracle_compute_intact_chain_tension(30.0, 1.0, 21)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
    )
    return [
        {
            "setup": setup,
            "call": "_sig(compute_intact_chain_tension(29.0, 1.0013368, 51), 7)",
            "gold_call": "_sig(_oracle_compute_intact_chain_tension(29.0, 1.0013368, 51), 7)",
        },
        {
            "setup": setup,
            "call": "_sig(compute_intact_chain_tension(0.5, 1.0, 51), 7)",
            "gold_call": "_sig(_oracle_compute_intact_chain_tension(0.5, 1.0, 51), 7)",
        },
        {
            "setup": setup,
            "call": "_sig(compute_intact_chain_tension(0.0, 1.0, 51), 7)",
            "gold_call": "_sig(_oracle_compute_intact_chain_tension(0.0, 1.0, 51), 7)",
        },
        {
            "setup": setup,
            "call": "_sig(compute_intact_chain_tension(19.5, 1.2, 21), 7)",
            "gold_call": "_sig(_oracle_compute_intact_chain_tension(19.5, 1.2, 21), 7)",
        },
        {
            "setup": setup,
            "call": "_sig(compute_intact_chain_tension(395.0, 1.02, 401), 7)",
            "gold_call": "_sig(_oracle_compute_intact_chain_tension(395.0, 1.02, 401), 7)",
        },
        {
            "setup": status,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
