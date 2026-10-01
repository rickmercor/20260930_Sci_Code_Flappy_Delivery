"""
Construct quadratic nodal coordinates and their one-dimensional control intervals.

Each primary interval $[a,b]$ has nodes at $a,(a+b)/2,b$ and cuts at $a+p(b-a)$ and $a+(1-p)(b-a)$.

Pieces at a shared primary endpoint are joined into one control interval.

The returned rows are $(x_i,\ell_i,r_i)$ in increasing node order, including boundary nodes.

Tensor products of these intervals define the two-dimensional nodal control volumes.

Returns
-------
A float array containing the node coordinate and two control-interval bounds in increasing node order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_dual_partition(knots: np.ndarray, partition: float) -> np.ndarray:
    r"""Construct quadratic nodal coordinates and their one-dimensional control intervals.

    Parameters
    ----------
    knots : np.ndarray
        Strictly increasing finite vector of at least two primary coordinates.
    partition : float
        Fraction $p$ satisfying $0 < p < 1/2$.

    Returns
    -------
    np.ndarray
        Shape $(2m-1,3)$ for $m$ knots; coordinate, left cut, right cut.

    Raises
    ------
    ValueError
        If knots are not a finite increasing vector or partition is outside $(0,1/2)$.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _finite_array(value, name, shape=None):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numerical array") from exc
    if not np.all(np.isfinite(result)) or (shape is not None and result.shape != shape):
        raise ValueError(f"{name} has invalid shape or non-finite entries")
    return result


def _finite_scalar(value, name):
    result = _finite_array(value, name, ())
    return float(result)


def _positive_scalar(value, name):
    result = _finite_scalar(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _oracle_build_dual_partition(knots: np.ndarray, partition: float) -> np.ndarray:
    """Evaluate the reference numerical operation."""
    knots = _finite_array(knots, "knots")
    partition = _finite_scalar(partition, "partition")
    if knots.ndim != 1 or knots.size < 2 or np.any(np.diff(knots) <= 0):
        raise ValueError("knots must be a strictly increasing vector")
    if not 0 < partition < 0.5:
        raise ValueError("partition must lie in (0, 0.5)")
    nodes = np.empty(2 * knots.size - 1)
    nodes[::2] = knots
    nodes[1::2] = (knots[:-1] + knots[1:]) / 2
    cuts = np.empty(2 * knots.size)
    cuts[0], cuts[-1] = knots[0], knots[-1]
    cuts[1:-1:2] = knots[:-1] + partition * np.diff(knots)
    cuts[2:-1:2] = knots[:-1] + (1 - partition) * np.diff(knots)
    return np.column_stack((nodes, cuts[:-1], cuts[1:]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nknots = np.array([0., .22, .57, 1.])\np = .27\n",
            "call": "build_dual_partition(knots.copy(), p)",
            "gold_call": "_oracle_build_dual_partition(knots.copy(), p)",
        },
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nknots = np.array([-1., 2.])\np = 1/3\n",
            "call": "build_dual_partition(knots.copy(), p)",
            "gold_call": "_oracle_build_dual_partition(knots.copy(), p)",
        },
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nknots = np.array([0., 1e-4, .8, 1.])\np = .49\n",
            "call": "build_dual_partition(knots.copy(), p)",
            "gold_call": "_oracle_build_dual_partition(knots.copy(), p)",
        },
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nknots = np.array([0., .4, .4])\np = .27\ndef _exception_code(function):\n    try:\n        function(knots.copy(), p)\n    except ValueError:\n        return 1\n    return 0\n\n",
            "call": "_exception_code(build_dual_partition)",
            "gold_call": "_exception_code(_oracle_build_dual_partition)",
        },
    ]
