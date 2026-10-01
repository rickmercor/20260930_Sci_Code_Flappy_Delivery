"""
Compute the weights that combine the values of a smooth function at several shifted arguments into an estimate of one of its derivatives.

Every extra shift in a finite-difference formula costs one more simulated path in every replication, so a sensitivity study works with the narrowest stencil that still resolves the derivative it targets.

Returns
-------
np.ndarray: float array (J,) of stencil weights in the order of the supplied offsets.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_stencil_weights(offsets: "np.ndarray", derivative_order: int) -> "np.ndarray":
    """Return the finite-difference weights of a stencil for one derivative order.

    Write ``J = len(offsets)``, ``a_r = offsets[r]`` and
    ``k = derivative_order``. Return the weights ``c_r`` for which
    ``sum_r c_r g(theta + a_r eps) / eps^k`` equals ``d^k g / d theta^k`` for
    every ``eps > 0`` whenever ``g`` is a polynomial of degree at most
    ``J - 1``. Distinct offsets and ``k <= J - 1`` make these weights unique.
    They must be accurate to a relative precision of ``1e-9``.

    Parameters
    ----------
    offsets : np.ndarray
        Shape ``(J,)`` with ``J >= 2`` finite distinct offsets ``a_r``.
    derivative_order : int
        Order ``k`` of the derivative, an integer with ``1 <= k <= J - 1``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(J,)`` holding ``c_r`` in the order of
        ``offsets``.

    Raises
    ------
    ValueError
        If ``offsets`` is not one-dimensional with at least two finite
        distinct entries, or if ``derivative_order`` is not an integer with
        ``1 <= derivative_order <= J - 1``.
    """
    return weights

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_build_stencil_weights(offsets: "np.ndarray", derivative_order: int) -> "np.ndarray":
    """Reference implementation: solve the transposed Vandermonde system of the offsets."""
    import math

    import numpy as np

    a = np.asarray(offsets, dtype=float)
    if a.ndim != 1 or a.size < 2 or not np.all(np.isfinite(a)):
        raise ValueError("offsets must be one-dimensional with at least two finite entries")
    if np.unique(a).size != a.size:
        raise ValueError("offsets must be distinct")
    if isinstance(derivative_order, bool) or not isinstance(derivative_order, (int, np.integer)) \
            or not 1 <= int(derivative_order) <= a.size - 1:
        raise ValueError("derivative_order must be an integer between 1 and len(offsets) - 1")
    k = int(derivative_order)
    # Row j collects the j-th term of the expansion, which every weight set must cancel except row k.
    system = np.vander(a, a.size, increasing=True).T
    wanted = np.zeros(a.size)
    wanted[k] = float(math.factorial(k))
    return np.linalg.solve(system, wanted)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    digest = (
        "import numpy as np\n"
        "def _digest(w):\n"
        "    w = np.asarray(w, dtype=float)\n"
        "    if w.ndim != 1:\n"
        "        return -1.0\n"
        "    scale = np.sqrt(np.arange(1.0, w.size + 1.0))\n"
        "    return float(10.0 + w.size + 0.01 * np.sum(scale * np.arcsinh(1.0e6 * w)))\n"
    )

    raises = (
        "import numpy as np\n"
        "def _candidate():\n    try:\n        build_stencil_weights({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n"
        "def _reference():\n    try:\n        _oracle_build_stencil_weights({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n"
    )
    return [
        {   # Normal: four symmetric shifts for a third derivative.
            "setup": digest,
            "call": "_digest(build_stencil_weights(np.array([2.0, 1.0, -1.0, -2.0]), 3))",
            "gold_call": "_digest(_oracle_build_stencil_weights(np.array([2.0, 1.0, -1.0, -2.0]), 3))",
        },
        {   # Boundary: the narrowest stencil, two shifts one of which is zero.
            "setup": digest,
            "call": "_digest(build_stencil_weights(np.array([1.0, 0.0]), 1))",
            "gold_call": "_digest(_oracle_build_stencil_weights(np.array([1.0, 0.0]), 1))",
        },
        {   # Edge: more shifts than the derivative order needs.
            "setup": digest,
            "call": "_digest(build_stencil_weights(np.array([2.0, 1.0, -1.0, -2.0]), 1))",
            "gold_call": "_digest(_oracle_build_stencil_weights(np.array([2.0, 1.0, -1.0, -2.0]), 1))",
        },
        {   # Edge: the highest order five shifts can reach.
            "setup": digest,
            "call": "_digest(build_stencil_weights(np.array([2.0, 1.0, 0.0, -1.0, -2.0]), 4))",
            "gold_call": "_digest(_oracle_build_stencil_weights(np.array([2.0, 1.0, 0.0, -1.0, -2.0]), 4))",
        },
        {   # Edge: unequally spaced one-sided shifts.
            "setup": digest,
            "call": "_digest(build_stencil_weights(np.array([0.0, 0.5, 2.0]), 2))",
            "gold_call": "_digest(_oracle_build_stencil_weights(np.array([0.0, 0.5, 2.0]), 2))",
        },
        {   # Invalid: a repeated offset.
            "setup": raises.replace("{args}", "np.array([1.0, 0.0, 1.0]), 2"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        },
        {   # Invalid: an order the stencil cannot reach.
            "setup": raises.replace("{args}", "np.array([1.0, -1.0]), 2"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        },
    ]
