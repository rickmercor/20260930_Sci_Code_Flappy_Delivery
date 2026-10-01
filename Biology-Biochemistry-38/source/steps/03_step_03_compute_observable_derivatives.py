"""
Compute the expected value of an observable of a finite continuous-time Markov chain at a fixed time, together with its exact derivatives of every order up to a given one with respect to one rate constant.

On a finite state space, the expected value of an observable at a fixed time is a smooth function of each rate constant, and its high-order derivatives are what set the bias of a finite-difference sensitivity estimator.

Returns
-------
np.ndarray: float array [g, g', ..., g^(max_order)] of exact derivatives in the chosen rate constant.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_observable_derivatives(
    generators: "np.ndarray",
    rates: "np.ndarray",
    initial_index: int,
    observable: "np.ndarray",
    horizon: float,
    channel: int,
    max_order: int,
) -> "np.ndarray":
    """Return the derivatives of an expected observable with respect to one rate constant.

    The chain with rate vector ``theta`` has generator
    ``Q(theta) = sum_l theta[l] * generators[l]`` and starts in state
    ``initial_index``. With ``g(theta) = E[f(X(T))]``, ``f = observable`` and
    ``T = horizon``, return ``d^m g / d theta[channel]^m`` at
    ``theta = rates`` for ``m = 0, 1, ..., max_order``. The values are exact
    derivatives of ``g``, not finite-difference approximations, and must be
    accurate to a relative precision of ``1e-9``.

    Parameters
    ----------
    generators : np.ndarray
        Float array of shape ``(R, M, M)`` of unit-rate channel generators.
    rates : np.ndarray
        Shape ``(R,)``, finite and non-negative.
    initial_index : int
        Row of the initial state, ``0 <= initial_index < M``.
    observable : np.ndarray
        Shape ``(M,)``, finite values of ``f`` on the states.
    horizon : float
        Finite non-negative time ``T``.
    channel : int
        Zero-based index of the differentiated rate constant.
    max_order : int
        Highest derivative order, at least 0.

    Returns
    -------
    np.ndarray
        Float array of shape ``(max_order + 1,)``; entry ``m`` is the ``m``-th
        derivative (entry 0 is ``g`` itself).

    Raises
    ------
    ValueError
        If ``generators`` is not of shape ``(R, M, M)``, if ``rates`` or
        ``observable`` has the wrong shape, if an input is not finite, if a
        rate is negative, if ``initial_index`` or ``channel`` is not an
        integer in range, if ``horizon`` is negative, or if ``max_order`` is
        not a non-negative integer.
    """
    return derivatives

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _validate_chain_inputs(generators, rates, initial_index, observable, horizon, channel):
    """Validate the chain description shared by the expectation and coupling steps."""
    import numpy as np

    stack = np.asarray(generators, dtype=float)
    if stack.ndim != 3 or stack.shape[1] != stack.shape[2] or min(stack.shape) < 1:
        raise ValueError("generators must have shape (R, M, M)")
    theta = np.asarray(rates, dtype=float)
    values = np.asarray(observable, dtype=float)
    if theta.shape != (stack.shape[0],) or values.shape != (stack.shape[1],):
        raise ValueError("rates and observable must match the generator stack")
    if not (np.all(np.isfinite(stack)) and np.all(np.isfinite(theta)) and np.all(np.isfinite(values))):
        raise ValueError("inputs must be finite")
    if np.any(theta < 0.0):
        raise ValueError("rates must be non-negative")
    for name, value, bound in (("initial_index", initial_index, stack.shape[1]), ("channel", channel, stack.shape[0])):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value < bound:
            raise ValueError(f"{name} must be an integer in range")
    if isinstance(horizon, bool) or not np.isfinite(float(horizon)) or float(horizon) < 0.0:
        raise ValueError("horizon must be finite and non-negative")
    return stack, theta, values

def _oracle_compute_observable_derivatives(
    generators: "np.ndarray",
    rates: "np.ndarray",
    initial_index: int,
    observable: "np.ndarray",
    horizon: float,
    channel: int,
    max_order: int,
) -> "np.ndarray":
    """Reference implementation: exponential of a block upper-bidiagonal generator."""
    import numpy as np
    from scipy.linalg import expm

    stack, theta, values = _validate_chain_inputs(generators, rates, initial_index, observable, horizon, channel)
    if isinstance(max_order, bool) or not isinstance(max_order, (int, np.integer)) or max_order < 0:
        raise ValueError("max_order must be a non-negative integer")
    size, order = stack.shape[1], int(max_order)
    base = np.tensordot(theta, stack, axes=1)
    block = np.zeros(((order + 1) * size, (order + 1) * size))
    for k in range(order + 1):
        block[k * size:(k + 1) * size, k * size:(k + 1) * size] = base
        if k < order:
            block[k * size:(k + 1) * size, (k + 1) * size:(k + 2) * size] = stack[channel]
    # Block (0, k) of the exponential is the k-th Taylor coefficient of exp(T Q) in theta[channel].
    top = expm(float(horizon) * block)[int(initial_index)]
    derivatives, factorial = np.empty(order + 1), 1.0
    for k in range(order + 1):
        factorial *= max(k, 1)
        derivatives[k] = factorial * float(top[k * size:(k + 1) * size] @ values)
    return derivatives

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    helpers = (
        "import numpy as np\n"
        "def _isomerization(total):\n"
        "    # State index a = number of A molecules; channel 0 is A -> B, channel 1 is B -> A.\n"
        "    g = np.zeros((2, total + 1, total + 1))\n"
        "    for a in range(total + 1):\n"
        "        if a > 0:\n"
        "            g[0, a, a - 1], g[0, a, a] = a, -a\n"
        "        if a < total:\n"
        "            g[1, a, a + 1], g[1, a, a] = total - a, -(total - a)\n"
        "    return g\n"
        "def _digest(v):\n"
        "    v = np.asarray(v, dtype=float)\n"
        "    if v.ndim != 1:\n"
        "        return -1.0\n"
        "    w = np.sqrt(np.arange(1.0, v.size + 1.0))\n"
        "    return float(v.size + 1.0e-3 * np.sum(w * v) + 1.0e-3 * np.sum(v ** 2 / w))\n"
    )

    raises = (
        "def _candidate():\n    try:\n        compute_observable_derivatives({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n"
        "def _reference():\n    try:\n        _oracle_compute_observable_derivatives({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n"
    )
    iso3 = "G = _isomerization(3)\nf = np.arange(4.0)\n"
    return [
        {   # Normal: derivatives up to fifth order.
            "setup": helpers + iso3,
            "call": "_digest(compute_observable_derivatives(G.copy(), np.array([0.7, 0.4]), 3, f.copy(), 1.5, 0, 5))",
            "gold_call": "_digest(_oracle_compute_observable_derivatives(G.copy(), np.array([0.7, 0.4]), 3, f.copy(), 1.5, 0, 5))",
        },
        {   # Boundary: the mean alone, of a squared observable.
            "setup": helpers + iso3,
            "call": "_digest(compute_observable_derivatives(G.copy(), np.array([0.7, 0.4]), 1, f.copy() ** 2, 2.0, 1, 0))",
            "gold_call": "_digest(_oracle_compute_observable_derivatives(G.copy(), np.array([0.7, 0.4]), 1, f.copy() ** 2, 2.0, 1, 0))",
        },
        {   # Edge: a zero horizon, where every derivative vanishes.
            "setup": helpers + iso3,
            "call": "_digest(compute_observable_derivatives(G.copy(), np.array([0.7, 0.4]), 2, f.copy(), 0.0, 0, 3))",
            "gold_call": "_digest(_oracle_compute_observable_derivatives(G.copy(), np.array([0.7, 0.4]), 2, f.copy(), 0.0, 0, 3))",
        },
        {   # Edge: differentiate a channel whose rate constant is zero.
            "setup": helpers + "G = np.concatenate([_isomerization(2), _isomerization(2)[:1]])\nf = np.array([0.0, 1.0, 4.0])\n",
            "call": "_digest(compute_observable_derivatives(G.copy(), np.array([0.3, 0.9, 0.0]), 2, f.copy(), 3.0, 2, 4))",
            "gold_call": "_digest(_oracle_compute_observable_derivatives(G.copy(), np.array([0.3, 0.9, 0.0]), 2, f.copy(), 3.0, 2, 4))",
        },
        {   # Invalid: the channel index is out of range.
            "setup": helpers + iso3 + raises.replace("{args}", "G.copy(), np.array([0.7, 0.4]), 3, f.copy(), 1.5, 2, 3"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        },
        {   # Invalid: a negative horizon.
            "setup": helpers + iso3 + raises.replace("{args}", "G.copy(), np.array([0.7, 0.4]), 3, f.copy(), -1.0, 0, 3"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        },
    ]
