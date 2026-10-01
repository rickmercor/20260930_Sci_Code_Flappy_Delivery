"""
Propagate normalized forward and backward orientational messages through the link-specific log kernels of a chain and return, for every bond, the angular weight imposed by the rest of the intact chain.

Nearest-neighbour bending correlations make a chain a one-dimensional transfer-matrix problem. Stiff or heterogeneous links create exponentially separated angular channels, so both local weights and transfer kernels must remain in the log domain while each directional message is normalized.

Returns
-------
np.ndarray: finite nonnegative float array of shape (n_bonds, n_theta) with the product of normalized forward and backward shapes per bond.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_angular_weights(log_intact: "np.ndarray", log_kernels: "np.ndarray") -> "np.ndarray":
    """Return the angular weight imposed on each bond by the rest of the chain.

    Angles are the ``n_theta`` Gauss-Legendre nodes of ``[0, pi]`` in
    ``leggauss`` order, and every angular integral uses that rule.
    ``log_intact[j]`` is the log intact weight of bond ``j`` on the grid, and
    ``log_kernels[q, a, b]`` is the log coupling at link ``q`` between bond
    ``q + 1`` at node ``a`` and predecessor bond ``q`` at node ``b``. Thus a
    chain of ``n_bonds`` uses exactly ``n_bonds - 1`` link kernels. For bond
    ``i`` (0-based), the forward shape is the intact-basin
    weight of bonds ``0..i-1`` conditioned on the polar angle of bond ``i``, and
    the backward shape is that of bonds ``i+1..n-1``; each shape is normalized
    to unit integral over the angle, and a side with no bonds has the uniform
    shape ``1 / pi``. The returned weight of bond ``i`` is the product of its
    forward and backward shapes. Perform the two recurrences with log-sum-exp,
    including the Gauss-Legendre integration weights inside each reduction,
    and normalize every message in the log domain. Adding an arbitrary finite
    constant to any row of ``log_intact`` or to any full link kernel must not
    change the result, even when ordinary exponentiation would overflow or
    underflow.

    Parameters
    ----------
    log_intact : np.ndarray
        Finite float array of shape ``(n_bonds, n_theta)``.
    log_kernels : np.ndarray
        Finite float array of shape ``(n_bonds - 1, n_theta, n_theta)``. For a
        one-bond chain this has shape ``(0, n_theta, n_theta)``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n_bonds, n_theta)``.

    Raises
    ------
    ValueError
        If ``log_intact`` is not a finite two-dimensional array with at least
        one row and two columns, or if ``log_kernels`` is not a finite array
        with the required link and angle dimensions.
    """
    return weights

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_angular_weights(log_intact: "np.ndarray", log_kernels: "np.ndarray") -> "np.ndarray":
    """Reference implementation with log-domain directional messages."""
    import numpy as np
    from scipy.special import logsumexp

    local = np.asarray(log_intact, dtype=float)
    coupling = np.asarray(log_kernels, dtype=float)
    if local.ndim != 2 or local.shape[0] < 1 or local.shape[1] < 2 or not np.all(np.isfinite(local)):
        raise ValueError("log_intact must be a finite 2D array with at least one row and two columns")
    n_bonds, n_theta = local.shape
    if coupling.shape != (n_bonds - 1, n_theta, n_theta) or not np.all(np.isfinite(coupling)):
        raise ValueError("log_kernels must contain one finite square log kernel per chain link")
    _, nodes_weights = np.polynomial.legendre.leggauss(n_theta)
    quad = 0.5 * np.pi * nodes_weights
    log_quad = np.log(quad)
    forward = np.empty((n_bonds, n_theta), dtype=float)
    backward = np.empty((n_bonds, n_theta), dtype=float)
    forward[0] = -np.log(np.pi)
    backward[-1] = -np.log(np.pi)
    for i in range(n_bonds - 1):
        source = log_quad + forward[i] + local[i]
        message = logsumexp(coupling[i] + source[None, :], axis=1)
        forward[i + 1] = message - logsumexp(log_quad + message)
    for i in range(n_bonds - 1, 0, -1):
        source = log_quad + backward[i] + local[i]
        message = logsumexp(coupling[i - 1] + source[:, None], axis=0)
        backward[i - 1] = message - logsumexp(log_quad + message)
    return np.exp(forward + backward)

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    digest = (
        "import numpy as np\n"
        "def _sig(values):\n"
        "    arr = np.asarray(values, dtype=float)\n"
        "    flat = 1.0e3 * arr.ravel()\n"
        "    weight = 1.0 + 0.5 * np.cos(0.7 * np.arange(flat.size))\n"
        "    return float(arr.shape[0] + flat.size + np.sum(weight * flat))\n"
        "t, w = np.polynomial.legendre.leggauss(10)\n"
        "theta = 0.5 * np.pi * (t + 1.0)\n"
        "base = -((theta[:, None] - theta[None, :] - 0.3) ** 2) / 0.08\n"
        "links = np.stack([base, 0.7 * base.T - 300.0, 1.4 * base + 725.0])\n"
        "logs = np.array([30.0 * np.cos(theta) + np.log(np.sin(theta)),\n"
        "                 45.0 * np.cos(theta) + np.log(np.sin(theta)) + 7.0,\n"
        "                 np.log(np.sin(theta)) - 2.0 * theta,\n"
        "                 20.0 * np.cos(theta) ** 2 + np.log(np.sin(theta))])\n"
    )
    raises = (
        "import numpy as np\n"
        "def _candidate():\n    try:\n        compute_angular_weights({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
        "def _reference():\n    try:\n        _oracle_compute_angular_weights({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
    )
    invalid = [
        "np.zeros((3, 6)), np.zeros((1, 6, 6))",
        "np.zeros((3, 6)), np.zeros((2, 5, 5))",
        "np.zeros(6), np.zeros((0, 6, 6))",
        "np.full((2, 6), np.inf), np.zeros((1, 6, 6))",
        "np.zeros((2, 6)), np.full((1, 6, 6), np.inf)",
    ]
    return [
        {
            "setup": digest,
            "call": "_sig(compute_angular_weights(logs.copy(), links.copy()))",
            "gold_call": "_sig(_oracle_compute_angular_weights(logs.copy(), links.copy()))",
        },
        {
            "setup": digest,
            "call": "_sig(compute_angular_weights(logs[:2].copy(), links[:1].transpose(0, 2, 1).copy()))",
            "gold_call": "_sig(_oracle_compute_angular_weights(logs[:2].copy(), links[:1].transpose(0, 2, 1).copy()))",
        },
        {
            "setup": digest,
            "call": "_sig(compute_angular_weights(logs[2:3].copy(), np.empty((0, 10, 10))))",
            "gold_call": "_sig(_oracle_compute_angular_weights(logs[2:3].copy(), np.empty((0, 10, 10))))",
        },
        {
            "setup": digest,
            "call": "_sig(compute_angular_weights(logs.copy(), np.zeros((3, 10, 10))))",
            "gold_call": "_sig(_oracle_compute_angular_weights(logs.copy(), np.zeros((3, 10, 10))))",
        },
        {
            "setup": digest,
            "call": "_sig(compute_angular_weights(logs[::-1] - 400.0, links[::-1] + np.array([900.0, -1200.0, 300.0])[:, None, None]))",
            "gold_call": "_sig(_oracle_compute_angular_weights(logs[::-1] - 400.0, links[::-1] + np.array([900.0, -1200.0, 300.0])[:, None, None]))",
        },
        {
            "setup": digest + "extreme_logs = logs + np.array([1200.0, -1200.0, 900.0, -900.0])[:, None]\n",
            "call": "_sig(compute_angular_weights(extreme_logs.copy(), links - 5000.0))",
            "gold_call": "_sig(_oracle_compute_angular_weights(extreme_logs.copy(), links - 5000.0))",
        },
        {
            "setup": digest + "long_logs = np.vstack([logs[i % 4] + (1500.0 if i % 2 else -1500.0) for i in range(36)])\nlong_links = np.stack([links[i % 3] + 400.0 * i for i in range(35)])\n",
            "call": "_sig(compute_angular_weights(long_logs.copy(), long_links.copy()))",
            "gold_call": "_sig(_oracle_compute_angular_weights(long_logs.copy(), long_links.copy()))",
        },
    ] + [
        {"setup": raises.replace("{args}", args), "call": "_candidate()", "gold_call": "_reference()"}
        for args in invalid
    ]
