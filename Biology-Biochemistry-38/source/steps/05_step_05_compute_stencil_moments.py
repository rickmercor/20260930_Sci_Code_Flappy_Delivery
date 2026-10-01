"""
Compute the exact mean and second moment of a finite-difference numerator built from several copies of a finite chain whose perturbed rate constants follow a stencil and which are generated jointly by stacking channel intensities on one Poisson point process.

A finite-difference sensitivity estimator combines outputs of several nearby parameterized paths per replication, and its mean and spread are governed by how strongly those paths are coupled.

Returns
-------
np.ndarray: float array [E[N], E[N^2]] for the stencil numerator N over jointly generated copies.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_stencil_moments(
    generators: "np.ndarray",
    rates: "np.ndarray",
    initial_index: int,
    observable: "np.ndarray",
    horizon: float,
    channel: int,
    offsets: "np.ndarray",
    coefficients: "np.ndarray",
    eps: float,
) -> "np.ndarray":
    """Return the mean and second moment of a stencil numerator over jointly generated copies.

    With ``J = len(offsets)``, the copies ``X_1, ..., X_J`` of the chain
    described in ``compute_observable_derivatives`` all start in state
    ``initial_index``; copy ``r`` uses ``rates`` with ``rates[channel]``
    replaced by ``rates[channel] + offsets[r] * eps``. All copies are driven
    by one unit-rate Poisson point process on ``[0, inf) x [0, inf)``. At each
    time every channel ``l`` is allotted its own strip of the vertical axis,
    disjoint from the strips of the other channels, whose height is the
    largest of the ``J`` copies' current intensities for channel ``l``. A
    point in that strip moves copy ``r`` through channel ``l`` exactly when
    its height above the bottom of the strip is below copy ``r``'s own
    current intensity for channel ``l``. With ``f = observable``,
    ``T = horizon`` and ``N = sum_r coefficients[r] f(X_r(T))``, return
    ``[E[N], E[N^2]]``, each accurate to a relative precision of ``1e-9``.

    Parameters
    ----------
    generators, rates, initial_index, observable, horizon, channel
        As in ``compute_observable_derivatives``. Each row of each generator
        has at most one positive off-diagonal entry, equal to minus its
        diagonal entry.
    offsets : np.ndarray
        Shape ``(J,)`` with ``J >= 2`` finite offsets ``a_r``.
    coefficients : np.ndarray
        Shape ``(J,)`` of finite coefficients ``c_r``.
    eps : float
        Finite positive perturbation size.

    Returns
    -------
    np.ndarray
        Float array ``[E[N], E[N^2]]``.

    Raises
    ------
    ValueError
        For any invalid chain input listed in ``compute_observable_derivatives``,
        if a generator row moves its state to more than one other state or does
        not have its off-diagonal entry equal to minus its diagonal, if
        ``offsets`` and ``coefficients`` are not one-dimensional of equal
        length at least 2 with finite entries, if ``eps`` is not finite and
        positive, or if some copy's perturbed rate is negative.
    """
    return moments

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _channel_moves(stack):
    """Return the target state and combinatorial factor of every channel at every state."""
    import numpy as np

    count, size = stack.shape[0], stack.shape[1]
    targets = np.full((count, size), -1, dtype=np.int64)
    factors = np.zeros((count, size))
    for l in range(count):
        for i in range(size):
            off = stack[l, i].copy()
            off[i] = 0.0
            nonzero, factor = np.flatnonzero(off), -stack[l, i, i]
            if nonzero.size == 0 and factor == 0.0:
                continue
            if nonzero.size != 1 or off[nonzero[0]] <= 0.0 or abs(off[nonzero[0]] - factor) > 1e-12 * max(1.0, factor):
                raise ValueError("each channel must move a state to at most one other state")
            targets[l, i], factors[l, i] = nonzero[0], factor
    return targets, factors

def _pair_gap(stack, targets, factors, theta_x, theta_y, initial_index, values, horizon):
    """Return E[(f(X(T)) - f(Y(T)))^2] for two copies of the chain sharing every channel strip."""
    import numpy as np
    import scipy.sparse as sparse
    from scipy.sparse.linalg import expm_multiply

    size = stack.shape[1]
    xs, ys = (grid.ravel() for grid in np.meshgrid(np.arange(size), np.arange(size), indexing="ij"))
    here = xs * size + ys
    rows, cols, vals = [here], [here], [np.zeros(size * size)]
    for l in range(stack.shape[0]):
        lam_x, lam_y = theta_x[l] * factors[l, xs], theta_y[l] * factors[l, ys]
        low = np.minimum(lam_x, lam_y)
        # Below the smaller intensity both copies move; above it only the copy with the larger one moves.
        for rate, keep, dest in ((low, low > 0, targets[l, xs] * size + targets[l, ys]),
                                 (lam_x - low, lam_x > low, targets[l, xs] * size + ys),
                                 (lam_y - low, lam_y > low, xs * size + targets[l, ys])):
            rows.append(here[keep])
            cols.append(dest[keep])
            vals.append(rate[keep])
        vals[0] = vals[0] - np.maximum(lam_x, lam_y)
    pair = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(size * size, size * size))
    squared = (np.subtract.outer(values, values) ** 2).ravel()
    result = expm_multiply(float(horizon) * pair, squared)
    return float(result[int(initial_index) * size + int(initial_index)])

def _oracle_compute_stencil_moments(
    generators: "np.ndarray",
    rates: "np.ndarray",
    initial_index: int,
    observable: "np.ndarray",
    horizon: float,
    channel: int,
    offsets: "np.ndarray",
    coefficients: "np.ndarray",
    eps: float,
) -> "np.ndarray":
    """Reference implementation: single-copy moments plus the mean squared differences of every pair."""
    import numpy as np

    stack, theta, values = _validate_chain_inputs(generators, rates, initial_index, observable, horizon, channel)
    a = np.asarray(offsets, dtype=float)
    c = np.asarray(coefficients, dtype=float)
    if a.ndim != 1 or c.shape != a.shape or a.size < 2 or not (np.all(np.isfinite(a)) and np.all(np.isfinite(c))):
        raise ValueError("offsets and coefficients must be finite one-dimensional arrays of equal length at least 2")
    if isinstance(eps, bool) or not np.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and positive")
    targets, factors = _channel_moves(stack)
    shifted = []
    for offset in a:
        perturbed = theta.copy()
        perturbed[channel] = theta[channel] + offset * float(eps)
        if perturbed[channel] < 0.0:
            raise ValueError("a perturbed rate constant is negative")
        shifted.append(perturbed)
    means = np.array([_oracle_compute_observable_derivatives(generators, th, initial_index, values, horizon,
                                                             channel, 0)[0] for th in shifted])
    squares = np.array([_oracle_compute_observable_derivatives(generators, th, initial_index, values ** 2, horizon,
                                                               channel, 0)[0] for th in shifted])
    # Restricted to any two copies the joint construction is the two-copy one, and
    # f_r f_s = (f_r^2 + f_s^2 - (f_r - f_s)^2) / 2 turns every cross moment into pair quantities.
    second = float(np.sum(c ** 2 * squares))
    for r in range(a.size):
        for s in range(r + 1, a.size):
            gap = _pair_gap(stack, targets, factors, shifted[r], shifted[s], initial_index, values, horizon)
            second += float(c[r] * c[s]) * (squares[r] + squares[s] - gap)
    return np.array([float(c @ means), second])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    helpers = (
        "import numpy as np\n"
        "def _isomerization(total):\n"
        "    g = np.zeros((2, total + 1, total + 1))\n"
        "    for a in range(total + 1):\n"
        "        if a > 0:\n"
        "            g[0, a, a - 1], g[0, a, a] = a, -a\n"
        "        if a < total:\n"
        "            g[1, a, a + 1], g[1, a, a] = total - a, -(total - a)\n"
        "    return g\n"
        "def _dimerization(total):\n"
        "    g = np.zeros((2, total + 1, total + 1))\n"
        "    for c in range(total + 1):\n"
        "        if c < total:\n"
        "            g[0, c, c + 1], g[0, c, c] = (total - c) ** 2, -(total - c) ** 2\n"
        "        if c > 0:\n"
        "            g[1, c, c - 1], g[1, c, c] = c, -c\n"
        "    return g\n"
        "def _digest(v):\n"
        "    v = np.asarray(v, dtype=float)\n"
        "    if v.shape != (2,):\n"
        "        return -1.0\n"
        "    return float(50.0 + np.arcsinh(1.0e6 * v[0]) + 3.0 * np.arcsinh(1.0e6 * v[1]))\n"
    )

    raises = (
        "def _candidate():\n    try:\n        compute_stencil_moments({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n"
        "def _reference():\n    try:\n        _oracle_compute_stencil_moments({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n"
    )
    dimer = "G = _dimerization(3)\nf = np.arange(4.0)\n"
    return [
        {   # Normal: the four-point third-derivative stencil.
            "setup": helpers + dimer,
            "call": "_digest(compute_stencil_moments(G.copy(), np.array([0.6, 0.9]), 0, f.copy(), 2.5, 0, np.array([2.0, 1.0, -1.0, -2.0]), np.array([0.5, -1.0, 1.0, -0.5]), 0.1))",
            "gold_call": "_digest(_oracle_compute_stencil_moments(G.copy(), np.array([0.6, 0.9]), 0, f.copy(), 2.5, 0, np.array([2.0, 1.0, -1.0, -2.0]), np.array([0.5, -1.0, 1.0, -0.5]), 0.1))",
        },
        {   # Boundary: only two copies, one of them unperturbed.
            "setup": helpers + "G = _isomerization(2)\nf = np.arange(3.0)\n",
            "call": "_digest(compute_stencil_moments(G.copy(), np.array([0.7, 0.3]), 2, f.copy(), 1.0, 1, np.array([1.0, 0.0]), np.array([1.0, -1.0]), 0.2))",
            "gold_call": "_digest(_oracle_compute_stencil_moments(G.copy(), np.array([0.7, 0.3]), 2, f.copy(), 1.0, 1, np.array([1.0, 0.0]), np.array([1.0, -1.0]), 0.2))",
        },
        {   # Edge: coefficients that do not sum to zero.
            "setup": helpers + "G = _isomerization(3)\nf = np.arange(4.0) ** 2\n",
            "call": "_digest(compute_stencil_moments(G.copy(), np.array([0.5, 0.8]), 3, f.copy(), 1.5, 0, np.array([0.0, 1.0, -1.0]), np.array([1.0, 2.0, -0.5]), 0.3))",
            "gold_call": "_digest(_oracle_compute_stencil_moments(G.copy(), np.array([0.5, 0.8]), 3, f.copy(), 1.5, 0, np.array([0.0, 1.0, -1.0]), np.array([1.0, 2.0, -0.5]), 0.3))",
        },
        {   # Edge: a perturbed rate constant reaches zero.
            "setup": helpers + dimer,
            "call": "_digest(compute_stencil_moments(G.copy(), np.array([0.6, 0.5]), 1, f.copy(), 4.0, 1, np.array([1.0, -1.0]), np.array([0.5, -0.5]), 0.5))",
            "gold_call": "_digest(_oracle_compute_stencil_moments(G.copy(), np.array([0.6, 0.5]), 1, f.copy(), 4.0, 1, np.array([1.0, -1.0]), np.array([0.5, -0.5]), 0.5))",
        },
        {   # Edge: five copies on thirty states, a joint chain of 24 million states.
            "setup": helpers + "G = _isomerization(29)\nf = np.arange(30.0)\n",
            "call": "_digest(compute_stencil_moments(G.copy(), np.array([0.3, 0.25]), 29, f.copy(), 4.0, 0, np.array([2.0, 1.0, 0.0, -1.0, -2.0]), np.array([1.0, -4.0, 6.0, -4.0, 1.0]), 0.1))",
            "gold_call": "_digest(_oracle_compute_stencil_moments(G.copy(), np.array([0.3, 0.25]), 29, f.copy(), 4.0, 0, np.array([2.0, 1.0, 0.0, -1.0, -2.0]), np.array([1.0, -4.0, 6.0, -4.0, 1.0]), 0.1))",
        },
        {   # Edge: four copies on two hundred states, so each pair carries forty thousand states.
            "setup": helpers + "G = _isomerization(199)\nf = np.arange(200.0)\n",
            "call": "_digest(compute_stencil_moments(G.copy(), np.array([0.3, 0.25]), 199, f.copy(), 0.25, 0, np.array([2.0, 1.0, -1.0, -2.0]), np.array([0.5, -1.0, 1.0, -0.5]), 0.1))",
            "gold_call": "_digest(_oracle_compute_stencil_moments(G.copy(), np.array([0.3, 0.25]), 199, f.copy(), 0.25, 0, np.array([2.0, 1.0, -1.0, -2.0]), np.array([0.5, -1.0, 1.0, -0.5]), 0.1))",
        },
        {   # Invalid: a perturbed rate constant goes negative.
            "setup": helpers + dimer + raises.replace("{args}", "G.copy(), np.array([0.6, 0.5]), 1, f.copy(), 4.0, 1, np.array([1.0, -2.0]), np.array([0.5, -0.5]), 0.5"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        },
        {   # Invalid: a zero perturbation size.
            "setup": helpers + dimer + raises.replace("{args}", "G.copy(), np.array([0.6, 0.5]), 1, f.copy(), 4.0, 1, np.array([1.0, -1.0]), np.array([0.5, -0.5]), 0.0"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        },
        {   # Invalid: a channel row that moves one state to two others.
            "setup": helpers + "G = _isomerization(2)\nG[0, 2, 0], G[0, 2, 2] = 1.0, -3.0\nf = np.arange(3.0)\n"
                     + raises.replace("{args}", "G.copy(), np.array([0.5, 0.5]), 0, f.copy(), 1.0, 1, np.array([1.0, -1.0]), np.array([0.5, -0.5]), 0.1"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        },
    ]
