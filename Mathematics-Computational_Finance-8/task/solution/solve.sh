#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

# ORACLE SOLUTION


def integrated_kernels(y: np.ndarray, c: float) -> np.ndarray:
    import numpy as np

    if isinstance(c, bool) or not isinstance(c, (int, float, np.integer, np.floating)):
        raise ValueError("c must be a real number")
    c = float(c)
    if not np.isfinite(c) or c <= 0.0:
        raise ValueError("c must be a finite positive number")
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if y.ndim != 1 or y.size < 1:
        raise ValueError("y must be a one-dimensional array with at least one entry")
    if not np.all(np.isfinite(y)):
        raise ValueError("y must contain finite values only")

    q = y * y + c * c
    # Kernel of the source: a variant of the inverse multiquadric.
    phi = q ** -2.5
    # Odd first antiderivative, vanishing at the origin.
    phi1 = (3.0 * c * c * y + 2.0 * y ** 3) / (3.0 * c ** 4 * q ** 1.5)
    # Second antiderivative in the source's closed form. Its value at the
    # origin is 1/(3 c^3); that constant is kept because the interpolation
    # uses no polynomial augmentation.
    phi2 = (c * c + 2.0 * y * y) / (3.0 * c ** 4 * np.sqrt(q))
    return np.column_stack([phi, phi1, phi2]).astype(float)

# ORACLE SOLUTION


def integrated_kernel_weights(nodes: np.ndarray, centre: float, c: float) -> np.ndarray:
    import numpy as np

    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 1 or nodes.size < 2:
        raise ValueError("nodes must be a one-dimensional array with at least two entries")
    if not np.all(np.isfinite(nodes)):
        raise ValueError("nodes must be finite")
    if np.unique(nodes).size != nodes.size:
        raise ValueError("nodes must be distinct")
    if isinstance(centre, bool) or not np.isfinite(float(centre)):
        raise ValueError("centre must be finite")
    if isinstance(c, bool) or not np.isfinite(float(c)) or float(c) <= 0.0:
        raise ValueError("c must be a finite positive number")
    centre = float(centre)
    c = float(c)

    m = nodes.size
    # Interpolation matrix built from the second antiderivative (step 01).
    offsets = (nodes[None, :] - nodes[:, None]).ravel()
    matrix = integrated_kernels(offsets, c)[:, 2].reshape(m, m)
    # Right-hand sides: the first and second derivatives of each translate
    # phi2(. - y_k) at the evaluation point are phi1 and phi.
    rhs = integrated_kernels(centre - nodes, c)
    w1 = np.linalg.solve(matrix, rhs[:, 1])
    w2 = np.linalg.solve(matrix, rhs[:, 0])
    return np.vstack([w1, w2]).astype(float)

# ORACLE SOLUTION


def analytic_interior_weights(h: float, c: float) -> np.ndarray:
    import numpy as np

    for name, value in (("h", h), ("c", c)):
        if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
    h = float(h)
    c = float(c)

    # First derivative, negative-side weights; the positive side is their
    # negative and the centre weight vanishes.
    b3 = -11192299.0 / 71680.0 * h ** 3 / c ** 4 + 3953.0 / 3360.0 * h / c ** 2 - 1.0 / (60.0 * h)
    b2 = 12571299.0 / 17920.0 * h ** 3 / c ** 4 - 3953.0 / 840.0 * h / c ** 2 + 3.0 / (20.0 * h)
    b1 = -13398699.0 / 14336.0 * h ** 3 / c ** 4 + 3953.0 / 672.0 * h / c ** 2 - 3.0 / (4.0 * h)
    first = np.array([b3, b2, b1, 0.0, -b1, -b2, -b3])

    # Second derivative, symmetric about the centre.
    s3 = (149380759993.0 * h ** 2 - 704267904.0 * c ** 2) / (371589120.0 * c ** 4) + 1.0 / (90.0 * h ** 2)
    s2 = (704267904.0 * c ** 2 - 162082276345.0 * h ** 2) / (61931520.0 * c ** 4) - 3.0 / (20.0 * h ** 2)
    s1 = (848515930781.0 * h ** 2 - 3521339520.0 * c ** 2) / (123863040.0 * c ** 4) + 3.0 / (2.0 * h ** 2)
    s0 = (3521339520.0 * c ** 2 - 861217447133.0 * h ** 2) / (92897280.0 * c ** 4) - 49.0 / (18.0 * h ** 2)
    second = np.array([s3, s2, s1, s0, s1, s2, s3])
    return np.vstack([first, second]).astype(float)

# ORACLE SOLUTION


def assemble_differentiation_matrices(n_nodes: int, s_max: float, c_over_h: float) -> np.ndarray:
    import numpy as np

    if isinstance(n_nodes, bool) or not isinstance(n_nodes, (int, np.integer)) or int(n_nodes) < 7:
        raise ValueError("n_nodes must be an integer >= 7")
    for name, value in (("s_max", s_max), ("c_over_h", c_over_h)):
        if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
    n = int(n_nodes)
    grid = np.linspace(0.0, float(s_max), n)
    h = grid[1] - grid[0]
    c = float(c_over_h) * h

    mats = np.zeros((2, n, n))
    centred = analytic_interior_weights(h, c)
    for i in range(1, n - 1):
        if 3 <= i <= n - 4:
            idx = np.arange(i - 3, i + 4)
            w = centred
        else:
            idx = np.arange(0, 7) if i < 3 else np.arange(n - 7, n)
            w = integrated_kernel_weights(grid[idx], grid[i], c)
        mats[0, i, idx] = w[0]
        mats[1, i, idx] = w[1]
    return mats

# ORACLE SOLUTION


def l1_caputo_coefficients(alpha: float, n_level: int) -> np.ndarray:
    import numpy as np
    from math import gamma

    if isinstance(alpha, bool) or not np.isfinite(float(alpha)) or not (0.0 < float(alpha) <= 1.0):
        raise ValueError("alpha must lie in (0, 1]")
    if isinstance(n_level, bool) or not isinstance(n_level, (int, np.integer)) or int(n_level) < 1:
        raise ValueError("n_level must be an integer >= 1")
    alpha = float(alpha)
    n = int(n_level)
    p = 1.0 - alpha

    # j**(1-alpha) with 0**(1-alpha) read as its limit 0 for all alpha in (0,1].
    j = np.arange(0, n + 2, dtype=float)
    pw = np.where(j > 0.0, j ** p, 0.0)

    coeff = np.empty(n + 1)
    coeff[0] = 1.0
    if n >= 2:
        k = np.arange(1, n)
        coeff[1:n] = pw[k - 1] - 2.0 * pw[k] + pw[k + 1]
    coeff[n] = pw[n - 1] - pw[n]
    return coeff / gamma(2.0 - alpha)

# ORACLE SOLUTION


def fractional_bs_operator(s_grid: np.ndarray, diff_matrices: np.ndarray, sigma: float,
                                   rate: float, dividend: float) -> np.ndarray:
    import numpy as np

    s = np.asarray(s_grid, dtype=float)
    if s.ndim != 1 or s.size < 3 or not np.all(np.isfinite(s)) or np.any(np.diff(s) <= 0.0):
        raise ValueError("s_grid must be a finite, strictly increasing array of at least 3 nodes")
    n = s.size
    d = np.asarray(diff_matrices, dtype=float)
    if d.shape != (2, n, n) or not np.all(np.isfinite(d)):
        raise ValueError("diff_matrices must be a finite array of shape (2, N, N)")
    for name, value in (("sigma", sigma), ("rate", rate), ("dividend", dividend)):
        if isinstance(value, bool) or not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(sigma) < 0.0:
        raise ValueError("sigma must be non-negative")

    op = (0.5 * float(sigma) ** 2 * (s ** 2)[:, None] * d[1]
          + (float(rate) - float(dividend)) * s[:, None] * d[0]
          - float(rate) * np.eye(n))
    # End rows are replaced by boundary data in the time stepper.
    op[0, :] = 0.0
    op[-1, :] = 0.0
    return op

# ORACLE SOLUTION


def l1_time_march(operator: np.ndarray, initial_values: np.ndarray, left_values: np.ndarray,
                          right_values: np.ndarray, alpha: float, maturity: float) -> np.ndarray:
    import numpy as np

    op = np.asarray(operator, dtype=float)
    u0 = np.asarray(initial_values, dtype=float)
    left = np.asarray(left_values, dtype=float)
    right = np.asarray(right_values, dtype=float)
    if u0.ndim != 1 or u0.size < 3:
        raise ValueError("initial_values must be one-dimensional with at least 3 entries")
    n_nodes = u0.size
    if op.shape != (n_nodes, n_nodes):
        raise ValueError("operator must have shape (N, N)")
    if left.ndim != 1 or left.shape != right.shape or left.size < 2:
        raise ValueError("boundary arrays must be one-dimensional of equal length n + 1 >= 2")
    for arr in (op, u0, left, right):
        if not np.all(np.isfinite(arr)):
            raise ValueError("inputs must be finite")
    if isinstance(alpha, bool) or not np.isfinite(float(alpha)) or not (0.0 < float(alpha) <= 1.0):
        raise ValueError("alpha must lie in (0, 1]")
    if isinstance(maturity, bool) or not np.isfinite(float(maturity)) or float(maturity) <= 0.0:
        raise ValueError("maturity must be positive")

    alpha = float(alpha)
    n_steps = left.size - 1
    delta = float(maturity) / n_steps
    scale = delta ** (-alpha)

    history = np.empty((n_steps + 1, n_nodes))
    history[0] = u0
    eye = np.eye(n_nodes)
    for m in range(1, n_steps + 1):
        coeff = l1_caputo_coefficients(alpha, m)
        # Past levels U^{m-1}, ..., U^0 weighted by l_{1,m}, ..., l_{m,m}.
        rhs = -scale * (coeff[1:] @ history[m - 1::-1])
        system = scale * coeff[0] * eye - op
        system[0, :] = 0.0
        system[0, 0] = 1.0
        system[-1, :] = 0.0
        system[-1, -1] = 1.0
        rhs[0] = left[m]
        rhs[-1] = right[m]
        history[m] = np.linalg.solve(system, rhs)
    return history[n_steps].astype(float)

# ORACLE SOLUTION


def price_at_strike(strike: float = 100.0, rate: float = 0.05, dividend: float = 0.0,
                            sigma: float = 0.4, alpha: float = 0.8, maturity: float = 1.0,
                            n_nodes: int = 61, n_steps: int = 400, smax_factor: float = 3.0,
                            c_over_h: float = 4.0) -> float:
    import numpy as np

    for name, value in (("strike", strike), ("sigma", sigma), ("maturity", maturity)):
        if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
    for name, value in (("rate", rate), ("dividend", dividend)):
        if isinstance(value, bool) or not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if isinstance(smax_factor, bool) or not np.isfinite(float(smax_factor)) or float(smax_factor) <= 1.0:
        raise ValueError("smax_factor must exceed 1")
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or int(n_steps) < 1:
        raise ValueError("n_steps must be an integer >= 1")

    k = float(strike)
    s_max = float(smax_factor) * k
    # -- Sub-problem 04: the seven-band differentiation matrices of the
    # integrated-kernel scheme.
    mats = assemble_differentiation_matrices(n_nodes, s_max, c_over_h)
    n = int(n_nodes)
    grid = np.linspace(0.0, s_max, n)
    h = grid[1] - grid[0]
    c = float(c_over_h) * h
    i_strike = int(round(k / h))
    if abs(grid[i_strike] - k) > 1e-9 * k:
        raise ValueError("the strike must fall on a grid node")

    # -- Sub-problems 03, 02 and 01: consistency gate on the assembled rows.
    # The centred rows must carry the closed-form weights, the rows next to
    # the boundaries the integrated-kernel weights, and those weights must
    # satisfy the exactness conditions on the kernel translates.
    centred = analytic_interior_weights(h, c)
    edge_nodes = grid[:7]
    edge = integrated_kernel_weights(edge_nodes, grid[1], c)
    if n >= 7 and not (np.allclose(mats[:, 1, :7], edge, rtol=1e-12, atol=0.0)):
        raise ValueError("one-sided rows disagree with the integrated-kernel weights")
    if n >= 7 and 3 <= n - 4 and not np.allclose(mats[:, 3, 0:7], centred, rtol=1e-12, atol=0.0):
        raise ValueError("centred rows disagree with the closed-form weights")
    kern = integrated_kernels((edge_nodes[None, :] - edge_nodes[:, None]).ravel(), c)[:, 2].reshape(7, 7)
    target = integrated_kernels(grid[1] - edge_nodes, c)
    residual = np.max(np.abs(kern @ edge[1] - target[:, 0])) / np.max(np.abs(target[:, 0]))
    if not residual < 1e-6:
        raise ValueError("integrated-kernel exactness conditions are not met")

    # -- Sub-problem 06: the semi-discrete spatial operator.
    op = fractional_bs_operator(grid, mats, sigma, rate, dividend)

    # -- Sub-problem 05: the L1 coefficients must sum to zero (a constant state
    # has zero Caputo derivative) before they are used in the march.
    last = l1_caputo_coefficients(alpha, int(n_steps))
    if abs(float(np.sum(last))) > 1e-10 * float(np.max(np.abs(last))):
        raise ValueError("L1 coefficients do not sum to zero")

    # Payoff at tau = 0 and call boundary data at every level.
    tau = np.linspace(0.0, float(maturity), int(n_steps) + 1)
    u0 = np.maximum(grid - k, 0.0)
    left = np.zeros_like(tau)
    right = s_max * np.exp(-float(dividend) * tau) - k * np.exp(-float(rate) * tau)

    # -- Sub-problem 07: implicit L1 march to maturity.
    terminal = l1_time_march(op, u0, left, right, alpha, maturity)
    return float(terminal[i_strike])
SCICODE_GOLD_EOF
