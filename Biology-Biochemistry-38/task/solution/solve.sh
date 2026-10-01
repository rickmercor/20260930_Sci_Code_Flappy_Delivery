#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np
def _processive_reactions(n_sites):
    """Return reactant counts and state changes of the 4n + 2 channels, rows in rate-vector order."""
    import numpy as np

    n = int(n_sites)
    s_0, s_n, kinase, phosphatase = 0, 1, 2, 3

    def _bound_k(i):  # Column of S_iK, i = 0, ..., n - 1.
        return 4 + i

    def _bound_f(j):  # Column of S_jF, j = 1, ..., n.
        return 3 + n + j

    channels = [((s_0, kinase), (_bound_k(0),)), ((_bound_k(0),), (s_0, kinase))]
    for i in range(1, n):
        channels += [((_bound_k(i - 1),), (_bound_k(i),)), ((_bound_k(i),), (_bound_k(i - 1),))]
    channels.append(((_bound_k(n - 1),), (s_n, kinase)))
    channels.append(((_bound_f(1),), (s_0, phosphatase)))
    for i in range(1, n):
        channels += [((_bound_f(i),), (_bound_f(i + 1),)), ((_bound_f(i + 1),), (_bound_f(i),))]
    channels += [((_bound_f(n),), (s_n, phosphatase)), ((s_n, phosphatase), (_bound_f(n),))]
    reactants = np.zeros((len(channels), 2 * n + 4), dtype=np.int64)
    change = np.zeros_like(reactants)
    for row, (inputs, outputs) in enumerate(channels):
        for species in inputs:
            reactants[row, species] += 1
            change[row, species] -= 1
        for species in outputs:
            change[row, species] += 1
    return reactants, change

def enumerate_processive_states(n_sites: int, initial_state: "np.ndarray") -> "np.ndarray":
    """Reference implementation: depth-first closure under the firing channels."""
    import numpy as np

    if isinstance(n_sites, bool) or not isinstance(n_sites, (int, np.integer)) or n_sites < 1:
        raise ValueError("n_sites must be an integer of at least 1")
    start = np.asarray(initial_state)
    if start.ndim != 1 or start.shape[0] != 2 * int(n_sites) + 4:
        raise ValueError("initial_state must have 2 * n_sites + 4 entries")
    if not np.all(np.isfinite(start.astype(float))) or np.any(start.astype(float) != np.round(start.astype(float))):
        raise ValueError("initial_state must hold integer copy numbers")
    if np.any(start < 0):
        raise ValueError("initial_state must be non-negative")
    reactants, change = _processive_reactions(n_sites)
    first = tuple(int(v) for v in start)
    seen, stack = {first}, [first]
    while stack:
        state = np.array(stack.pop(), dtype=np.int64)
        for row in range(reactants.shape[0]):
            if np.all(state >= reactants[row]):
                successor = tuple(int(v) for v in state + change[row])
                if successor not in seen:
                    seen.add(successor)
                    stack.append(successor)
    return np.array(sorted(seen), dtype=np.int64)

import numpy as np
def build_channel_generators(n_sites: int, states: "np.ndarray") -> "np.ndarray":
    """Reference implementation: one sparse row per state and channel."""
    import numpy as np

    if isinstance(n_sites, bool) or not isinstance(n_sites, (int, np.integer)) or n_sites < 1:
        raise ValueError("n_sites must be an integer of at least 1")
    table = np.asarray(states)
    if table.ndim != 2 or table.shape[1] != 2 * int(n_sites) + 4 or table.shape[0] < 1:
        raise ValueError("states must have shape (M, 2 * n_sites + 4)")
    as_float = table.astype(float)
    if not np.all(np.isfinite(as_float)) or np.any(as_float != np.round(as_float)) or np.any(as_float < 0):
        raise ValueError("states must hold non-negative integers")
    table = table.astype(np.int64)
    index = {tuple(int(v) for v in row): i for i, row in enumerate(table)}
    if len(index) != table.shape[0]:
        raise ValueError("states must be distinct")
    reactants, change = _processive_reactions(n_sites)
    size = table.shape[0]
    generators = np.zeros((reactants.shape[0], size, size))
    for channel in range(reactants.shape[0]):
        for i, state in enumerate(table):
            factor = 1.0
            for species in np.flatnonzero(reactants[channel]):
                for k in range(int(reactants[channel, species])):
                    factor *= float(max(int(state[species]) - k, 0))
            if factor == 0.0:
                continue
            target = index.get(tuple(int(v) for v in state + change[channel]))
            if target is None:
                raise ValueError("a firing channel leads out of the supplied states")
            generators[channel, i, target] += factor
            generators[channel, i, i] -= factor
    return generators

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

def compute_observable_derivatives(
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

import numpy as np
def build_stencil_weights(offsets: "np.ndarray", derivative_order: int) -> "np.ndarray":
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
    means = np.array([compute_observable_derivatives(generators, th, initial_index, values, horizon,
                                                             channel, 0)[0] for th in shifted])
    squares = np.array([compute_observable_derivatives(generators, th, initial_index, values ** 2, horizon,
                                                               channel, 0)[0] for th in shifted])
    # Restricted to any two copies the joint construction is the two-copy one, and
    # f_r f_s = (f_r^2 + f_s^2 - (f_r - f_s)^2) / 2 turns every cross moment into pair quantities.
    second = float(np.sum(c ** 2 * squares))
    for r in range(a.size):
        for s in range(r + 1, a.size):
            gap = _pair_gap(stack, targets, factors, shifted[r], shifted[s], initial_index, values, horizon)
            second += float(c[r] * c[s]) * (squares[r] + squares[s] - gap)
    return np.array([float(c @ means), second])

import numpy as np
def minimize_stencil_rmse(
    generators: "np.ndarray",
    rates: "np.ndarray",
    initial_index: int,
    observable: "np.ndarray",
    horizon: float,
    channel: int,
    offsets: "np.ndarray",
    coefficients: "np.ndarray",
    derivative_order: int,
    replications: float,
    target: float,
    eps_bounds: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation: logarithmic scan followed by bounded Brent refinement."""
    import numpy as np
    from scipy.optimize import minimize_scalar

    if isinstance(derivative_order, bool) or not isinstance(derivative_order, (int, np.integer)) or derivative_order < 1:
        raise ValueError("derivative_order must be an integer of at least 1")
    if not np.isfinite(float(replications)) or float(replications) <= 0.0 or not np.isfinite(float(target)):
        raise ValueError("replications must be finite and positive and target finite")
    bounds = np.asarray(eps_bounds, dtype=float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or not 0.0 < bounds[0] < bounds[1]:
        raise ValueError("eps_bounds must be an increasing pair of finite positive numbers")
    k, n = int(derivative_order), float(replications)

    def _mse(eps):
        mean, second = compute_stencil_moments(generators, rates, initial_index, observable, horizon,
                                                       channel, offsets, coefficients, float(eps))
        return (second - mean ** 2) / (n * eps ** (2 * k)) + (mean / eps ** k - float(target)) ** 2

    grid = np.geomspace(bounds[0], bounds[1], 17)
    scan = np.array([_mse(eps) for eps in grid])
    best = int(np.argmin(scan))
    left, right = grid[max(best - 1, 0)], grid[min(best + 1, grid.size - 1)]
    refined = minimize_scalar(_mse, bounds=(left, right), method="bounded", options={"xatol": 1e-10 * right})
    eps_opt, mse_min = (refined.x, refined.fun) if refined.fun <= scan[best] else (grid[best], scan[best])
    return np.array([float(eps_opt), float(np.sqrt(mse_min))])

import numpy as np
def estimate_minimum_relative_rmse(
    n_sites: int = 5,
    initial_state: tuple = (3, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    rates: tuple = (0.02, 0.2, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 2.0,
                    1.5, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.15, 0.015),
    channel: int = 2,
    horizon: float = 30.0,
    offsets: tuple = (2.0, 1.0, -1.0, -2.0),
    derivative_order: int = 3,
    path_budget: int = 4096,
    eps_bounds: tuple = (0.01, 0.5),
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    start_state = np.asarray(initial_state)
    states = enumerate_processive_states(n_sites, start_state)
    generators = build_channel_generators(n_sites, states)
    theta = np.asarray(rates, dtype=float)
    if theta.shape != (generators.shape[0],):
        raise ValueError("rates must have 4 * n_sites + 2 entries")
    start = int(np.flatnonzero(np.all(states == start_state.astype(np.int64), axis=1))[0])
    observable = states[:, 0].astype(float)
    k = derivative_order
    target = compute_observable_derivatives(generators, theta, start, observable, horizon, channel, k)[k]
    if target == 0.0:
        raise ValueError("the target derivative is zero")
    paths = len(offsets)
    if isinstance(path_budget, bool) or not isinstance(path_budget, (int, np.integer)) or path_budget <= 0 \
            or path_budget % paths:
        raise ValueError("path_budget must be a positive multiple of the number of stencil paths")
    a = np.asarray(offsets, dtype=float)
    c = build_stencil_weights(a, k)
    replications = float(path_budget // paths)
    eps_opt, _ = minimize_stencil_rmse(generators, theta, start, observable, horizon, channel, a, c, k,
                                               replications, target, np.asarray(eps_bounds, dtype=float))
    # The error is re-assembled at the optimum from the numerator's own mean and second moment.
    mean, second = compute_stencil_moments(generators, theta, start, observable, horizon, channel, a, c,
                                                   float(eps_opt))
    mse = (second - mean ** 2) / (replications * eps_opt ** (2 * k)) + (mean / eps_opt ** k - target) ** 2
    return float(np.sqrt(mse) / abs(target))
SCICODE_GOLD_EOF
