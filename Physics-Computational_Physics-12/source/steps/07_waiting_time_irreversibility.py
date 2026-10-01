"""
Waiting-time irreversibility rate of a four-state semi-Markov record.

The waiting-time irreversibility rate of a record compares every ordered pair of consecutively registered transitions with its time reverse. The time reverse of the pair $(u \to v)$ is the pair $(\bar v \to \bar u)$, where the bar exchanges the two directions of the same junction, and for the record in which a registered transition of type $u$ survives with probability $q_u$,



$$\sigma_{\mathrm{WTD}} = \sum_{u,v}\int_0^\infty \nu^{q}_u\,\psi^{q}_{u\to v}(t)\,\ln\frac{\psi^{q}_{u\to v}(t)}{\psi^{q}_{\bar v\to\bar u}(t)}\,\mathrm{d}t .$$



The four pairs with $v = \bar u$ are their own reverses and cancel identically, so only the remaining twelve contribute. This requires the surviving kernel in the time domain, which obeys the renewal equation



$$\psi^{q}(t) = \psi(t)\,D_q + \int_0^t \psi(u)\,D_{1-q}\,\psi^{q}(t-u)\,\mathrm{d}u .$$



Numerical convention. Extend the tabulated densities to $t = 0$ with the short-time values, that is $\psi_{L_-\to L_+}(0) = c_{L+}$ and $\psi_{L_+\to L_-}(0) = c_{L-}$ for each junction and zero for every other density. Build a uniform working grid of exactly 401 points from $t = 0$ to the last tabulated bin center and obtain each density on it by linear interpolation of the extended tabulation. Solve the renewal equation on that grid with the trapezoidal rule, marching forward and inverting the four-by-four matrix that multiplies the unknown endpoint value. Evaluate all remaining integrals on the same grid with the trapezoidal rule, clip negative densities to zero, and floor the arguments of the logarithms.



Obtain the surviving frequencies from the surviving kernel exactly as in step 04, dividing each row of the integrated kernel by its own weight before solving for the stationary distribution.

Returns
-------
A NumPy array of shape `(5,)` holding `[nu_a_plus, nu_a_minus, nu_b_plus, nu_b_minus, wtd_rate]`, the frequencies in inverse microseconds and the irreversibility rate in Boltzmann constants per microsecond. Raise `ValueError` for a shape mismatch, a non-monotonic or non-positive time axis, a keep probability outside $(0,1]$, a row of zero weight, a non-positive mean waiting time, or a non-finite entry.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Waiting-time irreversibility rate of a four-state semi-Markov record."""

import numpy as np

_GRID_POINTS = 401
_REVERSE = (1, 0, 3, 2)


def waiting_time_irreversibility(node_times, recorded_density, short_time, keep_probabilities):
    """Evaluate the waiting-time irreversibility rate of a four-state kernel.

    Parameters
    ----------
    node_times : array_like, shape (N,)
        Bin-center times in microseconds, strictly increasing.
    recorded_density : array_like, shape (4, 4, N)
        Recorded waiting-time densities in inverse microseconds.
    short_time : array_like, shape (2, 4)
        ``[c_plus, c_minus, a_pp, a_mm]`` per junction for the same recording.
    keep_probabilities : array_like, shape (4,)
        Probability that a registered transition of each type survives the
        discarding, in (0, 1].

    Returns
    -------
    numpy.ndarray, shape (5,)
        ``[nu_a_plus, nu_a_minus, nu_b_plus, nu_b_minus, wtd_rate]``: the four
        surviving event frequencies in inverse microseconds and the
        waiting-time irreversibility rate in Boltzmann constants per
        microsecond.
    """
    return np.zeros(5)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _working_kernel(times, density, short_time):
    import numpy as np
    augmented_times = np.concatenate([[0.0], times])
    augmented = np.zeros((4, 4, augmented_times.size))
    augmented[:, :, 1:] = density
    for link in range(2):
        plus, minus = 2 * link, 2 * link + 1
        augmented[minus, plus, 0] = short_time[link, 0]
        augmented[plus, minus, 0] = short_time[link, 1]
    grid = np.linspace(0.0, augmented_times[-1], 401)
    resampled = np.zeros((4, 4, 401))
    for previous in range(4):
        for following in range(4):
            resampled[previous, following] = np.interp(
                grid, augmented_times, augmented[previous, following]
            )
    return grid, resampled


def _survivor_kernel(grid, kernel, keep):
    import numpy as np
    step = grid[1] - grid[0]
    kept = np.diag(keep)
    dropped = np.diag(1.0 - keep)
    source = np.einsum('rsn,st->nrt', kernel, kept)
    feedback = np.einsum('rsn,st->nrt', kernel, dropped)
    surviving = np.zeros_like(source)
    surviving[0] = source[0]
    core = np.eye(4) - 0.5 * step * feedback[0]
    inverse = np.linalg.inv(core)
    for index in range(1, grid.size):
        weights = np.full(index, step)
        weights[-1] = 0.5 * step
        accumulated = np.einsum(
            'k,kij,kjl->il', weights, feedback[1:index + 1], surviving[index - 1::-1]
        )
        surviving[index] = inverse @ (source[index] + accumulated)
    return np.moveaxis(surviving, 0, 2)


def _integrate(values, grid):
    import numpy as np
    spacing = np.diff(grid)
    return np.sum(0.5 * spacing * (values[..., 1:] + values[..., :-1]), axis=-1)


def _stationary(matrix):
    import numpy as np
    system = np.vstack([matrix.T - np.eye(4), np.ones(4)])
    solution, _, _, _ = np.linalg.lstsq(system, np.array([0.0, 0.0, 0.0, 0.0, 1.0]), rcond=None)
    return solution


def _oracle_waiting_time_irreversibility(node_times, recorded_density, short_time, keep_probabilities):
    import numpy as np
    times = np.asarray(node_times, dtype=float)
    density = np.asarray(recorded_density, dtype=float)
    coefficients = np.asarray(short_time, dtype=float)
    keep = np.asarray(keep_probabilities, dtype=float).reshape(-1)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("node_times must be a one-dimensional array with at least two entries")
    if density.shape != (4, 4, times.size):
        raise ValueError("recorded_density must have shape (4, 4, len(node_times))")
    if coefficients.shape != (2, 4) or keep.size != 4:
        raise ValueError("short_time must have shape (2, 4) and keep_probabilities four entries")
    if not (np.all(np.isfinite(times)) and np.all(np.isfinite(density))
            and np.all(np.isfinite(coefficients)) and np.all(np.isfinite(keep))):
        raise ValueError("all inputs must be finite")
    if np.any(np.diff(times) <= 0.0) or times[0] <= 0.0:
        raise ValueError("node_times must be positive and strictly increasing")
    if np.any(keep <= 0.0) or np.any(keep > 1.0):
        raise ValueError("keep probabilities must lie in (0, 1]")
    grid, kernel = _working_kernel(times, density, coefficients)
    surviving = _survivor_kernel(grid, kernel, keep)
    mass = _integrate(surviving, grid)
    first = _integrate(surviving * grid, grid)
    row_total = mass.sum(axis=1)
    if np.any(row_total <= 0.0):
        raise ValueError("each row of the surviving kernel must carry positive weight")
    scale = row_total.reshape(4, 1)
    weights = _stationary(mass / scale)
    mean_time = float(weights @ (first / scale).sum(axis=1))
    if mean_time <= 0.0:
        raise ValueError("mean surviving waiting time must be positive")
    frequencies = weights / mean_time
    values = np.clip(surviving, 0.0, None)
    floor = 1e-300
    total = 0.0
    for previous in range(4):
        for following in range(4):
            back_previous = (1, 0, 3, 2)[following]
            back_following = (1, 0, 3, 2)[previous]
            if back_previous == previous and back_following == following:
                continue
            forward = values[previous, following]
            reverse = values[back_previous, back_following]
            total += float(_integrate(
                frequencies[previous] * forward * np.log((forward + floor) / (reverse + floor)), grid
            ))
    return np.array([frequencies[0], frequencies[1], frequencies[2], frequencies[3], total], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def _kernel_setup(rate_rows, etas, keeps):
    return (
        "import numpy as np\n"
        "K = np.array(%s, dtype=float)\n"
        "L = K - np.diag(K.sum(axis=1))\n"
        "eta = np.array(%r, dtype=float)\n"
        "pair = [(0, 1), (1, 0), (2, 3), (3, 2)]\n"
        "S = L.copy()\n"
        "for (a, b), e in zip(pair, eta):\n"
        "    S[a, b] -= e * L[a, b]\n"
        "w = np.concatenate([np.full(6, 0.004), np.full(5, 0.03), np.full(5, 0.16), np.full(4, 0.9)])\n"
        "bounds = np.concatenate([[0.0], np.cumsum(w)])\n"
        "t = 0.5 * (bounds[:-1] + bounds[1:])\n"
        "vals, vecs = np.linalg.eig(S)\n"
        "back = np.linalg.inv(vecs)\n"
        "flow = np.array([(vecs @ np.diag(np.exp(vals * x)) @ back).real for x in bounds])\n"
        "block = (np.linalg.inv(S) @ (flow[1:] - flow[:-1])).real\n"
        "rate = np.array([e * L[a, b] for (a, b), e in zip(pair, eta)])\n"
        "psi = np.zeros((4, 4, t.size))\n"
        "for u, (au, bu) in enumerate(pair):\n"
        "    for v, (av, bv) in enumerate(pair):\n"
        "        psi[u, v] = block[:, bu, av] * rate[v]\n"
        "psi = psi / w\n"
        "short_time = np.zeros((2, 4))\n"
        "for _l in range(2):\n"
        "    _p, _m = 2 * _l, 2 * _l + 1\n"
        "    short_time[_l] = [eta[_p] * L[pair[_p]], eta[_m] * L[pair[_m]], 0.0, 0.0]\n"
        "keep = np.array(%r, dtype=float)\n"
        % (rate_rows, etas, keeps)
    )


_NET_ONE = [[0.0, 3.4, 3.8, 0.0, 2.6],
            [5.5, 0.0, 3.1, 0.0, 0.0],
            [2.7, 4.2, 0.0, 5.6, 0.0],
            [0.0, 0.0, 3.7, 0.0, 6.2],
            [7.4, 0.0, 0.0, 7.1, 0.0]]
_NET_TWO = [[0.0, 2.2, 5.0, 0.0, 3.6],
            [6.8, 0.0, 2.4, 0.0, 0.0],
            [3.6, 5.1, 0.0, 7.0, 0.0],
            [0.0, 0.0, 2.8, 0.0, 5.5],
            [6.0, 0.0, 0.0, 8.4, 0.0]]
_NET_THREE = [[0.0, 4.1, 3.2, 0.0, 4.4],
              [4.9, 0.0, 3.8, 0.0, 0.0],
              [2.5, 3.9, 0.0, 4.8, 0.0],
              [0.0, 0.0, 4.6, 0.0, 7.6],
              [5.2, 0.0, 0.0, 6.4, 0.0]]


def test_cases():
    """Differential cases for the waiting-time irreversibility rate."""
    return [
        {
            "setup": _kernel_setup(_NET_ONE, [0.45, 0.85, 0.70, 0.50],
                                   [1.0, 0.5294117647058824, 0.7142857142857143, 1.0]),
            "call": "waiting_time_irreversibility(t, psi, short_time, keep)",
            "gold_call": "_oracle_waiting_time_irreversibility(t, psi, short_time, keep)",
        },
        {
            "setup": _kernel_setup(_NET_TWO, [0.62, 0.80, 0.55, 0.90], [1.0, 1.0, 1.0, 1.0]),
            "call": "waiting_time_irreversibility(t, psi, short_time, keep)",
            "gold_call": "_oracle_waiting_time_irreversibility(t, psi, short_time, keep)",
        },
        {
            "setup": _kernel_setup(_NET_THREE, [0.28, 0.92, 0.86, 0.40], [0.4, 0.12, 0.16, 0.35]),
            "call": "waiting_time_irreversibility(t, psi, short_time, keep)",
            "gold_call": "_oracle_waiting_time_irreversibility(t, psi, short_time, keep)",
        },
    ]
