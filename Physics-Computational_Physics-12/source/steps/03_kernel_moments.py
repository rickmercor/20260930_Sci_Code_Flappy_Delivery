"""
Row-normalized zeroth, first and second time moments of a recorded kernel.

The registered record is a semi-Markov process whose four states are the four registered transition types. It is completely characterized by the kernel $\psi_{u\to v}(t)$, and every quantity used later depends on the kernel only through its first three time moments. Because each tabulated value is the mean density over its bin and every event in a bin is dated by that bin center, the exact histogram sums are



$$P_{uv} = \frac{1}{Z_u}\sum_l \psi_{uv,l}\,w_l, \qquad M^{(1)}_{uv} = \frac{1}{Z_u}\sum_l \psi_{uv,l}\,w_l\,t_l, \qquad M^{(2)}_{uv} = \frac{1}{Z_u}\sum_l \psi_{uv,l}\,w_l\,t_l^2,$$



with the row weight $Z_u = \sum_v \sum_l \psi_{uv,l}\,w_l$. Dividing by $Z_u$ enforces the normalization $\sum_v P_{uv} = 1$ that a waiting-time kernel must satisfy. Use the bin-center time $t_l$ in the moment weights, use the tabulated bins as the entire support, and add no continuation beyond the last bin.



$P$ is the transition matrix of the embedded chain of registered transition types, $M^{(1)}$ carries the mean waiting times and $M^{(2)}$ their second moments.

Returns
-------
A NumPy array of shape `(3, 4, 4)` stacking `[P, M1, M2]` in that order, dimensionless for `P`, microseconds for `M1` and microseconds squared for `M2`. Raise `ValueError` for a shape mismatch, a non-positive bin width, a negative density, a non-monotonic time axis, a non-finite entry, or a row of zero weight.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Row-normalized zeroth, first and second time moments of a recorded kernel."""

import numpy as np


def kernel_moments(node_times, bin_widths, recorded_density):
    """Reduce a binned four-state waiting-time kernel to its first three moments.

    Parameters
    ----------
    node_times : array_like, shape (N,)
        Bin-center times in microseconds, strictly increasing.
    bin_widths : array_like, shape (N,)
        Positive widths of the same bins in microseconds.
    recorded_density : array_like, shape (4, 4, N)
        Recorded waiting-time densities in inverse microseconds, indexed as
        ``[previous transition type, next transition type, bin]``.

    Returns
    -------
    numpy.ndarray, shape (3, 4, 4)
        ``[P, M1, M2]``: the embedded transition matrix of the registered
        transition types and the first and second time moments of the same
        kernel, in microseconds and microseconds squared, each row divided by
        the total recorded weight of that row.
    """
    return np.zeros((3, 4, 4))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_kernel_moments(node_times, bin_widths, recorded_density):
    import numpy as np
    times = np.asarray(node_times, dtype=float)
    widths = np.asarray(bin_widths, dtype=float)
    density = np.asarray(recorded_density, dtype=float)
    if times.ndim != 1 or widths.shape != times.shape:
        raise ValueError("node_times and bin_widths must be one-dimensional arrays of equal length")
    if density.shape != (4, 4, times.size):
        raise ValueError("recorded_density must have shape (4, 4, len(node_times))")
    if not (np.all(np.isfinite(times)) and np.all(np.isfinite(widths)) and np.all(np.isfinite(density))):
        raise ValueError("all inputs must be finite")
    if np.any(widths <= 0.0) or np.any(density < 0.0) or np.any(np.diff(times) <= 0.0):
        raise ValueError("bin widths must be positive, densities non-negative, node times increasing")
    weight = density * widths
    mass = weight.sum(axis=2)
    first = (weight * times).sum(axis=2)
    second = (weight * times ** 2).sum(axis=2)
    row_total = mass.sum(axis=1)
    if np.any(row_total <= 0.0):
        raise ValueError("each row of the recorded kernel must carry positive weight")
    scale = row_total.reshape(4, 1)
    return np.stack([mass / scale, first / scale, second / scale])

# =============================================================================
# TEST CASES
# =============================================================================

def _instance(rate_rows, etas):
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
        % (rate_rows, etas)
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
    """Differential cases for the binned moment reduction."""
    return [
        {
            "setup": _instance(_NET_ONE, [0.45, 0.85, 0.70, 0.50]),
            "call": "kernel_moments(t, w, psi)",
            "gold_call": "_oracle_kernel_moments(t, w, psi)",
        },
        {
            "setup": _instance(_NET_TWO, [0.62, 0.80, 0.55, 0.90]),
            "call": "kernel_moments(t, w, psi)",
            "gold_call": "_oracle_kernel_moments(t, w, psi)",
        },
        {
            "setup": _instance(_NET_THREE, [1.0, 1.0, 1.0, 1.0]),
            "call": "kernel_moments(t, w, psi)",
            "gold_call": "_oracle_kernel_moments(t, w, psi)",
        },
    ]
