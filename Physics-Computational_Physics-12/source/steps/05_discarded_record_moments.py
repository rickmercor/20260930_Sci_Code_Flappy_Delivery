"""
Kernel-level resummation of a record with registered transitions removed at random.

Removing registered transitions from a record at random leaves a record that is again a renewal process on the same four types. Keep a registered transition of type $u$ with probability $q_u \le 1$, independently of every other event.



Discarding acts directly on the kernel. Between two surviving transitions there may be any number of discarded ones, and because each registered transition renews the process the sum over those runs is a geometric series in Laplace space. With $\hat\psi(s) = \int_0^\infty e^{-st}\psi(t)\,\mathrm{d}t$, $D_q = \mathrm{diag}(q_0,\dots,q_3)$ and $D_{1-q} = \mathrm{diag}(1-q_0,\dots,1-q_3)$,



$$\hat\psi^{q}(s) = \left[\mathbb{1} - \hat\psi(s)\,D_{1-q}\right]^{-1}\hat\psi(s)\,D_q .$$



Only the first three moments are needed, so expand $\hat\psi(s) = P - sM^{(1)} + \tfrac{1}{2}s^2 M^{(2)} + O(s^3)$, propagate that series through the matrix inverse, and read off



$$P^{q} = \hat\psi^{q}(0), \qquad M^{(1)q} = -\left.\frac{\mathrm{d}\hat\psi^{q}}{\mathrm{d}s}\right|_{s=0}, \qquad M^{(2)q} = \left.\frac{\mathrm{d}^2\hat\psi^{q}}{\mathrm{d}s^2}\right|_{s=0}.$$

Returns
-------
A NumPy array of shape `(3, 4, 4)` stacking `[P, M1, M2]` of the surviving kernel, in the same units as the input. Raise `ValueError` for a shape mismatch, a keep probability outside $(0,1]$, a singular resummation, or a non-finite entry.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Kernel-level resummation of a record with registered transitions removed at random."""

import numpy as np


def discarded_record_moments(moments, keep_probabilities):
    """Resum a four-state semi-Markov kernel under independent random discarding.

    Parameters
    ----------
    moments : array_like, shape (3, 4, 4)
        ``[P, M1, M2]`` of the kernel before any discarding.
    keep_probabilities : array_like, shape (4,)
        Probability that a registered transition of each type survives the
        discarding, in (0, 1].

    Returns
    -------
    numpy.ndarray, shape (3, 4, 4)
        ``[P, M1, M2]`` of the surviving kernel, in the same units.
    """
    return np.zeros((3, 4, 4))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_discarded_record_moments(moments, keep_probabilities):
    import numpy as np
    stack = np.asarray(moments, dtype=float)
    keep = np.asarray(keep_probabilities, dtype=float).reshape(-1)
    if stack.shape != (3, 4, 4) or not np.all(np.isfinite(stack)):
        raise ValueError("moments must be a finite array of shape (3, 4, 4)")
    if keep.size != 4 or not np.all(np.isfinite(keep)):
        raise ValueError("keep_probabilities must hold four finite numbers")
    if np.any(keep <= 0.0) or np.any(keep > 1.0):
        raise ValueError("keep probabilities must lie in (0, 1]")
    transition, first, second = stack
    kept = np.diag(keep)
    dropped = np.diag(1.0 - keep)
    a0, a1, a2 = transition, -first, 0.5 * second
    b0, b1, b2 = a0 @ dropped, a1 @ dropped, a2 @ dropped
    n0, n1, n2 = a0 @ kept, a1 @ kept, a2 @ kept
    core = np.eye(4) - b0
    if abs(np.linalg.det(core)) < 1e-14:
        raise ValueError("surviving kernel resummation is singular")
    y0 = np.linalg.inv(core)
    y1 = y0 @ b1 @ y0
    y2 = y0 @ b2 @ y0 + y0 @ b1 @ y0 @ b1 @ y0
    c0 = y0 @ n0
    c1 = y0 @ n1 + y1 @ n0
    c2 = y0 @ n2 + y1 @ n1 + y2 @ n0
    return np.stack([c0, -c1, 2.0 * c2])

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
        "weight = psi * w\n"
        "mass = weight.sum(axis=2)\n"
        "first = (weight * t).sum(axis=2)\n"
        "second = (weight * t ** 2).sum(axis=2)\n"
        "scale = mass.sum(axis=1).reshape(4, 1)\n"
        "moments = np.stack([mass / scale, first / scale, second / scale])\n"
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
    """Differential cases for the discarding resummation."""
    return [
        {
            "setup": _kernel_setup(_NET_ONE, [0.45, 0.85, 0.70, 0.50],
                                   [1.0, 0.5294117647058824, 0.7142857142857143, 1.0]),
            "call": "discarded_record_moments(moments, keep)",
            "gold_call": "_oracle_discarded_record_moments(moments, keep)",
        },
        {
            "setup": _kernel_setup(_NET_TWO, [0.62, 0.80, 0.55, 0.90], [1.0, 1.0, 1.0, 1.0]),
            "call": "discarded_record_moments(moments, keep)",
            "gold_call": "_oracle_discarded_record_moments(moments, keep)",
        },
        {
            "setup": _kernel_setup(_NET_THREE, [0.28, 0.92, 0.86, 0.40], [0.4, 0.12, 0.16, 0.35]),
            "call": "discarded_record_moments(moments, keep)",
            "gold_call": "_oracle_discarded_record_moments(moments, keep)",
        },
    ]
