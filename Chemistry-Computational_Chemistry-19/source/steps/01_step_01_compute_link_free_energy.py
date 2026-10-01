"""
Evaluate the free energy of a displacement-controlled chain with one breakable segment as a function of that segment's length.

With both chain ends held fixed, the length of the breakable segment is an internal coordinate whose free energy combines the bond energy of that segment with the orientationally averaged entropy of the rigid remainder of the chain. Marginalizing the three-dimensional breaking-segment vector at fixed length contributes the spherical radial factor $x_bar**2$ . The dimensionless free energy contains the term $-2 * ln(x_bar)$ in addition to the bond and orientational terms.

Returns
-------
np.ndarray: free energy in k_B T at each breakable-segment length (additive constant depends only on n_segments), +inf where the rigid segments cannot close the chain.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_link_free_energy(x_bar: "np.ndarray", y_bar: float, n_segments: int, bond_energy: float) -> "np.ndarray":
    """Return the free energy profile along the length of the breakable segment.

    Lengths are in Kuhn lengths and energies in k_B T. The chain has
    ``n_segments`` freely jointed segments. One of them is breakable, with the
    12-6 Lennard-Jones energy ``bond_energy * (x**-12 - 2 * x**-6)`` of its
    length ``x``. The other ``m = n_segments - 1`` segments are rigid. Their
    combined end-to-end vector, which joins the breakable segment to the fixed
    end-to-end vector of length ``y_bar``, has length ``r`` and carries the
    Kuhn-Grun free energy ``m * (F / tanh(F) + ln(F / sinh(F)))`` with
    ``F = e * (3 - e**2) / (1 - e**2)`` and ``e = r / m``. The rigid segments
    cannot reach ``r >= m``. The scalar-length marginal includes the spherical
    Jacobian ``x_bar**2``, so the dimensionless free energy includes
    ``-2 * ln(x_bar)`` in addition to the Lennard-Jones and orientational terms.

    The returned array is the free energy of the chain as a function of the
    length ``x_bar`` of the breakable segment at fixed ``y_bar``. It is defined
    up to one additive constant that may depend on ``n_segments`` but not on
    ``x_bar``, ``y_bar`` or ``bond_energy``. Finite entries must be accurate to
    1e-9, including near full extension where the rigid-segment Boltzmann
    weight is smaller than the smallest positive double.

    Parameters
    ----------
    x_bar : np.ndarray
        One-dimensional array of positive breakable-segment lengths.
    y_bar : float
        Fixed end-to-end distance, at least 0.
    n_segments : int
        Number of Kuhn segments, at least 2.
    bond_energy : float
        Lennard-Jones well depth in k_B T, positive.

    Returns
    -------
    free_energy : np.ndarray
        Free energy for each entry of ``x_bar`` (same shape), equal to
        ``+inf`` where ``abs(y_bar - x_bar) >= n_segments - 1``.

    Raises
    ------
    ValueError
        If ``n_segments`` is not an integer of at least 2, if ``bond_energy``
        is not positive, if ``y_bar`` is negative, or if ``x_bar`` is not a
        one-dimensional array of positive values.
    """
    return free_energy

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_link_free_energy(x_bar: "np.ndarray", y_bar: float, n_segments: int, bond_energy: float) -> "np.ndarray":
    """Reference implementation using stable log-space radial quadrature."""
    import numpy as np

    x = np.asarray(x_bar, dtype=float)
    if (
        isinstance(n_segments, bool)
        or not isinstance(n_segments, (int, np.integer))
        or n_segments < 2
        or not bond_energy > 0.0
        or not y_bar >= 0.0
        or x.ndim != 1
        or x.size == 0
        or not np.all(x > 0.0)
    ):
        raise ValueError(
            "need integer n_segments >= 2, bond_energy > 0, "
            "y_bar >= 0 and positive 1-D x_bar"
        )

    m = int(n_segments) - 1
    y = float(y_bar)
    nodes_1d, weights_1d = np.polynomial.legendre.leggauss(16)
    drops = np.array([
        0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0,
        14.0, 19.0, 25.0, 32.0, 40.0, 50.0
    ])

    def _rigid_energy(r):
        gap = (m - r) / m
        e = r / m
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            force = e * (3.0 - e * e) / (gap * (2.0 - gap))
            small = force < 1e-4
            safe = np.where(small, 1.0, force)
            h = np.where(
                small,
                1.0 + force * force / 6.0,
                np.log(2.0 * safe)
                + 2.0 * safe / np.expm1(2.0 * safe)
                - np.log1p(-np.exp(-2.0 * safe)),
            )
        return np.where(gap > 0.0, m * h, np.inf)

    def _log_weight(r):
        with np.errstate(divide="ignore", invalid="ignore"):
            return (
                np.where(
                    r > 0.0,
                    np.log(np.where(r > 0.0, r, 1.0)),
                    -np.inf,
                )
                - _rigid_energy(r)
            )

    probe = np.linspace(0.0, m, 801)[1:-1]
    r_mode = probe[int(np.argmax(_log_weight(probe)))]

    def _log_integral(lo, hi):
        peak_r = np.clip(r_mode, lo, hi)
        target = _log_weight(peak_r)[:, None] - drops[None, :]
        up_in = np.repeat(peak_r[:, None], drops.size, 1)
        up_out = np.repeat(hi[:, None], drops.size, 1)
        down_out = np.repeat(lo[:, None], drops.size, 1)
        down_in = np.repeat(peak_r[:, None], drops.size, 1)

        for _ in range(14):
            mid = 0.5 * (up_in + up_out)
            keep = _log_weight(mid) >= target
            up_in, up_out = (
                np.where(keep, mid, up_in),
                np.where(keep, up_out, mid),
            )
            mid = 0.5 * (down_out + down_in)
            keep = _log_weight(mid) >= target
            down_out, down_in = (
                np.where(keep, down_out, mid),
                np.where(keep, mid, down_in),
            )

        edges = np.sort(
            np.concatenate(
                [lo[:, None], down_in, peak_r[:, None], up_in, hi[:, None]],
                axis=1,
            ),
            axis=1,
        )
        left, right = edges[:, :-1], edges[:, 1:]
        nodes = (
            0.5 * (left + right)[..., None]
            + 0.5 * (right - left)[..., None] * nodes_1d
        )
        weights = 0.5 * (right - left)[..., None] * weights_1d
        values = _log_weight(nodes)
        top = np.max(
            np.where(weights > 0.0, values, -np.inf),
            axis=(1, 2),
        )
        with np.errstate(invalid="ignore", over="ignore"):
            total = np.sum(
                np.where(
                    weights > 0.0,
                    weights * np.exp(values - top[:, None, None]),
                    0.0,
                ),
                axis=(1, 2),
            )
        return top + np.log(total)

    bond = bond_energy * (x ** -12.0 - 2.0 * x ** -6.0)
    out = np.full(x.shape, np.inf)

    if y == 0.0:
        reach = x < m
        out[reach] = (
            bond[reach]
            - 2.0 * np.log(x[reach])
            - np.log(2.0)
            + _rigid_energy(x[reach])
        )
        return out

    reach = np.abs(y - x) < m
    if np.any(reach):
        xs = x[reach]
        lo = np.abs(y - xs)
        hi = np.minimum(xs + y, m)
        collapsed = hi <= lo
        log_angular = np.empty_like(xs)

        if np.any(~collapsed):
            log_angular[~collapsed] = (
                _log_integral(lo[~collapsed], hi[~collapsed])
                - np.log(xs[~collapsed] * y)
            )

        if np.any(collapsed):
            log_angular[collapsed] = (
                np.log(2.0) - _rigid_energy(xs[collapsed])
            )

        out[reach] = bond[reach] - 2.0 * np.log(xs) - log_angular

    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    setup = """import numpy as np
def _profile(fn, values, y, n, b):
    x = np.array(values, dtype=float)
    a = np.asarray(fn(x.copy(), y, n, b), dtype=float)
    if a.shape != x.shape or np.any(np.isnan(a)) or np.any(np.isneginf(a)):
        raise AssertionError("Expected a matching vector of finite values or +inf")
    finite = np.isfinite(a)
    centered = np.zeros_like(a)
    if np.any(finite):
        centered[finite] = a[finite] - a[np.flatnonzero(finite)[0]]
    return np.concatenate((finite.astype(float), centered))
"""
    return [
        {
            "setup": setup,
            "tol": 2e-9,
            "call": "_profile(compute_link_free_energy, [1.0, 2.06, 5.0], 29.0, 51, 35.0)",
            "gold_call": "_profile(_oracle_compute_link_free_energy, [1.0, 2.06, 5.0], 29.0, 51, 35.0)",
        },
        {
            "setup": setup,
            "tol": 2e-9,
            "call": "_profile(compute_link_free_energy, [1.0, 5.5, 12.0], 0.0, 51, 35.0)",
            "gold_call": "_profile(_oracle_compute_link_free_energy, [1.0, 5.5, 12.0], 0.0, 51, 35.0)",
        },
        {
            "setup": setup,
            "tol": 2e-9,
            "call": "_profile(compute_link_free_energy, [1.0, 5.5, 12.0], 1e-20, 51, 35.0)",
            "gold_call": "_profile(_oracle_compute_link_free_energy, [1.0, 5.5, 12.0], 1e-20, 51, 35.0)",
        },
        {
            "setup": setup,
            "tol": 2e-9,
            "call": "_profile(compute_link_free_energy, [1.02, 1.3, 2.0], 395.0, 401, 50.0)",
            "gold_call": "_profile(_oracle_compute_link_free_energy, [1.02, 1.3, 2.0], 395.0, 401, 50.0)",
        },
        {
            "setup": setup,
            "tol": 2e-9,
            "call": "_profile(compute_link_free_energy, [1.0, 10.0, 29.6], 20.0, 31, 40.0)",
            "gold_call": "_profile(_oracle_compute_link_free_energy, [1.0, 10.0, 29.6], 20.0, 31, 40.0)",
        },
        {
            "setup": setup,
            "tol": 2e-9,
            "call": "_profile(compute_link_free_energy, [1.0, 3.0, 22.0], 11.5, 11, 30.0)",
            "gold_call": "_profile(_oracle_compute_link_free_energy, [1.0, 3.0, 22.0], 11.5, 11, 30.0)",
        },
        {
            "setup": setup,
            "tol": 2e-9,
            "call": "_profile(compute_link_free_energy, [1.0, 49.999999, 49.999999999, 49.9999999999], 0.0, 51, 35.0)",
            "gold_call": "_profile(_oracle_compute_link_free_energy, [1.0, 49.999999, 49.999999999, 49.9999999999], 0.0, 51, 35.0)",
        },
    ]
