"""
Scan the contrast on a uniform parameter grid and select the highest, well-separated grid local maxima as starting points for continuous refinement.

The averaged-state contrast is generally multimodal over a bounded parameter box, so the global maximizer is sought by locating several candidate peaks on a grid before refining each of them.

Returns
-------
np.ndarray: shape (m, 2), accepted (alpha, beta) starting points in acceptance order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def locate_contrast_candidates(
    contrast_fn: "Callable[[float, float], float]",
    alpha_bounds: tuple,
    beta_bounds: tuple,
    spacing: float = 0.05,
    max_candidates: int = 8,
    min_separation: float = 0.15,
) -> 'np.ndarray':
    """Return the selected grid local maxima of the contrast.

    The grid is ``alpha_k = alpha_bounds[0] + k * spacing`` for
    ``k = 0, ..., K`` with ``K = round((alpha_bounds[1] - alpha_bounds[0]) / spacing)``,
    and likewise for ``beta``; each range must be a whole number of
    spacings to within ``1e-9 * max(1, range)``. The last point on each
    axis is set to its exact upper bound after this check. A grid point is a candidate
    when ``contrast_fn`` there is no smaller than at each of its (up to
    eight) neighbouring grid points. Candidates are ranked by decreasing
    contrast, ties keeping row-major grid order (``alpha`` index major).
    Walking down that ranking, a candidate is accepted when its Euclidean
    distance to every previously accepted point is at least
    ``min_separation - 1e-9``; the walk stops once ``max_candidates`` points
    are accepted.

    The supported numerical grid has bounds within ``[-100, 100]``,
    ``1e-4 <= spacing <= 200``, at most 2000 intervals on either axis,
    and at most 250000 grid points in total (endpoints included).

    Parameters
    ----------
    contrast_fn : callable
        Function of ``(alpha, beta)`` returning a finite real number.
    alpha_bounds, beta_bounds : tuple
        ``(low, high)`` with ``-100 <= low < high <= 100``.
    spacing : float
        Shared grid spacing in ``[1e-4, 200]``.
    max_candidates : int
        Maximum number of accepted points, ``>= 1``.
    min_separation : float
        Finite minimum distance between accepted points, ``>= 0``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(m, 2)`` holding the accepted ``(alpha, beta)``
        points in acceptance order, ``1 <= m <= max_candidates``.

    Raises
    ------
    ValueError
        If ``contrast_fn`` is not callable or returns a non-finite or
        non-scalar value, a bound pair is not two finite numbers with
        ``low < high``, ``spacing`` is not a finite positive number that
        divides both ranges, ``max_candidates`` is not an integer ``>= 1``
        (booleans are rejected), or ``min_separation`` is negative or not
        finite. Unsupported numerical bounds, spacing, axis sizes or total
        grid size, and nonfinite intermediate/result arrays also raise
        ``ValueError``.
    """
    return candidates

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_locate_contrast_candidates(
    contrast_fn: "Callable[[float, float], float]",
    alpha_bounds: tuple,
    beta_bounds: tuple,
    spacing: float = 0.05,
    max_candidates: int = 8,
    min_separation: float = 0.15,
) -> 'np.ndarray':
    """Reference implementation (eight-neighbour peak test, greedy separation)."""
    import numpy as np

    def _is_number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            return False
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError, OverflowError):
            return False

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    def _is_function(value):
        return callable(value)

    def _axis(bounds, name):
        try:
            low, high = bounds
        except (TypeError, ValueError):
            raise ValueError(f"{name} must hold two numbers") from None
        if not (_is_number(low) and _is_number(high) and low < high):
            raise ValueError(f"{name} must be finite with low < high")
        if low < -100.0 or high > 100.0:
            raise ValueError(f"{name} must lie within [-100, 100]")
        width = float(high) - float(low)
        count = round(width / spacing)
        if count < 1 or count > 2000 or abs(count * spacing - width) > 1e-9 * max(1.0, width):
            raise ValueError(f"spacing must divide the {name} range")
        axis = float(low) + spacing * np.arange(count + 1)
        axis[-1] = float(high)
        if not np.all(np.isfinite(axis)):
            raise ValueError("nonfinite grid axis")
        return axis

    if not _is_function(contrast_fn):
        raise ValueError("contrast_fn must be callable")
    if not (_is_number(spacing) and 1e-4 <= spacing <= 200.0):
        raise ValueError("spacing must lie in [1e-4, 200]")
    if not (_is_integer(max_candidates) and max_candidates >= 1):
        raise ValueError("max_candidates must be an integer >= 1")
    if not (_is_number(min_separation) and min_separation >= 0.0):
        raise ValueError("min_separation must be finite and >= 0")
    spacing = float(spacing)
    alphas = _axis(alpha_bounds, "alpha_bounds")
    betas = _axis(beta_bounds, "beta_bounds")
    if alphas.size * betas.size > 250000:
        raise ValueError("the grid must contain at most 250000 points")

    values = np.empty((alphas.size, betas.size))
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            try:
                v = np.asarray(contrast_fn(float(a), float(b)), dtype=complex)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("contrast_fn must return a finite real scalar") from exc
            if v.shape != () or not np.isfinite(v):
                raise ValueError("contrast_fn must return a finite scalar")
            if v.imag != 0.0:
                raise ValueError("contrast_fn must return a real scalar")
            values[i, j] = float(v.real)

    padded = np.full((alphas.size + 2, betas.size + 2), -np.inf)
    padded[1:-1, 1:-1] = values
    peak = np.ones(values.shape, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            shifted = padded[1 + di:1 + di + alphas.size, 1 + dj:1 + dj + betas.size]
            peak &= values >= shifted
    index = np.flatnonzero(peak.ravel())
    # Stable sort on the negated value keeps row-major order among ties.
    order = index[np.argsort(-values.ravel()[index], kind="stable")]
    chosen = []
    for flat in order:
        i, j = divmod(int(flat), betas.size)
        point = np.array([alphas[i], betas[j]])
        if all(np.hypot(*(point - q)) >= min_separation - 1e-9 for q in chosen):
            chosen.append(point)
            if len(chosen) == max_candidates:
                break
    result = np.array(chosen, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite selected candidates")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    helpers = (
        "import numpy as np\n"
        "def _bumps(a, b):\n"
        "    return (np.exp(-((a - 0.7) ** 2 + (b - 1.1) ** 2) / 0.2)\n"
        "            + 0.8 * np.exp(-((a + 1.3) ** 2 + (b - 0.6) ** 2) / 0.1)\n"
        "            + 0.3 * np.exp(-((a - 2.2) ** 2 + (b - 0.3) ** 2) / 0.05))\n"
        "def _wavy(a, b):\n"
        "    return np.cos(3.0 * a) * np.cos(2.0 * b) + 0.05 * a\n"
        "def _ridge(a, b):\n"
        "    return a + 0.3 * np.sin(4.0 * b)\n"
        "def _flat(a, b):\n"
        "    return 0.0\n"
        "def _diag(a, b):\n"
        "    return np.exp(-((a - b) ** 2) / 0.02) + 0.1 * (a + b)\n"
        "def _csig(c):\n"
        "    c = np.asarray(c, dtype=float)\n"
        "    if c.ndim != 2 or c.shape[1] != 2:\n"
        "        return -1.0\n"
        "    flat = c.ravel()\n"
        "    weights = np.cos(np.arange(1, flat.size + 1, dtype=float))\n"
        "    return float(10.0 * c.shape[0] + np.sum(np.abs(flat)) + np.sum(flat * weights))\n"
    )
    status = (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        {
            "setup": helpers,
            "call": "_csig(locate_contrast_candidates(_bumps, (-2.0, 3.0), (0.0, 2.0), 0.1))",
            "gold_call": "_csig(_oracle_locate_contrast_candidates(_bumps, (-2.0, 3.0), (0.0, 2.0), 0.1))",
        },
        {
            "setup": helpers,
            "call": "_csig(locate_contrast_candidates(_wavy, (-3.0, 3.0), (-1.5, 1.5)))",
            "gold_call": "_csig(_oracle_locate_contrast_candidates(_wavy, (-3.0, 3.0), (-1.5, 1.5)))",
        },
        {
            "setup": helpers,
            "call": "_csig(locate_contrast_candidates(_flat, (0.0, 1.0), (0.0, 0.5), 0.1, 5, 0.25))",
            "gold_call": "_csig(_oracle_locate_contrast_candidates(_flat, (0.0, 1.0), (0.0, 0.5), 0.1, 5, 0.25))",
        },
        {
            "setup": helpers,
            "call": "_csig(locate_contrast_candidates(_ridge, (-1.0, 1.0), (0.0, 2.0), 0.05, 4, 0.15))",
            "gold_call": "_csig(_oracle_locate_contrast_candidates(_ridge, (-1.0, 1.0), (0.0, 2.0), 0.05, 4, 0.15))",
        },
        {
            "setup": helpers,
            "call": "_csig(locate_contrast_candidates(_diag, (-1.0, 1.0), (-1.0, 1.0), 0.1))",
            "gold_call": "_csig(_oracle_locate_contrast_candidates(_diag, (-1.0, 1.0), (-1.0, 1.0), 0.1))",
        },
        {
            "setup": helpers,
            "call": "_csig(locate_contrast_candidates(_bumps, (-2.0, 3.0), (0.0, 2.0), 0.05, 1))",
            "gold_call": "_csig(_oracle_locate_contrast_candidates(_bumps, (-2.0, 3.0), (0.0, 2.0), 0.05, 1))",
        },
        {
            "setup": helpers,
            "call": "_csig(locate_contrast_candidates(_wavy, (-3.0, 3.0), (-1.5, 1.5), 0.1, 20, 1.0))",
            "gold_call": "_csig(_oracle_locate_contrast_candidates(_wavy, (-3.0, 3.0), (-1.5, 1.5), 0.1, 20, 1.0))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: locate_contrast_candidates(_bumps, (0.0, 1.0), (0.0, 1.0), 0.07))",
            "gold_call": "_status(lambda: _oracle_locate_contrast_candidates(_bumps, (0.0, 1.0), (0.0, 1.0), 0.07))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: locate_contrast_candidates(_bumps, (1.0, 1.0), (0.0, 1.0), 0.1))",
            "gold_call": "_status(lambda: _oracle_locate_contrast_candidates(_bumps, (1.0, 1.0), (0.0, 1.0), 0.1))",
        },
        {
            "setup": helpers,
            "call": "_csig(locate_contrast_candidates(_flat, (-100., 100.), (-100., 100.), 200.))",
            "gold_call": "_csig(_oracle_locate_contrast_candidates(_flat, (-100., 100.), (-100., 100.), 200.))",
        },
        {
            "setup": helpers,
            "call": "_csig(locate_contrast_candidates(_flat, (0., .2), (0., 1e-4), 1e-4, 1))",
            "gold_call": "_csig(_oracle_locate_contrast_candidates(_flat, (0., .2), (0., 1e-4), 1e-4, 1))",
        },
        {
            "setup": helpers,
            "call": "_csig(locate_contrast_candidates(_flat, (0., 49.9), (0., 49.9), .1, 1))",
            "gold_call": "_csig(_oracle_locate_contrast_candidates(_flat, (0., 49.9), (0., 49.9), .1, 1))",
        },
        {
            "setup": helpers,
            "call": "_csig(locate_contrast_candidates(lambda a, b: a + b, (-5., 5.), (0., 3.), .100000000005))",
            "gold_call": "_csig(_oracle_locate_contrast_candidates(lambda a, b: a + b, (-5., 5.), (0., 3.), .100000000005))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: locate_contrast_candidates(_flat, (-1e308, 1e308), (0., 1e308), 1e308))",
            "gold_call": "_status(lambda: _oracle_locate_contrast_candidates(_flat, (-1e308, 1e308), (0., 1e308), 1e308))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: locate_contrast_candidates(_flat, (-100., np.nextafter(100., np.inf)), (0., 1.), 1.))",
            "gold_call": "_status(lambda: _oracle_locate_contrast_candidates(_flat, (-100., np.nextafter(100., np.inf)), (0., 1.), 1.))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: locate_contrast_candidates(_flat, (0., 1.), (0., 1.), np.nextafter(1e-4, 0.)))",
            "gold_call": "_status(lambda: _oracle_locate_contrast_candidates(_flat, (0., 1.), (0., 1.), np.nextafter(1e-4, 0.)))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: locate_contrast_candidates(_flat, (0., 1.), (0., 1.), np.nextafter(200., np.inf)))",
            "gold_call": "_status(lambda: _oracle_locate_contrast_candidates(_flat, (0., 1.), (0., 1.), np.nextafter(200., np.inf)))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: locate_contrast_candidates(_flat, (0., .2001), (0., .0001), .0001))",
            "gold_call": "_status(lambda: _oracle_locate_contrast_candidates(_flat, (0., .2001), (0., .0001), .0001))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: locate_contrast_candidates(_flat, (0., 50.), (0., 49.9), .1))",
            "gold_call": "_status(lambda: _oracle_locate_contrast_candidates(_flat, (0., 50.), (0., 49.9), .1))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: locate_contrast_candidates(lambda a, b: np.inf, (0., 1.), (0., 1.), 1.))",
            "gold_call": "_status(lambda: _oracle_locate_contrast_candidates(lambda a, b: np.inf, (0., 1.), (0., 1.), 1.))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: locate_contrast_candidates(lambda a, b: 1j, (0., 1.), (0., 1.), 1.))",
            "gold_call": "_status(lambda: _oracle_locate_contrast_candidates(lambda a, b: 1j, (0., 1.), (0., 1.), 1.))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: locate_contrast_candidates(lambda a, b: 10**400, (0., 1.), (0., 1.), 1.))",
            "gold_call": "_status(lambda: _oracle_locate_contrast_candidates(lambda a, b: 10**400, (0., 1.), (0., 1.), 1.))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: locate_contrast_candidates(_flat, (-10**400, 1.), (0., 1.), 1.))",
            "gold_call": "_status(lambda: _oracle_locate_contrast_candidates(_flat, (-10**400, 1.), (0., 1.), 1.))",
        },
    ]
