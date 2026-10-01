"""
Find the stationary splay of the polarities of an isolated adhering cell pair arranged mirror-symmetrically.

Polarity regulation turns each polarity away from its partner while the adhesion torque favours polarities perpendicular to the pair axis, and their balance fixes how strongly neighbouring polarities fan out, which sets the curvature of a growing cell sheet.

Returns
-------
float: the stationary splay angle psi of the relaxed pair, in radians.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def compute_pair_splay_angle(
    magnitude: float,
    regulation_time: float,
    kernel_range: float,
    cutoff: float,
    tau_v: float,
    rotation_fn: "Callable[..., np.ndarray]",
) -> float:
    """Return the stationary splay angle of a relaxed mirror-symmetric cell pair.

    Two cells of polarity magnitude ``magnitude`` and regulation constant
    ``regulation_time`` sit at the separation where ``U(r) = exp(-r) -
    exp(-r / kernel_range)`` is minimal. Their polarities are mirror images
    of each other across the perpendicular bisector of the pair: if cell 1's
    polarity makes the angle ``psi`` with the direction from cell 1 to cell
    2, cell 2's polarity makes the angle ``pi - psi`` with that same
    direction. The polarity angles rotate at the rates ``rotation_fn(positions,
    angles, magnitudes, regulation_times, kernel_range, cutoff, tau_v)``,
    which follows the contract of ``compute_polarity_rotation_rates`` for
    two cells. Return the angle ``psi`` in ``(pi / 2, pi)`` at which the
    polarity angles are stationary and stable, i.e. the equilibrium in which
    the two polarities splay away from each other, accurate to ``1e-12``.

    Parameters
    ----------
    magnitude : float
        Positive polarity magnitude of both cells.
    regulation_time : float
        Positive regulation constant of both cells.
    kernel_range : float
        Range parameter of ``U``; must exceed 1 so that ``U`` has a minimum.
    cutoff : float
        Interaction range (``np.inf`` allowed); must exceed the separation
        minimising ``U``.
    tau_v : float
        Positive mobility constant passed to ``rotation_fn``.
    rotation_fn : callable
        Rotation-rate function with the signature of
        ``compute_polarity_rotation_rates``.

    Returns
    -------
    float
        The splay angle ``psi`` in radians.

    Raises
    ------
    ValueError
        If ``magnitude``, ``regulation_time`` or ``tau_v`` is not a finite
        positive number, ``kernel_range`` is not a finite number above 1,
        ``cutoff`` does not exceed the separation minimising ``U``,
        ``rotation_fn`` is not callable or returns anything other than two
        finite rates, or if the pair has no stable stationary angle in
        ``(pi / 2, pi)``.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_pair_splay_angle(
    magnitude: float,
    regulation_time: float,
    kernel_range: float,
    cutoff: float,
    tau_v: float,
    rotation_fn: "Callable[..., np.ndarray]",
) -> float:
    """Reference implementation (bisection on the rotation rate of cell 1)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_function(value):
        return callable(value)

    for value in (magnitude, regulation_time, tau_v):
        if not (_is_number(value) and value > 0.0):
            raise ValueError("magnitude, regulation_time and tau_v must be finite and positive")
    if not (_is_number(kernel_range) and kernel_range > 1.0):
        raise ValueError("kernel_range must be a finite number above 1")
    if not _is_function(rotation_fn):
        raise ValueError("rotation_fn must be callable")
    beta = float(kernel_range)
    rest = beta * np.log(beta) / (beta - 1.0)
    if not ((cutoff == np.inf or _is_number(cutoff)) and cutoff > rest):
        raise ValueError("cutoff must exceed the rest separation of the pair")
    positions = np.array([[0.0, 0.0], [rest, 0.0]])
    strengths = np.full(2, float(magnitude))
    regulation = np.full(2, float(regulation_time))

    def _rate(psi):
        angles = np.array([psi, np.pi - psi])
        out = rotation_fn(positions, angles, strengths, regulation, beta, cutoff, tau_v)
        out = np.asarray(out, dtype=float)
        if out.shape != (2,) or not np.all(np.isfinite(out)):
            raise ValueError("rotation_fn must return two finite rates")
        return float(out[0])

    # The rate of cell 1 is positive just past pi / 2 and must turn negative
    # before pi for a stable splayed rest angle to exist.
    low, high = 0.5 * np.pi, np.pi - 1e-9
    if not (_rate(low) > 0.0 and _rate(high) < 0.0):
        raise ValueError("the pair has no stable splayed rest angle")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        if _rate(middle) > 0.0:
            low = middle
        else:
            high = middle
    return float(0.5 * (low + high))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    model = (
        "import numpy as np\n"
        "def _rot(X, T, M, B, b, c, tv):\n"
        "    X = np.asarray(X, dtype=float); n = len(X)\n"
        "    W = np.zeros(n)\n"
        "    for i in range(n):\n"
        "        for j in range(n):\n"
        "            if i == j:\n"
        "                continue\n"
        "            d = X[j] - X[i]; r = np.hypot(d[0], d[1])\n"
        "            if not r < c:\n"
        "                continue\n"
        "            e = d / r\n"
        "            ui = np.array([np.cos(T[i]), np.sin(T[i])])\n"
        "            ni = np.array([-np.sin(T[i]), np.cos(T[i])])\n"
        "            nj = np.array([-np.sin(T[j]), np.cos(T[j])])\n"
        "            U = np.exp(-r) - np.exp(-r / b)\n"
        "            W[i] += tv * U * M[i] * M[j] * (ui @ e) * (nj @ e) - B[i] * (ni @ e)\n"
        "    return W\n"
        "def _toy(X, T, M, B, b, c, tv):\n"
        "    return np.array([np.sin(T[0]) * (0.3 + 0.5 * np.cos(T[0])), 0.0])\n"
        "def _matches_balance(psi, m, tb, b, tv):\n"
        "    rest = b * np.log(b) / (b - 1.0)\n"
        "    depth = np.exp(-rest) - np.exp(-rest / b)\n"
        "    gap = abs(np.cos(psi) - tb / (tv * depth * m * m))\n"
        "    return 1.0 if 0.5 * np.pi < psi < np.pi and gap < 1e-10 else 0.0\n"
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
            "setup": model,
            "call": "compute_pair_splay_angle(0.8, 1.1, 3.6, 2.4, 5.0, _rot)",
            "gold_call": "_oracle_compute_pair_splay_angle(0.8, 1.1, 3.6, 2.4, 5.0, _rot)",
        },
        {
            "setup": model,
            "call": "compute_pair_splay_angle(0.55, 0.35, 6.3, np.inf, 4.0, _rot)",
            "gold_call": "_oracle_compute_pair_splay_angle(0.55, 0.35, 6.3, np.inf, 4.0, _rot)",
        },
        {
            "setup": model,
            "call": "_matches_balance(compute_pair_splay_angle(1.3, 0.02, 2.9, 2.2, 1.0, _rot), 1.3, 0.02, 2.9, 1.0)",
            "gold_call": "_matches_balance(_oracle_compute_pair_splay_angle(1.3, 0.02, 2.9, 2.2, 1.0, _rot), 1.3, 0.02, 2.9, 1.0)",
        },
        {
            "setup": model,
            "call": "compute_pair_splay_angle(0.4, 0.5, 5.5, 3.0, 8.0, _toy)",
            "gold_call": "_oracle_compute_pair_splay_angle(0.4, 0.5, 5.5, 3.0, 8.0, _toy)",
        },
        {
            "setup": model,
            "call": "float(np.cos(compute_pair_splay_angle(0.7, 0.9, 4.4, 2.6, 6.0, _rot)))",
            "gold_call": "float(np.cos(_oracle_compute_pair_splay_angle(0.7, 0.9, 4.4, 2.6, 6.0, _rot)))",
        },
        {
            "setup": model + status,
            "call": "_status(lambda: compute_pair_splay_angle(0.3, 2.0, 3.6, 2.4, 5.0, _rot))",
            "gold_call": "_status(lambda: _oracle_compute_pair_splay_angle(0.3, 2.0, 3.6, 2.4, 5.0, _rot))",
        },
        {
            "setup": model + status,
            "call": "_status(lambda: compute_pair_splay_angle(0.8, 1.1, 3.6, 1.6, 5.0, _rot))",
            "gold_call": "_status(lambda: _oracle_compute_pair_splay_angle(0.8, 1.1, 3.6, 1.6, 5.0, _rot))",
        },
    ]
