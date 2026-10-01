"""
Find the stable non-collinear equilibrium of three adhering cells whose polarities are frozen, parallel and of a given magnitude.

Whether a third cell can rest on top of two neighbours rather than join their row decides between multilayer and monolayer growth, and polarity-dependent adhesion makes that stacked arrangement less favourable as polarity strengthens.

Returns
-------
np.ndarray: [base separation, common separation of the two other pairs].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def solve_stacked_triad(
    magnitude: float,
    kernel_range: float,
    cutoff: float,
    tau_v: float,
    velocity_fn: "Callable[..., np.ndarray]",
) -> np.ndarray:
    """Return the pair separations of the stable non-collinear three-cell equilibrium.

    Three cells carry frozen polarities of common magnitude ``magnitude``,
    all pointing along ``+y`` (every angle is ``pi / 2``), and move with the
    velocities ``velocity_fn(positions, angles, magnitudes, kernel_range,
    cutoff, tau_v)``, which follows the contract of
    ``compute_adhesion_velocities`` (``positions`` of shape ``(3, 2)``, a
    ``(3, 2)`` result, and ``np.inf`` accepted as ``cutoff``). Positions
    evolve; polarities do not.

    Return the equilibrium, under the given ``cutoff``, in which the three
    cells are not collinear and which is stable: an isosceles triangle whose
    base is perpendicular to the polarity. It is the equilibrium continuously
    connected, as the magnitude is raised from zero, to the equilateral
    triangle whose side is the distance at which ``U(r) = exp(-r) -
    exp(-r / kernel_range)`` is minimal. Separations must be accurate to
    ``1e-12``.

    Parameters
    ----------
    magnitude : float
        Nonnegative common polarity magnitude.
    kernel_range : float
        Range parameter of ``U``; must exceed 1 so that ``U`` has a minimum.
    cutoff : float
        Finite interaction range, larger than the distance minimising ``U``.
    tau_v : float
        Positive mobility constant passed to ``velocity_fn``.
    velocity_fn : callable
        Velocity function with the signature of
        ``compute_adhesion_velocities``.

    Returns
    -------
    np.ndarray
        Float array ``[base, side]``: the separation of the pair whose axis
        is perpendicular to the polarity, then the common separation of the
        two other pairs.

    Raises
    ------
    ValueError
        If ``magnitude`` is not a finite nonnegative number, ``kernel_range``
        is not a finite number above 1, ``cutoff`` is not a finite number
        above the distance minimising ``U``, ``tau_v`` is not a finite
        positive number, ``velocity_fn`` is not callable or returns an array
        of the wrong shape or with non-finite entries, or if at this
        magnitude the three cells have no stable non-collinear equilibrium
        under the given ``cutoff``.
    """
    return separations

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_solve_stacked_triad(
    magnitude: float,
    kernel_range: float,
    cutoff: float,
    tau_v: float,
    velocity_fn: "Callable[..., np.ndarray]",
) -> np.ndarray:
    """Reference implementation (continuation in magnitude with Newton steps)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_function(value):
        return callable(value)

    if not (_is_number(magnitude) and magnitude >= 0.0):
        raise ValueError("magnitude must be a finite nonnegative number")
    if not (_is_number(kernel_range) and kernel_range > 1.0):
        raise ValueError("kernel_range must be a finite number above 1")
    if not (_is_number(tau_v) and tau_v > 0.0):
        raise ValueError("tau_v must be a finite positive number")
    if not _is_function(velocity_fn):
        raise ValueError("velocity_fn must be callable")
    beta = float(kernel_range)
    rest = beta * np.log(beta) / (beta - 1.0)
    if not (_is_number(cutoff) and cutoff > rest):
        raise ValueError("cutoff must be a finite number above the rest distance")
    angles = np.full(3, 0.5 * np.pi)

    def _layout(state):
        base, height = state
        return np.array([[-0.5 * base, 0.0], [0.5 * base, 0.0], [0.0, height]])

    def _velocities(state, strength, reach):
        out = velocity_fn(_layout(state), angles, np.full(3, strength), beta, reach, tau_v)
        out = np.asarray(out, dtype=float)
        if out.shape != (3, 2) or not np.all(np.isfinite(out)):
            raise ValueError("velocity_fn must return a finite (3, 2) array")
        return out

    def _residual(state, strength):
        # Rates of the base length and of the apex height above the base
        # midpoint on the smooth branch (every pair interacting).
        v = _velocities(state, strength, np.inf)
        return np.array([v[1, 0] - v[0, 0], v[2, 1] - 0.5 * (v[0, 1] + v[1, 1])])

    def _jacobian(state, strength):
        matrix = np.empty((2, 2))
        for k in range(2):
            step = np.zeros(2)
            step[k] = 1e-6 * max(1.0, abs(state[k]))
            matrix[:, k] = (_residual(state + step, strength)
                            - _residual(state - step, strength)) / (2.0 * step[k])
        return matrix

    def _newton(start, strength):
        # Stay on the stacked branch: the apex keeps a clear height and no
        # continuation step may jump to a distant (for example collinear) root.
        state = start.copy()
        for _ in range(60):
            try:
                delta = np.linalg.solve(_jacobian(state, strength), -_residual(state, strength))
            except np.linalg.LinAlgError:
                return None
            state = state + delta
            if not (np.all(np.isfinite(state)) and state[0] > 0.0 and state[1] > 0.1 * rest):
                return None
            if np.max(np.abs(state - start)) > 0.5 * rest:
                return None
            if np.max(np.abs(delta)) <= 1e-14 * (1.0 + np.max(np.abs(state))):
                return state
        return None

    state = np.array([rest, 0.5 * np.sqrt(3.0) * rest])
    target = float(magnitude)
    for k in range(1, 25):
        state = _newton(state, target * np.sqrt(k / 24.0))
        if state is None:
            raise ValueError("no stacked equilibrium at this magnitude")
    base = float(state[0])
    side = float(np.hypot(0.5 * state[0], state[1]))
    if not (base < cutoff and side < cutoff):
        raise ValueError("no stacked equilibrium with every pair interacting")
    matrix = _jacobian(state, target)
    if not (np.trace(matrix) < 0.0 and np.linalg.det(matrix) > 0.0):
        raise ValueError("the stacked equilibrium is not stable")
    check = _velocities(state, target, float(cutoff))
    if np.max(np.abs(check)) > 1e-9 * max(1.0, float(tau_v)):
        raise ValueError("the stacked configuration is not at rest under the cutoff")
    return np.array([base, side])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    model = (
        "import numpy as np\n"
        "def _vel(X, T, M, b, c, tv):\n"
        "    X = np.asarray(X, dtype=float); n = len(X)\n"
        "    V = np.zeros((n, 2))\n"
        "    for i in range(n):\n"
        "        for j in range(n):\n"
        "            if i == j:\n"
        "                continue\n"
        "            d = X[j] - X[i]; r = np.hypot(d[0], d[1])\n"
        "            if not r < c:\n"
        "                continue\n"
        "            e = d / r\n"
        "            ni = np.array([-np.sin(T[i]), np.cos(T[i])])\n"
        "            nj = np.array([-np.sin(T[j]), np.cos(T[j])])\n"
        "            ci = M[i] * (ni @ e); cj = M[j] * (nj @ e)\n"
        "            dci = -(M[i] * ni - ci * e) / r\n"
        "            dcj = -(M[j] * nj - cj * e) / r\n"
        "            U = np.exp(-r) - np.exp(-r / b)\n"
        "            dU = -np.exp(-r) + np.exp(-r / b) / b\n"
        "            grad = -(ci * cj + 1.0) * dU * e + U * (dci * cj + ci * dcj)\n"
        "            V[i] -= tv * grad\n"
        "    return V\n"
        "def _central(X, T, M, b, c, tv):\n"
        "    return _vel(X, T, np.zeros(len(X)), b, c, tv)\n"
        "def _pair(a):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (2,):\n"
        "        return -1.0\n"
        "    return float(a[0] + 3.0 * a[1])\n"
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
            "call": "_pair(solve_stacked_triad(0.22, 3.6, 2.2, 1.0, _vel))",
            "gold_call": "_pair(_oracle_solve_stacked_triad(0.22, 3.6, 2.2, 1.0, _vel))",
        },
        {
            "setup": model,
            "call": "float(solve_stacked_triad(0.3354, 3.6, 2.2, 1.0, _vel)[0])",
            "gold_call": "float(_oracle_solve_stacked_triad(0.3354, 3.6, 2.2, 1.0, _vel)[0])",
        },
        {
            "setup": model,
            "call": "_pair(solve_stacked_triad(0.3, 6.3, 3.4, 2.5, _vel))",
            "gold_call": "_pair(_oracle_solve_stacked_triad(0.3, 6.3, 3.4, 2.5, _vel))",
        },
        {
            "setup": model,
            "call": "_pair(solve_stacked_triad(0.0, 4.4, 2.6, 1.0, _vel))",
            "gold_call": "_pair(_oracle_solve_stacked_triad(0.0, 4.4, 2.6, 1.0, _vel))",
        },
        {
            "setup": model,
            "call": "_pair(solve_stacked_triad(0.35, 4.4, 2.6, 1.0, _central))",
            "gold_call": "_pair(_oracle_solve_stacked_triad(0.35, 4.4, 2.6, 1.0, _central))",
        },
        {
            "setup": model + "fast = lambda X, T, M, b, c, tv: 4.0 * _vel(X, T, M, b, c, tv)\n",
            "call": "_pair(solve_stacked_triad(0.18, 3.6, 2.2, 0.5, fast))",
            "gold_call": "_pair(_oracle_solve_stacked_triad(0.18, 3.6, 2.2, 0.5, _vel))",
        },
        {
            "setup": model + status,
            "call": "_status(lambda: solve_stacked_triad(0.342, 3.6, 2.2, 1.0, _vel))",
            "gold_call": "_status(lambda: _oracle_solve_stacked_triad(0.342, 3.6, 2.2, 1.0, _vel))",
        },
        {
            "setup": model + status,
            "call": "_status(lambda: solve_stacked_triad(0.9, 6.3, 3.4, 2.5, _vel))",
            "gold_call": "_status(lambda: _oracle_solve_stacked_triad(0.9, 6.3, 3.4, 2.5, _vel))",
        },
        {
            "setup": model + status,
            "call": "_status(lambda: solve_stacked_triad(0.2, 3.6, 1.5, 1.0, _vel))",
            "gold_call": "_status(lambda: _oracle_solve_stacked_triad(0.2, 3.6, 1.5, 1.0, _vel))",
        },
    ]
