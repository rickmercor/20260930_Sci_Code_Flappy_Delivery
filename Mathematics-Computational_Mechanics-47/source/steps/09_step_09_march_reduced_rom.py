"""
March the reduced state of the projected polynomial model with a reduced Newton solver.

Each implicit level is reached by iterating the reduced Newton system to a

prescribed convergence measure using only the offline Gram matrix, so the

online cost per iteration is independent of the full-order dimension.

Returns
-------
np.ndarray, the reduced trajectory of shape (n, n_steps + 1) whose column m holds the reduced state at time m * time_step.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral, Real

import math
import numpy as np


def march_reduced_rom(
    gram: np.ndarray,
    initial_reduced_state: np.ndarray,
    time_step: float,
    n_steps: int,
    amplitudes: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    scheme: str,
    residual_tolerance: float,
    max_newton_iterations: int,
) -> np.ndarray:
    """Advance the reduced state with the reduced Newton solver.

    Time levels are t_m = m * time_step for m = 0, ..., n_steps, and the input
    at level m is

        u(t_m) = [amplitudes[0] * sin(2 * pi * t_m),
                  amplitudes[1] * sin(4 * pi * t_m)].

    The linear multistep weights alphas and betas have tau + 1 entries, index j
    referring to level m - j; any level with a negative index is replaced by
    level 0. At each step the iterate starts from the previous accepted level.
    One iteration evaluates the residual coefficients of the current iterate
    and the corresponding Newton direction with its convergence measure; if
    that measure is below residual_tolerance the iterate is accepted, otherwise
    the direction is added with unit step length and the iteration repeats, up
    to max_newton_iterations times. Column 0 of the output is
    initial_reduced_state.

    Parameters
    ----------
    gram : np.ndarray
        Finite real symmetric array of shape (d, d) with
        d = 2 * n + n**2 + 2 * n + 1 + 2 for two inputs.
    initial_reduced_state : np.ndarray
        Finite real array of shape (n,) with n >= 1.
    time_step : float
        Finite strictly positive step size.
    n_steps : int
        Non-negative number of time steps to take.
    amplitudes : np.ndarray
        Finite real array of shape (2,) holding the two source amplitudes.
    alphas : np.ndarray
        Finite real array of shape (tau + 1,) holding the state coefficients.
    betas : np.ndarray
        Finite real array of shape (tau + 1,) holding the rate coefficients.
    scheme : str
        Either "galerkin" or "lspg".
    residual_tolerance : float
        Finite strictly positive convergence tolerance.
    max_newton_iterations : int
        Positive cap on iterations per time step.

    Returns
    -------
    trajectory : np.ndarray
        Float array of shape (n, n_steps + 1).

    Raises
    ------
    ValueError
        If any array has the wrong rank or an inconsistent shape, if any entry
        is not finite, if a scalar control violates its stated sign or type
        requirement, or if scheme is not one of the two recognised names.
    """
    return np.empty((0, 0), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_march_reduced_rom(
    gram: np.ndarray,
    initial_reduced_state: np.ndarray,
    time_step: float,
    n_steps: int,
    amplitudes: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    scheme: str,
    residual_tolerance: float,
    max_newton_iterations: int,
    coefficient_function=None,
    direction_function=None,
    kronecker_function=None,
) -> np.ndarray:
    """Reference implementation."""
    from numbers import Integral, Real

    import math
    import numpy as np

    def _is_number(value) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool)

    def _is_integer(value) -> bool:
        return isinstance(value, Integral) and not isinstance(value, bool)

    def _finite(name: str, values, ndim: int) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != ndim:
            raise ValueError(f"{name} must have ndim={ndim}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} entries must be finite")
        return array

    matrix = _finite("gram", gram, 2)
    start = _finite("initial_reduced_state", initial_reduced_state, 1)
    amps = _finite("amplitudes", amplitudes, 1)
    state_weights = _finite("alphas", alphas, 1)
    rate_weights = _finite("betas", betas, 1)
    if not isinstance(scheme, str) or scheme not in ("galerkin", "lspg"):
        raise ValueError("scheme must be 'galerkin' or 'lspg'")
    if start.shape[0] < 1:
        raise ValueError("initial_reduced_state must be non-empty")
    if amps.shape[0] != 2:
        raise ValueError("amplitudes must have exactly two entries")
    if state_weights.shape[0] < 1 or state_weights.shape[0] != rate_weights.shape[0]:
        raise ValueError("alphas and betas must be non-empty and equally long")
    if not _is_number(time_step) or not math.isfinite(float(time_step)) or float(time_step) <= 0.0:
        raise ValueError("time_step must be a finite positive real scalar")
    if not _is_integer(n_steps) or int(n_steps) < 0:
        raise ValueError("n_steps must be a non-negative integer")
    if (not _is_number(residual_tolerance) or not math.isfinite(float(residual_tolerance))
            or float(residual_tolerance) <= 0.0):
        raise ValueError("residual_tolerance must be a finite positive real scalar")
    if not _is_integer(max_newton_iterations) or int(max_newton_iterations) < 1:
        raise ValueError("max_newton_iterations must be a positive integer")

    n_reduced = start.shape[0]
    levels = state_weights.shape[0]
    total = 2 * n_reduced + n_reduced * n_reduced + 2 * n_reduced + 1 + 2
    if matrix.shape != (total, total):
        raise ValueError("gram must be square with the column-basis dimension")

    step = float(time_step)
    horizon = int(n_steps)
    tolerance = float(residual_tolerance)
    cap = int(max_newton_iterations)
    make_coefficients = (_oracle_residual_coefficient_vector
                         if coefficient_function is None else coefficient_function)
    make_direction = (_oracle_reduced_newton_direction
                      if direction_function is None else direction_function)

    def _input_at(level: int) -> np.ndarray:
        moment = level * step
        return np.array([amps[0] * math.sin(2.0 * math.pi * moment),
                         amps[1] * math.sin(4.0 * math.pi * moment)], dtype=float)

    trajectory = np.empty((n_reduced, horizon + 1), dtype=float)
    trajectory[:, 0] = start
    for index in range(1, horizon + 1):
        current = trajectory[:, index - 1].copy()
        past = [trajectory[:, max(index - j, 0)] for j in range(1, levels)]
        controls = np.array([_input_at(max(index - j, 0)) for j in range(levels)], dtype=float)
        for _ in range(cap):
            history = np.array([current] + past, dtype=float)
            coefficients = make_coefficients(
                history, controls, step, state_weights, rate_weights)
            direction, measure = make_direction(
                matrix, coefficients, current, controls[0], step,
                float(state_weights[0]), float(rate_weights[0]), scheme,
                kronecker_function=kronecker_function)
            if float(measure) < tolerance:
                break
            current = current + np.asarray(direction, dtype=float)
        trajectory[:, index] = current
    return trajectory

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return a list of test case specifications."""
    reducer = (
        "import numpy as np\n"
        "def _sig(value, scale):\n"
        "    a = np.asarray(value, dtype=float)\n"
        "    b = np.concatenate([np.asarray([a.ndim, *a.shape], dtype=float), a.ravel()])\n"
        "    k = np.arange(1.0, b.size + 1.0)\n"
        "    return float((np.sum(np.abs(b)) + np.sum(b * np.cos(k))) / scale)\n"
        "def _gram(full, red, seed):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    basis, _ = np.linalg.qr(rng.standard_normal((full, red)))\n"
        "    c = 0.2 * rng.standard_normal(full)\n"
        "    a = -np.eye(full) + 0.2 * rng.standard_normal((full, full))\n"
        "    f = np.zeros((full, full * full))\n"
        "    for r in range(full):\n"
        "        for p in range(full):\n"
        "            for q in range(p, full):\n"
        "                if (r + p + q) % 4 == 0:\n"
        "                    f[r, p * full + q] = 0.15 * rng.standard_normal()\n"
        "    b = 0.5 * rng.standard_normal((full, 2))\n"
        "    nn = 0.1 * rng.standard_normal((full, 2 * full))\n"
        "    cols = np.concatenate([basis, a @ basis, f @ np.kron(basis, basis),\n"
        "                           nn @ np.kron(np.eye(2), basis),\n"
        "                           c.reshape(-1, 1), b], axis=1)\n"
        "    return cols.T @ cols, basis\n"
        "gram, basis = _gram(9, 3, 41)\n"
        "x0 = np.array([0.3, -0.2, 0.1])\n"
        "amps = np.array([1.5, 0.5])\n"
        "be_a = np.array([1.0, -1.0])\n"
        "be_b = np.array([1.0, 0.0])\n"
    )
    return [
        # Case 1: a short least-squares march with backward Euler.
        {
            "setup": reducer,
            "call": (
                "_sig(march_reduced_rom(gram, x0, 0.02, 25, amps, be_a, be_b, 'lspg', 1e-12, 50), 1e0)"
            ),
            "gold_call": (
                "_sig(_oracle_march_reduced_rom(gram, x0, 0.02, 25, amps, be_a, be_b, 'lspg', 1e-12, 50), 1e0)"
            ),
        },
        # Case 2: the same march under the Galerkin scheme.
        {
            "setup": reducer,
            "call": (
                "_sig(march_reduced_rom(gram, x0, 0.02, 25, amps, be_a, be_b, 'galerkin', 1e-12, 50), 1e0)"
            ),
            "gold_call": (
                "_sig(_oracle_march_reduced_rom(gram, x0, 0.02, 25, amps, be_a, be_b, 'galerkin', 1e-12, 50), 1e0)"
            ),
        },
        # Case 3: zero steps returns the initial column only.
        {
            "setup": reducer,
            "call": (
                "_sig(march_reduced_rom(gram, x0, 0.02, 0, amps, be_a, be_b, 'lspg', 1e-12, 50), 1e0)"
            ),
            "gold_call": (
                "_sig(_oracle_march_reduced_rom(gram, x0, 0.02, 0, amps, be_a, be_b, 'lspg', 1e-12, 50), 1e0)"
            ),
        },
        # Case 4: a three-level scheme, which exercises the clamped start-up
        #     where the two levels before the initial one coincide with it.
        {
            "setup": reducer + (
                "bdf_a = np.array([1.5, -2.0, 0.5])\n"
                "bdf_b = np.array([1.0, 0.0, 0.0])\n"
            ),
            "call": (
                "_sig(march_reduced_rom(gram, x0, 0.02, 15, amps, bdf_a, bdf_b, 'lspg', 1e-12, 50), 1e0)"
            ),
            "gold_call": (
                "_sig(_oracle_march_reduced_rom(gram, x0, 0.02, 15, amps, bdf_a, bdf_b, 'lspg', 1e-12, 50), 1e0)"
            ),
        },
        # --- Decisive: every accepted level must already satisfy the stated
        #     convergence measure, and the two schemes must produce different
        #     trajectories on this instance.
        {
            "setup": reducer + (
                "def converged():\n"
                "    dt = 0.02\n"
                "    tr = march_reduced_rom(gram, x0, dt, 20, amps, be_a, be_b, 'lspg', 1e-12, 50)\n"
                "    tg = march_reduced_rom(gram, x0, dt, 20, amps, be_a, be_b, 'galerkin', 1e-12, 50)\n"
                "    worst = 0.0\n"
                "    for m in range(1, 21):\n"
                "        u = np.array([[amps[0] * np.sin(2.0 * np.pi * k * dt),\n"
                "                       amps[1] * np.sin(4.0 * np.pi * k * dt)]\n"
                "                      for k in (m, m - 1)])\n"
                "        hist = np.array([tr[:, m], tr[:, m - 1]])\n"
                "        coeff = residual_coefficient_vector(hist, u, dt, be_a, be_b)\n"
                "        _, measure = reduced_newton_direction(gram, coeff, tr[:, m], u[0], dt,\n"
                "                                              1.0, 1.0, 'lspg')\n"
                "        worst = max(worst, float(measure))\n"
                "    return int(worst < 1e-12) + 2 * int(float(np.max(np.abs(tr - tg))) > 1e-6)\n"
            ),
            "call": "converged()",
            "gold_call": "3",
        },
        # --- Decisive: the input must be sampled at the implicit level, so a
        #     march over a whole forcing period must not coincide with the march
        #     that samples the input one level early.
        {
            "setup": reducer + (
                "def input_level():\n"
                "    dt = 0.05\n"
                "    tr = march_reduced_rom(gram, x0, dt, 20, amps, be_a, be_b, 'lspg', 1e-12, 50)\n"
                "    shifted = march_reduced_rom(gram, x0, dt, 20,\n"
                "                                np.array([amps[0], -amps[1]]), be_a, be_b,\n"
                "                                'lspg', 1e-12, 50)\n"
                "    return int(float(np.max(np.abs(tr - shifted))) > 1e-6)\n"
            ),
            "call": "input_level()",
            "gold_call": "1",
        },
        # Case 7: invalid scheme name.
        {
            "setup": reducer + (
                "def run_model():\n"
                "    try:\n"
                "        march_reduced_rom(gram, x0, 0.02, 5, amps, be_a, be_b, 'lsq', 1e-12, 50)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_march_reduced_rom(gram, x0, 0.02, 5, amps, be_a, be_b, 'lsq', 1e-12, 50)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Case 8: invalid amplitude count.
        {
            "setup": reducer + (
                "def run_model():\n"
                "    try:\n"
                "        march_reduced_rom(gram, x0, 0.02, 5, np.array([1.0]), be_a, be_b, 'lspg', 1e-12, 50)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_march_reduced_rom(gram, x0, 0.02, 5, np.array([1.0]), be_a, be_b, 'lspg', 1e-12, 50)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
