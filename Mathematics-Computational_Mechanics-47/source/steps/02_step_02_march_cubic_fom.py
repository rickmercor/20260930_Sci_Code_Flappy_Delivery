"""
Advance the full-order cubic-reaction model in time with an implicit one-step scheme solved by a Newton iteration.

Each time level is obtained by driving the backward-Euler residual of the semi-discrete system to a prescribed tolerance, so the recorded trajectory is the reference against which every reduced model is later measured.

Returns
-------
np.ndarray, the nodal trajectory with shape (n, n_steps + 1) whose column m holds the field at time m * time_step.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral, Real

import math
import numpy as np


def march_cubic_fom(
    initial_state: np.ndarray,
    time_step: float,
    n_steps: int,
    amplitudes: np.ndarray,
    domain_length: float,
    diffusivity: float,
    boundary_values: np.ndarray,
    residual_tolerance: float,
    max_newton_iterations: int,
) -> np.ndarray:
    """March the semi-discrete cubic-reaction model with backward Euler.

    Time levels are t_m = m * time_step for m = 0, ..., n_steps. At each level
    the implicit state q_m solves

        q_m - q_{m-1} - time_step * cubic_heat_rhs(q_m, t_m, ...) = 0,

    which is solved by Newton iteration started from q_{m-1} with unit step
    length. The iteration stops as soon as the 2-norm of that residual is
    below residual_tolerance, and in any case after max_newton_iterations
    updates. Column 0 of the output is initial_state.

    Parameters
    ----------
    initial_state : np.ndarray
        Finite real nodal field of shape (n,) with n >= 1.
    time_step : float
        Finite strictly positive step size.
    n_steps : int
        Non-negative number of time steps to take.
    amplitudes : np.ndarray
        Finite real array of shape (2,) holding the two source amplitudes.
    domain_length : float
        Finite strictly positive length of the spatial domain.
    diffusivity : float
        Finite strictly positive diffusion coefficient.
    boundary_values : np.ndarray
        Finite real array of shape (2,) holding the Dirichlet values at
        x = 0 and x = domain_length, in that order.
    residual_tolerance : float
        Finite strictly positive Newton stopping tolerance.
    max_newton_iterations : int
        Positive cap on Newton updates per time step.

    Returns
    -------
    trajectory : np.ndarray
        Float array of shape (n, n_steps + 1) holding the nodal field at every
        time level.

    Raises
    ------
    ValueError
        If any array has the wrong rank or shape, if any entry is not finite,
        or if a scalar control violates the stated sign or type requirement.
    """
    return np.empty((0, 0), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_march_cubic_fom(
    initial_state: np.ndarray,
    time_step: float,
    n_steps: int,
    amplitudes: np.ndarray,
    domain_length: float,
    diffusivity: float,
    boundary_values: np.ndarray,
    residual_tolerance: float,
    max_newton_iterations: int,
    rhs_function=None,
) -> np.ndarray:
    """Reference implementation."""
    from numbers import Integral, Real

    import math
    import numpy as np

    def _is_number(value) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool)

    def _is_integer(value) -> bool:
        return isinstance(value, Integral) and not isinstance(value, bool)

    field = np.asarray(initial_state, dtype=float)
    if field.ndim != 1 or field.shape[0] < 1:
        raise ValueError("initial_state must be a 1-D array with at least one entry")
    if not np.all(np.isfinite(field)):
        raise ValueError("initial_state entries must be finite")
    n_nodes = field.shape[0]

    bounds = np.asarray(boundary_values, dtype=float)
    if bounds.ndim != 1 or bounds.shape[0] != 2 or not np.all(np.isfinite(bounds)):
        raise ValueError("boundary_values must be a finite 1-D array of length 2")
    if not _is_number(time_step) or not math.isfinite(float(time_step)) or float(time_step) <= 0.0:
        raise ValueError("time_step must be a finite positive real scalar")
    if not _is_integer(n_steps) or int(n_steps) < 0:
        raise ValueError("n_steps must be a non-negative integer")
    if not _is_number(domain_length) or not math.isfinite(float(domain_length)) or float(domain_length) <= 0.0:
        raise ValueError("domain_length must be a finite positive real scalar")
    if not _is_number(diffusivity) or not math.isfinite(float(diffusivity)) or float(diffusivity) <= 0.0:
        raise ValueError("diffusivity must be a finite positive real scalar")
    if (not _is_number(residual_tolerance) or not math.isfinite(float(residual_tolerance))
            or float(residual_tolerance) <= 0.0):
        raise ValueError("residual_tolerance must be a finite positive real scalar")
    if not _is_integer(max_newton_iterations) or int(max_newton_iterations) < 1:
        raise ValueError("max_newton_iterations must be a positive integer")

    step = float(time_step)
    total = int(n_steps)
    tolerance = float(residual_tolerance)
    cap = int(max_newton_iterations)
    rhs = _oracle_cubic_heat_rhs if rhs_function is None else rhs_function
    spacing = float(domain_length) / (n_nodes + 1.0)
    curvature_scale = float(diffusivity) / spacing ** 2

    stencil = np.diag(-2.0 * np.ones(n_nodes, dtype=float))
    if n_nodes > 1:
        off = np.ones(n_nodes - 1, dtype=float)
        stencil += np.diag(off, 1) + np.diag(off, -1)
    diffusion = curvature_scale * stencil
    identity = np.eye(n_nodes, dtype=float)

    trajectory = np.empty((n_nodes, total + 1), dtype=float)
    trajectory[:, 0] = field
    for index in range(1, total + 1):
        moment = index * step
        previous = trajectory[:, index - 1]
        current = previous.copy()
        for _ in range(cap):
            rate = rhs(current, moment, amplitudes, domain_length,
                       diffusivity, boundary_values)
            residual = current - previous - step * np.asarray(rate, dtype=float)
            if float(np.linalg.norm(residual)) < tolerance:
                break
            jacobian = identity - step * (diffusion - np.diag(3.0 * current ** 2))
            current = current - np.linalg.solve(jacobian, residual)
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
        "base = dict(time_step=0.01, n_steps=12, amplitudes=np.array([1.5, 0.5]),\n"
        "            domain_length=1.0, diffusivity=0.005,\n"
        "            boundary_values=np.array([0.0, 1.0]),\n"
        "            residual_tolerance=1e-12, max_newton_iterations=50)\n"
        "q0 = np.linspace(0.0, 1.0, 9)[1:]\n"
    )
    return [
        # Case 1: a short march on a smooth initial profile.
        {
            "setup": reducer,
            "call": "_sig(march_cubic_fom(q0, **base), 1e1)",
            "gold_call": "_sig(_oracle_march_cubic_fom(q0, **base), 1e1)",
        },
        # Case 2: zero steps returns the initial column only.
        {
            "setup": reducer + "z = dict(base)\nz['n_steps'] = 0\n",
            "call": "_sig(march_cubic_fom(q0, **z), 1e0)",
            "gold_call": "_sig(_oracle_march_cubic_fom(q0, **z), 1e0)",
        },
        # Case 3: a large step size makes the cubic sink dominate, so the Newton
        #     iteration is far from its linear regime.
        {
            "setup": reducer + (
                "stiff = dict(base)\n"
                "stiff['time_step'] = 0.5\n"
                "stiff['n_steps'] = 6\n"
                "big = 3.0 * np.ones(6)\n"
            ),
            "call": "_sig(march_cubic_fom(big, **stiff), 1e0)",
            "gold_call": "_sig(_oracle_march_cubic_fom(big, **stiff), 1e0)",
        },
        # Case 4: the march must relax an initially uniform field towards the
        #     inhomogeneous right boundary rather than towards zero.
        {
            "setup": reducer + (
                "def drift():\n"
                "    quiet = dict(base)\n"
                "    quiet['amplitudes'] = np.array([0.0, 0.0])\n"
                "    quiet['time_step'] = 0.5\n"
                "    quiet['n_steps'] = 60\n"
                "    tr = march_cubic_fom(np.zeros(7), **quiet)\n"
                "    last = tr[:, -1]\n"
                "    return int(np.all(np.diff(last) > 0.0)) + 2 * int(last[-1] > last[0])\n"
            ),
            "call": "drift()",
            "gold_call": "3",
        },
        # Case 5: a single interior node keeps both ghost values in one stencil.
        {
            "setup": reducer + "one = np.array([0.25])\n",
            "call": "_sig(march_cubic_fom(one, **base), 1e0)",
            "gold_call": "_sig(_oracle_march_cubic_fom(one, **base), 1e0)",
        },
        # --- Decisive: every accepted level must satisfy the backward-Euler
        #     residual to the requested tolerance, with the right-hand side
        #     evaluated at the implicit level. Stopping the iteration early, or
        #     sampling the source at the preceding level, breaks this.
        {
            "setup": reducer + (
                "def implicit_residual():\n"
                "    n, dl, kap, dt = 7, 1.0, 0.005, 0.05\n"
                "    bv = np.array([0.0, 1.0])\n"
                "    amps = np.array([1.5, 0.5])\n"
                "    q0 = np.linspace(0.1, 0.9, n)\n"
                "    tr = np.asarray(march_cubic_fom(q0, dt, 20, amps, dl, kap, bv, 1e-12, 50),\n"
                "                    dtype=float)\n"
                "    dx = dl / (n + 1.0)\n"
                "    s = dx * np.arange(1, n + 1) / dl\n"
                "    worst = 0.0\n"
                "    for m in range(1, 21):\n"
                "        q = tr[:, m]\n"
                "        t = m * dt\n"
                "        lap = np.empty(n)\n"
                "        lap[0] = bv[0] - 2.0 * q[0] + q[1]\n"
                "        lap[1:-1] = q[:-2] - 2.0 * q[1:-1] + q[2:]\n"
                "        lap[-1] = q[-2] - 2.0 * q[-1] + bv[1]\n"
                "        src = (amps[0] * np.sin(2.0 * np.pi * t) / (1.0 + 100.0 * (s - 0.25) ** 2)\n"
                "               + amps[1] * np.sin(4.0 * np.pi * t) / (1.0 + 100.0 * (s - 0.75) ** 2))\n"
                "        rate = kap / dx ** 2 * lap - q ** 3 + src\n"
                "        worst = max(worst, float(np.linalg.norm(q - tr[:, m - 1] - dt * rate)))\n"
                "    return int(worst < 1e-12) + 2 * int(tr.shape == (n, 21))\n"
            ),
            "call": "implicit_residual()",
            "gold_call": "3",
        },
        # Case 6: invalid negative step count.
        {
            "setup": reducer + (
                "bad = dict(base)\n"
                "bad['n_steps'] = -1\n"
                "def run_model():\n"
                "    try:\n"
                "        march_cubic_fom(q0, **bad)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_march_cubic_fom(q0, **bad)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Case 7: invalid non-positive time step.
        {
            "setup": reducer + (
                "bad = dict(base)\n"
                "bad['time_step'] = 0.0\n"
                "def run_model():\n"
                "    try:\n"
                "        march_cubic_fom(q0, **bad)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_march_cubic_fom(q0, **bad)\n"
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
