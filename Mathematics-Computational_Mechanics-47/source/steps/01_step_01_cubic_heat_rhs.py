"""
Evaluate the semi-discrete right-hand side of a diffusion equation carrying a cubic sink and a two-lobe unsteady source.

Second-order central differences on the interior nodes of a uniform grid, with both Dirichlet values supplied through ghost nodes, reduce the partial differential equation to a system of ordinary differential equations whose right-hand side is evaluated here at one instant.

Returns
-------
np.ndarray, the nodal time derivative with shape (n,) as a float array.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Real

import math
import numpy as np


def cubic_heat_rhs(
    state: np.ndarray,
    time: float,
    amplitudes: np.ndarray,
    domain_length: float,
    diffusivity: float,
    boundary_values: np.ndarray,
) -> np.ndarray:
    """Evaluate the semi-discrete right-hand side of the cubic-reaction model.

    The field lives on the n interior nodes x_i = i * dx of (0, domain_length)
    with dx = domain_length / (n + 1) and i = 1, ..., n. The governing equation
    is

        dq/dt = diffusivity * d2q/dx2 - q**3 + s(x, time),

    the second derivative is approximated by the second-order central
    difference that uses boundary_values[0] as the ghost value at x = 0 and
    boundary_values[1] as the ghost value at x = domain_length, and the source
    is

        s(x, t) = amplitudes[0] * sin(2 * pi * t) / (1 + 100 * (x / L - 1 / 4)**2)
                + amplitudes[1] * sin(4 * pi * t) / (1 + 100 * (x / L - 3 / 4)**2)

    with L = domain_length.

    Parameters
    ----------
    state : np.ndarray
        Finite real nodal field of shape (n,) with n >= 1.
    time : float
        Finite real evaluation time.
    amplitudes : np.ndarray
        Finite real array of shape (2,) holding the two source amplitudes.
    domain_length : float
        Finite strictly positive length of the spatial domain.
    diffusivity : float
        Finite strictly positive diffusion coefficient.
    boundary_values : np.ndarray
        Finite real array of shape (2,) holding the Dirichlet values at
        x = 0 and x = domain_length, in that order.

    Returns
    -------
    derivative : np.ndarray
        Float array of shape (n,) holding dq/dt at the interior nodes.

    Raises
    ------
    ValueError
        If any array has the wrong rank or shape, if any entry is not finite,
        or if domain_length or diffusivity is not finite and positive.
    """
    return np.empty(0, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_cubic_heat_rhs(
    state: np.ndarray,
    time: float,
    amplitudes: np.ndarray,
    domain_length: float,
    diffusivity: float,
    boundary_values: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    from numbers import Real

    import math
    import numpy as np

    def _is_number(value) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool)

    def _finite_vector(name: str, values, size: int) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or array.shape[0] != size:
            raise ValueError(f"{name} must be a 1-D array of length {size}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} entries must be finite")
        return array

    field = np.asarray(state, dtype=float)
    if field.ndim != 1 or field.shape[0] < 1:
        raise ValueError("state must be a 1-D array with at least one entry")
    if not np.all(np.isfinite(field)):
        raise ValueError("state entries must be finite")
    n_nodes = field.shape[0]

    amps = _finite_vector("amplitudes", amplitudes, 2)
    bounds = _finite_vector("boundary_values", boundary_values, 2)
    if not _is_number(time) or not math.isfinite(float(time)):
        raise ValueError("time must be a finite real scalar")
    if not _is_number(domain_length) or not math.isfinite(float(domain_length)):
        raise ValueError("domain_length must be a finite real scalar")
    if not _is_number(diffusivity) or not math.isfinite(float(diffusivity)):
        raise ValueError("diffusivity must be a finite real scalar")
    length = float(domain_length)
    kappa = float(diffusivity)
    if length <= 0.0:
        raise ValueError("domain_length must be positive")
    if kappa <= 0.0:
        raise ValueError("diffusivity must be positive")

    spacing = length / (n_nodes + 1.0)
    positions = spacing * np.arange(1, n_nodes + 1, dtype=float)
    scaled = positions / length

    curvature = np.empty(n_nodes, dtype=float)
    if n_nodes == 1:
        curvature[0] = bounds[0] - 2.0 * field[0] + bounds[1]
    else:
        curvature[0] = bounds[0] - 2.0 * field[0] + field[1]
        curvature[1:-1] = field[:-2] - 2.0 * field[1:-1] + field[2:]
        curvature[-1] = field[-2] - 2.0 * field[-1] + bounds[1]

    moment = float(time)
    lobe_left = 1.0 / (1.0 + 100.0 * (scaled - 0.25) ** 2)
    lobe_right = 1.0 / (1.0 + 100.0 * (scaled - 0.75) ** 2)
    source = (amps[0] * math.sin(2.0 * math.pi * moment) * lobe_left
              + amps[1] * math.sin(4.0 * math.pi * moment) * lobe_right)

    derivative = kappa / spacing ** 2 * curvature - field ** 3 + source
    return np.asarray(derivative, dtype=float)

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
    )
    return [
        # Case 1: interior stencil with both lobes active.
        {
            "setup": reducer + (
                "state = np.array([0.2, -0.4, 0.9, 1.3, 0.05, -1.1], dtype=float)\n"
                "time = 0.37\n"
                "amplitudes = np.array([1.5, 0.5], dtype=float)\n"
                "domain_length = 1.0\n"
                "diffusivity = 0.005\n"
                "boundary_values = np.array([0.0, 1.0], dtype=float)"
            ),
            "call": (
                "_sig(cubic_heat_rhs(state, time, amplitudes, domain_length, diffusivity, "
                "boundary_values), 1e2)"
            ),
            "gold_call": (
                "_sig(_oracle_cubic_heat_rhs(state, time, amplitudes, domain_length, diffusivity, "
                "boundary_values), 1e2)"
            ),
        },
        # Case 2: single interior node, where both ghost values enter one stencil.
        {
            "setup": reducer + (
                "state = np.array([0.6], dtype=float)\n"
                "time = 0.125\n"
                "amplitudes = np.array([-2.0, 3.0], dtype=float)\n"
                "domain_length = 2.0\n"
                "diffusivity = 0.02\n"
                "boundary_values = np.array([-0.5, 1.25], dtype=float)"
            ),
            "call": (
                "_sig(cubic_heat_rhs(state, time, amplitudes, domain_length, diffusivity, "
                "boundary_values), 1e0)"
            ),
            "gold_call": (
                "_sig(_oracle_cubic_heat_rhs(state, time, amplitudes, domain_length, diffusivity, "
                "boundary_values), 1e0)"
            ),
        },
        # Case 3: the source vanishes at integer times, isolating diffusion and reaction.
        {
            "setup": reducer + (
                "state = np.linspace(0.05, 0.95, 9)\n"
                "time = 2.0\n"
                "amplitudes = np.array([7.0, -9.0], dtype=float)\n"
                "domain_length = 1.0\n"
                "diffusivity = 0.005\n"
                "boundary_values = np.array([0.0, 1.0], dtype=float)"
            ),
            "call": (
                "_sig(cubic_heat_rhs(state, time, amplitudes, domain_length, diffusivity, "
                "boundary_values), 1e0)"
            ),
            "gold_call": (
                "_sig(_oracle_cubic_heat_rhs(state, time, amplitudes, domain_length, diffusivity, "
                "boundary_values), 1e0)"
            ),
        },
        # Case 4: asymmetric boundary data separates the two ghost contributions.
        {
            "setup": reducer + (
                "state = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)\n"
                "time = 0.05\n"
                "amplitudes = np.array([0.0, 0.0], dtype=float)\n"
                "domain_length = 1.0\n"
                "diffusivity = 0.5\n"
                "boundary_values = np.array([3.0, -2.0], dtype=float)"
            ),
            "call": (
                "_sig(cubic_heat_rhs(state, time, amplitudes, domain_length, diffusivity, "
                "boundary_values), 1e1)"
            ),
            "gold_call": (
                "_sig(_oracle_cubic_heat_rhs(state, time, amplitudes, domain_length, diffusivity, "
                "boundary_values), 1e1)"
            ),
        },
        # Case 5: only the second lobe is driven, and the two lobes are not mirror images.
        {
            "setup": reducer + (
                "state = np.zeros(11, dtype=float)\n"
                "time = 0.3\n"
                "amplitudes = np.array([0.0, 1.0], dtype=float)\n"
                "domain_length = 1.0\n"
                "diffusivity = 0.005\n"
                "boundary_values = np.array([0.0, 0.0], dtype=float)"
            ),
            "call": (
                "_sig(cubic_heat_rhs(state, time, amplitudes, domain_length, diffusivity, "
                "boundary_values), 1e0)"
            ),
            "gold_call": (
                "_sig(_oracle_cubic_heat_rhs(state, time, amplitudes, domain_length, diffusivity, "
                "boundary_values), 1e0)"
            ),
        },
        # --- Decisive: with a zero field and homogeneous ghost values the whole
        #     right-hand side is the source, which pins both lobe centres, the
        #     factor 100 in each denominator and the two forcing frequencies.
        {
            "setup": reducer + (
                "def source_only():\n"
                "    n, dl = 12, 1.6\n"
                "    amps = np.array([1.5, -0.75])\n"
                "    t = 0.31\n"
                "    got = cubic_heat_rhs(np.zeros(n), t, amps, dl, 0.005, np.zeros(2))\n"
                "    dx = dl / (n + 1.0)\n"
                "    s = dx * np.arange(1, n + 1) / dl\n"
                "    want = (amps[0] * np.sin(2.0 * np.pi * t) / (1.0 + 100.0 * (s - 0.25) ** 2)\n"
                "            + amps[1] * np.sin(4.0 * np.pi * t) / (1.0 + 100.0 * (s - 0.75) ** 2))\n"
                "    return int(np.max(np.abs(np.asarray(got, dtype=float) - want)) < 1e-12)\n"
            ),
            "call": "source_only()",
            "gold_call": "1",
        },
        # --- Decisive: with no source the right-hand side is diffusion minus the
        #     cubic sink, which pins the grid spacing and the ghost-node closure.
        {
            "setup": reducer + (
                "def quiet_field():\n"
                "    n, dl, kap = 7, 1.0, 0.05\n"
                "    bv = np.array([0.4, -0.9])\n"
                "    q = np.array([0.3, -0.5, 0.8, 1.1, -0.2, 0.05, 0.65])\n"
                "    got = np.asarray(cubic_heat_rhs(q, 2.0, np.array([5.0, -4.0]), dl, kap, bv),\n"
                "                     dtype=float)\n"
                "    dx = dl / (n + 1.0)\n"
                "    lap = np.empty(n)\n"
                "    lap[0] = bv[0] - 2.0 * q[0] + q[1]\n"
                "    lap[1:-1] = q[:-2] - 2.0 * q[1:-1] + q[2:]\n"
                "    lap[-1] = q[-2] - 2.0 * q[-1] + bv[1]\n"
                "    want = kap / dx ** 2 * lap - q ** 3\n"
                "    return int(np.max(np.abs(got - want)) < 1e-12)\n"
            ),
            "call": "quiet_field()",
            "gold_call": "1",
        },
        # Case 6: invalid non-positive diffusivity.
        {
            "setup": reducer + (
                "state = np.array([0.1, 0.2], dtype=float)\n"
                "time = 0.1\n"
                "amplitudes = np.array([1.0, 1.0], dtype=float)\n"
                "domain_length = 1.0\n"
                "diffusivity = 0.0\n"
                "boundary_values = np.array([0.0, 1.0], dtype=float)\n"
                "def run_model():\n"
                "    try:\n"
                "        cubic_heat_rhs(state, time, amplitudes, domain_length, diffusivity, boundary_values)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_cubic_heat_rhs(state, time, amplitudes, domain_length, diffusivity, boundary_values)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Case 7: invalid boundary_values length.
        {
            "setup": reducer + (
                "state = np.array([0.1, 0.2], dtype=float)\n"
                "time = 0.1\n"
                "amplitudes = np.array([1.0, 1.0], dtype=float)\n"
                "domain_length = 1.0\n"
                "diffusivity = 0.005\n"
                "boundary_values = np.array([0.0, 1.0, 2.0], dtype=float)\n"
                "def run_model():\n"
                "    try:\n"
                "        cubic_heat_rhs(state, time, amplitudes, domain_length, diffusivity, boundary_values)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_cubic_heat_rhs(state, time, amplitudes, domain_length, diffusivity, boundary_values)\n"
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
