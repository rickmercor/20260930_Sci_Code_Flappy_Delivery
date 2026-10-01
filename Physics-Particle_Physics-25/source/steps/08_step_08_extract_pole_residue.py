"""
Extract the residue of a function at a simple pole from its values on a small circle about that pole.

The residue of an amplitude at a resonance pole is the product of the couplings of that resonance to the initial and final states, so it is the quantity a parametrisation valid at the pole is meant to deliver.

Returns
-------
complex: the residue of the sampled function at the given simple pole.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def extract_pole_residue(
    pole_position: complex,
    function_fn: "Callable[[np.ndarray], np.ndarray]",
    radius: float,
    nodes: int,
) -> complex:
    """Return the residue of a function at a simple pole.

    ``function_fn`` maps a one-dimensional complex array of arguments to the
    array of function values. On the closed disc of radius ``radius`` about
    ``pole_position`` the function is analytic apart from a simple pole at
    the centre. Sample it at ``nodes`` equally spaced points on that circle,
    the first of them at ``pole_position + radius``, and return the residue.

    Parameters
    ----------
    pole_position : complex
        Location of the simple pole.
    function_fn : callable
        Function of a complex array returning an array of the same length.
    radius : float
        Finite positive radius of the sampling circle.
    nodes : int
        Number of sample points, at least two.

    Returns
    -------
    complex
        The residue at ``pole_position``.

    Raises
    ------
    ValueError
        If ``pole_position`` is not a finite number, if ``function_fn`` is
        not callable or returns the wrong shape, if ``radius`` is not a
        finite positive number, if ``nodes`` is not an integer of at least
        two, or if the sampled values are not finite.
    """
    return 0j

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_extract_pole_residue(
    pole_position: complex,
    function_fn: "Callable[[np.ndarray], np.ndarray]",
    radius: float,
    nodes: int,
) -> complex:
    """Reference implementation (trapezoidal Cauchy integral on a circle)."""
    import numpy as np

    def _is_real(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_number(value):
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(np.isfinite(value))
        if isinstance(value, (complex, np.complexfloating)):
            return bool(np.isfinite(value.real) and np.isfinite(value.imag))
        return False

    def _is_function(value):
        return callable(value)

    def _is_integer(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, np.integer)))

    if not _is_number(pole_position):
        raise ValueError("pole_position must be a finite number")
    if not _is_function(function_fn):
        raise ValueError("function_fn must be callable")
    if not (_is_real(radius) and radius > 0.0):
        raise ValueError("radius must be a finite positive number")
    if not _is_integer(nodes):
        raise ValueError("nodes must be an integer")
    count = int(nodes)
    if count < 2:
        raise ValueError("nodes must be at least two")

    centre = complex(pole_position)
    angles = 2.0 * np.pi * np.arange(count, dtype=float) / float(count)
    offsets = float(radius) * np.exp(1j * angles)
    samples = np.atleast_1d(np.asarray(function_fn(centre + offsets), dtype=complex))
    if samples.shape != offsets.shape:
        raise ValueError("function_fn must return one value per sample point")
    if not np.all(np.isfinite(samples.real) & np.isfinite(samples.imag)):
        raise ValueError("the sampled values must be finite")

    # (1 / 2 pi i) * contour integral of f, with ds = i * offset * dtheta.
    result = complex(np.sum(samples * offsets) / float(count))
    if not (np.isfinite(result.real) and np.isfinite(result.imag)):
        raise ValueError("the residue estimate is not finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    common = (
        "import numpy as np\n"
        "def _scal(value):\n"
        "    arr = np.atleast_1d(np.asarray(value, dtype=complex)).ravel()\n"
        "    idx = np.arange(1.0, arr.size + 1.0)\n"
        "    return float(np.sum(np.abs(arr))\n"
        "                 + np.sum(arr.real * np.cos(idx))\n"
        "                 + np.sum(arr.imag * np.sin(idx)))\n"
        "def _simple(centre, weight, background):\n"
        "    def fn(points):\n"
        "        arr = np.asarray(points, dtype=complex)\n"
        "        return weight / (arr - centre) + background * arr\n"
        "    return fn\n"
        "def _two_poles(points):\n"
        "    arr = np.asarray(points, dtype=complex)\n"
        "    return (1.3 - 0.7j) / (arr - (0.14 - 0.25j)) + 0.9 / (arr - (0.99 + 0.31j))\n"
    )
    status = common + (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def _short(points):\n"
        "    return np.ones(1, dtype=complex)\n"
        "def _blows_up(points):\n"
        "    return np.full(np.size(points), np.inf, dtype=complex)\n"
    )
    return [
        {   # isolated simple pole with a regular background
            "setup": common,
            "call": "_scal(extract_pole_residue(0.1400030 - 0.2504040j, _simple(0.1400030 - 0.2504040j, -0.75 - 0.036j, 2.4), 0.01, 128))",
            "gold_call": "_scal(_oracle_extract_pole_residue(0.1400030 - 0.2504040j, _simple(0.1400030 - 0.2504040j, -0.75 - 0.036j, 2.4), 0.01, 128))",
        },
        {   # a second pole outside the contour must not contribute
            "setup": common,
            "call": "_scal(extract_pole_residue(0.14 - 0.25j, _two_poles, 0.05, 96))",
            "gold_call": "_scal(_oracle_extract_pole_residue(0.14 - 0.25j, _two_poles, 0.05, 96))",
        },
        {   # a much larger radius, still enclosing only the one pole
            "setup": common,
            "call": "_scal(extract_pole_residue(0.14 - 0.25j, _two_poles, 0.4, 256))",
            "gold_call": "_scal(_oracle_extract_pole_residue(0.14 - 0.25j, _two_poles, 0.4, 256))",
        },
        {   # boundary: the smallest admissible number of nodes
            "setup": common,
            "call": "_scal(extract_pole_residue(0.0 + 0.0j, _simple(0.0 + 0.0j, 1.0, 0.0), 0.25, 2))",
            "gold_call": "_scal(_oracle_extract_pole_residue(0.0 + 0.0j, _simple(0.0 + 0.0j, 1.0, 0.0), 0.25, 2))",
        },
        {   # a real pole position
            "setup": common,
            "call": "_scal(extract_pole_residue(0.7, _simple(0.7, 3.25, -1.1), 0.02, 64))",
            "gold_call": "_scal(_oracle_extract_pole_residue(0.7, _simple(0.7, 3.25, -1.1), 0.02, 64))",
        },
        {   # a tiny contour radius
            "setup": common,
            "call": "_scal(extract_pole_residue(-0.3 + 1.2j, _simple(-0.3 + 1.2j, 0.004, 5.0), 1.0e-4, 32))",
            "gold_call": "_scal(_oracle_extract_pole_residue(-0.3 + 1.2j, _simple(-0.3 + 1.2j, 0.004, 5.0), 1.0e-4, 32))",
        },
        {   # invalid: non-positive radius
            "setup": status,
            "call": "_status(lambda: extract_pole_residue(0.2 - 0.1j, _two_poles, 0.0, 64))",
            "gold_call": "_status(lambda: _oracle_extract_pole_residue(0.2 - 0.1j, _two_poles, 0.0, 64))",
        },
        {   # invalid: fewer than two nodes
            "setup": status,
            "call": "_status(lambda: extract_pole_residue(0.2 - 0.1j, _two_poles, 0.01, 1))",
            "gold_call": "_status(lambda: _oracle_extract_pole_residue(0.2 - 0.1j, _two_poles, 0.01, 1))",
        },
        {   # invalid: the callback returns the wrong number of values
            "setup": status,
            "call": "_status(lambda: extract_pole_residue(0.2 - 0.1j, _short, 0.01, 64))",
            "gold_call": "_status(lambda: _oracle_extract_pole_residue(0.2 - 0.1j, _short, 0.01, 64))",
        },
        {   # invalid: the callback returns values that are not finite
            "setup": status,
            "call": "_status(lambda: extract_pole_residue(0.2 - 0.1j, _blows_up, 0.01, 64))",
            "gold_call": "_status(lambda: _oracle_extract_pole_residue(0.2 - 0.1j, _blows_up, 0.01, 64))",
        },
    ]
