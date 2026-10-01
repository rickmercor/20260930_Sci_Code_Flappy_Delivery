"""
Turn the two channel masses and the two reference points into the four real numbers that fix the geometry of the expansion. The two thresholds are four times the squared mass of the lighter and the heavier channel respectively, and the cut opening and the normalization point are passed through unchanged. The returned tuple carries the lower threshold, the upper threshold, the cut opening and the normalization point in that order, all in squared energy units. Invalid input raises ValueError when either mass is not a finite positive scalar, when either reference point is not a finite real scalar, when the two masses do not place the lower threshold strictly below the upper one or when a squared threshold overflows.

A form factor is analytic in the squared centre-of-mass energy except on cuts that open wherever a physical channel becomes accessible. A two-body channel of equal masses opens at four times the squared mass, so a system with a light and a heavy channel carries two right-hand branch points fixed entirely by those masses. The left-hand cut is inherited from the crossed process and opens at its own point on the negative or zero side. A fourth number, the normalization point, is not a feature of the amplitude at all but a choice: it is where the bounded expansion variable is defined to vanish, and moving it changes the coefficients of an expansion without changing the function being expanded.

Returns
-------
tuple of four floats: lower threshold, upper threshold, cut opening, normalization point
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def channel_geometry(light_mass, heavy_mass, cut_opening, normalization_point):
    """Return the two thresholds, the cut opening and the normalization point.

    Parameters
    ----------
    light_mass : float
        Finite positive mass of the lighter channel constituent, in energy units.
    heavy_mass : float
        Finite positive mass of the heavier channel constituent, in energy units.
    cut_opening : float
        Finite real squared energy at which the left-hand cut opens.
    normalization_point : float
        Finite real squared energy at which the expansion variable vanishes.

    Returns
    -------
    tuple of float
        The lower threshold, the upper threshold, the cut opening and the
        normalization point, in that order, in squared energy units.

    Raises
    ------
    ValueError
        If either mass is not a finite positive scalar, if either reference
        point is not a finite real scalar, or if four times the squared light
        mass is not strictly below four times the squared heavy mass, or if
        either squared threshold overflows.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_channel_geometry(light_mass, heavy_mass, cut_opening, normalization_point):
    def _real(value, name):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or isinstance(value, (str, bytes)):
            raise ValueError(name + " must be a finite real scalar")
        if np.iscomplexobj(value) and np.imag(value) != 0.0:
            raise ValueError(name + " must be a finite real scalar")
        try:
            numeric = float(np.real(value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must be a finite real scalar") from exc
        if not np.isfinite(numeric):
            raise ValueError(name + " must be a finite real scalar")
        return numeric

    light = _real(light_mass, "light_mass")
    heavy = _real(heavy_mass, "heavy_mass")
    if light <= 0.0:
        raise ValueError("light_mass must be a finite positive scalar")
    if heavy <= 0.0:
        raise ValueError("heavy_mass must be a finite positive scalar")

    opening = _real(cut_opening, "cut_opening")
    normalization = _real(normalization_point, "normalization_point")

    lower = 4.0 * light * light
    upper = 4.0 * heavy * heavy
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("the squared thresholds must be finite")
    if not lower < upper:
        raise ValueError("light_mass must place the lower threshold strictly below the upper one")

    return (float(lower), float(upper), float(opening), float(normalization))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    status_setup = (
        "import numpy as np\n"
        "def status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "    return 0\n"
    )

    def _status_case(args):
        # call runs the model's function and gold_call runs the oracle through
        # the same try/except, so the exception contract is compared against the
        # oracle rather than asserted against a constant.
        return {
            "setup": status_setup,
            "call": "status(lambda: channel_geometry" + args + ")",
            "gold_call": "status(lambda: _oracle_channel_geometry" + args + ")",
        }

    return [
        {
            "setup": "import numpy as np\n",
            "call": "[round(v, 12) for v in channel_geometry(0.13957, 0.493677, 0.0, -0.60)]",
            "gold_call": "[round(v, 12) for v in _oracle_channel_geometry(0.13957, 0.493677, 0.0, -0.60)]",
        },
        {
            "setup": "import numpy as np\n",
            "call": "[round(v, 12) for v in channel_geometry(0.13957, 0.493677, 0.0, 0.0)]",
            "gold_call": "[round(v, 12) for v in _oracle_channel_geometry(0.13957, 0.493677, 0.0, 0.0)]",
        },
        {
            "setup": "import numpy as np\n",
            "call": "[round(v, 12) for v in channel_geometry(0.5, 0.5000001, -1.25, -3.5)]",
            "gold_call": "[round(v, 12) for v in _oracle_channel_geometry(0.5, 0.5000001, -1.25, -3.5)]",
        },
        {
            "setup": "import numpy as np\n",
            "call": "[round(v, 12) for v in channel_geometry(1e-06, 1000.0, 0.0, -1.0)]",
            "gold_call": "[round(v, 12) for v in _oracle_channel_geometry(1e-06, 1000.0, 0.0, -1.0)]",
        },
        # invalid: complex cut opening
        _status_case("(0.13957, 0.493677, np.complex64(.1 + .2j), -0.60)"),
        # invalid: complex normalization point
        _status_case("(0.13957, 0.493677, 0.0, np.complex64(.1 + .2j))"),
        # invalid: complex mass
        _status_case("(np.complex64(.1 + .2j), 0.493677, 0.0, -0.60)"),
        # invalid: squared threshold overflows
        _status_case("(0.13957, 1e200, 0.0, -0.60)"),
        # invalid: masses swapped, thresholds unordered
        _status_case("(0.493677, 0.13957, 0.0, -0.60)"),
        # invalid: negative mass
        _status_case("(-0.13957, 0.493677, 0.0, -0.60)"),
        # invalid: non-finite cut opening
        _status_case("(0.13957, 0.493677, float('nan'), -0.60)"),
    ]
