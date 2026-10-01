"""
Return the image of point under the conformal map from the unit disc slit along [-1, cut_branch_point] onto the unit disc, normalized to preserve the real axis within its domain, send origin_point to zero, and fix +1.

A series in a conformal variable converges only up to the nearest singularity, so a cut left inside the disc must be moved onto its boundary before the expansion can describe the whole interior.

Returns
-------
complex: the image of the point under the slit-disc conformal map.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def map_left_hand_cut(
    point: complex,
    cut_branch_point: float,
    origin_point: float,
) -> complex:
    """Return the image of a point under the slit-disc conformal map.

    Let ``D`` be the open unit disc cut along the real segment that joins
    ``-1`` to ``cut_branch_point``. Return the image of ``point`` under the
    conformal map of ``D`` onto the unit disc that

    * takes real values on the real axis,
    * sends ``origin_point`` to ``0``,
    * leaves ``+1`` fixed.

    These three conditions fix the map uniquely. It carries the slit onto an
    arc of the unit circle, with ``cut_branch_point`` going to ``-1``. Only
    arguments of modulus at most one are accepted.

    Parameters
    ----------
    point : complex
        Point of the closed unit disc, ``abs(point) <= 1``.
    cut_branch_point : float
        Finite real number with ``-1 < cut_branch_point < origin_point``,
        where the slit opens.
    origin_point : float
        Finite real number with ``cut_branch_point < origin_point < 1``,
        mapped to the origin.

    Returns
    -------
    complex
        The image of ``point``.

    Raises
    ------
    ValueError
        If ``point`` is not a finite number of modulus at most one, if
        ``cut_branch_point`` and ``origin_point`` are not finite real
        numbers obeying ``-1 < cut_branch_point < origin_point < 1``, or if
        the image is not finite.
    """
    return 0j

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_map_left_hand_cut(
    point: complex,
    cut_branch_point: float,
    origin_point: float,
) -> complex:
    """Reference implementation of the slit-disc map."""
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

    if not _is_number(point):
        raise ValueError("point must be a finite real or complex number")
    if not (_is_real(cut_branch_point) and _is_real(origin_point)):
        raise ValueError("cut_branch_point and origin_point must be finite real numbers")
    if not (-1.0 < float(cut_branch_point) < float(origin_point) < 1.0):
        raise ValueError("require -1 < cut_branch_point < origin_point < 1")
    value = complex(point)
    if abs(value) > 1.0:
        raise ValueError("point must satisfy abs(point) <= 1")

    edge = float(cut_branch_point)
    centre = float(origin_point)

    # The slit disc is carried to an auxiliary plane in which the slit is a
    # subthreshold cut and then back with a second square-root map; composing
    # the two leaves a ratio of square roots whose arguments are
    #   A ~ (s_a - s)  and  B ~ (s_a - s_0)
    # up to one common factor, with s the image of `point`.
    first = (value * (edge - 1.0) ** 2 - edge * (value - 1.0) ** 2) * (centre - 1.0) ** 2
    second = (centre * (edge - 1.0) ** 2 - edge * (centre - 1.0) ** 2) * (value - 1.0) ** 2
    root_first = np.sqrt(complex(first))
    root_second = np.sqrt(complex(second))
    total = root_first + root_second
    if total == 0.0:
        raise ValueError("the slit-disc map is not finite at this point")
    result = (root_first - root_second) / total
    if not (np.isfinite(result.real) and np.isfinite(result.imag)):
        raise ValueError("the slit-disc map is not finite at this point")
    return complex(result)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    reduce_setup = (
        "import numpy as np\n"
        "def _scal(value):\n"
        "    arr = np.atleast_1d(np.asarray(value, dtype=complex)).ravel()\n"
        "    idx = np.arange(1.0, arr.size + 1.0)\n"
        "    return float(np.sum(np.abs(arr))\n"
        "                 + np.sum(arr.real * np.cos(idx))\n"
        "                 + np.sum(arr.imag * np.sin(idx)))\n"
        "EDGE = -0.144301069324419\n"
        "CENTRE = 0.2960981094855836\n"
    )
    status_setup = (
        "import numpy as np\n"
        "EDGE = -0.144301069324419\n"
        "CENTRE = 0.2960981094855836\n"
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
        {   # generic interior point
            "setup": reduce_setup,
            "call": "_scal(map_left_hand_cut(-0.18 - 0.84j, EDGE, CENTRE))",
            "gold_call": "_scal(_oracle_map_left_hand_cut(-0.18 - 0.84j, EDGE, CENTRE))",
        },
        {   # real interior point on the far side of the slit
            "setup": reduce_setup,
            "call": "_scal(map_left_hand_cut(0.54, EDGE, CENTRE))",
            "gold_call": "_scal(_oracle_map_left_hand_cut(0.54, EDGE, CENTRE))",
        },
        {   # the normalisation point itself
            "setup": reduce_setup,
            "call": "_scal(map_left_hand_cut(CENTRE, EDGE, CENTRE) + 0.4)",
            "gold_call": "_scal(_oracle_map_left_hand_cut(CENTRE, EDGE, CENTRE) + 0.4)",
        },
        {   # the fixed point at +1
            "setup": reduce_setup,
            "call": "_scal(map_left_hand_cut(1.0, EDGE, CENTRE))",
            "gold_call": "_scal(_oracle_map_left_hand_cut(1.0, EDGE, CENTRE))",
        },
        {   # the branch point of the slit
            "setup": reduce_setup,
            "call": "_scal(map_left_hand_cut(EDGE, EDGE, CENTRE))",
            "gold_call": "_scal(_oracle_map_left_hand_cut(EDGE, EDGE, CENTRE))",
        },
        {   # boundary of the disc, off the slit
            "setup": reduce_setup,
            "call": "_scal(map_left_hand_cut(0.6 + 0.8j, EDGE, CENTRE))",
            "gold_call": "_scal(_oracle_map_left_hand_cut(0.6 + 0.8j, EDGE, CENTRE))",
        },
        {   # a different slit and normalisation
            "setup": reduce_setup,
            "call": "_scal(map_left_hand_cut(-0.35 + 0.42j, -0.62, 0.0))",
            "gold_call": "_scal(_oracle_map_left_hand_cut(-0.35 + 0.42j, -0.62, 0.0))",
        },
        {   # small-slit limit, close to the identity
            "setup": reduce_setup,
            "call": "_scal(map_left_hand_cut(0.2 - 0.3j, -0.99, 0.95))",
            "gold_call": "_scal(_oracle_map_left_hand_cut(0.2 - 0.3j, -0.99, 0.95))",
        },
        {   # invalid: argument outside the closed disc
            "setup": status_setup,
            "call": "_status(lambda: map_left_hand_cut(1.3 - 0.2j, EDGE, CENTRE))",
            "gold_call": "_status(lambda: _oracle_map_left_hand_cut(1.3 - 0.2j, EDGE, CENTRE))",
        },
        {   # invalid: normalisation point on the wrong side of the slit
            "setup": status_setup,
            "call": "_status(lambda: map_left_hand_cut(0.2, 0.5, -0.3))",
            "gold_call": "_status(lambda: _oracle_map_left_hand_cut(0.2, 0.5, -0.3))",
        },
    ]
