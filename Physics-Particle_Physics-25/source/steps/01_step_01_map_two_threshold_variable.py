"""
Map the invariant mass squared onto the uniformising variable that opens the elastic and the inelastic two-particle cut together, on a selected Riemann sheet.

One conformal variable can open both two-particle cuts of a two-channel amplitude at once, so that the four Riemann sheets they generate become four disjoint regions of a single complex plane.

Returns
-------
complex: the two-threshold uniformising variable phi on the requested sheet.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def map_two_threshold_variable(
    s: complex,
    s_plus: float,
    s_inelastic: float,
    sheet: str,
) -> complex:
    """Return the two-threshold uniformising variable on one Riemann sheet.

    The variable ``phi`` inverts

        ``s = (s_plus * (1 + phi**2)**2 - 4 * s_inelastic * phi**2)
              / (1 - phi**2)**2``

    whose four branches are the four Riemann sheets opened by the two
    thresholds. ``sheet`` selects one of them:

    * ``"11"`` -- the physical branch, fixed by ``0 < phi < 1`` for real
      ``s < s_plus``;
    * ``"21"`` -- reached from ``"11"`` by continuing through the cut
      between the two thresholds, where the elastic-channel square root
      changes sign;
    * ``"22"`` -- reached from ``"11"`` by continuing through the cut above
      the inelastic threshold, where both channel square roots change sign;
    * ``"12"`` -- the remaining branch, on which only the inelastic-channel
      square root changes sign.

    A real ``s`` lying on a cut is taken as the limit from the upper half of
    the complex ``s`` plane, which is the boundary value carrying physical
    amplitudes. The two branches inside the unit disc both take the value
    zero at ``s = s_plus``, where the other two are infinite.

    Parameters
    ----------
    s : complex
        Invariant mass squared; real values are accepted.
    s_plus : float
        Elastic threshold, finite and positive.
    s_inelastic : float
        Inelastic threshold, finite and larger than ``s_plus``.
    sheet : str
        One of ``"11"``, ``"21"``, ``"22"`` and ``"12"``.

    Returns
    -------
    complex
        The variable ``phi`` on the requested sheet.

    Raises
    ------
    ValueError
        If ``s`` is not a finite number, if ``s_plus`` or ``s_inelastic`` is
        not a finite real number with ``0 < s_plus < s_inelastic``, if
        ``sheet`` is not one of the four labels, or if the requested branch
        is not finite at ``s``.
    """
    return 0j

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_map_two_threshold_variable(
    s: complex,
    s_plus: float,
    s_inelastic: float,
    sheet: str,
) -> complex:
    """Reference implementation (principal roots, upper-half-plane boundary values)."""
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

    if not _is_number(s):
        raise ValueError("s must be a finite real or complex number")
    if not (_is_real(s_plus) and s_plus > 0.0):
        raise ValueError("s_plus must be a finite positive number")
    if not (_is_real(s_inelastic) and s_inelastic > float(s_plus)):
        raise ValueError("s_inelastic must be a finite number above s_plus")
    if sheet not in ("11", "21", "22", "12"):
        raise ValueError("sheet must be one of '11', '21', '22', '12'")

    point = complex(s)
    low, high = float(s_plus), float(s_inelastic)

    def _root(value):
        # Principal square root, except that a negative real argument is the
        # limit reached from s + i0, which sits just below the branch cut.
        value = complex(value)
        if value.imag == 0.0 and value.real < 0.0:
            return complex(0.0, -np.sqrt(-value.real))
        return np.sqrt(value)

    # phi_(11) = sqrt(s_plus - s) / (sqrt(s_in - s) + sqrt(s_in - s_plus)); this
    # rationalised form stays accurate at the elastic threshold, where the
    # difference form collapses to 0/0.
    denominator = _root(high - point) + np.sqrt(high - low)
    if denominator == 0.0:
        raise ValueError("the uniformising variable is not finite at this s")
    physical = _root(low - point) / denominator

    if sheet == "11":
        result = physical
    elif sheet == "21":
        result = -physical
    else:
        if physical == 0.0:
            raise ValueError("the requested branch is not finite at this s")
        result = 1.0 / physical if sheet == "22" else -1.0 / physical

    if not (np.isfinite(result.real) and np.isfinite(result.imag)):
        raise ValueError("the requested branch is not finite at this s")
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
        "SP = 4.0 * 0.13957 ** 2\n"
        "SIN = 4.0 * 0.493677 ** 2\n"
    )
    status_setup = (
        "import numpy as np\n"
        "SP = 4.0 * 0.13957 ** 2\n"
        "SIN = 4.0 * 0.493677 ** 2\n"
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
        {   # spacelike point on the physical sheet
            "setup": reduce_setup,
            "call": "_scal(map_two_threshold_variable(-2.0, SP, SIN, '11'))",
            "gold_call": "_scal(_oracle_map_two_threshold_variable(-2.0, SP, SIN, '11'))",
        },
        {   # elastic-region boundary value: fixes the side of the cut
            "setup": reduce_setup,
            "call": "_scal(map_two_threshold_variable(0.8, SP, SIN, '11'))",
            "gold_call": "_scal(_oracle_map_two_threshold_variable(0.8, SP, SIN, '11'))",
        },
        {   # above the inelastic threshold, second sheet
            "setup": reduce_setup,
            "call": "_scal(map_two_threshold_variable(1.4, SP, SIN, '21'))",
            "gold_call": "_scal(_oracle_map_two_threshold_variable(1.4, SP, SIN, '21'))",
        },
        {   # boundary: exactly at the elastic threshold
            "setup": reduce_setup,
            "call": "_scal(map_two_threshold_variable(SP, SP, SIN, '11') + 0.25)",
            "gold_call": "_scal(_oracle_map_two_threshold_variable(SP, SP, SIN, '11') + 0.25)",
        },
        {   # complex resonance pole below the real axis, sheet (22)
            "setup": reduce_setup,
            "call": "_scal(map_two_threshold_variable(0.9375800 - 0.1271355j, SP, SIN, '22'))",
            "gold_call": "_scal(_oracle_map_two_threshold_variable(0.9375800 - 0.1271355j, SP, SIN, '22'))",
        },
        {   # remaining branch at a complex point
            "setup": reduce_setup,
            "call": "_scal(map_two_threshold_variable(0.3675 - 0.49j, SP, SIN, '12'))",
            "gold_call": "_scal(_oracle_map_two_threshold_variable(0.3675 - 0.49j, SP, SIN, '12'))",
        },
        {   # edge: exactly at the inelastic threshold on the physical sheet
            "setup": reduce_setup,
            "call": "_scal(map_two_threshold_variable(SIN, SP, SIN, '11'))",
            "gold_call": "_scal(_oracle_map_two_threshold_variable(SIN, SP, SIN, '11'))",
        },
        {   # invalid: reciprocal branch is singular at the elastic threshold
            "setup": status_setup,
            "call": "_status(lambda: map_two_threshold_variable(SP, SP, SIN, '22'))",
            "gold_call": "_status(lambda: _oracle_map_two_threshold_variable(SP, SP, SIN, '22'))",
        },
        {   # invalid: unknown sheet label
            "setup": status_setup,
            "call": "_status(lambda: map_two_threshold_variable(0.5, SP, SIN, '13'))",
            "gold_call": "_status(lambda: _oracle_map_two_threshold_variable(0.5, SP, SIN, '13'))",
        },
        {   # invalid: thresholds out of order
            "setup": status_setup,
            "call": "_status(lambda: map_two_threshold_variable(0.5, SIN, SP, '11'))",
            "gold_call": "_status(lambda: _oracle_map_two_threshold_variable(0.5, SIN, SP, '11'))",
        },
    ]
