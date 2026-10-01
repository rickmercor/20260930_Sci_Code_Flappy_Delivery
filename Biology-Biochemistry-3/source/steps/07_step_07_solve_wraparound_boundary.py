"""
Locate the polarity-regulation constant at which a three-cell arc of relaxed cell pairs first brings its end cells into interaction range.

If the ends of a short arc of relaxed pairs stay out of range, a longer sheet can be assembled pair by pair and curls into a closed ring by wraparound; once they interact, that construction fails, which is read as the few-cell boundary between wraparound and inflation.

Returns
-------
float: the regulation constant at the wraparound-inflation boundary.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def solve_wraparound_boundary(
    magnitude: float,
    kernel_range: float,
    cutoff: float,
    splay_fn: "Callable[[float, float], float]",
    regulation_bracket: tuple = (1e-6, 10.0),
    tolerance: float = 1e-12,
) -> float:
    """Return the largest regulation constant for which the arc's end cells do not interact.

    ``splay_fn(magnitude, regulation_time)`` follows the contract of
    ``compute_pair_splay_angle`` with its remaining arguments fixed: it
    returns the splay angle ``psi`` in ``[pi / 2, pi)`` of a relaxed
    mirror-symmetric pair and raises ``ValueError`` when the pair has no
    splayed rest angle. A three-cell arc consists of cells 1, 2 and 3 such
    that the pairs (1, 2) and (2, 3) are both such relaxed pairs, sitting at
    the separation where ``U(r) = exp(-r) - exp(-r / kernel_range)`` is
    minimal and sharing cell 2 and its polarity, with cells 1 and 3
    distinct. Return the largest regulation constant in
    ``regulation_bracket`` for which cells 1 and 3 of the arc are not
    partners (their separation is at least ``cutoff``), counting a
    regulation constant at which ``splay_fn`` raises ``ValueError`` as one
    at which no such arc exists, with absolute accuracy ``tolerance``.

    Parameters
    ----------
    magnitude : float
        Positive polarity magnitude of all three cells.
    kernel_range : float
        Range parameter of ``U``; must exceed 1 so that ``U`` has a minimum.
    cutoff : float
        Finite interaction range, larger than the separation minimising ``U``.
    splay_fn : callable
        Function ``(magnitude, regulation_time) -> psi``.
    regulation_bracket : tuple
        ``(low, high)`` with ``0 < low < high``.
    tolerance : float
        Positive absolute accuracy of the returned regulation constant.

    Returns
    -------
    float
        The boundary regulation constant.

    Raises
    ------
    ValueError
        If ``magnitude`` is not a finite positive number, ``kernel_range``
        is not a finite number above 1, ``cutoff`` is not a finite number
        above the separation minimising ``U``, ``splay_fn`` is not callable
        or returns a value outside ``[pi / 2, pi)``, the bracket is not two
        finite numbers with ``0 < low < high``, ``tolerance`` is not a
        finite positive number, or if the end cells already interact at
        ``low`` or still do not interact at ``high``.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_solve_wraparound_boundary(
    magnitude: float,
    kernel_range: float,
    cutoff: float,
    splay_fn: "Callable[[float, float], float]",
    regulation_bracket: tuple = (1e-6, 10.0),
    tolerance: float = 1e-12,
) -> float:
    """Reference implementation (bisection on the arc end separation)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_function(value):
        return callable(value)

    if not (_is_number(magnitude) and magnitude > 0.0):
        raise ValueError("magnitude must be a finite positive number")
    if not (_is_number(kernel_range) and kernel_range > 1.0):
        raise ValueError("kernel_range must be a finite number above 1")
    if not _is_function(splay_fn):
        raise ValueError("splay_fn must be callable")
    beta = float(kernel_range)
    rest = beta * np.log(beta) / (beta - 1.0)
    if not (_is_number(cutoff) and cutoff > rest):
        raise ValueError("cutoff must be a finite number above the rest separation")
    try:
        low, high = regulation_bracket
    except (TypeError, ValueError):
        raise ValueError("regulation_bracket must hold two numbers") from None
    if not (_is_number(low) and _is_number(high) and 0.0 < low < high):
        raise ValueError("regulation_bracket must satisfy 0 < low < high")
    if not (_is_number(tolerance) and tolerance > 0.0):
        raise ValueError("tolerance must be a finite positive number")
    low, high = float(low), float(high)

    def _ends_apart(regulation):
        try:
            psi = splay_fn(float(magnitude), regulation)
        except ValueError:
            return False
        if not (_is_number(psi) and 0.5 * np.pi <= psi < np.pi):
            raise ValueError("splay_fn must return an angle in [pi / 2, pi)")
        # Cell 2's polarity makes pi - psi with the bond from cell 1 and psi
        # with the bond to cell 3, so the bonds turn by 2 psi - pi and the
        # end cells are 2 * rest * sin(psi) apart.
        return 2.0 * rest * np.sin(psi) >= cutoff

    if not _ends_apart(low):
        raise ValueError("the end cells already interact at the lower bracket end")
    if _ends_apart(high):
        raise ValueError("the end cells still do not interact at the upper bracket end")
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        if _ends_apart(middle):
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    stand_ins = (
        "import numpy as np\n"
        "def _splay(scale):\n"
        "    def fn(m, tb):\n"
        "        ratio = tb / (scale * m * m)\n"
        "        if ratio >= 1.0:\n"
        "            raise ValueError('no splayed rest angle')\n"
        "        return float(np.arccos(-ratio))\n"
        "    return fn\n"
        "def _rest(b):\n"
        "    return b * np.log(b) / (b - 1.0)\n"
        "def _matches_closure(tb, scale, m, b, c):\n"
        "    target = scale * m * m * np.sqrt(1.0 - (c / (2.0 * _rest(b))) ** 2)\n"
        "    return 1.0 if abs(tb - target) < 1e-10 else 0.0\n"
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
            "setup": stand_ins,
            "call": "solve_wraparound_boundary(0.6, 3.6, 2.3, _splay(4.0), (1e-6, 10.0), 1e-12)",
            "gold_call": "_oracle_solve_wraparound_boundary(0.6, 3.6, 2.3, _splay(4.0), (1e-6, 10.0), 1e-12)",
        },
        {
            "setup": stand_ins,
            "call": "_matches_closure(solve_wraparound_boundary(0.45, 6.3, 3.1, _splay(7.5)), 7.5, 0.45, 6.3, 3.1)",
            "gold_call": "_matches_closure(_oracle_solve_wraparound_boundary(0.45, 6.3, 3.1, _splay(7.5)), 7.5, 0.45, 6.3, 3.1)",
        },
        {
            "setup": stand_ins,
            "call": "solve_wraparound_boundary(1.2, 2.9, 1.8, _splay(0.9), (1e-4, 5.0), 1e-13)",
            "gold_call": "_oracle_solve_wraparound_boundary(1.2, 2.9, 1.8, _splay(0.9), (1e-4, 5.0), 1e-13)",
        },
        {
            "setup": stand_ins + "slow = lambda m, tb: float(np.pi / 2 + 0.5 * np.arctan(tb))\n",
            "call": "solve_wraparound_boundary(0.8, 4.4, 2.9, slow, (1e-6, 50.0), 1e-12)",
            "gold_call": "_oracle_solve_wraparound_boundary(0.8, 4.4, 2.9, slow, (1e-6, 50.0), 1e-12)",
        },
        {
            "setup": stand_ins + status,
            "call": "_status(lambda: solve_wraparound_boundary(0.6, 3.6, 3.6, _splay(4.0)))",
            "gold_call": "_status(lambda: _oracle_solve_wraparound_boundary(0.6, 3.6, 3.6, _splay(4.0)))",
        },
        {
            "setup": stand_ins + status,
            "call": "_status(lambda: solve_wraparound_boundary(0.6, 3.6, 2.3, _splay(4.0), (1e-6, 0.5)))",
            "gold_call": "_status(lambda: _oracle_solve_wraparound_boundary(0.6, 3.6, 2.3, _splay(4.0), (1e-6, 0.5)))",
        },
        {
            "setup": stand_ins + status,
            "call": "_status(lambda: solve_wraparound_boundary(0.6, 3.6, 1.2, _splay(4.0)))",
            "gold_call": "_status(lambda: _oracle_solve_wraparound_boundary(0.6, 3.6, 1.2, _splay(4.0)))",
        },
    ]
