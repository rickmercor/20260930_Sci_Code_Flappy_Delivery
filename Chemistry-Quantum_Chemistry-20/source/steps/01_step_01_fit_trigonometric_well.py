"""
Fit the exactly solvable trigonometric double well to a symmetric proton-transfer scan and return the confining half-width with the two potential parameters.

The proton coordinate between the two heavy atoms is mapped onto a finite interval, and the trigonometric double well keeps its exact eigenfunctions only for an integer order parameter, so the width of that interval is the quantity adjusted to the scan.

Returns
-------
np.ndarray: float array [L_angstrom, m, p] of the fitted trigonometric double well.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fit_trigonometric_well(
    barrier_kcal_per_mol: float,
    minimum_offset_angstrom: float,
    donor_acceptor_distance_angstrom: float,
) -> "np.ndarray":
    """Return the confining half-width and the parameters of the fitted trigonometric double well.

    The proton, of mass ``M``, moves on ``-L <= X <= L`` between the heavy
    atoms. With ``x = pi X / (2 L)`` and energies measured in units of
    ``hbar**2 pi**2 / (8 M L**2)``, the Schrodinger equation becomes
    ``psi'' + (eps - U(x)) psi = 0`` on ``-pi/2 < x < pi/2`` with the
    potential ``U(x) = (m**2 - 1/4) tan(x)**2 - p**2 sin(x)**2``, where the
    order ``m`` is a positive integer and ``p > 0``. The fitted potential
    reproduces exactly the barrier height ``U(0) - U(x_min)`` and the
    position ``X_min`` of the minimum with ``X_min > 0``. For a trial
    half-width these two conditions give ``m`` as a real number that grows
    monotonically with ``L``; the returned ``L`` is the half-width closest to
    half the donor-acceptor distance at which ``m`` is an integer of at least
    one (the smaller half-width on an exact tie), and ``p`` is the value
    that goes with it. Use ``hbar = 1.054571817e-34`` J s,
    ``M = 1.67262192369e-27`` kg, the Avogadro constant
    ``6.02214076e23`` per mol and ``1 kcal = 4184`` J.

    Parameters
    ----------
    barrier_kcal_per_mol : float
        Barrier height of the scan at the midpoint, in kcal/mol.
    minimum_offset_angstrom : float
        Distance of each minimum from the midpoint, in angstrom.
    donor_acceptor_distance_angstrom : float
        Approximate heavy-atom distance, in angstrom.

    Returns
    -------
    np.ndarray
        Float array ``[L, m, p]`` with ``L`` in angstrom, relative accuracy
        of ``1e-12`` or better in ``L`` and ``p``.

    Raises
    ------
    ValueError
        If an argument is not a finite positive real number (booleans are
        rejected), or if the minimum offset is not smaller than half the
        donor-acceptor distance.
    """
    return fit

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_fit_trigonometric_well(
    barrier_kcal_per_mol: float,
    minimum_offset_angstrom: float,
    donor_acceptor_distance_angstrom: float,
) -> "np.ndarray":
    """Reference implementation: bracketed root search for each integer order next to the trial one."""
    import math
    import numpy as np
    from scipy.optimize import brentq

    hbar, proton_mass, _, avogadro, kcal = _well_constants()
    inputs = (barrier_kcal_per_mol, minimum_offset_angstrom, donor_acceptor_distance_angstrom)
    for value in inputs:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("scan inputs must be real numbers")
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError("scan inputs must be finite and positive")
    offset = float(minimum_offset_angstrom)
    target = 0.5 * float(donor_acceptor_distance_angstrom)
    if offset >= target:
        raise ValueError("the minimum offset must be smaller than half the distance")
    barrier = float(barrier_kcal_per_mol) * kcal / avogadro

    def _reduced_well(width):
        # Barrier height B, cos^2 of the minimum position and the real order m at half-width `width`.
        unit = hbar ** 2 * math.pi ** 2 / (8.0 * proton_mass * (width * 1.0e-10) ** 2)
        height = barrier / unit
        cos2 = math.cos(0.5 * math.pi * offset / width) ** 2
        return height, cos2, math.sqrt(height * cos2 * cos2 / (1.0 - cos2) ** 2 + 0.25)

    trial = _reduced_well(target)[2]
    best = None
    for order in sorted({math.floor(trial), math.ceil(trial)}):
        if order < 1:
            continue
        low, high = offset * (1.0 + 1.0e-12), max(target, 2.0 * offset)
        while _reduced_well(high)[2] <= order:
            high *= 2.0
        width = brentq(lambda w: _reduced_well(w)[2] - order, low, high,
                       xtol=1.0e-15, rtol=4.0 * np.finfo(float).eps, maxiter=500)
        key = (abs(width - target), width)
        if best is None or key < best[0]:
            best = (key, width, order)
    _, width, order = best
    height, cos2, _ = _reduced_well(width)
    return np.array([width, float(order), math.sqrt(height) / (1.0 - cos2)])


def _well_constants() -> tuple:
    """CODATA 2018 hbar (J s), proton mass (kg), Boltzmann constant (J/K), Avogadro constant (1/mol) and kcal (J)."""
    return 1.054571817e-34, 1.67262192369e-27, 1.380649e-23, 6.02214076e23, 4184.0

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    reduce = (
        "import numpy as np\n"
        "def _sig(v):\n"
        "    v = np.asarray(v, dtype=float)\n"
        "    if v.shape != (3,):\n"
        "        return -1.0\n"
        "    return float(v.size + 1.0e3 * v[0] + v[1] + 10.0 * v[2])\n"
    )
    status = (
        "import numpy as np\n"
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
            "setup": reduce,
            "call": "_sig(fit_trigonometric_well(5.0, 0.55, 2.78))",
            "gold_call": "_sig(_oracle_fit_trigonometric_well(5.0, 0.55, 2.78))",
        },
        {
            "setup": reduce,
            "call": "_sig(fit_trigonometric_well(8.0, 0.42, 2.80))",
            "gold_call": "_sig(_oracle_fit_trigonometric_well(8.0, 0.42, 2.80))",
        },
        {
            "setup": reduce,
            "call": "_sig(fit_trigonometric_well(0.05, 0.30, 1.30))",
            "gold_call": "_sig(_oracle_fit_trigonometric_well(0.05, 0.30, 1.30))",
        },
        {
            "setup": reduce,
            "call": "_sig(fit_trigonometric_well(1.3, 1.0, 3.15))",
            "gold_call": "_sig(_oracle_fit_trigonometric_well(1.3, 1.0, 3.15))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "float(fit_trigonometric_well(5.0, 0.55, 2.7955549381646616)[2])",
            "gold_call": "float(_oracle_fit_trigonometric_well(5.0, 0.55, 2.7955549381646616)[2])",
        },
        {
            "setup": status,
            "call": "_status(lambda: fit_trigonometric_well(5.0, 1.39, 2.78))",
            "gold_call": "_status(lambda: _oracle_fit_trigonometric_well(5.0, 1.39, 2.78))",
        },
        {
            "setup": status,
            "call": "_status(lambda: fit_trigonometric_well(True, 0.55, 2.78))",
            "gold_call": "_status(lambda: _oracle_fit_trigonometric_well(True, 0.55, 2.78))",
        },
    ]
