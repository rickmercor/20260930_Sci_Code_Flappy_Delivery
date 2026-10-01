"""
Short-circuit loss budget of the fourth step against the balanced mobility for a cell with symmetric barriers phi and reduction factor zeta under unity generation (material as in the seventh step): find by bisection in log10 of the mobility (cm^2/(V s)), on the bracket from three decades below the design mobility up to the ideal mobility, to an interval width of 1e-9, the crossover mobility at which the second-order bulk loss equals the sum of the first-order bulk loss and the two contact extraction losses. Return [log10 of the crossover mobility, then the five budget fractions (collected, first-order, second-order, anode, cathode) at the design mobility, then the same five at the ideal mobility]. Raise ValueError if the mobilities are not positive with the design mobility below the ideal one, the barrier is negative or twice the barrier reaches the gap, the reduction factor is not positive, material does not have 7 entries or N is not an even integer of at least 4.

Below a crossover mobility the collection loss is dominated by bimolecular recombination between photogenerated carriers and improves with faster transport; above it the loss is fixed by the contacts and no improvement of transport or reduction of the bimolecular coefficient can remove it.

Returns
-------
An (11,) float64 array [log10 mu_x, collected, first-order, second-order, anode, cathode at the design mobility, the same five at the ideal mobility].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def transport_crossover(mobility_ideal: float, mobility_design: float, barrier: float, reduction_factor: float, intervals: int, material: "np.ndarray") -> "np.ndarray":
    """Short-circuit loss budget of the fourth step against the balanced mobility for a cell
        with symmetric barriers phi and reduction factor zeta under unity generation (material
        as in the seventh step): find by bisection in log10 of the mobility (cm^2/(V s)), on the
        bracket from three decades below the design mobility up to the ideal mobility, to an
        interval width of 1e-9, the crossover mobility at which the second-order bulk loss
        equals the sum of the first-order bulk loss and the two contact extraction losses. An
        (11,) float64 array [log10 mu_x, collected, first-order, second-order, anode, cathode at
        the design mobility, the same five at the ideal mobility].
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import solve_banded


def _bisect(f, a, b, tol):
    """Root of f on [a, b] (sign change required) to an interval width below tol."""
    fa = f(a); fb = f(b)
    if not (fa * fb < 0):
        raise ValueError("no sign change on the bracket")
    while b - a > tol:
        m = 0.5 * (a + b); fm = f(m)
        if fa * fm <= 0: b, fb = m, fm
        else: a, fa = m, fm
    return 0.5 * (a + b)

def _oracle_transport_crossover(mobility_ideal: float, mobility_design: float, barrier: float, reduction_factor: float, intervals: int, material: "np.ndarray") -> "np.ndarray":
    """Short-circuit loss _budget against the balanced mobility: the mobility at which the second-order (bimolecular)
    loss equals the sum of the first-order and contact losses, and the budgets at the design and ideal mobilities."""
    mui = float(mobility_ideal); mud = float(mobility_design); phi = float(barrier); g = float(reduction_factor)
    N = int(intervals); mat = [float(v) for v in np.asarray(material, dtype=float).ravel()]
    if mui <= 0 or mud <= 0 or mud >= mui or phi < 0 or g <= 0 or len(mat) != 7 or N < 4 or N % 2 != 0:
        raise ValueError("0 < design mobility < ideal mobility, non-negative barrier, positive reduction factor, material the 7-entry tuple, intervals an even integer of at least 4")
    d_nm, eps, T, Nc, Nv, Eg, Gex = mat
    if 2.0 * phi >= Eg:
        raise ValueError("twice the barrier must lie below the _gap")
    def _budget(mu):
        pv = _oracle_device_parameters(d_nm, eps, T, Nc, Nv, Eg, phi, phi, mu, mu, g, Gex)
        sl = _oracle_steady_state(0.0, 1.0, N, pv); sd = _oracle_steady_state(0.0, 0.0, N, pv)
        return _oracle_loss_budget(sl, sd, 0.0, 1.0, pv)
    def _gap(lm):
        b = _budget(10.0 ** lm)
        return b[4] - (b[3] + b[5] + b[6])
    lm = _bisect(_gap, np.log10(mud) - 3.0, np.log10(mui), 1e-9)
    bd = _budget(mud); bi = _budget(mui)
    return np.array([lm] + list(bd[2:7]) + list(bi[2:7]), dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nmaterial = (100.0, 3.5, 300.0, 1.0e20, 1.0e20, 1.4, 1.0e22)\nmobility_ideal = 1.0\nmobility_design = 2.0e-4\nbarrier = 0.25\nreduction_factor = 1.0\nintervals = 200\n',
         'call': 'transport_crossover(mobility_ideal, mobility_design, barrier, reduction_factor, intervals, material)',
         'gold_call': '_oracle_transport_crossover(mobility_ideal, mobility_design, barrier, reduction_factor, intervals, material)'},
        {'setup': 'import numpy as np\nmaterial = (100.0, 3.5, 300.0, 1.0e20, 1.0e20, 1.4, 1.0e22)\nmobility_ideal = 0.1\nmobility_design = 1.0e-4\nbarrier = 0.15\nreduction_factor = 0.5\nintervals = 200\n',
         'call': 'transport_crossover(mobility_ideal, mobility_design, barrier, reduction_factor, intervals, material)',
         'gold_call': '_oracle_transport_crossover(mobility_ideal, mobility_design, barrier, reduction_factor, intervals, material)'},
        {'setup': 'import numpy as np\nmaterial = (150.0, 3.0, 290.0, 1.0e21, 5.0e20, 1.5, 5.0e21)\nmobility_ideal = 1.0\nmobility_design = 5.0e-4\nbarrier = 0.30\nreduction_factor = 0.2\nintervals = 200\n',
         'call': 'transport_crossover(mobility_ideal, mobility_design, barrier, reduction_factor, intervals, material)',
         'gold_call': '_oracle_transport_crossover(mobility_ideal, mobility_design, barrier, reduction_factor, intervals, material)'},
    ]
