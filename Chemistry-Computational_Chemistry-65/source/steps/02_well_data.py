"""
Locate both minima of the potential and report their harmonic data. SPECIFICATION: each minimum is found by Newton iteration started from the point with x equal to minus one and then plus one, each with y equal to zero; at every iteration the Hessian is diagonalised, its eigenvalues are replaced by 1e-12 wherever they fall below that value, and the step is minus the inverse so modified applied to the gradient; iteration stops once the largest absolute gradient component is below 1e-15 or after three hundred iterations. The normal mode frequencies are the square roots of the Hessian eigenvalues at the converged point divided by the mass, which is one, with any negative eigenvalue first replaced by zero; they are reported in ascending order. The energy of a well is its potential plus hbar over two times the sum of its two frequencies. The result is a real array of shape (2, 6) whose first row is the well reached from negative x and whose second row is the well reached from positive x, with columns in this order: the x coordinate of the minimum, its y coordinate, the lower frequency, the higher frequency, the potential at the minimum, and the energy of the well.

Before any tunnelling calculation the two wells have to be characterised on their own, because the observable splits into a part carried by the difference between the wells and a part carried by the motion between them. The cubic term shifts each minimum slightly away from plus or minus one and changes both curvatures, so neither the positions nor the frequencies can be written down by inspection. Adding half of hbar times the sum of the frequencies to the depth is what turns a classical well depth into an energy that a quantum level can be measured against, and the difference between the two such energies is the entire asymmetry of the problem.

Returns
-------
numpy.ndarray of shape (2, 6) and real dtype.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def well_data(a: float, c: float, wy: float, hbar: float) -> "np.ndarray":
    """Locate both minima of the potential and report their harmonic data. SPECIFICATION: each minimum is found by Newton iteration started from the point with x equal to minus one and then plus one, each with y equal to zero; at every iteration the Hessian is diagonalised, its eigenvalues are replaced by 1e-12 wherever they fall below that value, and the step is minus the inverse so modified applied to the gradient; iteration stops once the largest absolute gradient component is below 1e-15 or after three hundred iterations. The normal mode frequencies are the square roots of the Hessian eigenvalues at the converged point divided by the mass, which is one, with any negative eigenvalue first replaced by zero; they are reported in ascending order. The energy of a well is its potential plus hbar over two times the sum of its two frequencies. The result is a real array of shape (2, 6) whose first row is the well reached from negative x and whose second row is the well reached from positive x, with columns in this order: the x coordinate of the minimum, its y coordinate, the lower frequency, the higher frequency, the potential at the minimum, and the energy of the well.

    Returns
    -------
    numpy.ndarray of shape (2, 6) and real dtype.

    Raises
    ------
    ValueError: if hbar is not positive and finite, or if any value it forwards to an earlier step is invalid.
    """
    return wells  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def _oracle_well_data(a: float, c: float, wy: float, hbar: float) -> "np.ndarray":
    mass = 1.0
    if not np.isfinite(hbar) or hbar <= 0.0:
        raise ValueError("hbar must be positive and finite")
    rows = []
    for start in ([-1.0, 0.0], [1.0, 0.0]):
        p = np.array(start, dtype=float)
        for _ in range(300):
            row = _oracle_potential_and_derivatives(p[None, :], a, c, wy)[0]
            g = row[1:3]
            H = np.array([[row[3], row[4]], [row[4], row[5]]])
            if np.max(np.abs(g)) < 1e-15:
                break
            w, U = np.linalg.eigh(H)
            w = np.where(w < 1e-12, 1e-12, w)
            p = p - U @ ((U.T @ g) / w)
        row = _oracle_potential_and_derivatives(p[None, :], a, c, wy)[0]
        H = np.array([[row[3], row[4]], [row[4], row[5]]])
        om = np.sqrt(np.maximum(np.linalg.eigvalsh(H), 0.0) / mass)
        rows.append([p[0], p[1], om[0], om[1], row[0],
                     row[0] + 0.5 * hbar * float(om.sum())])
    return np.array(rows, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np',
         "call": 'well_data(5e-10, 1.6, 0.55, 0.04)',
         "gold_call": '_oracle_well_data(5e-10, 1.6, 0.55, 0.04)'},   # normal: the task couplings
        {"setup": 'import numpy as np',
         "call": 'well_data(2.5e-8, 1.0, 0.8, 0.04)',
         "gold_call": '_oracle_well_data(2.5e-8, 1.0, 0.8, 0.04)'},   # edge: reference couplings, larger asymmetry
        {"setup": 'import numpy as np',
         "call": 'well_data(0.01, 1.2, 0.9, 0.2)',
         "gold_call": '_oracle_well_data(0.01, 1.2, 0.9, 0.2)'},   # edge: large asymmetry, the depth difference is resolved by the comparator
        {"setup": 'import numpy as np',
         "call": 'well_data(0.0, 1.0, 0.8, 0.04)',
         "gold_call": '_oracle_well_data(0.0, 1.0, 0.8, 0.04)'},   # boundary: symmetric limit, degenerate wells
        {"setup": 'import numpy as np\ndef _c():\n    try:\n        well_data(5e-10, 1.6, 0.55, 0.0)\n        return 0\n    except ValueError:\n        return 1\ndef _g():\n    try:\n        _oracle_well_data(5e-10, 1.6, 0.55, 0.0)\n        return 0\n    except ValueError:\n        return 1',
         "call": '_c()',
         "gold_call": '_g()'},   # invalid: hbar not positive
    ]
