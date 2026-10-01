"""
Assemble the multirate-infinitesimal coupling vector that carries the advective and reactive operators into a single stage of the partitioned step, and the analogous vector used by the embedded solution.

The tableau is an implicit-explicit pair of square arrays A_impl and A_expl of the same shape, with s stages and shared abscissae, together with embedding weight vectors d_impl and d_expl of length s. The stage values already computed in the current step are supplied as f_impl and f_expl, each of shape (s, m): row j holds the reactive right-hand side and the advective right-hand side respectively, evaluated at stage j. Rows beyond those already computed are not read.

The vector is the weighted combination of those stage values in which each stage value is weighted by the corresponding first-level coupling coefficient of the multirate-infinitesimal form of the tableau pair, the implicit table supplying the weights for the reactive values and the explicit table those for the advective values, with the whole sum scaled by h. These coefficients are not the tableau entries themselves; they are the quantities the multirate-infinitesimal construction derives from a pair of tableau rows for the stage in question.

For a stage row_index between 1 and s - 1, the two rows entering that construction are tableau rows row_index and row_index - 1, and the sum runs over the stage values with index strictly below row_index.

For row_index equal to s, the vector is the one used by the embedded solution. The upper row is then the embedding weight vector rather than a tableau row, the lower row is tableau row s - 2, and the sum runs over all s stage values.

The returned vector is scaled by h in both cases and is not divided by any abscissa increment.

The function raises ValueError when A_impl is not a square two-dimensional array; when A_expl does not have the same shape as A_impl; when the tableau has fewer than 3 stages; when d_impl or d_expl is not a one-dimensional array of length s; when f_impl is not a two-dimensional array with one row per tableau stage; when f_expl does not have the same shape as f_impl; when any tableau or stage-value input contains a non-finite value; when h is not a finite positive scalar; when row_index is not an integer; or when row_index lies outside the range from 1 to s.

Partitioned Runge-Kutta methods and multirate infinitesimal methods differ in how the slowly evaluated operators reach the stage being computed, and the difference is exactly the structure assembled here. In a partitioned method each stage is written as the initial value plus a combination of previously computed slopes weighted by tableau entries, so consecutive stages are independent combinations anchored at the step's starting value. In a multirate infinitesimal method each stage is instead reached by integrating a modified sub-problem forward from the previous stage value, over the interval between consecutive abscissae. What the slow operators contribute is therefore not the tableau row itself but the increment between consecutive rows, since the previous stage has already absorbed everything the earlier row contributed.

This is why the coefficients that couple the operators are differences. In the general formulation the coupling coefficients form a three-index array whose first level is exactly the consecutive row difference, with higher levels carrying polynomial dependence on time within the sub-interval; for a second-order method built from an implicit-explicit pair only the first level is nonzero, and the forcing is constant across the sub-interval.

The internal consistency condition of the framework is a statement about these differences: for each stage, the difference coefficients of each table must sum to the corresponding abscissa increment. This ensures the modified sub-problem carries the correct constant forcing to reproduce the intended stage time, and it is what allows the coupled method to inherit second order from a second-order inner solver and a second-order tableau pair. Assembling the stage contribution from undifferenced tableau rows still produces a bounded, plausible computation but violates this condition and changes the method.

The embedded solution reuses the same construction with the embedding weights in place of the final tableau row. Because the last two abscissae of a stiffly accurate padded tableau coincide, the interval for that contribution has zero length, and the sub-problem degenerates to an algebraic update anchored at the second-to-last stage rather than an integration. The difference between the resulting embedded value and the solution provides the temporal error estimate that adaptive implementations use.

Returns
-------
np.ndarray of shape (m,), the coupling vector for the requested row, scaled by the step size
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_coupling_vector(h: float, A_impl: np.ndarray, A_expl: np.ndarray,
                             d_impl: np.ndarray, d_expl: np.ndarray,
                             row_index: int, f_impl: np.ndarray,
                             f_expl: np.ndarray) -> np.ndarray:
    '''Assemble the multirate-infinitesimal coupling vector for one stage row.

    Parameters
    ----------
    h : float
        Step size, positive.
    A_impl : np.ndarray
        Implicit Butcher matrix, shape (s, s).
    A_expl : np.ndarray
        Explicit Butcher matrix, shape (s, s).
    d_impl : np.ndarray
        Implicit embedding weights, shape (s,).
    d_expl : np.ndarray
        Explicit embedding weights, shape (s,).
    row_index : int
        Stage row to assemble, between 1 and s. A value below s selects the
        stage coupling vector; the value s selects the embedding coupling vector.
    f_impl : np.ndarray
        Reactive right-hand-side values at the stages, shape (s, m). Only the
        rows entering the requested sum are read.
    f_expl : np.ndarray
        Advective right-hand-side values at the stages, shape (s, m).

    Returns
    -------
    g : np.ndarray
        Coupling vector of shape (m,), already scaled by h.
    '''
    return g  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_assemble_coupling_vector(h: float, A_impl: np.ndarray,
                                     A_expl: np.ndarray, d_impl: np.ndarray,
                                     d_expl: np.ndarray, row_index: int,
                                     f_impl: np.ndarray,
                                     f_expl: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    A_impl = np.asarray(A_impl, dtype=float)
    A_expl = np.asarray(A_expl, dtype=float)
    d_impl = np.asarray(d_impl, dtype=float)
    d_expl = np.asarray(d_expl, dtype=float)
    f_impl = np.asarray(f_impl, dtype=float)
    f_expl = np.asarray(f_expl, dtype=float)
    if A_impl.ndim != 2 or A_impl.shape[0] != A_impl.shape[1]:
        raise ValueError("A_impl must be a square 2D array")
    if A_expl.shape != A_impl.shape:
        raise ValueError("A_expl must have the same shape as A_impl")
    s = A_impl.shape[0]
    if s < 3:
        raise ValueError("the tableau must have at least 3 stages")
    if d_impl.ndim != 1 or d_impl.size != s or d_expl.shape != d_impl.shape:
        raise ValueError("d_impl and d_expl must be 1D arrays of length s")
    if f_impl.ndim != 2 or f_impl.shape[0] != s:
        raise ValueError("f_impl must be a 2D array with one row per tableau stage")
    if f_expl.shape != f_impl.shape:
        raise ValueError("f_expl must have the same shape as f_impl")
    if not (np.isfinite(A_impl).all() and np.isfinite(A_expl).all()
            and np.isfinite(d_impl).all() and np.isfinite(d_expl).all()
            and np.isfinite(f_impl).all() and np.isfinite(f_expl).all()):
        raise ValueError("all tableau and stage-value inputs must be finite")
    if not (np.isscalar(h) or np.ndim(h) == 0) or not np.isfinite(float(h)) \
            or float(h) <= 0.0:
        raise ValueError("h must be a finite positive scalar")
    if int(row_index) != row_index:
        raise ValueError("row_index must be an integer")
    row_index = int(row_index)
    if row_index < 1 or row_index > s:
        raise ValueError("row_index must lie between 1 and the number of stages")
    h = float(h)

    if row_index < s:
        upper_i = A_impl[row_index]
        upper_e = A_expl[row_index]
        lower_i = A_impl[row_index - 1]
        lower_e = A_expl[row_index - 1]
        n_terms = row_index
    else:
        upper_i = d_impl
        upper_e = d_expl
        lower_i = A_impl[s - 2]
        lower_e = A_expl[s - 2]
        n_terms = s

    g = np.zeros(f_impl.shape[1])
    for j in range(n_terms):
        g = g + ((upper_i[j] - lower_i[j]) * f_impl[j]
                 + (upper_e[j] - lower_e[j]) * f_expl[j])

    return h * g

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: locked tableau, first super-time-stepping stage row ---
        {
            "setup": """import numpy as np
sq2 = np.sqrt(2.0)
gam = (2.0 - sq2) / 2.0
dl = sq2 / 4.0
A_impl = np.zeros((6, 6))
A_expl = np.zeros((6, 6))
A_expl[1, 0] = 2.0 * gam
A_expl[2, 0] = 2.0 * gam
A_expl[3, 0] = (3.0 - 2.0 * sq2) / 6.0
A_expl[3, 2] = (3.0 + 2.0 * sq2) / 6.0
A_expl[4, 0] = (3.0 - 2.0 * sq2) / 6.0
A_expl[4, 2] = (3.0 + 2.0 * sq2) / 6.0
A_expl[5, 0] = dl
A_expl[5, 2] = dl
A_expl[5, 4] = gam
A_impl[1, 0] = 2.0 * gam
A_impl[2, 0] = gam
A_impl[2, 2] = gam
A_impl[3, 2] = 1.0
A_impl[4, 0] = dl
A_impl[4, 2] = dl
A_impl[4, 4] = gam
A_impl[5, 0] = dl
A_impl[5, 2] = dl
A_impl[5, 4] = gam
d_impl = np.array([(4.0 - sq2) / 8.0, 0.0, (4.0 - sq2) / 8.0, 0.0, dl, 0.0])
d_expl = d_impl.copy()
h = 1.5 / 73.0
rng = np.random.default_rng(2)
f_impl = rng.standard_normal((6, 12))
f_expl = rng.standard_normal((6, 12))
""",
            "call": "assemble_coupling_vector(h, A_impl, A_expl, d_impl, d_expl, 1, f_impl, f_expl)",
            "gold_call": "_oracle_assemble_coupling_vector(h, A_impl, A_expl, d_impl, d_expl, 1, f_impl, f_expl)",
        },
        # --- Normal: locked tableau, second super-time-stepping stage row ---
        {
            "setup": """import numpy as np
sq2 = np.sqrt(2.0)
gam = (2.0 - sq2) / 2.0
dl = sq2 / 4.0
A_impl = np.zeros((6, 6))
A_expl = np.zeros((6, 6))
A_expl[1, 0] = 2.0 * gam
A_expl[2, 0] = 2.0 * gam
A_expl[3, 0] = (3.0 - 2.0 * sq2) / 6.0
A_expl[3, 2] = (3.0 + 2.0 * sq2) / 6.0
A_expl[4, 0] = (3.0 - 2.0 * sq2) / 6.0
A_expl[4, 2] = (3.0 + 2.0 * sq2) / 6.0
A_expl[5, 0] = dl
A_expl[5, 2] = dl
A_expl[5, 4] = gam
A_impl[1, 0] = 2.0 * gam
A_impl[2, 0] = gam
A_impl[2, 2] = gam
A_impl[3, 2] = 1.0
A_impl[4, 0] = dl
A_impl[4, 2] = dl
A_impl[4, 4] = gam
A_impl[5, 0] = dl
A_impl[5, 2] = dl
A_impl[5, 4] = gam
d_impl = np.array([(4.0 - sq2) / 8.0, 0.0, (4.0 - sq2) / 8.0, 0.0, dl, 0.0])
d_expl = d_impl.copy()
h = 1.5 / 73.0
rng = np.random.default_rng(2)
f_impl = rng.standard_normal((6, 12))
f_expl = rng.standard_normal((6, 12))
""",
            "call": "assemble_coupling_vector(h, A_impl, A_expl, d_impl, d_expl, 3, f_impl, f_expl)",
            "gold_call": "_oracle_assemble_coupling_vector(h, A_impl, A_expl, d_impl, d_expl, 3, f_impl, f_expl)",
        },
        # --- Normal: locked tableau, final algebraic stage row ---
        {
            "setup": """import numpy as np
sq2 = np.sqrt(2.0)
gam = (2.0 - sq2) / 2.0
dl = sq2 / 4.0
A_impl = np.zeros((6, 6))
A_expl = np.zeros((6, 6))
A_expl[1, 0] = 2.0 * gam
A_expl[2, 0] = 2.0 * gam
A_expl[3, 0] = (3.0 - 2.0 * sq2) / 6.0
A_expl[3, 2] = (3.0 + 2.0 * sq2) / 6.0
A_expl[4, 0] = (3.0 - 2.0 * sq2) / 6.0
A_expl[4, 2] = (3.0 + 2.0 * sq2) / 6.0
A_expl[5, 0] = dl
A_expl[5, 2] = dl
A_expl[5, 4] = gam
A_impl[1, 0] = 2.0 * gam
A_impl[2, 0] = gam
A_impl[2, 2] = gam
A_impl[3, 2] = 1.0
A_impl[4, 0] = dl
A_impl[4, 2] = dl
A_impl[4, 4] = gam
A_impl[5, 0] = dl
A_impl[5, 2] = dl
A_impl[5, 4] = gam
d_impl = np.array([(4.0 - sq2) / 8.0, 0.0, (4.0 - sq2) / 8.0, 0.0, dl, 0.0])
d_expl = d_impl.copy()
h = 1.5 / 73.0
rng = np.random.default_rng(2)
f_impl = rng.standard_normal((6, 12))
f_expl = rng.standard_normal((6, 12))
""",
            "call": "assemble_coupling_vector(h, A_impl, A_expl, d_impl, d_expl, 5, f_impl, f_expl)",
            "gold_call": "_oracle_assemble_coupling_vector(h, A_impl, A_expl, d_impl, d_expl, 5, f_impl, f_expl)",
        },
        # --- Normal: locked tableau, embedding coupling vector ---
        {
            "setup": """import numpy as np
sq2 = np.sqrt(2.0)
gam = (2.0 - sq2) / 2.0
dl = sq2 / 4.0
A_impl = np.zeros((6, 6))
A_expl = np.zeros((6, 6))
A_expl[1, 0] = 2.0 * gam
A_expl[2, 0] = 2.0 * gam
A_expl[3, 0] = (3.0 - 2.0 * sq2) / 6.0
A_expl[3, 2] = (3.0 + 2.0 * sq2) / 6.0
A_expl[4, 0] = (3.0 - 2.0 * sq2) / 6.0
A_expl[4, 2] = (3.0 + 2.0 * sq2) / 6.0
A_expl[5, 0] = dl
A_expl[5, 2] = dl
A_expl[5, 4] = gam
A_impl[1, 0] = 2.0 * gam
A_impl[2, 0] = gam
A_impl[2, 2] = gam
A_impl[3, 2] = 1.0
A_impl[4, 0] = dl
A_impl[4, 2] = dl
A_impl[4, 4] = gam
A_impl[5, 0] = dl
A_impl[5, 2] = dl
A_impl[5, 4] = gam
d_impl = np.array([(4.0 - sq2) / 8.0, 0.0, (4.0 - sq2) / 8.0, 0.0, dl, 0.0])
d_expl = d_impl.copy()
h = 1.5 / 73.0
rng = np.random.default_rng(2)
f_impl = rng.standard_normal((6, 12))
f_expl = rng.standard_normal((6, 12))
""",
            "call": "assemble_coupling_vector(h, A_impl, A_expl, d_impl, d_expl, 6, f_impl, f_expl)",
            "gold_call": "_oracle_assemble_coupling_vector(h, A_impl, A_expl, d_impl, d_expl, 6, f_impl, f_expl)",
        },
        # --- Boundary: repeated-abscissa row whose difference coefficients cancel ---
        {
            "setup": """import numpy as np
sq2 = np.sqrt(2.0)
gam = (2.0 - sq2) / 2.0
dl = sq2 / 4.0
A_impl = np.zeros((6, 6))
A_expl = np.zeros((6, 6))
A_expl[1, 0] = 2.0 * gam
A_expl[2, 0] = 2.0 * gam
A_expl[3, 0] = (3.0 - 2.0 * sq2) / 6.0
A_expl[3, 2] = (3.0 + 2.0 * sq2) / 6.0
A_expl[4, 0] = (3.0 - 2.0 * sq2) / 6.0
A_expl[4, 2] = (3.0 + 2.0 * sq2) / 6.0
A_expl[5, 0] = dl
A_expl[5, 2] = dl
A_expl[5, 4] = gam
A_impl[1, 0] = 2.0 * gam
A_impl[2, 0] = gam
A_impl[2, 2] = gam
A_impl[3, 2] = 1.0
A_impl[4, 0] = dl
A_impl[4, 2] = dl
A_impl[4, 4] = gam
A_impl[5, 0] = dl
A_impl[5, 2] = dl
A_impl[5, 4] = gam
d_impl = np.array([(4.0 - sq2) / 8.0, 0.0, (4.0 - sq2) / 8.0, 0.0, dl, 0.0])
d_expl = d_impl.copy()
h = 1.5 / 73.0
rng = np.random.default_rng(2)
f_impl = rng.standard_normal((6, 12))
f_expl = rng.standard_normal((6, 12))
""",
            "call": "assemble_coupling_vector(h, A_impl, A_expl, d_impl, d_expl, 4, f_impl, f_expl)",
            "gold_call": "_oracle_assemble_coupling_vector(h, A_impl, A_expl, d_impl, d_expl, 4, f_impl, f_expl)",
        },
        # --- Boundary: minimal three-stage tableau ---
        {
            "setup": """import numpy as np
A_impl = np.array([[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.1, 0.4, 0.5]])
A_expl = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.25, 0.75, 0.0]])
d_impl = np.array([0.2, 0.3, 0.5])
d_expl = np.array([0.4, 0.4, 0.2])
h = 0.05
f_impl = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
f_expl = np.array([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]])
""",
            "call": "assemble_coupling_vector(h, A_impl, A_expl, d_impl, d_expl, 2, f_impl, f_expl)",
            "gold_call": "_oracle_assemble_coupling_vector(h, A_impl, A_expl, d_impl, d_expl, 2, f_impl, f_expl)",
        },
        # --- Edge: identical tableau rows give a vanishing coupling vector ---
        {
            "setup": """import numpy as np
A_impl = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.3, 0.0, 0.0]])
A_expl = np.array([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [0.6, 0.0, 0.0]])
d_impl = np.array([0.5, 0.25, 0.25])
d_expl = np.array([0.5, 0.5, 0.0])
h = 0.02
f_impl = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0], [7.0, 8.0, 9.0]])
f_expl = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, -9.0]])
""",
            "call": "assemble_coupling_vector(h, A_impl, A_expl, d_impl, d_expl, 2, f_impl, f_expl)",
            "gold_call": "_oracle_assemble_coupling_vector(h, A_impl, A_expl, d_impl, d_expl, 2, f_impl, f_expl)",
        },
        # --- Edge: single-component stage values ---
        {
            "setup": """import numpy as np
A_impl = np.array([[0.0, 0.0, 0.0], [0.4, 0.1, 0.0], [0.2, 0.3, 0.5]])
A_expl = np.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [0.3, 0.7, 0.0]])
d_impl = np.array([0.1, 0.6, 0.3])
d_expl = np.array([0.2, 0.5, 0.3])
h = 0.125
f_impl = np.array([[2.0], [-3.0], [5.0]])
f_expl = np.array([[1.0], [4.0], [-6.0]])
""",
            "call": "assemble_coupling_vector(h, A_impl, A_expl, d_impl, d_expl, 3, f_impl, f_expl)",
            "gold_call": "_oracle_assemble_coupling_vector(h, A_impl, A_expl, d_impl, d_expl, 3, f_impl, f_expl)",
        },
        # --- Invalid: non-square implicit tableau ---
        {
            "setup": """import numpy as np
A = np.zeros((4, 4))
d = np.zeros(4)
fI = np.zeros((4, 5))
fE = np.zeros((4, 5))
def run_model():
    try:
        assemble_coupling_vector(0.1, np.zeros((4, 3)), A, d, d, 1, fI, fE)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_coupling_vector(0.1, np.zeros((4, 3)), A, d, d, 1, fI, fE)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: mismatched stage-value row count ---
        {
            "setup": """import numpy as np
A = np.zeros((4, 4))
d = np.zeros(4)
fE = np.zeros((4, 5))
def run_model():
    try:
        assemble_coupling_vector(0.1, A, A, d, d, 1, np.zeros((3, 5)), fE)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_coupling_vector(0.1, A, A, d, d, 1, np.zeros((3, 5)), fE)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: embedding weight vector of the wrong length ---
        {
            "setup": """import numpy as np
A = np.zeros((4, 4))
d = np.zeros(4)
fI = np.zeros((4, 5))
fE = np.zeros((4, 5))
def run_model():
    try:
        assemble_coupling_vector(0.1, A, A, np.zeros(3), d, 1, fI, fE)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_coupling_vector(0.1, A, A, np.zeros(3), d, 1, fI, fE)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: row index above the number of stages ---
        {
            "setup": """import numpy as np
A = np.zeros((4, 4))
d = np.zeros(4)
fI = np.zeros((4, 5))
fE = np.zeros((4, 5))
def run_model():
    try:
        assemble_coupling_vector(0.1, A, A, d, d, 5, fI, fE)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_coupling_vector(0.1, A, A, d, d, 5, fI, fE)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive step size ---
        {
            "setup": """import numpy as np
A = np.zeros((4, 4))
d = np.zeros(4)
fI = np.zeros((4, 5))
fE = np.zeros((4, 5))
def run_model():
    try:
        assemble_coupling_vector(0.0, A, A, d, d, 1, fI, fE)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_coupling_vector(0.0, A, A, d, d, 1, fI, fE)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
