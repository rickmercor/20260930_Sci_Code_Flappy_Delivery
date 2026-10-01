"""
Evaluate the shifted root-mean-square loss of a program with optimized per-element energy shifts.

Reference energies of each molecule are measured from the independent-atom zero of energy, the sum of its isolated-atom energies, while the program predictions are shifted by a molecule-specific constant assembled from one contribution per element type, each counted as many times as that element occurs in the molecule. The loss is the root-mean-square difference between shifted predictions and shifted references over every retained training geometry of every molecule, with the per-element contributions chosen to make that loss as small as possible for the program at hand.

Returns
-------
np.ndarray, shape (1 + T,) float array with the minimized loss first and the fitted per-element contributions after it
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fit_shifted_loss(pred: list, ref: list, ref_shifts: "np.ndarray", counts: "np.ndarray") -> "np.ndarray":
    """Return the minimized shifted RMSE loss and the fitted per-element shifts.

    Parameters
    ----------
    pred : list
        Length N_m list; entry i is a shape (N_i,) array of program-predicted
        total energies for the retained configurations of molecule i.
    ref : list
        Length N_m list; entry i is a shape (N_i,) array of reference total
        energies for the same configurations, in the same order.
    ref_shifts : np.ndarray
        Shape (N_m,) reference shifts D_i (independent-atom energies).
    counts : np.ndarray
        Shape (N_m, T) integer numbers of atoms of each of the T element
        types in each molecule; must have full column rank T.

    Returns
    -------
    result : np.ndarray
        Shape (1 + T,) float array whose first entry is the smallest
        achievable loss and whose remaining entries are the per-element
        contributions that achieve it. The loss is the root of the mean, over
        all configurations of all molecules, of the squared difference between
        the prediction minus the molecule's assembled shift and the reference
        minus the molecule's independent-atom energy.

    Raises
    ------
    ValueError
        If the list lengths or per-molecule array lengths disagree, if
        ``counts`` does not have full column rank, or if no configuration is
        supplied.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_fit_shifted_loss(pred: list, ref: list, ref_shifts: "np.ndarray", counts: "np.ndarray") -> "np.ndarray":
    D = np.asarray(ref_shifts, dtype=float)
    N = np.asarray(counts, dtype=float)
    n_mol = D.shape[0]
    if len(pred) != n_mol or len(ref) != n_mol or N.ndim != 2 or N.shape[0] != n_mol:
        raise ValueError("pred, ref, ref_shifts and counts must describe the same molecules")
    rows, targets = [], []
    for i in range(n_mol):
        p = np.asarray(pred[i], dtype=float).reshape(-1)
        r = np.asarray(ref[i], dtype=float).reshape(-1)
        if p.shape != r.shape:
            raise ValueError("pred and ref of one molecule must have the same length")
        for pj, rj in zip(p, r):
            rows.append(N[i])
            targets.append(pj - (rj - D[i]))
    if not rows:
        raise ValueError("at least one configuration is required")
    A = np.array(rows)
    y = np.array(targets)
    if np.linalg.matrix_rank(A) < N.shape[1]:
        raise ValueError("counts must have full column rank to determine every shift")
    d, *_ = np.linalg.lstsq(A, y, rcond=None)
    residual = y - A @ d
    loss = float(np.sqrt(np.mean(residual ** 2)))
    return np.concatenate([[loss], d])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test cases for this step."""
    return [
        # three molecules with unequal retained counts
        {
            "setup": """import numpy as np
import copy
pred = [np.array([-2.51, -2.50, -2.48, -2.47]), np.array([-4.19, -4.18, -4.16, -4.19]), np.array([-8.18, -8.17, -8.16, -8.15, -8.14])]
ref = [np.array([-2.5257, -2.5210, -2.5125, -2.5087]), np.array([-4.1941, -4.1856, -4.1803, -4.1903]), np.array([-8.1968, -8.1908, -8.1863, -8.1779, -8.1790])]
ref_shifts = np.array([-2.0, -3.2, -5.2])
counts = np.array([[2, 0], [0, 2], [2, 2]])
""",
            "call": "fit_shifted_loss(*copy.deepcopy((pred, ref, ref_shifts, counts)))",
            "gold_call": "_oracle_fit_shifted_loss(pred, ref, ref_shifts, counts)",
        },
        # predictions differ from the references by exactly a per-element shift
        {
            "setup": """import numpy as np
import copy
ref = [np.array([-2.5257, -2.5210, -2.5125]), np.array([-4.1941, -4.1856]), np.array([-8.1968, -8.1908, -8.1863])]
ref_shifts = np.array([-2.0, -3.2, -5.2])
counts = np.array([[2, 0], [0, 2], [2, 2]])
pred = [ref[0] + 2 * 0.3, ref[1] - 2 * 0.1, ref[2] + 2 * 0.3 - 2 * 0.1]
""",
            "call": "fit_shifted_loss(*copy.deepcopy((pred, ref, ref_shifts, counts)))",
            "gold_call": "_oracle_fit_shifted_loss(pred, ref, ref_shifts, counts)",
        },
        # three element types, one molecule per type plus a mixed molecule
        {
            "setup": """import numpy as np
import copy
pred = [np.array([-1.0, -0.9]), np.array([-2.0, -1.8, -1.7]), np.array([-3.0]), np.array([-6.5, -6.4, -6.3, -6.1])]
ref = [np.array([-1.2, -1.15]), np.array([-2.4, -2.3, -2.1]), np.array([-3.3]), np.array([-7.0, -6.8, -6.75, -6.6])]
ref_shifts = np.array([-1.0, -2.0, -3.0, -6.0])
counts = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1], [1, 1, 1]])
""",
            "call": "fit_shifted_loss(*copy.deepcopy((pred, ref, ref_shifts, counts)))",
            "gold_call": "_oracle_fit_shifted_loss(pred, ref, ref_shifts, counts)",
        },
        # composition matrix cannot separate the two element shifts
        {
            "setup": """import numpy as np
import copy
pred = [np.array([-3.4, -3.3, -3.2]), np.array([-6.8, -6.7])]
ref = [np.array([-3.40, -3.31, -3.25]), np.array([-6.82, -6.71])]
ref_shifts = np.array([-2.6, -5.2])
counts = np.array([[1, 1], [2, 2]])
def run_model():
    try:
        fit_shifted_loss(pred, ref, ref_shifts, counts)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_fit_shifted_loss(pred, ref, ref_shifts, counts)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
