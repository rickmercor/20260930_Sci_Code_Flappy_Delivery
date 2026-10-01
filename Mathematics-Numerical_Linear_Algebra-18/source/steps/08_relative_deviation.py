"""
Express the Rayleigh quotient of a trial vector as a relative deviation from a reference eigenvalue.

Kp and Mp are the partitioned matrices, u0 is the trial vector and lam is the reference eigenvalue. Both quadratic forms of u0 are used as computed; the mass quadratic form is not assumed to equal one.

The function returns a float.

Raises ValueError if u0 is not one dimensional; if u0 does not have one entry per equation of the system; if lam is not strictly positive; or if the mass quadratic form of u0 is not strictly positive.

The Rayleigh quotient of a trial vector is the ratio of its stiffness quadratic form to its mass quadratic form, and it returns the corresponding eigenvalue exactly when the trial vector is an eigenvector. For any other vector it is stationary about the eigenvalues, so a trial vector close to an eigenvector yields a quotient close to that eigenvalue with an error of second order in the departure.

Comparing that quotient with a reference eigenvalue and dividing by the reference gives a dimensionless measure of how far the trial vector falls short of the mode the reference describes. Used with a trial vector obtained by stripping part of a reduced mode, this measures what the stripped part was contributing.

When the trial vector is a normalised eigenvector of the system, its mass quadratic form is unity and the expression simplifies. That simplification is a convenience and not an identity: a vector formed by deleting part of a normalised vector is not itself normalised, and the discrepancy grows precisely in the circumstances where the deleted part matters most. Carrying both quadratic forms costs one additional pair of products and removes the assumption.

Returns
-------
float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def relative_deviation(Kp: 'np.ndarray', Mp: 'np.ndarray', u0: 'np.ndarray',
                       lam: float) -> float:
    '''Rayleigh quotient of u0 as a relative deviation from lam.

    Parameters
    ----------
    Kp, Mp : np.ndarray
        (n, n) partitioned stiffness and mass matrices.
    u0 : np.ndarray
        (n,) trial vector.
    lam : float
        Reference eigenvalue, strictly positive.

    Returns
    -------
    float
    '''
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_relative_deviation(Kp: 'np.ndarray', Mp: 'np.ndarray', u0: 'np.ndarray',
                               lam: float) -> float:
    import numpy as np
    u0 = np.asarray(u0, dtype=float)
    if u0.ndim != 1:
        raise ValueError("u0 must be one dimensional")
    if u0.size != Kp.shape[0]:
        raise ValueError("u0 has the wrong length")
    if not lam > 0:
        raise ValueError("lam must be strictly positive")
    d = float(u0 @ Mp @ u0)
    if not d > 0:
        raise ValueError("base contribution has non-positive mass")
    return float((u0 @ Kp @ u0 - lam*d)/(lam*d))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    mode = """import numpy as np
def cand(i):
    counts = np.array([72, 72, 96, 48])
    nd = (5, 5, 4)
    S = assemble_partitioned_system(12, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)
    Kp, Mp = S[0], S[1]
    o = np.concatenate([[0], np.cumsum(counts)])
    cols = []; psi = np.zeros((Kp.shape[0], counts[3])); res = np.zeros((Kp.shape[0], counts[3]))
    for k in range(3):
        Phi = fixed_interface_modes(Kp, Mp, counts, k, nd[k])
        psi += constraint_modes(Kp, counts, k)
        res += residual_modes(Kp, Mp, counts, k, Phi)
        cols.append(Phi)
    psi[o[3]:o[4], :] = np.eye(counts[3])
    T = np.hstack(cols + [psi, res])
    lam = reduced_eigenvalues(Kp, Mp, T, 62)
    u0 = base_contribution(Kp, Mp, T, 62, i)
    return relative_deviation(Kp, Mp, u0, lam[i - 1])
def gold(i):
    counts = np.array([72, 72, 96, 48])
    nd = (5, 5, 4)
    S = _oracle_assemble_partitioned_system(12, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)
    Kp, Mp = S[0], S[1]
    o = np.concatenate([[0], np.cumsum(counts)])
    cols = []; psi = np.zeros((Kp.shape[0], counts[3])); res = np.zeros((Kp.shape[0], counts[3]))
    for k in range(3):
        Phi = _oracle_fixed_interface_modes(Kp, Mp, counts, k, nd[k])
        psi += _oracle_constraint_modes(Kp, counts, k)
        res += _oracle_residual_modes(Kp, Mp, counts, k, Phi)
        cols.append(Phi)
    psi[o[3]:o[4], :] = np.eye(counts[3])
    T = np.hstack(cols + [psi, res])
    lam = _oracle_reduced_eigenvalues(Kp, Mp, T, 62)
    u0 = _oracle_base_contribution(Kp, Mp, T, 62, i)
    return _oracle_relative_deviation(Kp, Mp, u0, lam[i - 1])
"""
    column = """import numpy as np
def cand(lam):
    counts = np.array([72, 72, 96, 48])
    S = assemble_partitioned_system(12, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)
    Kp, Mp = S[0], S[1]
    u0 = fixed_interface_modes(Kp, Mp, counts, 0, 5)[:, 0].copy()
    return relative_deviation(Kp, Mp, u0, lam)
def gold(lam):
    counts = np.array([72, 72, 96, 48])
    S = _oracle_assemble_partitioned_system(12, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)
    Kp, Mp = S[0], S[1]
    u0 = _oracle_fixed_interface_modes(Kp, Mp, counts, 0, 5)[:, 0].copy()
    return _oracle_relative_deviation(Kp, Mp, u0, lam)
"""
    invalid = """import numpy as np
def run_model(lam, two_d):
    counts = np.array([72, 72, 96, 48])
    S = assemble_partitioned_system(12, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)
    Kp, Mp = S[0], S[1]
    P = fixed_interface_modes(Kp, Mp, counts, 0, 5)
    u0 = P[:, :2].copy() if two_d else P[:, 0].copy()
    try:
        relative_deviation(Kp, Mp, u0, lam)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold(lam, two_d):
    counts = np.array([72, 72, 96, 48])
    S = _oracle_assemble_partitioned_system(12, 3, 1, 1.0, 0.6, 0.01, 210e9, 0.3, 7850.0)
    Kp, Mp = S[0], S[1]
    P = _oracle_fixed_interface_modes(Kp, Mp, counts, 0, 5)
    u0 = P[:, :2].copy() if two_d else P[:, 0].copy()
    try:
        _oracle_relative_deviation(Kp, Mp, u0, lam)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""
    return [
        # --- normal: base contribution of the fifth surrogate mode ---
        {
            "setup": mode,
            "call": "cand(5)",
            "gold_call": "gold(5)",
            "tol": 1e-7,
        },
        # --- boundary: lowest mode, where the deviation is smallest ---
        {
            "setup": mode,
            "call": "cand(1)",
            "gold_call": "gold(1)",
            "tol": 1e-7,
        },
        # --- edge: a single basis column as the trial vector ---
        {
            "setup": column,
            "call": "cand(1.0e7)",
            "gold_call": "gold(1.0e7)",
            "tol": 1e-7,
        },
        # --- invalid: non-positive reference eigenvalue ---
        {
            "setup": invalid,
            "call": "run_model(0.0, False)",
            "gold_call": "run_gold(0.0, False)",
        },
        # --- invalid: two-dimensional trial vector ---
        {
            "setup": invalid,
            "call": "run_model(1.0e7, True)",
            "gold_call": "run_gold(1.0e7, True)",
        },
    ]
