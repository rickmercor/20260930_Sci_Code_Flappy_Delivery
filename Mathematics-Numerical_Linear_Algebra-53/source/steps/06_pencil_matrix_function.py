"""
Compute A f(A^{-1} B) for f = natural log and B = A0 @ A0, using a method that remains numerically accurate even though A is severely ill-conditioned.

For a Hermitian-definite pencil (A, B) with A symmetric positive definite and B symmetric, the matrix function A f(A^{-1} B) (f applied entrywise to the eigenvalues of A^{-1}B) is well-defined whenever those eigenvalues are real and strictly positive. Here B = A0 @ A0, which is automatically symmetric positive semidefinite since A0 is symmetric.

Returns
-------
A 9x9 real symmetric matrix, A f(A^{-1} B).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pencil_matrix_function(A: "np.ndarray", A0: "np.ndarray") -> "np.ndarray":
    """Compute A f(A^{-1} B) for f = natural log and B = A0 @ A0, using a
    method that remains numerically accurate even though A is severely
    ill-conditioned.

    Parameters
    ----------
    A : np.ndarray
        (9, 9) real symmetric positive definite matrix.
    A0 : np.ndarray
        (9, 9) real symmetric matrix; B = A0 @ A0 is used as the pencil's
        second matrix.

    Returns
    -------
    result : np.ndarray
        (9, 9) real symmetric matrix, A f(A^{-1} B).

    Raises
    ------
    ValueError
        If A and A0 are not square matrices of the same shape, if A is
        not symmetric, if A is not positive definite, or if A^{-1}B has
        a non-positive eigenvalue (log undefined).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_pencil_matrix_function(A: "np.ndarray", A0: "np.ndarray") -> "np.ndarray":
    import numpy as np
    A = np.asarray(A, dtype=float)
    A0 = np.asarray(A0, dtype=float)
    if A.shape[0] != A.shape[1] or A0.shape != A.shape:
        raise ValueError("A and A0 must be square matrices of the same shape")
    if not np.allclose(A, A.T, atol=1e-6):
        raise ValueError("A must be symmetric")
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        raise ValueError("A must be positive definite")
    B = A0 @ A0
    B = 0.5 * (B + B.T)
    RA = L.T
    RAinv = np.linalg.inv(RA)
    S3 = RAinv.T @ B @ RAinv
    S3 = 0.5 * (S3 + S3.T)
    lam, Q = np.linalg.eigh(S3)
    if np.any(lam <= 0):
        raise ValueError("A^{-1}B has a non-positive eigenvalue, log undefined")
    S4 = Q @ np.diag(np.log(lam)) @ Q.T
    result = RA.T @ S4 @ RA
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    import numpy as np
    return [
        # --- Normal case: the actual task instance's A (condition 1e8), A0 ---
        {
            "setup": """import numpy as np
A = np.array([
    [77462154.88024831, -19802833.84583554, -5529549.007432951, -16181017.667507108, 22469055.958806943, -14377048.612130528, -10524248.473056965, 6590879.313706085, -13631052.980479568],
    [-19802833.84583554, 6928091.579193714, 3031339.154191876, 4940586.043301099, -4247045.959258107, 3822712.134580205, 4687727.451046943, 345214.42171932256, 3749549.4616608415],
    [-5529549.007432951, 3031339.154191876, 2311512.7724471553, 1600686.5274684187, 56950.94341247021, 1211319.2555199391, 2843485.946870081, 1530790.5469621862, 1705167.2540615643],
    [-16181017.667507108, 4940586.043301099, 1600686.5274684187, 3892870.307478407, -4283512.983337201, 3041680.4218957834, 2874618.1884204703, -659882.80917917, 2685509.710360666],
    [22469055.958806943, -4247045.959258107, 56950.94341247021, -4283512.983337201, 8104173.534632624, -4014429.579914466, -1170744.916929069, 3797276.0542691587, -3325422.191303737],
    [-14377048.612130528, 3822712.134580205, 1211319.2555199391, 3041680.4218957834, -4014429.579914466, 2686930.7564331456, 2151136.9817418903, -1037746.2175213767, 2605150.0041041006],
    [-10524248.473056965, 4687727.451046943, 2843485.946870081, 2874618.1884204703, -1170744.916929069, 2151136.9817418903, 3826297.1773055927, 1463808.1752031452, 2498909.8106303234],
    [6590879.313706085, 345214.42171932256, 1530790.5469621862, -659882.80917917, 3797276.0542691587, -1037746.2175213767, 1463808.1752031452, 2941917.0799799506, -595301.3757800902],
    [-13631052.980479568, 3749549.4616608415, 1705167.2540615643, 2685509.710360666, -3325422.191303737, 2605150.0041041006, 2498909.8106303234, -595301.3757800902, 2957162.912281232],
])
A0 = np.array([
    [-1.3970930432779, -0.8261549771188, -1.8668714533688, -0.0827245613111, -2.476541141567, -1.1596456292301, -1.2423617321294, 0.2394564017417, -1.1055910110864],
    [-0.8261549771188, 2.7584892020148, -0.2675369818222, -0.0049264367502, 3.8329486735192, 3.5626550751361, 0.0571644546668, 0.8233789783511, 3.006460304955],
    [-1.8668714533688, -0.2675369818222, 0.1780305823815, 0.3016567041018, 1.5418394826134, 1.7573355167471, -1.9968757183944, 1.6239844816119, 0.6498518767468],
    [-0.0827245613111, -0.0049264367502, 0.3016567041018, 0.6879442599283, 1.6916439262655, 1.6263468979392, -0.1545505373096, 0.3246002518952, 1.4843369563505],
    [-2.476541141567, 3.8329486735192, 1.5418394826134, 1.6916439262655, 7.8756610284986, 5.0330235817508, 2.1882007920051, 2.5005648162565, 3.6371417468254],
    [-1.1596456292301, 3.5626550751361, 1.7573355167471, 1.6263468979392, 5.0330235817508, 5.2893874801108, 2.2529533659097, 1.4992517216744, 3.2570839450532],
    [-1.2423617321294, 0.0571644546668, -1.9968757183944, -0.1545505373096, 2.1882007920051, 2.2529533659097, -1.0502686512868, 0.4452991013106, 2.0797498922368],
    [0.2394564017417, 0.8233789783511, 1.6239844816119, 0.3246002518952, 2.5005648162565, 1.4992517216744, 0.4452991013106, 1.6287561352071, 1.3603154387317],
    [-1.1055910110864, 3.006460304955, 0.6498518767468, 1.4843369563505, 3.6371417468254, 3.2570839450532, 2.0797498922368, 1.3603154387317, 2.5210867264582],
])
A_ref = A.copy()
A0_ref = A0.copy()
""",
            "call": "pencil_matrix_function(A, A0)",
            "gold_call": "_oracle_pencil_matrix_function(A_ref, A0_ref)",
            "tol": 1.0e3,
        },
        # --- Boundary case: well-conditioned A, small A0 ---
        {
            "setup": """import numpy as np
A = np.diag([4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0])
A0 = np.array([
    [1.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.2, 1.5, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.1, 0.15, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.2, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.1, 1.3, 0.05, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.05, 1.1, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.4, 0.12, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12, 1.6, 0.08],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08, 1.25],
])
A_ref = A.copy()
A0_ref = A0.copy()
""",
            "call": "pencil_matrix_function(A, A0)",
            "gold_call": "_oracle_pencil_matrix_function(A_ref, A0_ref)",
            "tol": 1e-6,
        },
        # --- Edge case: non symmetric positive definite A should raise ---
        {
            "setup": """import numpy as np
A = np.eye(9)
A[0, 1] = 2.0
A0 = np.eye(9)
A_ref = A.copy()
A0_ref = A0.copy()
def run_model():
    try:
        pencil_matrix_function(A, A0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_pencil_matrix_function(A_ref, A0_ref)
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
