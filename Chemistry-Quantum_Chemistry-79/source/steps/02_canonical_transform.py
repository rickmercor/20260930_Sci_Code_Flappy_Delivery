"""
Canonical orthogonalisation of the non-orthogonal Gaussian basis, with the near-linear-dependent directions removed and a fixed column sign.

An even-tempered Gaussian set is strongly non-orthogonal, and its most diffuse members are almost linearly dependent on one another. Working directly with the generalised eigenvalue problem in such a basis is numerically unstable, so the standard remedy is canonical orthogonalisation. Diagonalising the overlap, S u_k = s_k u_k, one keeps only the directions whose overlap eigenvalue exceeds a threshold and forms the rectangular transformation whose columns are X_k = u_k / sqrt(s_k). It satisfies X^T S X = 1 on the retained subspace, so it turns any generalised problem in the primitive basis into an ordinary one of reduced dimension.

The threshold is not a cosmetic parameter. Discarding a direction removes one continuum-like function from the description of the unbound electron, which is precisely the part of the basis the absorbing potential acts on, so resonance parameters can depend on where the cut is placed.

Columns are ordered by increasing overlap eigenvalue. An eigenvector is only defined up to an overall sign, so the sign is fixed here by requiring that the entry of largest absolute value within each column be positive; should two entries tie in absolute value, the earlier index decides.

Returns
-------
numpy.ndarray of shape (n_basis, n_kept): the canonical orthogonalisation matrix, columns ordered by increasing overlap eigenvalue and sign-fixed as described
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def canonical_transform(overlap: "np.ndarray", threshold: float) -> "np.ndarray":
    '''Canonical orthogonalisation matrix of a non-orthogonal basis.

    Parameters
    ----------
    overlap : numpy.ndarray
        Real symmetric overlap matrix S of shape (n_basis, n_basis).
    threshold : float
        Overlap eigenvalues less than or equal to this value are discarded;
        must be positive.

    Returns
    -------
    transform : numpy.ndarray
        Real array of shape (n_basis, n_kept) whose column k is u_k / sqrt(s_k),
        ordered by increasing s_k and sign-fixed so that the entry of largest
        absolute value in each column is positive.

    Raises
    ------
    ValueError
        If overlap is not a square two-dimensional array, if threshold is not
        positive, or if no overlap eigenvalue exceeds the threshold.
    '''
    return transform

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_canonical_transform(overlap: "np.ndarray", threshold: float) -> "np.ndarray":
    import numpy as np
    S = np.asarray(overlap, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("overlap must be a square two-dimensional array")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be a positive finite number")

    s, U = np.linalg.eigh(S)
    keep = s > float(threshold)
    if not keep.any():
        raise ValueError("no overlap eigenvalue exceeds the threshold")

    X = U[:, keep] / np.sqrt(s[keep])
    for k in range(X.shape[1]):
        col = X[:, k]
        lead = int(np.argmax(np.abs(col)))
        if col[lead] < 0.0:
            X[:, k] = -col
    return X

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():

    return [

        # --- Normal: the production overlap, where every direction survives ---

        {

            "setup": """import numpy as np

S_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)[0]
S_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)[0]

""",

            "call": 'canonical_transform(S_model.copy(), 1e-8)',

            "gold_call": '_oracle_canonical_transform(S_gold.copy(), 1e-8)',

            "tol": 1e-09,

        },

        # --- Normal: a compact well-conditioned set ---

        {

            "setup": """import numpy as np

S_model = basis_integrals(0.05, 2.5, 9, 2.0, 0.8, 3.5)[0]
S_gold = _oracle_basis_integrals(0.05, 2.5, 9, 2.0, 0.8, 3.5)[0]

""",

            "call": 'canonical_transform(S_model.copy(), 1e-8)',

            "gold_call": '_oracle_canonical_transform(S_gold.copy(), 1e-8)',

            "tol": 1e-10,

        },

        # --- Boundary: a threshold high enough to actually discard directions ---

        {

            "setup": """import numpy as np

S_model = basis_integrals(0.02, 1.6, 26, 1.2, 0.5, 2.0)[0]
S_gold = _oracle_basis_integrals(0.02, 1.6, 26, 1.2, 0.5, 2.0)[0]

""",

            "call": 'canonical_transform(S_model.copy(), 1e-6)',

            "gold_call": '_oracle_canonical_transform(S_gold.copy(), 1e-6)',

            "tol": 1e-09,

        },

        # --- Edge: a nearly linearly dependent set trimmed hard ---

        {

            "setup": """import numpy as np

S_model = basis_integrals(0.01, 1.35, 24, 1.2, 0.5, 2.0)[0]
S_gold = _oracle_basis_integrals(0.01, 1.35, 24, 1.2, 0.5, 2.0)[0]

""",

            "call": 'canonical_transform(S_model.copy(), 1e-4)',

            "gold_call": '_oracle_canonical_transform(S_gold.copy(), 1e-4)',

            "tol": 1e-07,

        },

        # --- Edge: a two-function basis, the smallest non-trivial case ---

        {

            "setup": """import numpy as np

S_model = basis_integrals(0.3, 1.05, 2, 0.75, 1.5, 0.9)[0]
S_gold = _oracle_basis_integrals(0.3, 1.05, 2, 0.75, 1.5, 0.9)[0]

""",

            "call": 'canonical_transform(S_model.copy(), 1e-8)',

            "gold_call": '_oracle_canonical_transform(S_gold.copy(), 1e-8)',

            "tol": 1e-12,

        },

        # --- Invalid: a threshold above every overlap eigenvalue leaves no basis ---

        {

            "setup": """import numpy as np

S_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)[0]
S_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)[0]

def run_model():

    try:

        canonical_transform(S_model.copy(), 1e6)

        return 0

    except ValueError:

        return 1

    except Exception:

        return 2

def run_oracle():

    try:

        _oracle_canonical_transform(S_gold.copy(), 1e6)

        return 0

    except ValueError:

        return 1

    except Exception:

        return 2

""",

            "call": 'run_model()',

            "gold_call": 'run_oracle()',

        },

        # --- Boundary: threshold exactly equal to an overlap eigenvalue.

        #     Use exactly representable diagonal entries so the strict > rule is

        #     tested without depending on eigensolver roundoff. ---

        {

            "setup": """import numpy as np

S = np.diag([0.25, 0.5, 1.0, 2.0])

thr_exact = 0.5

""",

            "call": "canonical_transform(S.copy(), thr_exact)",

            "gold_call": "_oracle_canonical_transform(S.copy(), thr_exact)",

            "tol": 1e-12,

        },

        # --- Invalid: a non-square overlap matrix ---

        {

            "setup": """import numpy as np

def run_model():

    try:

        canonical_transform(np.ones((3, 4)), 1e-8)

        return 0

    except ValueError:

        return 1

    except Exception:

        return 2

def run_oracle():

    try:

        _oracle_canonical_transform(np.ones((3, 4)), 1e-8)

        return 0

    except ValueError:

        return 1

    except Exception:

        return 2

""",

            "call": "run_model()",

            "gold_call": "run_oracle()",

        },

    ]
