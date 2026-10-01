"""
Eliminate the pressure variable from the mixed system while retaining the continuous P1 zero-mean pressure constraint.

The mixed eigenproblem has a displacement mass form but no pressure mass contribution. The pressure therefore acts as an auxiliary variable in the coupled system. Restricting the pressure to the paper's zero-mean space and eliminating it gives an equivalent displacement-only generalized eigenproblem. The underlying mixed eigenproblem and zero-mean pressure space are given in the paper.

Returns
-------
dict, containing the zero-mean pressure transform and reduced operators
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def pressure_condense(
    B,
    C,
    pressure_mass,
):
    """
    Prepare the zero-mean pressure Schur complement.

    Parameters
    ----------
    B : scipy.sparse.spmatrix
        Global divergence matrix.
    C : scipy.sparse.spmatrix
        Global pressure bilinear-form matrix.
    pressure_mass : scipy.sparse.spmatrix
        Continuous P1 pressure mass matrix.

    Returns
    -------
    data : dict
        Reduced pressure operators for the zero-mean pressure subspace,
        containing the following required keys:

        - ``Z`` : scipy.sparse.spmatrix
            Zero-mean pressure transformation matrix mapping reduced
            pressure coordinates to the full pressure space.
        - ``Br`` : scipy.sparse.spmatrix
            Reduced displacement-pressure coupling matrix, defined by
            ``Br = Z.T @ B``.
        - ``Cr`` : scipy.sparse.spmatrix
            Reduced pressure matrix, defined by
            ``Cr = Z.T @ C @ Z``.

    Raises
    ------
    ValueError
        If the pressure mean constraint is invalid or if the pressure
        matrices have incompatible shapes.
    """
    return data

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_pressure_condense(B, C, pressure_mass):
    """Reference implementation."""
    import scipy.sparse as sp

    B = sp.csr_matrix(B, dtype=float)
    C = sp.csr_matrix(C, dtype=float)
    Q = sp.csr_matrix(pressure_mass, dtype=float)

    if C.shape[0] != C.shape[1]:
        raise ValueError("C must be square")

    if Q.shape != C.shape:
        raise ValueError("pressure_mass must match C")

    if B.shape[0] != C.shape[0]:
        raise ValueError("B and C dimensions are incompatible")

    r = np.asarray(Q.sum(axis=0), dtype=float).reshape(-1)

    if r.size == 0:
        raise ValueError("pressure space must be nonempty")

    if not np.all(np.isfinite(r)) or np.linalg.norm(r) <= 1e-14:
        raise ValueError("invalid pressure mean vector")

    if abs(r[-1]) <= 1e-14:
        raise ValueError("invalid pressure mean vector")

    n = r.size

    # p = Z q, enforcing r^T p = 0.
    Z = sp.lil_matrix((n, n - 1), dtype=float)

    for i in range(n - 1):
        Z[i, i] = 1.0
        Z[n - 1, i] = -r[i] / r[-1]

    Z = Z.tocsr()

    Br = Z.T @ B
    Cr = Z.T @ C @ Z

    return {
        "Z": Z,
        "Br": Br.tocsr(),
        "Cr": Cr.tocsr(),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
import scipy.sparse as sp

B = sp.csr_matrix([
    [1., 0., -1.],
    [0., 1., -1.]
])

C = sp.csr_matrix([
    [1., 0.],
    [0., 1.]
])

Q = sp.csr_matrix([
    [2., 1.],
    [1., 2.]
])

def pack_condensed(data):
    return np.concatenate([
        np.asarray(data["Z"].toarray(), dtype=float).ravel(),
        np.asarray(data["Br"].toarray(), dtype=float).ravel(),
        np.asarray(data["Cr"].toarray(), dtype=float).ravel(),
    ])
""",
            "call": "pack_condensed(pressure_condense(B, C, Q))",
            "gold_call": "pack_condensed(_oracle_pressure_condense(B, C, Q))",
        },
        {
            "setup": """import numpy as np
import scipy.sparse as sp

B = sp.csr_matrix([
    [1., 0.]
])

C = sp.csr_matrix([
    [2.]
])

Q = sp.csr_matrix([
    [3.]
])

def pack_condensed(data):
    return np.concatenate([
        np.asarray(data["Z"].toarray(), dtype=float).ravel(),
        np.asarray(data["Br"].toarray(), dtype=float).ravel(),
        np.asarray(data["Cr"].toarray(), dtype=float).ravel(),
    ])
""",
            "call": "pack_condensed(pressure_condense(B, C, Q))",
            "gold_call": "pack_condensed(_oracle_pressure_condense(B, C, Q))",
        },
        {
            "setup": """import numpy as np
import scipy.sparse as sp

B = sp.csr_matrix([
    [1., 2.],
    [2., 1.]
])

C = sp.eye(2, format="csr")

Q = sp.eye(2, format="csr")

def pack_condensed(data):
    return np.concatenate([
        np.asarray(data["Z"].toarray(), dtype=float).ravel(),
        np.asarray(data["Br"].toarray(), dtype=float).ravel(),
        np.asarray(data["Cr"].toarray(), dtype=float).ravel(),
    ])
""",
            "call": "pack_condensed(pressure_condense(B, C, Q))",
            "gold_call": "pack_condensed(_oracle_pressure_condense(B, C, Q))",
        },
        {
            "setup": """import numpy as np
import scipy.sparse as sp

B = sp.csr_matrix([[1., 2.]])
C = sp.csr_matrix([[1.]])
Q = sp.csr_matrix([[0.]])

def run_model():
    try:
        pressure_condense(B, C, Q)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_pressure_condense(B, C, Q)
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
