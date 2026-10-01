"""
Compute the finite-community feasibility and global-stability boundaries.

Positive equilibrium densities and global stability are distinct ecological requirements; the management lower bound must respect both. Returns
-------
return certificates

Returns
-------
return certificates
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def coexistence_certificates(competition: "numpy.ndarray", regulation: "numpy.ndarray") -> "numpy.ndarray":
    """Return one certificate row per condition for one management plan.

    Work in the diagonal regulation coordinates that turn the self-regulation
    term into effort times the identity while keeping the right-hand side
    equal to one. For each condition, F is the last nonnegative effort at
    which the finite-community equilibrium branch reached from large effort
    is not strictly positive, including losses through every boundary face.
    G is the separate sufficient global-stability boundary from the smallest
    eigenvalue of the symmetric part in the same coordinates. Return
    [F, G, alpha_0, alpha_1, ..., alpha_N], where alpha_0 is the interior
    branch obstruction and alpha_i is the governing loss through face i;
    record a nongoverning or negative candidate as zero. A root or eigenvalue
    counts as real when |Im| <= 1e-8 max(1, |Re|), and candidates within
    1e-10 of the interior obstruction are nongoverning.

    Raises
    ------
    ValueError
        If competition is not a finite nonnegative panel of square matrices
        with zero diagonals, regulation is not an aligned finite vector, or
        any regulation value is not strictly positive.
    """
    return certificates

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _largest_real_certificate(matrix):
    values = np.linalg.eigvals(matrix)
    real = values.real[np.abs(values.imag) <= 1e-8*np.maximum(1.0, np.abs(values.real))]
    return float(max(0.0, np.max(real, initial=0.0)))


def _certificate_parts(interactions):
    K = np.asarray(interactions, dtype=np.float64)
    n = len(K)
    mean = K.mean(axis=0)
    centered = K-mean[None, :]
    basis = np.ones((n, 1), dtype=np.float64)/np.sqrt(n)
    for j in range(n-1):
        vector = centered@basis[:, j]
        for _ in range(2):
            vector -= basis@(basis.T@vector)
        size = np.linalg.norm(vector)
        if size <= 1e-11*max(1.0, np.linalg.norm(centered)):
            break
        basis = np.column_stack([basis, vector/size])
    baseline = _largest_real_certificate(-basis.T@centered@basis)
    parts = [baseline]
    for i in range(n):
        keep = np.arange(n) != i
        reduced = (np.ones((n-1, 1))*K[i, keep][None, :]
                   - K[np.ix_(keep, keep)])
        value = _largest_real_certificate(reduced) if n > 1 else 0.0
        parts.append(value if value > baseline+1e-10 else 0.0)
    return np.asarray(parts, dtype=np.float64)


def _oracle_coexistence_certificates(competition: "numpy.ndarray", regulation: "numpy.ndarray") -> "numpy.ndarray":
    B = np.asarray(competition, dtype=np.float64)
    d = np.asarray(regulation, dtype=np.float64)
    if (B.ndim != 3 or B.shape[0] < 1 or B.shape[1] != B.shape[2]
            or B.shape[1] < 2 or d.shape != (B.shape[1],)
            or not np.all(np.isfinite(B)) or not np.all(np.isfinite(d))
            or np.any(B < 0.0) or np.any(d <= 0.0)
            or np.max(np.abs(np.diagonal(B, axis1=1, axis2=2))) > 1e-12):
        raise ValueError("a finite nonnegative condition panel and positive regulation are required")
    rows = []
    for base in B:
        interactions = base/d[None, :]
        parts = _certificate_parts(interactions)
        feasibility = float(parts.max())
        stability = float(max(0.0, -np.linalg.eigvalsh((interactions+interactions.T)/2.0)[0]))
        rows.append(np.r_[feasibility, stability, parts])
    return np.asarray(rows, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nB=np.array([[[0.,.4,.7],[.2,0.,.5],[.9,.3,0.]],[[0.,.6,.4],[.3,0.,.8],[.7,.2,0.]]])\nd=np.array([.8,1.2,.9])",
            "call": "coexistence_certificates(B,d)",
            "gold_call": "_oracle_coexistence_certificates(B,d)",
            "tol": 1e-8,
        },
        {
            "setup": "import numpy as np\nB=np.array([[[0.,.2],[.7,0.]]]);d=np.array([1.,1.])",
            "call": "coexistence_certificates(B,d)",
            "gold_call": "_oracle_coexistence_certificates(B,d)",
            "tol": 1e-9,
        },
        {
            "setup": "import numpy as np\nB=np.zeros((3,4,4));B[:,~np.eye(4,dtype=bool)]=np.array([.2,.5,.8])[:,None]\nd=np.array([.7,1.1,.9,1.3])",
            "call": "coexistence_certificates(B,d)",
            "gold_call": "_oracle_coexistence_certificates(B,d)",
            "tol": 1e-8,
        },
        {
            "setup": "import numpy as np\nB=np.array([[[0.,-.2],[.7,0.]]]);d=np.array([1.,1.])\ndef run(fn):\n    try: fn(B,d)\n    except ValueError: return 1.0\n    return 0.0",
            "call": "run(coexistence_certificates)",
            "gold_call": "run(_oracle_coexistence_certificates)",
            "tol": 0.0,
        },
    ]
