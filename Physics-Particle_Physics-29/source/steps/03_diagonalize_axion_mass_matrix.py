"""
Construct the coupled two-field mass matrix implied by the axion-mixing potential, and diagonalize it to obtain the physical mass eigenvalues and mixing angle at a given temperature.

The two canonically normalized flavor fields interact through the combined potential V = m_a(T)^2 fa^2 [1 - cos(a/fa)] + mS^2 fS^2 [1 - cos(aS/fS + a/fa)], with fS = Rf*fa. Expanding this potential to second order in the fields gives the squared-mass matrix in the (aS, a) field basis; its entries depend only on m_a(T)^2 (the input m_a_sq), mS and Rf. The matrix itself is not given here.

Diagonalizing this real symmetric 2x2 matrix gives two real eigenvalues, the physical squared masses m_H^2 (the larger) and m_L^2 (the smaller) of the instantaneous mass eigenstates, together with a mixing angle xi describing the rotation from the (aS, a) flavor basis into the mass-eigenstate basis. Convention for xi: the normalized eigenvector belonging to the heavy eigenvalue m_H^2, written in the (aS, a) basis, is (cos xi, sin xi), so the light eigenvector is (-sin xi, cos xi). An eigenvector's overall sign is arbitrary, so xi is defined only up to xi -> xi + pi; results are compared through cos(2*xi) and sin(2*xi), which this convention fixes uniquely. For example, when the heavy state is almost pure a, its eigenvector is close to (0, 1) and cos(2*xi) is close to -1. No closed-form expression for the eigenvalues or the mixing angle is given here; both must be obtained from the matrix.

Returns
-------
tuple of three floats: (m_H_sq, m_L_sq, xi), where m_H_sq >= m_L_sq are the two squared mass eigenvalues in GeV^2, and xi is the mixing angle in radians, defined so that the heavy eigenvector in the (aS, a) basis is (cos xi, sin xi).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def diagonalize_axion_mass_matrix(m_a_sq: float, mS: float, Rf: float) -> tuple:
    """Construct and diagonalize the 2x2 axion mass matrix at a given temperature.

    Parameters
    ----------
    m_a_sq : float
        The QCD-coupled field's squared mass m_a(T)^2 at the temperature of
        interest (GeV^2), a finite number > 0.
    mS : float
        The second flavor field's mass (GeV), a finite number > 0.
    Rf : float
        Dimensionless ratio fS/fa, a finite number > 0.

    Returns
    -------
    result : tuple
        (m_H_sq, m_L_sq, xi): the heavy and light squared mass eigenvalues
        (GeV^2, m_H_sq >= m_L_sq > 0) and the mixing angle xi (radians),
        defined so that the heavy eigenvector in the (aS, a) basis is
        (cos xi, sin xi).

    Raises
    ------
    ValueError
        If m_a_sq, mS, or Rf is not a finite number > 0.
    """
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_diagonalize_axion_mass_matrix(m_a_sq: float, mS: float, Rf: float) -> tuple:
    import numpy as np

    for name, value in (("m_a_sq", m_a_sq), ("mS", mS), ("Rf", Rf)):
        if not (isinstance(value, (int, float, np.floating)) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")

    m_a_sq = float(m_a_sq)
    mS = float(mS)
    Rf = float(Rf)

    maa2 = m_a_sq + mS ** 2 * Rf ** 2
    mss2 = mS ** 2
    mas2 = mS ** 2 * Rf

    M2 = np.array([[mss2, mas2], [mas2, maa2]], dtype=float)
    eigvals, eigvecs = np.linalg.eigh(M2)
    m_L_sq, m_H_sq = float(eigvals[0]), float(eigvals[1])

    v_H = eigvecs[:, 1]
    xi = float(np.arctan2(v_H[1], v_H[0]))

    return (m_H_sq, m_L_sq, xi)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark values, away from crossing ---
        {
            "setup": """import numpy as np
def to_invariant(result):
    # cos(2*xi), sin(2*xi) are invariant under xi -> xi + pi (the sign/branch
    # ambiguity of which eigenvector direction is called "positive"), unlike
    # raw xi itself, so physically equivalent mixing angles (e.g. pi/4 and
    # -3*pi/4) digest identically.
    mH2, mL2, xi = result
    return (mH2, mL2, np.cos(2.0 * xi), np.sin(2.0 * xi))

def digest(result):
    parts = to_invariant(result)
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-30
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
m_a_sq = 3.336e-29
mS = 5.775455048675713e-16
Rf = 0.02
""",
            "call": "digest(diagonalize_axion_mass_matrix(m_a_sq, mS, Rf))",
            "gold_call": "digest(_oracle_diagonalize_axion_mass_matrix(m_a_sq, mS, Rf))",
        },
        # --- Valid: a different, larger mixing ratio ---
        {
            "setup": """import numpy as np
def to_invariant(result):
    # cos(2*xi), sin(2*xi) are invariant under xi -> xi + pi (the sign/branch
    # ambiguity of which eigenvector direction is called "positive"), unlike
    # raw xi itself, so physically equivalent mixing angles (e.g. pi/4 and
    # -3*pi/4) digest identically.
    mH2, mL2, xi = result
    return (mH2, mL2, np.cos(2.0 * xi), np.sin(2.0 * xi))

def digest(result):
    parts = to_invariant(result)
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-30
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
m_a_sq = 1.0
mS = 0.7
Rf = 0.35
""",
            "call": "digest(diagonalize_axion_mass_matrix(m_a_sq, mS, Rf))",
            "gold_call": "digest(_oracle_diagonalize_axion_mass_matrix(m_a_sq, mS, Rf))",
        },
        # --- Boundary: exactly at the crossing, ordered squared masses and maximal mixing ---
        {
            "setup": """import numpy as np
mS = 1.0
Rf = 0.1
m_a_sq = mS**2*(1-Rf**2)
def compare_outputs(result):
    mH2, mL2, xi = result
    return (mH2/mS**2, mL2/mS**2, abs(np.sin(2.0*xi)))
""",
            "call": "compare_outputs(diagonalize_axion_mass_matrix(m_a_sq, mS, Rf))",
            "gold_call": "compare_outputs(_oracle_diagonalize_axion_mass_matrix(m_a_sq, mS, Rf))",
            "tol": 1e-9,
        },
        # --- Consistency: eigenvalues satisfy trace and determinant identities of the
        # original matrix (basis-independent invariants) ---
        {
            "setup": """import numpy as np
m_a_sq = 2.5
mS = 0.9
Rf = 0.22
def check(fn):
    mH2, mL2, xi = fn(m_a_sq, mS, Rf)
    maa2 = m_a_sq + mS**2*Rf**2
    mss2 = mS**2
    mas2 = mS**2*Rf
    trace_ok = abs((mH2+mL2) - (maa2+mss2)) < 1e-8*(maa2+mss2)
    det_ok = abs((mH2*mL2) - (maa2*mss2-mas2**2)) < 1e-6*abs(maa2*mss2)
    order_ok = mH2 >= mL2
    return int(trace_ok and det_ok and order_ok)
""",
            "call": "check(diagonalize_axion_mass_matrix)",
            "gold_call": "check(_oracle_diagonalize_axion_mass_matrix)",
        },
        # --- Consistency: at maximal mixing (crossing), |xi| corresponds to a 45-degree
        # rotation up to the standard pi/4 offset/sign conventions ---
        {
            "setup": """import numpy as np
mS = 1.0
Rf = 0.05
m_a_sq = mS**2*(1-Rf**2)
def check(fn):
    mH2, mL2, xi = fn(m_a_sq, mS, Rf)
    sin2xi = np.sin(2*xi)
    return int(abs(abs(sin2xi) - 1.0) < 1e-3)
""",
            "call": "check(diagonalize_axion_mass_matrix)",
            "gold_call": "check(_oracle_diagonalize_axion_mass_matrix)",
        },
        # --- Invalid: non-positive Rf ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        diagonalize_axion_mass_matrix(1.0, 1.0, -0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_diagonalize_axion_mass_matrix(1.0, 1.0, -0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive mS ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        diagonalize_axion_mass_matrix(1.0, 0.0, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_diagonalize_axion_mass_matrix(1.0, 0.0, 0.1)
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
