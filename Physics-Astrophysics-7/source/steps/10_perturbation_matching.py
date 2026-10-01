"""
Return the slow-mode perturbation and the slow-mode trace metric rate implied by matching at time t, given the exact perturbation dpsi and the exact trace metric rate hdot there, and the slow-mode background psi_s with its expansion rate. Write R and I for the real and imaginary parts of psi_s, c2 and s2 for cos and sin of 2*m*t, and define Bp = (3*hubble_s/(4*m))*c2 + (k**2/(4*m**2*a_s**2))*s2, Bm = (3*hubble_s/(4*m))*s2 - (k**2/(4*m**2*a_s**2))*c2, Dp = (c2*R + s2*I)/(8*m), Dm = (c2*I - s2*R)/(8*m), Ep = (3/m_pl**2)*(I*s2 + R*c2), Em = (3/m_pl**2)*(I*c2 - R*s2). Solve the real three by three system whose matrix rows are [1+Bm, -Bp, -Dm], [-Bp, 1-Bm, -Dp] and [-Em, -Ep, 1] against the right-hand side [Re(dpsi), Im(dpsi), hdot]. The three outputs are the real part of the slow-mode perturbation, its imaginary part, and the slow-mode trace metric rate, in that order. Raise ValueError if the system is singular.

Matching the perturbations across the transition is not a copy and not a separate problem from the metric: inverting the perturbation reconstruction introduces the slow-mode trace metric rate as a third unknown, so the perturbation and the metric have to be recovered together. Unlike the background matching, the coefficients here depend only on quantities already fixed, so a single linear solve suffices.

Returns
-------
ndarray of shape (3,), float64: the real and imaginary parts of the slow-mode perturbation followed by the slow-mode trace metric rate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def perturbation_matching(psi_s: 'np.ndarray', hubble_s: 'np.ndarray', a_s: 'np.ndarray', m: float, m_pl: float, k: float, t: float, dpsi: complex, hdot: float) -> 'np.ndarray':
    """Return the slow-mode perturbation and the slow-mode trace metric rate implied by matching at time t, given the exact perturbation dpsi and the exact trace metric rate hdot there, and the slow-mode background psi_s with its expansion rate. Write R and I for the real and imaginary parts of psi_s, c2 and s2 for cos and sin of 2*m*t, and define Bp = (3*hubble_s/(4*m))*c2 + (k**2/(4*m**2*a_s**2))*s2, Bm = (3*hubble_s/(4*m))*s2 - (k**2/(4*m**2*a_s**2))*c2, Dp = (c2*R + s2*I)/(8*m), Dm = (c2*I - s2*R)/(8*m), Ep = (3/m_pl**2)*(I*s2 + R*c2), Em = (3/m_pl**2)*(I*c2 - R*s2). Solve the real three by three system whose matrix rows are [1+Bm, -Bp, -Dm], [-Bp, 1-Bm, -Dp] and [-Em, -Ep, 1] against the right-hand side [Re(dpsi), Im(dpsi), hdot]. The three outputs are the real part of the slow-mode perturbation, its imaginary part, and the slow-mode trace metric rate, in that order. Raise ValueError if the system is singular.

    Returns
    -------
    ndarray of shape (3,), float64: the real and imaginary parts of the slow-mode perturbation followed by the slow-mode trace metric rate.

    Raises
    ------
    ValueError
        If m or m_pl is not strictly positive, if a_s is not strictly positive, or if the system is singular.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_perturbation_matching(
    psi_s: "np.ndarray",
    hubble_s: "np.ndarray",
    a_s: "np.ndarray",
    m: float,
    m_pl: float,
    k: float,
    t: float,
    dpsi: complex,
    hdot: float,
) -> "np.ndarray":
    psi_s = np.asarray(psi_s, dtype=complex)
    hubble_s = float(np.asarray(hubble_s, dtype=float))
    a_s = float(np.asarray(a_s, dtype=float))
    m = float(m)
    m_pl = float(m_pl)
    k = float(k)
    t = float(t)
    dpsi = complex(dpsi)
    hdot = float(hdot)
    if not (m > 0.0 and m_pl > 0.0):
        raise ValueError("m and m_pl must be strictly positive")
    if not (a_s > 0.0):
        raise ValueError("a_s must be strictly positive")
    R = float(np.real(psi_s))
    imag_part = float(np.imag(psi_s))
    c2 = np.cos(2.0 * m * t)
    s2 = np.sin(2.0 * m * t)
    # Eqs (46)-(49), (51), (52).
    b_plus = (3.0 * hubble_s / (4.0 * m)) * c2 + (
        k**2 / (4.0 * m**2 * a_s**2)
    ) * s2
    b_minus = (3.0 * hubble_s / (4.0 * m)) * s2 - (
        k**2 / (4.0 * m**2 * a_s**2)
    ) * c2
    d_plus = (1.0 / (8.0 * m)) * (c2 * R + s2 * imag_part)
    d_minus = (1.0 / (8.0 * m)) * (c2 * imag_part - s2 * R)
    e_plus = (3.0 / m_pl**2) * (imag_part * s2 + R * c2)
    e_minus = (3.0 / m_pl**2) * (imag_part * c2 - R * s2)
    # Eq (53).
    mat = np.array(
        [
            [1.0 + b_minus, -b_plus, -d_minus],
            [-b_plus, 1.0 - b_minus, -d_plus],
            [-e_minus, -e_plus, 1.0],
        ],
        dtype=float,
    )
    if abs(np.linalg.det(mat)) < 1e-14:
        raise ValueError("perturbation matching system is singular")
    rhs = np.array([dpsi.real, dpsi.imag, hdot], dtype=float)
    return np.linalg.solve(mat, rhs)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
        {
            "setup": "import numpy as np",
            "call": "perturbation_matching(0.105+0.04j, 0.09, 2.2, 1.0, 1.0, 0.5, 5.0, 1e-3+2e-4j, 3e-3)",
            "gold_call": "_oracle_perturbation_matching(0.105+0.04j, 0.09, 2.2, 1.0, 1.0, 0.5, 5.0, 1e-3+2e-4j, 3e-3)",
        },
        {
            "setup": "import numpy as np",
            "call": "perturbation_matching(0.02-0.01j, 0.05, 1.4, 2.0, 1.0, 0.8, 1.3, 2e-3-1e-4j, -1e-3)",
            "gold_call": "_oracle_perturbation_matching(0.02-0.01j, 0.05, 1.4, 2.0, 1.0, 0.8, 1.3, 2e-3-1e-4j, -1e-3)",
        },
        {
            "setup": "import numpy as np",
            "call": "perturbation_matching(0.105+0.04j, 0.09, 2.2, 1.0, 1.0, 0.0, 0.0, 1e-3+0j, 0.0)",
            "gold_call": "_oracle_perturbation_matching(0.105+0.04j, 0.09, 2.2, 1.0, 1.0, 0.0, 0.0, 1e-3+0j, 0.0)",
        },
        {
            "setup": "import numpy as np\ndef _raises(fn):\n    try:\n        fn(0.105+0.04j, 0.09, -2.2, 1.0, 1.0, 0.5, 5.0, 1e-3+0j, 3e-3)\n        return False\n    except ValueError:\n        return True",
            "call": "_raises(perturbation_matching)",
            "gold_call": "_raises(_oracle_perturbation_matching)",
        },
    ]
