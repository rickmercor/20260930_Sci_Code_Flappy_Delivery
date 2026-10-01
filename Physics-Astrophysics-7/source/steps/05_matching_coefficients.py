"""
Return the four coefficients that turn the reconstruction into a linear system for the real and imaginary parts of the slow-mode wavefunction at time t. The first two are the coefficients multiplying the unknown slow-mode parts; the last two collect everything that does not multiply an unknown linearly. Outputs are stacked along a new leading axis of length 4 in that order, all real. Write R and I for the real and imaginary parts of psi_s, c2 and s2 for cos and sin of 2*m*t, c4 and s4 for cos and sin of 4*m*t, and w for rho_other_s + p_other_s. The four are, in order: (3*hubble_s/(4*m))*s2; (3*hubble_s/(4*m))*c2; (3/(64*m*m_pl**2)) * (c4*R**3 + (4*s2 - s4)*I**3 - (4*s2 - 3*s4)*R**2*I + (8*c2 - 3*c4)*R*I**2) + (3/(16*m**2*m_pl**2))*w*(c2*R + s2*I); (3/(64*m*m_pl**2)) * ((4*s2 + s4)*R**3 + c4*I**3 - (8*c2 + 3*c4)*R**2*I - (4*s2 + 3*s4)*R*I**2) + (3/(16*m**2*m_pl**2))*w*(s2*R - c2*I). t may be a scalar or an array, and broadcasts against the other array inputs.

Matching between the exact and the effective regimes means inverting the reconstruction rather than simply copying values across, and once corrections beyond leading order are kept the inversion is no longer linear.

Returns
-------
ndarray of shape (4,) + np.shape(psi_s), float64: the two linear coefficients followed by the two collected terms, for the real and imaginary parts in turn.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def matching_coefficients(psi_s: 'np.ndarray', hubble_s: 'np.ndarray', rho_other_s: 'np.ndarray', p_other_s: 'np.ndarray', m: float, m_pl: float, t: 'np.ndarray') -> 'np.ndarray':
    """Return the four coefficients that turn the reconstruction into a linear system for the real and imaginary parts of the slow-mode wavefunction at time t. The first two are the coefficients multiplying the unknown slow-mode parts; the last two collect everything that does not multiply an unknown linearly. Outputs are stacked along a new leading axis of length 4 in that order, all real. Write R and I for the real and imaginary parts of psi_s, c2 and s2 for cos and sin of 2*m*t, c4 and s4 for cos and sin of 4*m*t, and w for rho_other_s + p_other_s. The four are, in order: (3*hubble_s/(4*m))*s2; (3*hubble_s/(4*m))*c2; (3/(64*m*m_pl**2)) * (c4*R**3 + (4*s2 - s4)*I**3 - (4*s2 - 3*s4)*R**2*I + (8*c2 - 3*c4)*R*I**2) + (3/(16*m**2*m_pl**2))*w*(c2*R + s2*I); (3/(64*m*m_pl**2)) * ((4*s2 + s4)*R**3 + c4*I**3 - (8*c2 + 3*c4)*R**2*I - (4*s2 + 3*s4)*R*I**2) + (3/(16*m**2*m_pl**2))*w*(s2*R - c2*I). t may be a scalar or an array, and broadcasts against the other array inputs.

    Returns
    -------
    ndarray of shape (4,) + np.shape(psi_s), float64: the two linear coefficients followed by the two collected terms, for the real and imaginary parts in turn.

    Raises
    ------
    ValueError
        If m or m_pl is not strictly positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_matching_coefficients(
    psi_s: "np.ndarray",
    hubble_s: "np.ndarray",
    rho_other_s: "np.ndarray",
    p_other_s: "np.ndarray",
    m: float,
    m_pl: float,
    t: "np.ndarray",
) -> "np.ndarray":
    psi_s = np.asarray(psi_s, dtype=complex)
    hubble_s = np.asarray(hubble_s, dtype=float)
    rho_other_s = np.asarray(rho_other_s, dtype=float)
    p_other_s = np.asarray(p_other_s, dtype=float)
    m = float(m)
    m_pl = float(m_pl)
    t = float(t) if isinstance(t, (int, float)) else np.asarray(t, dtype=float)
    if not (m > 0.0 and m_pl > 0.0):
        raise ValueError("m and m_pl must be strictly positive")
    R = np.real(psi_s)
    imag_part = np.imag(psi_s)
    c2 = np.cos(2.0 * m * t)
    s2 = np.sin(2.0 * m * t)
    c4 = np.cos(4.0 * m * t)
    s4 = np.sin(4.0 * m * t)
    # Eq (38).
    a_sin = (3.0 * hubble_s / (4.0 * m)) * s2
    a_cos = (3.0 * hubble_s / (4.0 * m)) * c2
    w = rho_other_s + p_other_s
    # Eq (39).
    c_re = (3.0 / (64.0 * m * m_pl**2)) * (
        c4 * R**3
        + (4.0 * s2 - s4) * imag_part**3
        - (4.0 * s2 - 3.0 * s4) * R**2 * imag_part
        + (8.0 * c2 - 3.0 * c4) * R * imag_part**2
    ) + (3.0 / (16.0 * m**2 * m_pl**2)) * w * (c2 * R + s2 * imag_part)
    # Eq (40).
    c_im = (3.0 / (64.0 * m * m_pl**2)) * (
        (4.0 * s2 + s4) * R**3
        + c4 * imag_part**3
        - (8.0 * c2 + 3.0 * c4) * R**2 * imag_part
        - (4.0 * s2 + 3.0 * s4) * R * imag_part**2
    ) + (3.0 / (16.0 * m**2 * m_pl**2)) * w * (s2 * R - c2 * imag_part)
    return np.stack(np.broadcast_arrays(a_sin, a_cos, c_re, c_im), axis=0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent test specifications."""
    return [
        {
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "matching_coefficients(\n"
                "    0.05 + 0.01j, 0.03, 0.004, 0.0005, 1.0, 1.0, 0.7\n"
                ")"
            ),
            "gold_call": (
                "_oracle_matching_coefficients(\n"
                "    0.05 + 0.01j, 0.03, 0.004, 0.0005, 1.0, 1.0, 0.7\n"
                ")"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "z = np.array([0.02 + 0j, 0.0 + 0.03j])\n"
                "h = np.array([0.01, 0.02])\n"
                "r = np.array([1e-3, 2e-3])\n"
                "p = np.array([1e-4, 2e-4])\n"
            ),
            "call": (
                "matching_coefficients(\n"
                "    z.copy(), h.copy(), r.copy(), p.copy(), 2.0, 1.0, 1.3\n"
                ")"
            ),
            "gold_call": (
                "_oracle_matching_coefficients(\n"
                "    z.copy(), h.copy(), r.copy(), p.copy(), 2.0, 1.0, 1.3\n"
                ")"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "matching_coefficients(\n"
                "    0.05 + 0.01j, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0\n"
                ")"
            ),
            "gold_call": (
                "_oracle_matching_coefficients(\n"
                "    0.05 + 0.01j, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0\n"
                ")"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "z = np.array([0.02 + 0j, 0.0 + 0.03j])\n"
                "h = np.array([0.01, 0.02])\n"
                "r = np.array([1e-3, 2e-3])\n"
                "p = np.array([1e-4, 2e-4])\n"
                "tv = np.array([0.4, 1.9])\n"
            ),
            "call": (
                "matching_coefficients(\n"
                "    z.copy(),\n"
                "    h.copy(),\n"
                "    r.copy(),\n"
                "    p.copy(),\n"
                "    2.0,\n"
                "    1.0,\n"
                "    tv.copy(),\n"
                ")"
            ),
            "gold_call": (
                "_oracle_matching_coefficients(\n"
                "    z.copy(),\n"
                "    h.copy(),\n"
                "    r.copy(),\n"
                "    p.copy(),\n"
                "    2.0,\n"
                "    1.0,\n"
                "    tv.copy(),\n"
                ")"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def _raises(fn):\n"
                "    try:\n"
                "        fn(\n"
                "            0.05 + 0.01j,\n"
                "            0.05,\n"
                "            0.004,\n"
                "            0.001,\n"
                "            -1.0,\n"
                "            1.0,\n"
                "            0.7,\n"
                "        )\n"
                "        return False\n"
                "    except ValueError:\n"
                "        return True\n"
            ),
            "call": (
                "_raises(matching_coefficients)"
            ),
            "gold_call": (
                "_raises(_oracle_matching_coefficients)"
            ),
        },
    ]
