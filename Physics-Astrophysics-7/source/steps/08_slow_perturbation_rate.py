"""
Return the proper-time derivative of the slow-mode wavefunction perturbation dpsi_s at comoving wavenumber k under the effective theory. It carries no explicit dependence on time. The derivative is the sum of five terms: -(1.5*hubble_s + 1j*k**2/(2*m*a_s**2))*dpsi_s, minus 0.25*hdot_s*psi_s, plus (3j*hubble_s/(8*m) + k**2/(16*m**2*a_s**2))*psi_s*hdot_s, plus (3j*psi_s**2/(16*m_pl**2))*conj(dpsi_s), plus (9j*hubble_s**2/(8*m) + 3j*abs(psi_s)**2/(8*m_pl**2) + 1j*k**4/(8*m**3*a_s**4))*dpsi_s. The result is complex and has the shape of dpsi_s.

Removing the explicit oscillation from the perturbation equation costs the same kind of relativistic corrections that the background equation acquires, with two differences: the corrections now also involve the wavenumber, reaching fourth order in it, and one of them couples the perturbation to its own complex conjugate through the square of the background wavefunction.

Returns
-------
ndarray with the shape of dpsi_s, complex128: the time derivative of dpsi_s.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def slow_perturbation_rate(
    dpsi_s: "np.ndarray",
    psi_s: "np.ndarray",
    hubble_s: "np.ndarray",
    hdot_s: "np.ndarray",
    a_s: "np.ndarray",
    m: float,
    m_pl: float,
    k: float,
) -> "np.ndarray":
    """Return the proper-time derivative of the slow-mode wavefunction
    perturbation dpsi_s at comoving wavenumber k under the effective
    theory. It carries no explicit dependence on time. The derivative is
    the sum of five terms: -(1.5*hubble_s + 1j*k**2/(2*m*a_s**2))*dpsi_s,
    minus 0.25*hdot_s*psi_s, plus (3j*hubble_s/(8*m) +
    k**2/(16*m**2*a_s**2))*psi_s*hdot_s, plus
    (3j*psi_s**2/(16*m_pl**2))*conj(dpsi_s), plus
    (9j*hubble_s**2/(8*m) + 3j*abs(psi_s)**2/(8*m_pl**2) +
    1j*k**4/(8*m**3*a_s**4))*dpsi_s. The result is complex and has the
    shape of dpsi_s.

    Returns
    -------
    ndarray with the shape of dpsi_s, complex128: the time derivative of
    dpsi_s.

    Raises
    ------
    ValueError
        If m or m_pl is not strictly positive, or if any entry of a_s is
        not strictly positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_slow_perturbation_rate(
    dpsi_s: "np.ndarray",
    psi_s: "np.ndarray",
    hubble_s: "np.ndarray",
    hdot_s: "np.ndarray",
    a_s: "np.ndarray",
    m: float,
    m_pl: float,
    k: float,
) -> "np.ndarray":
    dpsi_s = np.asarray(dpsi_s, dtype=complex)
    psi_s = np.asarray(psi_s, dtype=complex)
    hubble_s = np.asarray(hubble_s, dtype=float)
    hdot_s = np.asarray(hdot_s, dtype=float)
    a_s = np.asarray(a_s, dtype=float)
    m = float(m)
    m_pl = float(m_pl)
    k = float(k)
    if not (m > 0.0 and m_pl > 0.0):
        raise ValueError("m and m_pl must be strictly positive")
    if np.any(a_s <= 0.0):
        raise ValueError("a_s must be strictly positive")
    # Use the unstarred background square from the EFT mode derivation.
    # The printed Eq (8) conjugates it in error; Eq (17) fixes H_{-2}.
    return (
        -(1.5 * hubble_s + 1j * k**2 / (2.0 * m * a_s**2)) * dpsi_s
        - 0.25 * hdot_s * psi_s
        + (3j * hubble_s / (8.0 * m) + k**2 / (16.0 * m**2 * a_s**2))
        * psi_s
        * hdot_s
        + (3j * psi_s ** 2 / (16.0 * m_pl**2)) * np.conj(dpsi_s)
        + (
            9j * hubble_s**2 / (8.0 * m)
            + 3j * np.abs(psi_s) ** 2 / (8.0 * m_pl**2)
            + 1j * k**4 / (8.0 * m**3 * a_s**4)
        )
        * dpsi_s
    )

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
                "slow_perturbation_rate(\n"
                "    0.001 + 0.0002j,\n"
                "    0.05 + 0.01j,\n"
                "    0.03,\n"
                "    0.0001,\n"
                "    2.0,\n"
                "    1.0,\n"
                "    1.0,\n"
                "    0.5,\n"
                ")"
            ),
            "gold_call": (
                "_oracle_slow_perturbation_rate(\n"
                "    0.001 + 0.0002j,\n"
                "    0.05 + 0.01j,\n"
                "    0.03,\n"
                "    0.0001,\n"
                "    2.0,\n"
                "    1.0,\n"
                "    1.0,\n"
                "    0.5,\n"
                ")"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "dz = np.array([1e-3 + 0j, 0.0 + 2e-3j])\n"
                "z = np.array([0.02 + 0j, 0.0 + 0.03j])\n"
                "h = np.array([0.01, 0.02])\n"
                "hd = np.array([1e-4, 2e-4])\n"
                "aa = np.array([1.5, 2.5])\n"
            ),
            "call": (
                "slow_perturbation_rate(\n"
                "    dz.copy(),\n"
                "    z.copy(),\n"
                "    h.copy(),\n"
                "    hd.copy(),\n"
                "    aa.copy(),\n"
                "    2.0,\n"
                "    1.0,\n"
                "    0.8,\n"
                ")"
            ),
            "gold_call": (
                "_oracle_slow_perturbation_rate(\n"
                "    dz.copy(),\n"
                "    z.copy(),\n"
                "    h.copy(),\n"
                "    hd.copy(),\n"
                "    aa.copy(),\n"
                "    2.0,\n"
                "    1.0,\n"
                "    0.8,\n"
                ")"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "slow_perturbation_rate(\n"
                "    0.001 + 0j, 0.05 + 0.01j, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0\n"
                ")"
            ),
            "gold_call": (
                "_oracle_slow_perturbation_rate(\n"
                "    0.001 + 0j, 0.05 + 0.01j, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0\n"
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
                "            1e-3 + 0j,\n"
                "            0.05 + 0.01j,\n"
                "            0.03,\n"
                "            1e-4,\n"
                "            1.0,\n"
                "            -1.0,\n"
                "            1.0,\n"
                "            0.5,\n"
                "        )\n"
                "        return False\n"
                "    except ValueError:\n"
                "        return True\n"
            ),
            "call": (
                "_raises(slow_perturbation_rate)"
            ),
            "gold_call": (
                "_raises(_oracle_slow_perturbation_rate)"
            ),
        },
    ]
