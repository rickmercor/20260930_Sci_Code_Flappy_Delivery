"""
Return the exact complex wavefunction perturbation at proper time t and comoving wavenumber k, rebuilt from the slow-mode perturbation dpsi_s. Write E for exp(2j*m*t). The reconstruction is dpsi_s minus ((3j*hubble_s/(4*m) + k**2/(4*m**2*a_s**2))*conj(dpsi_s) + (1j*hdot_s/(8*m))*conj(psi_s)) * E. The result is complex and has the shape of dpsi_s. t may be a scalar or an array, and broadcasts against the other array inputs.

The perturbation is rebuilt to one order beyond leading, rather than the two the background is carried to, which is why this expression is so much shorter than the background one. Its single correction nevertheless mixes three separate first-order quantities: the expansion rate, the gradient and the metric rate, and the last of these enters through the background wavefunction rather than the perturbation.

Returns
-------
ndarray with the shape of dpsi_s, complex128: the reconstructed perturbation.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reconstruct_perturbation(dpsi_s: 'np.ndarray', psi_s: 'np.ndarray', hubble_s: 'np.ndarray', hdot_s: 'np.ndarray', a_s: 'np.ndarray', m: float, k: float, t: 'np.ndarray') -> 'np.ndarray':
    """Return the exact complex wavefunction perturbation at proper time t and comoving wavenumber k, rebuilt from the slow-mode perturbation dpsi_s. Write E for exp(2j*m*t). The reconstruction is dpsi_s minus ((3j*hubble_s/(4*m) + k**2/(4*m**2*a_s**2))*conj(dpsi_s) + (1j*hdot_s/(8*m))*conj(psi_s)) * E. The result is complex and has the shape of dpsi_s. t may be a scalar or an array, and broadcasts against the other array inputs.

    Returns
    -------
    ndarray with the shape of dpsi_s, complex128: the reconstructed perturbation.

    Raises
    ------
    ValueError
        If m is not strictly positive, or if any entry of a_s is not strictly positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_reconstruct_perturbation(dpsi_s: 'np.ndarray', psi_s: 'np.ndarray', hubble_s: 'np.ndarray', hdot_s: 'np.ndarray', a_s: 'np.ndarray', m: float, k: float, t: 'np.ndarray') -> 'np.ndarray':
    dpsi_s = np.asarray(dpsi_s, dtype=complex)
    psi_s = np.asarray(psi_s, dtype=complex)
    hubble_s = np.asarray(hubble_s, dtype=float)
    hdot_s = np.asarray(hdot_s, dtype=float)
    a_s = np.asarray(a_s, dtype=float)
    m = float(m); k = float(k)
    t = float(t) if isinstance(t, (int, float)) else np.asarray(t, dtype=float)
    if not (m > 0.0):
        raise ValueError("m must be strictly positive")
    if np.any(a_s <= 0.0):
        raise ValueError("a_s must be strictly positive")
    # Eq (16).  One correction only, unlike the four the background carries.
    e2 = np.exp(2j * m * t)
    return dpsi_s - ((3j * hubble_s / (4.0 * m) + k ** 2 / (4.0 * m ** 2 * a_s ** 2))
                     * np.conj(dpsi_s) + (1j * hdot_s / (8.0 * m)) * np.conj(psi_s)) * e2

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
                "reconstruct_perturbation(\n"
                "    0.001 + 0.0002j,\n"
                "    0.05 + 0.01j,\n"
                "    0.03,\n"
                "    0.0001,\n"
                "    2.0,\n"
                "    1.0,\n"
                "    0.5,\n"
                "    0.7,\n"
                ")"
            ),
            "gold_call": (
                "_oracle_reconstruct_perturbation(\n"
                "    0.001 + 0.0002j,\n"
                "    0.05 + 0.01j,\n"
                "    0.03,\n"
                "    0.0001,\n"
                "    2.0,\n"
                "    1.0,\n"
                "    0.5,\n"
                "    0.7,\n"
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
                "reconstruct_perturbation(\n"
                "    dz.copy(),\n"
                "    z.copy(),\n"
                "    h.copy(),\n"
                "    hd.copy(),\n"
                "    aa.copy(),\n"
                "    2.0,\n"
                "    0.8,\n"
                "    1.3,\n"
                ")"
            ),
            "gold_call": (
                "_oracle_reconstruct_perturbation(\n"
                "    dz.copy(),\n"
                "    z.copy(),\n"
                "    h.copy(),\n"
                "    hd.copy(),\n"
                "    aa.copy(),\n"
                "    2.0,\n"
                "    0.8,\n"
                "    1.3,\n"
                ")"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "tv = np.array([0.4, 1.9])\n"
                "dz = np.array([1e-3 + 0j, 0.0 + 2e-3j])\n"
                "z = np.array([0.02 + 0j, 0.0 + 0.03j])\n"
                "h = np.array([0.01, 0.02])\n"
                "hd = np.array([1e-4, 2e-4])\n"
                "aa = np.array([1.5, 2.5])\n"
            ),
            "call": (
                "reconstruct_perturbation(\n"
                "    dz.copy(),\n"
                "    z.copy(),\n"
                "    h.copy(),\n"
                "    hd.copy(),\n"
                "    aa.copy(),\n"
                "    2.0,\n"
                "    0.8,\n"
                "    tv.copy(),\n"
                ")"
            ),
            "gold_call": (
                "_oracle_reconstruct_perturbation(\n"
                "    dz.copy(),\n"
                "    z.copy(),\n"
                "    h.copy(),\n"
                "    hd.copy(),\n"
                "    aa.copy(),\n"
                "    2.0,\n"
                "    0.8,\n"
                "    tv.copy(),\n"
                ")"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "reconstruct_perturbation(\n"
                "    0.001 + 0j, 0.05 + 0.01j, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0\n"
                ")"
            ),
            "gold_call": (
                "_oracle_reconstruct_perturbation(\n"
                "    0.001 + 0j, 0.05 + 0.01j, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0\n"
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
                "            0.0,\n"
                "            1.0,\n"
                "            0.5,\n"
                "            0.7,\n"
                "        )\n"
                "        return False\n"
                "    except ValueError:\n"
                "        return True\n"
            ),
            "call": (
                "_raises(reconstruct_perturbation)"
            ),
            "gold_call": (
                "_raises(_oracle_reconstruct_perturbation)"
            ),
        },
    ]
