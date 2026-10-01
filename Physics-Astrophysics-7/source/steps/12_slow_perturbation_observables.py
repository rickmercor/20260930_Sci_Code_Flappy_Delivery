"""
Return the slow-mode axion density, pressure and velocity perturbations at comoving wavenumber k. Write P for conj(psi_s)*dpsi_s + psi_s*conj(dpsi_s) and M for conj(psi_s)*dpsi_s - psi_s*conj(dpsi_s). The density perturbation is m*Re(P). The pressure perturbation is (k**2/(4*m*a_s**2))*Re(P); it is not the slow part of the exact pressure perturbation, which carries the oscillating factor instead, and it vanishes with the wavenumber. The velocity perturbation is the real part of -0.5j*M + (3*hubble_s/(4*m))*P + (1j*k**2/(8*m**2*a_s**2))*M + (1/(8*m))*abs(psi_s)**2*hdot_s. The three are stacked along a new leading axis of length 3 in that order, all real.

Once the slow mode of the wavefunction is known the fluid description follows without any further solving, which is what lets the effective theory hand its results to a Boltzmann code. The pressure perturbation is the one that carries no memory of the oscillation at all: it is a pure gradient effect, and an axion whose wavelength is long compared with its Compton scale supports none of it.

Returns
-------
ndarray of shape (3,) + np.shape(psi_s), float64: the density, pressure and velocity perturbations of the axion slow mode.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def slow_perturbation_observables(psi_s: 'np.ndarray', dpsi_s: 'np.ndarray', hdot_s: 'np.ndarray', hubble_s: 'np.ndarray', a_s: 'np.ndarray', m: float, k: float) -> 'np.ndarray':
    """Return the slow-mode axion density, pressure and velocity perturbations at comoving wavenumber k. Write P for conj(psi_s)*dpsi_s + psi_s*conj(dpsi_s) and M for conj(psi_s)*dpsi_s - psi_s*conj(dpsi_s). The density perturbation is m*Re(P). The pressure perturbation is (k**2/(4*m*a_s**2))*Re(P); it is not the slow part of the exact pressure perturbation, which carries the oscillating factor instead, and it vanishes with the wavenumber. The velocity perturbation is the real part of -0.5j*M + (3*hubble_s/(4*m))*P + (1j*k**2/(8*m**2*a_s**2))*M + (1/(8*m))*abs(psi_s)**2*hdot_s. The three are stacked along a new leading axis of length 3 in that order, all real.

    Returns
    -------
    ndarray of shape (3,) + np.shape(psi_s), float64: the density, pressure and velocity perturbations of the axion slow mode.

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


def _oracle_slow_perturbation_observables(psi_s: 'np.ndarray', dpsi_s: 'np.ndarray', hdot_s: 'np.ndarray', hubble_s: 'np.ndarray', a_s: 'np.ndarray', m: float, k: float) -> 'np.ndarray':
    psi_s = np.asarray(psi_s, dtype=complex)
    dpsi_s = np.asarray(dpsi_s, dtype=complex)
    hdot_s = np.asarray(hdot_s, dtype=float)
    hubble_s = np.asarray(hubble_s, dtype=float)
    a_s = np.asarray(a_s, dtype=float)
    m = float(m); k = float(k)
    if not (m > 0.0):
        raise ValueError("m must be strictly positive")
    if np.any(a_s <= 0.0):
        raise ValueError("a_s must be strictly positive")
    plus = np.conj(psi_s) * dpsi_s + psi_s * np.conj(dpsi_s)
    minus = np.conj(psi_s) * dpsi_s - psi_s * np.conj(dpsi_s)
    # Eq (28).
    drho = m * np.real(plus)
    # Eq (29).  The slow pressure perturbation is NOT the slow part of Eq (24):
    # it is the same combination carrying the gradient factor instead.
    dp = (k ** 2 / (4.0 * m * a_s ** 2)) * np.real(plus)
    # Eq (30).
    du = np.real(-0.5j * minus + (3.0 * hubble_s / (4.0 * m)) * plus
                 + (1j * k ** 2 / (8.0 * m ** 2 * a_s ** 2)) * minus
                 + (1.0 / (8.0 * m)) * np.abs(psi_s) ** 2 * hdot_s)
    return np.stack(np.broadcast_arrays(drho, dp, du), axis=0)

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
                "slow_perturbation_observables(\n"
                "    0.05 + 0.01j,\n"
                "    0.001 + 0.0002j,\n"
                "    0.0001,\n"
                "    0.03,\n"
                "    2.0,\n"
                "    1.0,\n"
                "    0.5,\n"
                ")"
            ),
            "gold_call": (
                "_oracle_slow_perturbation_observables(\n"
                "    0.05 + 0.01j,\n"
                "    0.001 + 0.0002j,\n"
                "    0.0001,\n"
                "    0.03,\n"
                "    2.0,\n"
                "    1.0,\n"
                "    0.5,\n"
                ")"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "z = np.array([0.02 + 0j, 0.0 + 0.03j])\n"
                "dz = np.array([1e-3 + 0j, 0.0 + 2e-3j])\n"
                "hd = np.array([1e-4, 2e-4])\n"
                "h = np.array([0.01, 0.02])\n"
                "aa = np.array([1.5, 2.5])\n"
            ),
            "call": (
                "slow_perturbation_observables(\n"
                "    z.copy(),\n"
                "    dz.copy(),\n"
                "    hd.copy(),\n"
                "    h.copy(),\n"
                "    aa.copy(),\n"
                "    2.0,\n"
                "    0.8,\n"
                ")"
            ),
            "gold_call": (
                "_oracle_slow_perturbation_observables(\n"
                "    z.copy(),\n"
                "    dz.copy(),\n"
                "    hd.copy(),\n"
                "    h.copy(),\n"
                "    aa.copy(),\n"
                "    2.0,\n"
                "    0.8,\n"
                ")"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "slow_perturbation_observables(\n"
                "    0.05 + 0.01j,\n"
                "    0.001 + 0.0002j,\n"
                "    0.0001,\n"
                "    0.03,\n"
                "    2.0,\n"
                "    1.0,\n"
                "    0.0,\n"
                ")"
            ),
            "gold_call": (
                "_oracle_slow_perturbation_observables(\n"
                "    0.05 + 0.01j,\n"
                "    0.001 + 0.0002j,\n"
                "    0.0001,\n"
                "    0.03,\n"
                "    2.0,\n"
                "    1.0,\n"
                "    0.0,\n"
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
                "            1e-3 + 0j,\n"
                "            1e-4,\n"
                "            0.03,\n"
                "            2.0,\n"
                "            -1.0,\n"
                "            0.5,\n"
                "        )\n"
                "        return False\n"
                "    except ValueError:\n"
                "        return True\n"
            ),
            "call": (
                "_raises(slow_perturbation_observables)"
            ),
            "gold_call": (
                "_raises(_oracle_slow_perturbation_observables)"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def pressure_check(fn):\n"
                "    result = fn(\n"
                "        0.05 + 0.01j,\n"
                "        1e-3 + 2e-4j,\n"
                "        1e-4,\n"
                "        0.03,\n"
                "        2.0,\n"
                "        1.0,\n"
                "        0.4,\n"
                "    )\n"
                "    expected_density = 1.04e-4\n"
                "    expected_pressure = expected_density * 0.4**2 / 16.0\n"
                "    return bool(\n"
                "        np.isclose(\n"
                "            result[0],\n"
                "            expected_density,\n"
                "            rtol=1e-12,\n"
                "            atol=0.0,\n"
                "        )\n"
                "        and np.isclose(\n"
                "            result[1],\n"
                "            expected_pressure,\n"
                "            rtol=1e-12,\n"
                "            atol=0.0,\n"
                "        )\n"
                "    )\n"
            ),
            "call": (
                "pressure_check(slow_perturbation_observables)"
            ),
            "gold_call": (
                "pressure_check(_oracle_slow_perturbation_observables)"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def pressure_check(fn):\n"
                "    result = fn(\n"
                "        0.05 + 0.01j,\n"
                "        1e-3 + 2e-4j,\n"
                "        1e-4,\n"
                "        0.03,\n"
                "        2.0,\n"
                "        1.0,\n"
                "        0.8,\n"
                "    )\n"
                "    expected_density = 1.04e-4\n"
                "    expected_pressure = expected_density * 0.8**2 / 16.0\n"
                "    return bool(\n"
                "        np.isclose(\n"
                "            result[0],\n"
                "            expected_density,\n"
                "            rtol=1e-12,\n"
                "            atol=0.0,\n"
                "        )\n"
                "        and np.isclose(\n"
                "            result[1],\n"
                "            expected_pressure,\n"
                "            rtol=1e-12,\n"
                "            atol=0.0,\n"
                "        )\n"
                "    )\n"
            ),
            "call": (
                "pressure_check(slow_perturbation_observables)"
            ),
            "gold_call": (
                "pressure_check(_oracle_slow_perturbation_observables)"
            ),
        },
    ]
