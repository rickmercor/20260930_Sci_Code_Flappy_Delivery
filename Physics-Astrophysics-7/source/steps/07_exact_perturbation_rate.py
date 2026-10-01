"""
Return the proper-time derivative of the exact complex wavefunction perturbation dpsi at comoving wavenumber k, in the synchronous gauge. psi is the exact background wavefunction, hubble the exact expansion rate, hdot the proper-time derivative of the synchronous metric perturbation, and a the scale factor. Writing q for 1j*k**2/(2*m*a**2), the derivative is -(1.5*hubble + q)*dpsi - 0.25*psi*hdot + exp(2j*m*t) * ((1.5*hubble - q)*conj(dpsi) + 0.25*conj(psi)*hdot). Note the sign of the gradient term reverses inside the oscillating bracket. The result is complex and has the shape of dpsi. t may be a scalar or an array, and broadcasts against the other array inputs.

At the linear level the wavefunction perturbation obeys a first-order equation of the same shape as the background one, with the gradient of the perturbation adding a wavenumber-dependent term and the metric perturbation acting as a source. The explicit oscillating factor is what makes this equation stiff once the oscillation outruns the expansion, and it is the reason an effective description is needed at the perturbation level as well as at the background level.

Returns
-------
ndarray with the shape of dpsi, complex128: the time derivative of dpsi.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def exact_perturbation_rate(
    dpsi: "np.ndarray",
    psi: "np.ndarray",
    hubble: "np.ndarray",
    hdot: "np.ndarray",
    a: "np.ndarray",
    m: float,
    k: float,
    t: "np.ndarray",
) -> "np.ndarray":
    """Return the proper-time derivative of the exact complex wavefunction
    perturbation dpsi at comoving wavenumber k, in the synchronous gauge.
    psi is the exact background wavefunction, hubble the exact expansion
    rate, hdot the proper-time derivative of the synchronous metric
    perturbation, and a the scale factor. Writing q for 1j*k**2/(2*m*a**2),
    the derivative is -(1.5*hubble + q)*dpsi - 0.25*psi*hdot + exp(2j*m*t)
    * ((1.5*hubble - q)*conj(dpsi) + 0.25*conj(psi)*hdot). Note the sign of
    the gradient term reverses inside the oscillating bracket. The result
    is complex and has the shape of dpsi. t may be a scalar or an array,
    and broadcasts against the other array inputs.

    Returns
    -------
    ndarray with the shape of dpsi, complex128: the time derivative of
    dpsi.

    Raises
    ------
    ValueError
        If m is not strictly positive, or if any entry of a is not strictly
        positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_exact_perturbation_rate(
    dpsi: "np.ndarray",
    psi: "np.ndarray",
    hubble: "np.ndarray",
    hdot: "np.ndarray",
    a: "np.ndarray",
    m: float,
    k: float,
    t: "np.ndarray",
) -> "np.ndarray":
    dpsi = np.asarray(dpsi, dtype=complex)
    psi = np.asarray(psi, dtype=complex)
    hubble = np.asarray(hubble, dtype=float)
    hdot = np.asarray(hdot, dtype=float)
    a = np.asarray(a, dtype=float)
    m = float(m)
    k = float(k)
    t = float(t) if isinstance(t, (int, float)) else np.asarray(t, dtype=float)
    if not (m > 0.0):
        raise ValueError("m must be strictly positive")
    if np.any(a <= 0.0):
        raise ValueError("a must be strictly positive")
    # The Klein-Gordon equation (A2) fixes a negative metric term
    # outside the oscillatory bracket. The printed Eq (5) has a
    # sign error there; the conjugate metric term stays positive.
    e2 = np.exp(2j * m * t)
    q = 1j * k**2 / (2.0 * m * a**2)
    return (
        -(1.5 * hubble + q) * dpsi
        - 0.25 * psi * hdot
        + e2
        * ((1.5 * hubble - q) * np.conj(dpsi) + 0.25 * np.conj(psi) * hdot)
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
                "exact_perturbation_rate(\n"
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
                "_oracle_exact_perturbation_rate(\n"
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
                "exact_perturbation_rate(\n"
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
                "_oracle_exact_perturbation_rate(\n"
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
                "exact_perturbation_rate(\n"
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
                "_oracle_exact_perturbation_rate(\n"
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
                "exact_perturbation_rate(\n"
                "    0.001 + 0j, 0.05 + 0.01j, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0\n"
                ")"
            ),
            "gold_call": (
                "_oracle_exact_perturbation_rate(\n"
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
                "            -1.0,\n"
                "            1.0,\n"
                "            0.5,\n"
                "            0.7,\n"
                "        )\n"
                "        return False\n"
                "    except ValueError:\n"
                "        return True\n"
            ),
            "call": (
                "_raises(exact_perturbation_rate)"
            ),
            "gold_call": (
                "_raises(_oracle_exact_perturbation_rate)"
            ),
        },
    ]
