"""
Step 07 - Spatially averaged tunnelling conductance versus bias.

Spatially averaged tunnelling conductance of the condensate at zero temperature.

Within Fermi's golden rule the differential conductance between a tip and the sample is a sum over the
Bogoliubov quasiparticle states of the mean-field solution, each weighted by the squared amplitude with which the
tip's electron operator creates or destroys it, at the bias where its energy equals eV. Two cases are needed.
In a monolayer the tip couples to both bands, so the spatially averaged conductance measures the total density
of states of the two quasiparticle bands. In a bilayer the tip couples only to the conduction-band (top) layer,
so the weight of the upper band is u_k^2 and that of the lower band is v_k^2: the lower band then contributes at
negative bias only because of the condensate, producing the satellite feature. Conductances
are expressed in units of G_0 = (2 pi e^2/hbar) g_s t^2 nu_0 nu_t, the conductance into a parabolic band with the
reduced mass m (density of states nu_0 = m/(2 pi hbar^2)); with this normalisation the total conductance into a
bare band of mass m_c above its edge is (m_c/m) G_0. For an isotropic band the sum over states at fixed energy
reduces to a one-dimensional integral over the magnitude k, with a Jacobian from the band dispersion; the exact
expression is not restated. Inside the excitation gap the conductance vanishes. At a bias within 1e-9 Ry* of a
band edge return the limiting value of the conductance as the edge is approached from inside the band.

The quasiparticle bands must be monotonic on k >= 0 (E_{k,+} increasing, E_{k,-} decreasing); if either band of the
self-consistent solution is not monotonic on the momentum range that contributes, the bias-to-momentum map is
multivalued and a ValueError is raised.

Inputs: E_G (Ry*), d >= 0, r > 0 as in the previous steps; biases, a one-dimensional array of finite values of eV
in Ry*. Output: array of shape (2, len(biases)); row 0 the total conductance (both bands with unit weight) and
row 1 the conductance into the conduction-band layer (upper band weighted by u_k^2, lower band by v_k^2), both in
units of G_0 and converged to 1e-8 relative accuracy. Raises ValueError for invalid inputs, a non-converging
self-consistent iteration, or non-monotonic bands.

Returns
-------
numpy.ndarray of shape (2, len(biases)): [total conductance, conductance into the top layer] in units of G_0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def averaged_conductance(EG: float, d: float, r: float, biases: np.ndarray) -> np.ndarray:
    '''Spatially averaged zero-temperature tunnelling conductance dI/dV of the condensate.

    Parameters
    ----------
    EG : float
        Bare band gap E_G in Ry*, finite.
    d : float
        Interlayer distance in units of a_B*, >= 0.
    r : float
        Mass ratio m_v/m_c, > 0.
    biases : np.ndarray
        One-dimensional array of finite bias energies eV in Ry*, measured from the middle of the bare gap.

    Returns
    -------
    result : np.ndarray
        Shape (2, len(biases)) in units of G_0: row 0 the total conductance (both quasiparticle bands with unit
        weight), row 1 the conductance into the conduction-band layer only (upper band weighted by u_k^2, lower
        band by v_k^2). Zero inside the excitation gap; the inside limit within 1e-9 Ry* of an edge. Converged to
        1e-8 relative accuracy.

    Raises
    ------
    ValueError
        If E_G is not finite, d is negative or not finite, r is not positive and finite, biases is not a
        one-dimensional array of finite numbers, the self-consistent iteration does not converge, or either
        quasiparticle band is not monotonic on k >= 0.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def _band_curves(s):
    """Even-polynomial fits of E+, E-, v^2 near k = 0 and a monotonicity check on the trusted grid up to k = 60."""
    k = s["k"]
    sel = k < 60.0
    if np.any(np.diff(s["Ep"][sel]) <= 0) or np.any(np.diff(s["Em"][sel]) >= 0):
        raise ValueError("quasiparticle bands are not monotonic on k >= 0")
    return {nm: _even_fit(k, s[nm]) for nm in ("Ep", "Em", "v2")}


def _k_of_bias(s, band, e):
    """Momentum at which the band ('Ep' or 'Em') equals the energy e, by bracketing on the trusted grid."""
    k = s["k"]
    arr = s[band]
    if band == "Ep":
        j = int(np.searchsorted(arr, e))
    else:
        j = int(np.searchsorted(-arr, -e))
    lo = 0.0 if j == 0 else k[j - 1]
    hi = k[min(j, len(k) - 1)] if j < len(k) else k[-1] * 2
    row = 2 if band == "Ep" else 3
    g = lambda kk: _state_at(s, np.array([kk]))[row, 0] - e
    if lo == 0.0 and g(0.0) * g(hi) > 0:
        return 0.0
    return brentq(g, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=200)


def _band_and_weight(s, band, kk):
    """Value, derivative dE/dk (five-point stencil) and the projected weight (u^2 or v^2) of a band at kk."""
    h = 1e-3 * (1.0 + kk)
    pts = np.array([kk - 2 * h, kk - h, kk, kk + h, kk + 2 * h])
    if pts[0] < 0:
        pts = np.abs(pts)
    st = _state_at(s, pts)
    row = 2 if band == "Ep" else 3
    e = st[row]
    de = (e[0] - 8 * e[1] + 8 * e[3] - e[4]) / (12 * h)
    w = st[0, 2] if band == "Em" else 1 - st[0, 2]
    return e[2], de, w


def _oracle_averaged_conductance(EG: float, d: float, r: float, biases: np.ndarray) -> np.ndarray:
    """Reference implementation: (dI/dV)/G_0 = 2 k w_k / |dE/dk| at k(V) (Eqs. 38 and 49 of the source)."""
    EG = _check_scalar(EG, "EG")
    d = _check_scalar(d, "d", nonneg=True)
    r = _check_scalar(r, "r", positive=True)
    biases = np.asarray(biases, dtype=float)
    if biases.ndim != 1 or biases.size == 0 or not np.all(np.isfinite(biases)):
        raise ValueError("biases must be a one-dimensional array of finite numbers")
    s = _hf(EG, d, r)
    fits = _band_curves(s)
    E0p, E0m = fits["Ep"][0], fits["Em"][0]
    out = np.zeros((2, len(biases)))
    for i, e in enumerate(biases):
        if E0m + 1e-9 < e < E0p - 1e-9:
            continue
        band = "Ep" if e >= E0p - 1e-9 else "Em"
        edge = E0p if band == "Ep" else E0m
        if abs(e - edge) <= 1e-9:
            a2 = fits[band][1]
            w0 = 1 - fits["v2"][0] if band == "Ep" else fits["v2"][0]
            out[0, i] = 1.0 / abs(a2)
            out[1, i] = w0 / abs(a2)
            continue
        kk = _k_of_bias(s, band, e)
        _, de, w = _band_and_weight(s, band, kk)
        out[0, i] = 2 * kk / abs(de)
        out[1, i] = 2 * kk * w / abs(de)
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the heterobilayer of the task, biases across the satellite, the gap and the upper band ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "from scipy.optimize import brentq\n"
                     "biases = np.array([-3.0, -1.6, -1.2, -1.05, 0.0, 1.05, 1.3, 2.5])\n",
            "call": "averaged_conductance(1.7815, 0.25, 2.0, biases.copy())",
            "gold_call": "_oracle_averaged_conductance(1.7815, 0.25, 2.0, biases.copy())",
            "tol": 1e-6,
        },
        # --- Normal: monolayer, equal masses; the total conductance must approach 2 G_0 deep in either band ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "from scipy.optimize import brentq\n"
                     "biases = np.array([-12.0, -4.0, -2.3, 2.3, 2.6, 4.0, 12.0])\n",
            "call": "averaged_conductance(3.848585, 0.0, 1.0, biases.copy())",
            "gold_call": "_oracle_averaged_conductance(3.848585, 0.0, 1.0, biases.copy())",
            "tol": 1e-6,
        },
        # --- Boundary: biases exactly at the two band edges (limiting values), equal masses at half a Bohr radius ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "from scipy.optimize import brentq\n"
                     "biases = np.array([-0.6882517896, 0.6882517896, 0.0])\n",
            "call": "averaged_conductance(1.28, 0.5, 1.0, biases.copy())",
            "gold_call": "_oracle_averaged_conductance(1.28, 0.5, 1.0, biases.copy())",
            "tol": 1e-6,
        },
        # --- Edge: no condensate (gap above E_b); the projected conductance vanishes at negative bias ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "from scipy.optimize import brentq\n"
                     "biases = np.array([-3.0, -1.5, 1.5, 3.0])\n",
            "call": "averaged_conductance(2.3, 0.25, 2.0, biases.copy())",
            "gold_call": "_oracle_averaged_conductance(2.3, 0.25, 2.0, biases.copy())",
            "tol": 1e-6,
        },
        # --- Invalid: an inverted valence band (dense monolayer with a heavy valence band) must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "from scipy.optimize import brentq\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(3.70, 0.0, 3.0, np.array([-3.0, 3.0]))\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    return 0\n",
            "call": "_probe(averaged_conductance)",
            "gold_call": "_probe(_oracle_averaged_conductance)",
        },
    ]
