"""
Step 01: evaluate the prescribed static background on the height grid.

Contract
--------
The atmosphere is completely prescribed; nothing here is solved for. On a
height grid $z$ measured from the base of the domain, evaluate

- the hydrogen number density, falling off exponentially with scale height
$H$ from its base value $n_base$;
- the mass density, obtained from the hydrogen number density by adding the
helium mass, with a helium-to-hydrogen abundance ratio $A_He$, so that
$rho = m_p (1 + 4 A_He) n_H$;
- the temperature, rising from $T_base$ towards $T_base + dT$ as
1 - exp(-z / LT);
- the magnetic field strength, falling as the inverse square of the distance
from the solar centre, $B = B0 (R_sun / (z + R_sun))^2$ with
`R_sun = 695.7` Mm;
- the strand radius, which follows the field through conservation of the
magnetic flux threading one strand, R = R0 sqrt(B0 / B);
- the density contrast, relaxing from `zeta0` towards 1 as
zeta = (zeta0 - 1) exp(-z / Lzeta) + 1.

Conventions
-----------
Units: z, R0 and the length scales H, LT, Lzeta in Mm; $n_base$ in
1e15 m^-3; mass density in 1e-12 kg m^-3; $T_base$ and dT in MK; B0 in G. Use
`m_p = 1.6726219e-27` kg. Return a float array of shape (6, N) whose rows are,
in this order, the hydrogen number density, the mass density, the temperature,
the field strength, the strand radius and the density contrast.

Validation
----------
Raise ValueError if $z$ is not a one-dimensional array of at least two
finite, strictly increasing values with z[0] >= 0; if any of $n_base$, $H$,
`B0`, `R0`, `Lzeta`, $LT$ or $T_base$ is not finite or not strictly
positive; if `zeta0` is not finite or below 1; or if $dT$ or $A_He$ is
not finite or negative.

Returns
-------
A float array of shape (6, N) holding the hydrogen number density, the mass density, the temperature, the field strength, the strand radius and the density contrast on the grid.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stratified_background(z: "ArrayLike", n_base: float, H: float, B0: float,
                          R0: float, zeta0: float, Lzeta: float, T_base: float,
                          dT: float, LT: float, A_He: float) -> "np.ndarray":
    '''Evaluate the prescribed background profiles on the height grid.

    Parameters
    ----------
    z : array_like
        One-dimensional height grid in Mm, strictly increasing, z[0] >= 0.
    n_base : float
        Hydrogen number density at the base, in 1e15 m^-3. Finite and > 0.
    H : float
        Density scale height in Mm. Finite and > 0.
    B0 : float
        Field strength at the base in G. Finite and > 0.
    R0 : float
        Strand radius at the base in Mm. Finite and > 0.
    zeta0 : float
        Density contrast at the base. Finite and not less than 1.
    Lzeta : float
        Relaxation length of the density contrast in Mm. Finite and > 0.
    T_base : float
        Temperature at the base in MK. Finite and > 0.
    dT : float
        Temperature rise across the domain in MK. Finite and >= 0.
    LT : float
        Height scale of the temperature rise in Mm. Finite and > 0.
    A_He : float
        Helium-to-hydrogen abundance ratio. Finite and >= 0.

    Returns
    -------
    np.ndarray
        Shape (6, N) float array. Rows in order: hydrogen number density in
        1e15 m^-3, mass density in 1e-12 kg m^-3, temperature in MK, field
        strength in G, strand radius in Mm, density contrast (dimensionless).

    Raises
    ------
    ValueError
        On a malformed grid or a non-finite or out-of-range parameter.
    '''
    return out  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_grid(z):
    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or z.size < 2:
        raise ValueError("z must be a one-dimensional grid of at least two points")
    if not np.all(np.isfinite(z)):
        raise ValueError("z must be finite")
    if np.any(np.diff(z) <= 0.0):
        raise ValueError("z must be strictly increasing")
    if z[0] < 0.0:
        raise ValueError("z must start at or above the base")
    return z


def _check_positive(name, value):
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError("%s must be finite and strictly positive" % name)
    return v


def _check_nonnegative(name, value):
    v = float(value)
    if not np.isfinite(v) or v < 0.0:
        raise ValueError("%s must be finite and non-negative" % name)
    return v


def _oracle_stratified_background(z: "ArrayLike", n_base: float, H: float,
                                  B0: float, R0: float, zeta0: float,
                                  Lzeta: float, T_base: float, dT: float,
                                  LT: float, A_He: float) -> "np.ndarray":
    z = _check_grid(z)
    n_base = _check_positive("n_base", n_base)
    H = _check_positive("H", H)
    B0 = _check_positive("B0", B0)
    R0 = _check_positive("R0", R0)
    Lzeta = _check_positive("Lzeta", Lzeta)
    LT = _check_positive("LT", LT)
    T_base = _check_positive("T_base", T_base)
    dT = _check_nonnegative("dT", dT)
    A_He = _check_nonnegative("A_He", A_He)
    zeta0 = float(zeta0)
    if not np.isfinite(zeta0) or zeta0 < 1.0:
        raise ValueError("zeta0 must be finite and not less than 1")

    r_sun_mm = 695.7
    m_proton = 1.6726219e-27
    n_H = n_base * np.exp(-z / H)
    rho = m_proton * (1.0 + 4.0 * A_He) * n_H * 1.0e15 / 1.0e-12
    T = T_base + dT * (1.0 - np.exp(-z / LT))
    B = B0 * (r_sun_mm / (z + r_sun_mm)) ** 2
    R = R0 * np.sqrt(B0 / B)
    zeta = (zeta0 - 1.0) * np.exp(-z / Lzeta) + 1.0
    return np.asarray([n_H, rho, T, B, R, zeta], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    '''Differential test cases for this step.'''
    # Each call builds its own input arrays, so an implementation that works
    # in place on its arguments cannot change what the other call receives.
    benchmark = (
        "import numpy as np\n"
        "def _args():\n"
        "    z = np.linspace(0.0, 120.0, 9)\n"
        "    return (z, 1.5, 42.0, 12.5, 0.8, 3.6, 400.0, 0.62, 0.83, 45.0, 0.1)\n"
    )
    boundary = (
        "import numpy as np\n"
        "def _args():\n"
        "    z = np.array([0.0, 1.0e-6])\n"
        "    return (z, 1.5, 42.0, 12.5, 0.8, 1.0, 400.0, 0.62, 0.0, 45.0, 0.0)\n"
    )
    edge = (
        "import numpy as np\n"
        "def _code(fn):\n"
        "    try:\n"
        "        fn(np.array([0.0, 1.0]), 1.5, 42.0, 12.5, 0.8, 0.5, 400.0,\n"
        "           0.62, 0.83, 45.0, 0.1)\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "    return 0\n"
    )
    return [
        {"setup": benchmark,
         "call": "stratified_background(*_args())",
         "gold_call": "_oracle_stratified_background(*_args())"},
        {"setup": boundary,
         "call": "stratified_background(*_args())",
         "gold_call": "_oracle_stratified_background(*_args())"},
        {"setup": edge,
         "call": "_code(stratified_background)",
         "gold_call": "_code(_oracle_stratified_background)"},
    ]
