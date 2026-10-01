"""
Map the physical description of a yolk-shell nanoreactor onto the dimensionless parameters of the capture problem. The catalytic core is a sphere of radius R (nm) held concentrically inside the cavity of radius R0 > R (nm) of a thin permeable shell; the reactant B is held at its bulk concentration c_B outside the shell and enters the cavity through the shell with a permeability P (nm/us): the entry flux density is P (c_B - c) with c the concentration just inside the shell, P = inf meaning that the bulk concentration is maintained on the cavity wall. Inside the cavity the reactant diffuses through the corona grafted on the core with a position-dependent diffusivity D(r) = D (r / R)^p, where D (nm^2/us) is the diffusivity at the core surface and p >= 0 the hindrance exponent (p = 0 is a free cavity). The active site is an axisymmetric spherical cap covering the fraction site_fraction of the core surface area, the rest of the core being inert; on the site the reactant is consumed with a surface reactivity kappa (nm/us), kappa = inf denoting a perfect sink. Return the thickness ratio eps = R / R0, the relative shell thickness h = 1 - eps, the polar half-angle theta0 of the cap in radians (from its surface fraction), the Damkohler number of the site Da = kappa R / D, the shell Biot number Bi = P R / D, the hindrance exponent p, the Smoluchowski rate constant k_S = 4 pi R D of an isotropic perfect-sink sphere in an unbounded free medium of diffusivity D (nm^3/us), the rate correction factor J_shell of the confined isotropic perfect sink in the corona behind the permeable shell (the exact solution of the spherically symmetric problem; Berg's factor 1 / (1 - eps) when P = inf and p = 0), and the rate correction factor J_iso of the confined isotropic core whose whole surface reacts with the finite reactivity kappa in the same corona behind the same shell (equal to J_shell when kappa = inf); derive both closed forms. The rate correction factor J is defined throughout by k = k_S J, with k the total steady flux into the core per unit bulk concentration. Raise ValueError if R is not positive, if R0 is not larger than R, if site_fraction is not in (0, 1], if D, kappa or P is not positive, or if p is negative or not finite.

Berg's model of diffusion to capture places a perfectly absorbing isotropic sphere inside a spherical cavity whose wall acts as the source of the diffusing particles; the source treats the same geometry with the absorber replaced by a particle carrying one axisymmetric reactive patch. A yolk-shell nanoreactor, a catalytic core enclosed in a hollow permeable shell, is the materials realisation of this geometry: a shell of finite permeability supplies the reactant at a rate proportional to the concentration deficit just inside it, a polymer corona on the core hinders diffusion most near the core surface, and the radiation (Collins-Kimball) condition D dc/dr = kappa c on the active site generalises the perfect sink to a finite surface reactivity. None of the three is treated in the source.

Returns
-------
A (9,) float64 array [eps, h, theta0, Da, Bi, p, k_S, J_shell, J_iso] with theta0 in radians and k_S in nm^3/us; inf-safe in Da and Bi.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nanoreactor_parameters(core_radius, cavity_radius, site_fraction, diffusivity, reactivity, permeability, hindrance):
    """Map the physical description of a yolk-shell nanoreactor onto the dimensionless
    parameters of the capture problem. A (9,) float64 array [eps, h, theta0, Da, Bi, p, k_S,
    J_shell, J_iso] with theta0 in radians and k_S in nm^3/us; inf-safe in Da and Bi."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander


def _oracle_nanoreactor_parameters(core_radius, cavity_radius, site_fraction, diffusivity, reactivity, permeability, hindrance):
    R = float(core_radius); R0 = float(cavity_radius); phi = float(site_fraction)
    D = float(diffusivity); kappa = float(reactivity); perm = float(permeability); p = float(hindrance)
    if not (R > 0.0) or not (R0 > R):
        raise ValueError("need 0 < core radius < cavity radius")
    if not (0.0 < phi <= 1.0):
        raise ValueError("site fraction must lie in (0, 1]")
    if not (D > 0.0) or not (kappa > 0.0) or not (perm > 0.0):
        raise ValueError("diffusivity, reactivity and permeability must be positive")
    if not (p >= 0.0) or not np.isfinite(p):
        raise ValueError("hindrance exponent must be finite and non-negative")
    eps = R / R0
    h = 1.0 - eps
    theta0 = float(np.arccos(1.0 - 2.0 * phi))
    da = kappa * R / D
    bi = perm * R / D
    ks = 4.0 * np.pi * R * D
    eps0 = eps ** (1.0 + p) if not np.isfinite(bi) else eps ** (1.0 + p) - (1.0 + p) * eps * eps / bi
    j_shell = (1.0 + p) / (1.0 - eps0)
    j_iso = j_shell if not np.isfinite(da) else (1.0 + p) * da / ((1.0 + p) + da * (1.0 - eps0))
    return np.array([eps, h, theta0, da, bi, p, ks, j_shell, j_iso], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\ncore_radius = 2.5\ncavity_radius = 4.0\nsite_fraction = 0.15\ndiffusivity = 500.0\nreactivity = 600.0\npermeability = 500.0\nhindrance = 1.0\n',
         'call': 'nanoreactor_parameters(core_radius, cavity_radius, site_fraction, diffusivity, reactivity, permeability, hindrance)',
         'gold_call': '_oracle_nanoreactor_parameters(core_radius, cavity_radius, site_fraction, diffusivity, reactivity, permeability, hindrance)'},
        {'setup': 'import numpy as np\ncore_radius = 1.4\ncavity_radius = 4.0\nsite_fraction = 0.10\ndiffusivity = 500.0\nreactivity = 250.0\npermeability = 1.0e4\nhindrance = 0.0\n',
         'call': 'nanoreactor_parameters(core_radius, cavity_radius, site_fraction, diffusivity, reactivity, permeability, hindrance)',
         'gold_call': '_oracle_nanoreactor_parameters(core_radius, cavity_radius, site_fraction, diffusivity, reactivity, permeability, hindrance)'},
        {'setup': 'import numpy as np\ncore_radius = 3.0\ncavity_radius = 4.0\nsite_fraction = 1.0\ndiffusivity = 400.0\nreactivity = 1600.0\npermeability = 800.0\nhindrance = 0.5\n',
         'call': 'nanoreactor_parameters(core_radius, cavity_radius, site_fraction, diffusivity, reactivity, permeability, hindrance)',
         'gold_call': '_oracle_nanoreactor_parameters(core_radius, cavity_radius, site_fraction, diffusivity, reactivity, permeability, hindrance)'},
        {'setup': 'import numpy as np\ncore_radius = 2.0\ncavity_radius = 4.2\nsite_fraction = 0.5\ndiffusivity = 500.0\nreactivity = 125.0\npermeability = 125.0\nhindrance = 1.5\n',
         'call': 'nanoreactor_parameters(core_radius, cavity_radius, site_fraction, diffusivity, reactivity, permeability, hindrance)',
         'gold_call': '_oracle_nanoreactor_parameters(core_radius, cavity_radius, site_fraction, diffusivity, reactivity, permeability, hindrance)'},
    ]
