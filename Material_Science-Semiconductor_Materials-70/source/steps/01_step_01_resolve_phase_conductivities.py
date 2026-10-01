"""
The medium is a two-phase semiconductor film in which both phases are the same host crystal doped to different levels, so the only thing that distinguishes them is how many carriers each carries and how freely those carriers move. For an extrinsic n-type region in which every donor is ionised and minority carriers are negligible, the ohmic conductivity is

$$c = q * N * mu,$$

with q the elementary charge, N the donor density and mu the electron mobility at that donor density. The two factors do not move together. Raising the doping raises N, but it also raises the ionised-impurity scattering rate, so mu falls; over the range that separates a lightly doped region from a degenerately doped one the mobility can fall by a factor of five while the doping rises by a factor of five hundred. The conductivity contrast between the two regions is therefore not the doping ratio, and taking it to be the doping ratio puts the whole calculation on the wrong material. Both mobilities are supplied here, so no mobility model has to be fitted; what has to be done is to keep the two factors separate.

The contrast that results is the single number the rest of the problem depends on. Everything downstream is scale-free in the conductivity: multiplying both phases by a common factor multiplies the effective conductivity by that factor and leaves every relative error unchanged, so only the ratio of the two phase conductivities can affect a dimensionless answer.

Three classical estimates bracket the effective conductivity of any two-phase isotropic composite with the given area fraction f of the more conducting phase, and they are computed here so that the exact value obtained later has something to be tested against. The Wiener bounds are the arithmetic and harmonic means,

$$c_{W+} = f * c_2 + (1 - f) * c_1, c_{W-} = 1 / [f / c_2 + (1 - f) / c_1],$$

and they use no information beyond the volume fractions. The two-dimensional Hashin-Shtrikman bounds add the assumption that the composite is isotropic in the plane and are correspondingly tighter,

$$c_{HS-} = c_1 + f / [1 / (c_2 - c_1) + (1 - f) / (2 * c_1)],$$

$$c_{HS+} = c_2 + (1 - f) / [1 / (c_1 - c_2) + f / (2 * c_2)],$$

where the factor 2 in each denominator is the space dimension and would be 3 in three dimensions. At high contrast these four numbers are spread over more than an order of magnitude, which is the point: bounds of this kind confirm an arithmetic slip but they come nowhere near fixing the answer, and that is why an exact solution is needed at all.

Returns
-------
dict, the two phase conductivities in siemens per metre, their contrast, and the four classical bounds that bracket the composite.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def resolve_phase_conductivities(
    donor_density_matrix: float,
    mobility_matrix: float,
    donor_density_inclusion: float,
    mobility_inclusion: float,
    area_fraction: float,
) -> dict:
    """Turn the doping and mobility of each phase into its conductivity, and bracket the composite.

    Parameters
    ----------
    donor_density_matrix : float
        Ionised donor density of the surrounding phase in reciprocal cubic metre, above zero.
    mobility_matrix : float
        Electron mobility of the surrounding phase in square metre per volt second, above zero.
    donor_density_inclusion : float
        Ionised donor density of the embedded phase in reciprocal cubic metre, above zero.
    mobility_inclusion : float
        Electron mobility of the embedded phase in square metre per volt second, above zero.
    area_fraction : float
        Area fraction of the embedded phase, strictly between zero and one.

    Returns
    -------
    dict
        Under the keys c_matrix, c_inclusion, contrast, wiener_lower, wiener_upper, hs_lower and hs_upper.

    Raises
    ------
    ValueError
        When any argument fails to be finite, when any density or mobility fails to be above zero, when the area fraction falls outside the open interval from zero to one, or when the embedded phase fails to be the more conducting of the two.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

ELEMENTARY_CHARGE = 1.602176634e-19


def _positive(value, label):
    """Return an argument as a float once it is known to be finite and above zero."""
    out = float(value)
    if not math.isfinite(out):
        raise ValueError("%s must be finite" % label)
    if out <= 0.0:
        raise ValueError("%s must be above zero" % label)
    return out


def _oracle_resolve_phase_conductivities(
    donor_density_matrix: float,
    mobility_matrix: float,
    donor_density_inclusion: float,
    mobility_inclusion: float,
    area_fraction: float,
) -> dict:
    """Reference implementation."""
    nd1 = _positive(donor_density_matrix, "donor_density_matrix")
    mu1 = _positive(mobility_matrix, "mobility_matrix")
    nd2 = _positive(donor_density_inclusion, "donor_density_inclusion")
    mu2 = _positive(mobility_inclusion, "mobility_inclusion")
    f = float(area_fraction)
    if not math.isfinite(f):
        raise ValueError("area_fraction must be finite")
    if not 0.0 < f < 1.0:
        raise ValueError("area_fraction must lie strictly between zero and one")

    c1 = ELEMENTARY_CHARGE * nd1 * mu1
    c2 = ELEMENTARY_CHARGE * nd2 * mu2
    if c2 <= c1:
        raise ValueError("the embedded phase must be the more conducting of the two")

    wiener_upper = f * c2 + (1.0 - f) * c1
    wiener_lower = 1.0 / (f / c2 + (1.0 - f) / c1)
    # two dimensions, so the denominator carries 2 c_m rather than 3 c_m
    hs_lower = c1 + f / (1.0 / (c2 - c1) + (1.0 - f) / (2.0 * c1))
    hs_upper = c2 + (1.0 - f) / (1.0 / (c1 - c2) + f / (2.0 * c2))
    return {
        "c_matrix": c1,
        "c_inclusion": c2,
        "contrast": c2 / c1,
        "wiener_lower": wiener_lower,
        "wiener_upper": wiener_upper,
        "hs_lower": hs_lower,
        "hs_upper": hs_upper,
    }

# =============================================================================
# TEST CASES
# =============================================================================

FLAT = """
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        out = []
        for v in x:
            out.extend(flat(v))
        return tuple(out)
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    if isinstance(x, bool):
        return (int(x),)
    return (x,)
"""


def test_cases():
    return [
        {
            "setup": """
def digest(out):
    return (round(out["c_matrix"], 8), round(out["c_inclusion"], 6),
            round(out["contrast"], 10), round(out["wiener_lower"], 8),
            round(out["wiener_upper"], 8), round(out["hs_lower"], 8),
            round(out["hs_upper"], 8))
""" + FLAT,
            "call": "flat(digest(resolve_phase_conductivities(1.0e22, 0.1200, 5.0e24, 0.0240, 0.25)))",
            "gold_call": "flat(digest(_oracle_resolve_phase_conductivities(1.0e22, 0.1200, 5.0e24, 0.0240, 0.25)))",
        },
        {
            # the bounds must stay ordered, and the ordering is the only thing that
            # certifies the two-dimensional Hashin-Shtrikman denominators
            "setup": """
def digest(out):
    a = (out["wiener_lower"], out["hs_lower"], out["hs_upper"], out["wiener_upper"])
    ordered = int(a[0] <= a[1] <= a[2] <= a[3])
    return (ordered, round(out["contrast"], 10), round(out["hs_lower"], 8), round(out["hs_upper"], 8))
""" + FLAT,
            "call": "flat((digest(resolve_phase_conductivities(2.5e21, 0.135, 1.0e24, 0.030, 0.4)), digest(resolve_phase_conductivities(1.0e22, 0.1200, 1.1e22, 0.1190, 0.25)), digest(resolve_phase_conductivities(1.0e21, 0.1350, 2.0e25, 0.0180, 0.1))))",
            "gold_call": "flat((digest(_oracle_resolve_phase_conductivities(2.5e21, 0.135, 1.0e24, 0.030, 0.4)), digest(_oracle_resolve_phase_conductivities(1.0e22, 0.1200, 1.1e22, 0.1190, 0.25)), digest(_oracle_resolve_phase_conductivities(1.0e21, 0.1350, 2.0e25, 0.0180, 0.1))))",
        },
        {
            # scale freedom: multiplying both conductivities by a common factor must
            # multiply every bound by the same factor and leave the contrast alone
            "setup": """
def digest(out):
    return (round(out["contrast"], 10), out["hs_lower"], out["hs_upper"], out["wiener_lower"])
def ratio(fn):
    a = digest(fn(1.0e22, 0.1200, 5.0e24, 0.0240, 0.25))
    b = digest(fn(1.0e22, 0.6000, 5.0e24, 0.1200, 0.25))
    return (round(a[0] - b[0], 12), round(b[1] / a[1], 10), round(b[2] / a[2], 10), round(b[3] / a[3], 10))
""" + FLAT,
            "call": "flat(ratio(resolve_phase_conductivities))",
            "gold_call": "flat(ratio(_oracle_resolve_phase_conductivities))",
        },
        {
            "setup": """
def verdict(fn, nd1=1.0e22, mu1=0.12, nd2=5.0e24, mu2=0.024, f=0.25):
    try:
        fn(nd1, mu1, nd2, mu2, f)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(resolve_phase_conductivities, nd1=0.0), verdict(resolve_phase_conductivities, mu1=-1.0), verdict(resolve_phase_conductivities, nd2=float('nan')), verdict(resolve_phase_conductivities, f=0.0), verdict(resolve_phase_conductivities, f=1.0), verdict(resolve_phase_conductivities, nd2=1.0e21), verdict(resolve_phase_conductivities)))",
            "gold_call": "flat((verdict(_oracle_resolve_phase_conductivities, nd1=0.0), verdict(_oracle_resolve_phase_conductivities, mu1=-1.0), verdict(_oracle_resolve_phase_conductivities, nd2=float('nan')), verdict(_oracle_resolve_phase_conductivities, f=0.0), verdict(_oracle_resolve_phase_conductivities, f=1.0), verdict(_oracle_resolve_phase_conductivities, nd2=1.0e21), verdict(_oracle_resolve_phase_conductivities)))",
        },
    ]
