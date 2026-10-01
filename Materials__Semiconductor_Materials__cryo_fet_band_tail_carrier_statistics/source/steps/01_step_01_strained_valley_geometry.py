"""
The quantum well of a Si/SiGe field-effect heterostructure is a thin silicon layer grown pseudomorphically on a relaxed SiGe buffer. The silicon adopts the in-plane lattice constant of the buffer, so it is stretched biaxially in the growth plane and, to keep its stress free along the growth direction, contracts along [001]. For a cubic crystal the in-plane strain is eps_par = (a_sub - a_Si) / a_Si and the tetragonal response along the growth direction is eps_zz = -2 (C12 / C11) eps_par, with C11 and C12 the elastic constants of silicon. The shear components that this growth geometry produces are zero; a small shear strain in the plane, when present, enters later only through the Bloch-factor data and not through the geometry computed here.

The lattice constant of the relaxed alloy buffer is not the linear interpolation between silicon and germanium. It carries a quadratic bowing correction, a_sub(x) = a_Si + b x (1 - x) + (a_Ge - a_Si) x^2, with x the germanium fraction of the buffer and b the bowing length.

Strain matters to the valley problem because it moves the Brillouin zone. The six conduction-band minima of silicon lie on the Delta lines near the X points; biaxial tensile strain lowers the two valleys along the growth direction, at plus and minus k0 on the z axis, far below the four in-plane valleys, so the low-energy physics of the well is a two-valley problem. Along the growth direction the reciprocal-lattice vector that joins the images of these two valleys is G0 = (I - eps)(b1 + b2), which for the strain tensor above has the single non-zero component G0z = (4 pi / a_Si)(1 - eps_zz). Its half, G0z / 2 = (2 pi / a_Si)(1 - eps_zz), is the zone boundary along [001] in the strained crystal, and the valley-specific sector of the +k0 valley along the growth direction is the open interval of wave numbers between zero and that boundary. The valley minimum itself is at k0, quoted as a fraction of 2 pi / a_Si of the unstrained crystal, and k1 = G0z / 2 - k0 is its distance to the strained zone boundary, the wave number whose double, 2 k1, characterises the long-period resonance of an oscillating germanium profile.

Returns
-------
dict holding the floats eps_parallel and eps_zz, the two strain components; zone_vector, the growth-direction component G0z in reciprocal nm; sector_edge, its half, in reciprocal nm; valley_wavenumber, k0 in reciprocal nm; and edge_distance, k1 in reciprocal nm.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def strained_valley_geometry(
    lattice_si: float,
    lattice_ge: float,
    bowing: float,
    substrate_ge_fraction: float,
    c11: float,
    c12: float,
    valley_fraction: float,
) -> dict:
    """Strain the silicon layer to its buffer and locate the valley sector along the growth direction.

    Parameters
    ----------
    lattice_si : float
        Lattice constant of silicon in nm, above zero.
    lattice_ge : float
        Lattice constant of germanium in nm, above zero.
    bowing : float
        Bowing length of the alloy lattice constant in nm.
    substrate_ge_fraction : float
        Germanium fraction of the relaxed buffer, between zero and one.
    c11 : float
        Elastic constant C11 of silicon in GPa, above zero.
    c12 : float
        Elastic constant C12 of silicon in GPa, above zero and below C11.
    valley_fraction : float
        Valley minimum k0 as a fraction of 2 pi / a_Si, inside the strained sector.

    Returns
    -------
    dict
        Under the keys eps_parallel, eps_zz, zone_vector, sector_edge, valley_wavenumber and edge_distance.

    Raises
    ------
    ValueError
        When any argument fails to be finite, when a lattice or elastic constant fails to be above zero, when C12 fails to be below C11, when the buffer fraction falls outside the closed interval from zero to one, or when the valley minimum fails to lie strictly inside the strained sector.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math


def _finite(value, label):
    """Return an argument as a float once it is known to be finite."""
    out = float(value)
    if not math.isfinite(out):
        raise ValueError("%s must be finite" % label)
    return out


def _oracle_strained_valley_geometry(
    lattice_si: float,
    lattice_ge: float,
    bowing: float,
    substrate_ge_fraction: float,
    c11: float,
    c12: float,
    valley_fraction: float,
) -> dict:
    """Reference implementation."""
    a_si = _finite(lattice_si, "lattice_si")
    a_ge = _finite(lattice_ge, "lattice_ge")
    b = _finite(bowing, "bowing")
    x = _finite(substrate_ge_fraction, "substrate_ge_fraction")
    stiff11 = _finite(c11, "c11")
    stiff12 = _finite(c12, "c12")
    frac = _finite(valley_fraction, "valley_fraction")
    if a_si <= 0.0 or a_ge <= 0.0:
        raise ValueError("lattice constants must be above zero")
    if stiff11 <= 0.0 or stiff12 <= 0.0:
        raise ValueError("elastic constants must be above zero")
    if stiff12 >= stiff11:
        raise ValueError("C12 must be below C11")
    if not 0.0 <= x <= 1.0:
        raise ValueError("substrate_ge_fraction must lie between zero and one")

    a_sub = a_si + b * x * (1.0 - x) + (a_ge - a_si) * x * x
    eps_par = (a_sub - a_si) / a_si
    eps_zz = -2.0 * stiff12 / stiff11 * eps_par
    zone_vector = 4.0 * math.pi / a_si * (1.0 - eps_zz)
    sector_edge = 0.5 * zone_vector
    k0 = frac * 2.0 * math.pi / a_si
    if not 0.0 < k0 < sector_edge:
        raise ValueError("the valley minimum must lie strictly inside the strained sector")
    return {
        "eps_parallel": eps_par,
        "eps_zz": eps_zz,
        "zone_vector": zone_vector,
        "sector_edge": sector_edge,
        "valley_wavenumber": k0,
        "edge_distance": sector_edge - k0,
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
            # the task configuration: Si on relaxed Si(0.7)Ge(0.3)
            "setup": """
import math
def digest(out):
    unit = 2.0 * math.pi / 0.543
    return (round(out["eps_parallel"], 10), round(out["eps_zz"], 10),
            round(out["zone_vector"], 8), round(out["sector_edge"] / unit, 10),
            round(out["valley_wavenumber"], 8), round(2.0 * out["edge_distance"] / unit, 10))
""" + FLAT,
            "call": "flat(digest(strained_valley_geometry(0.543, 0.565, 0.0200326, 0.3, 167.5, 65.0, 0.8394)))",
            "gold_call": "flat(digest(_oracle_strained_valley_geometry(0.543, 0.565, 0.0200326, 0.3, 167.5, 65.0, 0.8394)))",
        },
        {
            # boundaries of the buffer fraction: pure silicon leaves the zone unstrained, and the
            # bowing term vanishes at both ends so it cannot be detected there
            "setup": """
import math
def digest(fn, x, b):
    out = fn(0.543, 0.565, b, x, 167.5, 65.0, 0.8394)
    return (round(out["eps_parallel"], 12), round(out["eps_zz"], 12),
            round(out["zone_vector"] * 0.543 / (4.0 * math.pi), 12), round(out["edge_distance"], 10))
""" + FLAT,
            "call": "flat((digest(strained_valley_geometry, 0.0, 0.0200326), digest(strained_valley_geometry, 1.0, 0.0200326), digest(strained_valley_geometry, 1.0, 0.0), digest(strained_valley_geometry, 0.5, 0.0)))",
            "gold_call": "flat((digest(_oracle_strained_valley_geometry, 0.0, 0.0200326), digest(_oracle_strained_valley_geometry, 1.0, 0.0200326), digest(_oracle_strained_valley_geometry, 1.0, 0.0), digest(_oracle_strained_valley_geometry, 0.5, 0.0)))",
        },
        {
            # tetragonal response: eps_zz / eps_par must equal -2 C12 / C11 whatever the buffer,
            # and the sector edge must move with eps_zz and not with eps_par
            "setup": """
def digest(fn):
    rows = []
    for x, c12 in ((0.2, 65.0), (0.45, 50.0), (0.3, 120.0)):
        out = fn(0.543, 0.565, 0.0200326, x, 167.5, c12, 0.83)
        rows.append((round(out["eps_zz"] / out["eps_parallel"], 12),
                     round(out["sector_edge"] * 0.543 / 6.283185307179586 - 1.0 + out["eps_zz"], 13)))
    return rows
""" + FLAT,
            "call": "flat(digest(strained_valley_geometry))",
            "gold_call": "flat(digest(_oracle_strained_valley_geometry))",
        },
        {
            "setup": """
def verdict(fn, **kw):
    args = dict(lattice_si=0.543, lattice_ge=0.565, bowing=0.0200326, substrate_ge_fraction=0.3,
                c11=167.5, c12=65.0, valley_fraction=0.8394)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(strained_valley_geometry, lattice_si=0.0), verdict(strained_valley_geometry, substrate_ge_fraction=1.2), verdict(strained_valley_geometry, c12=170.0), verdict(strained_valley_geometry, valley_fraction=1.01), verdict(strained_valley_geometry, valley_fraction=0.0), verdict(strained_valley_geometry, bowing=float('nan')), verdict(strained_valley_geometry, c11=-1.0), verdict(strained_valley_geometry)))",
            "gold_call": "flat((verdict(_oracle_strained_valley_geometry, lattice_si=0.0), verdict(_oracle_strained_valley_geometry, substrate_ge_fraction=1.2), verdict(_oracle_strained_valley_geometry, c12=170.0), verdict(_oracle_strained_valley_geometry, valley_fraction=1.01), verdict(_oracle_strained_valley_geometry, valley_fraction=0.0), verdict(_oracle_strained_valley_geometry, bowing=float('nan')), verdict(_oracle_strained_valley_geometry, c11=-1.0), verdict(_oracle_strained_valley_geometry)))",
        },
    ]
