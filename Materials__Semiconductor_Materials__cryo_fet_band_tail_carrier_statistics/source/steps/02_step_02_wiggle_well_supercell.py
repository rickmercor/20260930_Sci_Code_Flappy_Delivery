"""
The confinement the electron sees along the growth direction z is set by the germanium profile of the stack and by the vertical gate field. The silicon well occupies -h < z < 0, so z = 0 is its upper interface, and the SiGe barrier above it is a spacer that ends at an impenetrable gate dielectric at z = d. The nominal germanium fraction is X(z) = X_b (1 - Xi(z)) + X_mod(z), where Xi(z) = (1/2)[tanh((z + h) / sigma_l) - tanh(z / sigma_u)] is the smoothed well shape, equal to one deep inside the well and zero deep in either barrier, sigma_u and sigma_l are the widths of the upper and lower interfaces, and X_b is the barrier germanium fraction. A wiggle well adds germanium inside the well only, X_mod(z) = (X_w / 2)(1 + cos(q z)) Xi(z), with amplitude X_w and modulation wave number q = 2 pi / lambda for a stated oscillation period lambda. The well thickness h is given in silicon monolayers of a_Si / 4 each.

The conduction-band potential energy follows from a linear band offset in the germanium fraction and from the field, U(z) = Delta E_c X(z) - e F z, in electronvolts when z is in nm and the field F is given in volts per nm; with F above zero the energy falls with height, so the electron is pulled towards the upper interface and the spacer. This zero of energy, the conduction-band edge of the silicon well with the field potential vanishing at the upper interface, is the one the task states.

The equations of the next stages are solved on a periodic supercell whose Fourier grid is matched to the strained crystal. With G0z the growth-direction reciprocal-lattice vector of the strained crystal, N_FBZ grid wave numbers per Brillouin zone and N_BZ resolved zones, the cell length is L = 2 pi N_FBZ / G0z, the cell holds N = N_BZ N_FBZ points at spacing dz = L / N, and the grid wave numbers are k_m = 2 pi m / L for m = -N/2, ..., N/2 - 1 in the discrete Fourier transform order 0, 1, ..., N/2 - 1, -N/2, ..., -1. With N_FBZ even, both the zone centre and the zone boundary G0z / 2 fall exactly on grid points, which is what makes an open valley sector a well-defined set of grid wave numbers. The cell is placed to end at the dielectric: its points are z_j = d - L + j dz for j = 0, ..., N - 1. The periodic continuation then joins the low potential just below the dielectric to the high potential at the bottom of the buffer, a barrier that stands in for the hard wall and is invisible to a state confined in the well.

Returns
-------
dict holding the arrays z, the N grid positions in nm; wavenumbers, the grid wave numbers in reciprocal nm in discrete Fourier transform order; ge_fraction, the germanium fraction; well_indicator, the smoothed well shape Xi; and potential, U in eV; together with the floats spacing and length in nm.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def wiggle_well_supercell(
    zone_vector: float,
    n_bz: int,
    n_fbz: int,
    spacer: float,
    well_monolayers: float,
    lattice_si: float,
    sigma_upper: float,
    sigma_lower: float,
    x_barrier: float,
    x_wiggle: float,
    wiggle_period: float,
    band_offset: float,
    field: float,
) -> dict:
    """Lay out the periodic supercell and evaluate the wiggle-well germanium profile and confinement energy on it.

    Parameters
    ----------
    zone_vector : float
        Strained growth-direction reciprocal-lattice vector G0z in reciprocal nm, above zero.
    n_bz : int
        Number of resolved Brillouin zones, at least two.
    n_fbz : int
        Grid wave numbers per zone, even and at least four.
    spacer : float
        Distance from the upper interface to the dielectric in nm, above zero.
    well_monolayers : float
        Well thickness in monolayers of a_Si / 4, above zero.
    lattice_si : float
        Lattice constant of silicon in nm, above zero.
    sigma_upper : float
        Upper interface width in nm, above zero.
    sigma_lower : float
        Lower interface width in nm, above zero.
    x_barrier : float
        Barrier germanium fraction, between zero and one.
    x_wiggle : float
        Amplitude of the in-well oscillation, between zero and one.
    wiggle_period : float
        Oscillation period in nm, above zero.
    band_offset : float
        Conduction-band offset per unit germanium fraction in eV.
    field : float
        Vertical field in volts per nm.

    Returns
    -------
    dict
        Under the keys z, wavenumbers, ge_fraction, well_indicator, potential, spacing and length.

    Raises
    ------
    ValueError
        When any float argument fails to be finite, when a length, width, period or the zone vector fails to be above zero, when a germanium fraction falls outside the closed interval from zero to one, when n_bz fails to be an integer of at least two, when n_fbz fails to be an even integer of at least four, or when the cell fails to be longer than the well and the spacer together.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _finite(value, label):
    """Return an argument as a float once it is known to be finite."""
    out = float(value)
    if not math.isfinite(out):
        raise ValueError("%s must be finite" % label)
    return out


def _positive(value, label):
    """Return an argument as a float once it is known to be finite and above zero."""
    out = _finite(value, label)
    if out <= 0.0:
        raise ValueError("%s must be above zero" % label)
    return out


def _integer(value, label, least):
    """Return an argument as an int once it is known to be an integer of at least the stated size."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("%s must be an integer" % label)
    out = int(value)
    if out < least:
        raise ValueError("%s must be at least %d" % (label, least))
    return out


def _oracle_wiggle_well_supercell(
    zone_vector: float,
    n_bz: int,
    n_fbz: int,
    spacer: float,
    well_monolayers: float,
    lattice_si: float,
    sigma_upper: float,
    sigma_lower: float,
    x_barrier: float,
    x_wiggle: float,
    wiggle_period: float,
    band_offset: float,
    field: float,
) -> dict:
    """Reference implementation."""
    g0 = _positive(zone_vector, "zone_vector")
    zones = _integer(n_bz, "n_bz", 2)
    per_zone = _integer(n_fbz, "n_fbz", 4)
    if per_zone % 2 != 0:
        raise ValueError("n_fbz must be even")
    d = _positive(spacer, "spacer")
    monolayers = _positive(well_monolayers, "well_monolayers")
    a_si = _positive(lattice_si, "lattice_si")
    s_u = _positive(sigma_upper, "sigma_upper")
    s_l = _positive(sigma_lower, "sigma_lower")
    x_b = _finite(x_barrier, "x_barrier")
    x_w = _finite(x_wiggle, "x_wiggle")
    period = _positive(wiggle_period, "wiggle_period")
    offset = _finite(band_offset, "band_offset")
    f = _finite(field, "field")
    if not 0.0 <= x_b <= 1.0:
        raise ValueError("x_barrier must lie between zero and one")
    if not 0.0 <= x_w <= 1.0:
        raise ValueError("x_wiggle must lie between zero and one")

    h = monolayers * a_si / 4.0
    n = zones * per_zone
    length = 2.0 * math.pi * per_zone / g0
    if length <= d + h:
        raise ValueError("the cell must be longer than the well and the spacer together")
    dz = length / n
    z = d - length + dz * np.arange(n)
    wavenumbers = 2.0 * math.pi / length * np.fft.fftfreq(n, d=1.0 / n)

    indicator = 0.5 * (np.tanh((z + h) / s_l) - np.tanh(z / s_u))
    modulation = 0.5 * x_w * (1.0 + np.cos(2.0 * math.pi / period * z)) * indicator
    ge = x_b * (1.0 - indicator) + modulation
    potential = offset * ge - f * z
    return {
        "z": z,
        "wavenumbers": wavenumbers,
        "ge_fraction": ge,
        "well_indicator": indicator,
        "potential": potential,
        "spacing": dz,
        "length": length,
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
            # the task configuration: grid layout and the profile sampled at fixed indices
            "setup": """
import math
import numpy as np
G0 = 4.0 * math.pi / 0.543 * (1.0 + 0.008842567)
def digest(out):
    z = out["z"]
    pick = [0, 1050, 1070, 1100, 1250, 1300, 1400, 1440, 1450, 1700, 2559]
    return (len(z), round(out["length"], 9), round(out["spacing"], 12), round(z[0], 9), round(z[-1], 9),
            round(float(out["wavenumbers"][1]) * out["length"] / (2.0 * math.pi), 9),
            round(float(out["wavenumbers"][128]) / G0, 9), round(float(out["wavenumbers"][1280]) / G0, 9),
            [round(float(out["ge_fraction"][i]), 9) for i in pick],
            [round(float(out["well_indicator"][i]), 9) for i in pick],
            [round(float(out["potential"][i]), 9) for i in pick])
""" + FLAT,
            "call": "flat(digest(wiggle_well_supercell(G0, 10, 256, 30.0, 75, 0.543, 0.5, 0.5, 0.3, 0.15, 1.629, 0.5, 3.0e-3)))",
            "gold_call": "flat(digest(_oracle_wiggle_well_supercell(G0, 10, 256, 30.0, 75, 0.543, 0.5, 0.5, 0.3, 0.15, 1.629, 0.5, 3.0e-3)))",
        },
        {
            # a flat well without field: the profile must be even about the well centre up to the
            # sampling, the barrier must return X_b, and the potential must be Delta E_c X exactly
            "setup": """
import math
import numpy as np
def digest(fn):
    out = fn(4.0 * math.pi / 0.543, 4, 64, 6.0, 40, 0.543, 0.3, 0.7, 0.25, 0.0, 2.0, 0.6, 0.0)
    z = out["z"]; x = out["ge_fraction"]; u = out["potential"]
    deep = int(np.argmin(np.abs(z + 20 * 0.543 / 4.0)))
    return (len(z), round(out["length"], 10), round(float(np.max(np.abs(u - 0.6 * x))), 14),
            round(float(x[0]), 10), round(float(x[deep]), 10), round(float(np.sum(out["well_indicator"]) * out["spacing"]), 8))
""" + FLAT,
            "call": "flat(digest(wiggle_well_supercell))",
            "gold_call": "flat(digest(_oracle_wiggle_well_supercell))",
        },
        {
            # the field term alone: removing the germanium leaves U = -F z to rounding, and the
            # modulation must vanish outside the well and oscillate with the stated period inside
            "setup": """
import math
import numpy as np
def digest(fn):
    a = fn(4.0 * math.pi / 0.543, 6, 32, 3.0, 20, 0.543, 0.2, 0.2, 0.0, 0.0, 1.0, 0.5, -2.0e-3)
    b = fn(4.0 * math.pi / 0.543, 6, 32, 3.0, 20, 0.543, 0.2, 0.2, 0.0, 0.4, 1.0, 0.5, 0.0)
    z = a["z"]
    return (round(float(np.max(np.abs(a["potential"] - 2.0e-3 * z))), 14),
            round(float(np.max(b["ge_fraction"][z > 1.5])), 12),
            round(float(np.max(b["ge_fraction"])), 8), round(float(np.sum(b["ge_fraction"]) * b["spacing"]), 8))
""" + FLAT,
            "call": "flat(digest(wiggle_well_supercell))",
            "gold_call": "flat(digest(_oracle_wiggle_well_supercell))",
        },
        {
            "setup": """
def verdict(fn, **kw):
    args = dict(zone_vector=23.35, n_bz=10, n_fbz=256, spacer=30.0, well_monolayers=75, lattice_si=0.543,
                sigma_upper=0.5, sigma_lower=0.5, x_barrier=0.3, x_wiggle=0.15, wiggle_period=1.629,
                band_offset=0.5, field=3.0e-3)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(wiggle_well_supercell, n_fbz=255), verdict(wiggle_well_supercell, n_bz=1), verdict(wiggle_well_supercell, n_bz=10.0), verdict(wiggle_well_supercell, n_fbz=128), verdict(wiggle_well_supercell, x_barrier=1.5), verdict(wiggle_well_supercell, sigma_upper=0.0), verdict(wiggle_well_supercell, field=float('inf')), verdict(wiggle_well_supercell, wiggle_period=-1.0), verdict(wiggle_well_supercell)))",
            "gold_call": "flat((verdict(_oracle_wiggle_well_supercell, n_fbz=255), verdict(_oracle_wiggle_well_supercell, n_bz=1), verdict(_oracle_wiggle_well_supercell, n_bz=10.0), verdict(_oracle_wiggle_well_supercell, n_fbz=128), verdict(_oracle_wiggle_well_supercell, x_barrier=1.5), verdict(_oracle_wiggle_well_supercell, sigma_upper=0.0), verdict(_oracle_wiggle_well_supercell, field=float('inf')), verdict(_oracle_wiggle_well_supercell, wiggle_period=-1.0), verdict(_oracle_wiggle_well_supercell)))",
        },
    ]
