"""
Compute the squared critical stretch of every directed bond from the critical energy density.

A bond fails once the energy density stored in it reaches a critical value w_c. That critical value is fixed by equating the total work required to break every bond crossing a unit fracture area to the fracture energy G_c, an integration that in two dimensions depends only on the horizon and yields a w_c independent of the influence function.

Because the stored energy density is quadratic in the bond stretch, the criterion is equivalent to a threshold on the squared stretch, and that threshold depends on the individual bond through both its reference length and the influence function evaluated at that length. The function returns this threshold, not the energy density itself, so that the failure test downstream is a direct comparison against the squared stretch.

The energy a bond stores is a physical quantity and does not depend on how the internal energy of the body is written as a sum. The micromodulus supplied to this function is the constant appearing in the micro-potential of the double-sum form, in which each bond appears twice, so the energy stored in one bond is obtained by accounting for both appearances before the threshold is formed.

The horizon and micromodulus used for a given bond are those of the point that parameterises that bond's micro-potential, consistent with the convention used everywhere else in the pipeline; for a spatially varying horizon these differ from the corresponding quantities of the bond's owner.

The influence function is the power law omega(r) = r ** (-influence_exponent). Bonds are indexed as in the directed bond list, and the returned array has one entry per directed bond, in that same order.

Raises ValueError if: lattice does not have shape (N, 5) with N >= 2; lattice contains non-finite values; bonds does not have shape (B, 2); any bond index lies outside the range of lattice; micromodulus does not have shape (N,) matching lattice; micromodulus contains non-finite or non-positive values; fracture_energy is not finite and strictly positive; influence_exponent is not finite; influence_exponent is greater than or equal to 3; any bond has non-positive reference length.

Two bond-failure criteria dominate peridynamic fracture modelling. The critical stretch criterion breaks a bond when its stretch reaches a fixed material constant. The critical energy density criterion breaks a bond when the energy stored in it reaches a threshold. Both constants are calibrated the same way, by requiring that the model dissipate the correct energy per unit of created crack surface: one integrates the energy carried by all bonds crossing a fracture plane, in their assumed critical state, over the volume swept by that plane, and equates the result to the fracture energy.

The two criteria are not interchangeable. Dividing the critical energy density by the stored energy per unit squared stretch recasts the energy criterion as a threshold on stretch, but that threshold now carries the bond's own reference length and the influence function evaluated there, rather than being a single constant for the material. The two formulations coincide for exactly one family of influence functions, the reciprocal weighting; for any other choice they break different sets of bonds, and which bonds break first, the short ones or the long ones, is determined by the influence function. This distinction propagates directly into the damage field, the crack path, and the crack-tip speed.

A second difference follows from the quadratic dependence on stretch: the energy criterion is insensitive to the sign of the stretch, so a bond in sufficient compression stores enough energy to fail, whereas a criterion phrased as a signed stretch threshold never breaks a compressed bond.

Two bookkeeping questions attend the calibration. The energy released when a bond breaks is a physical quantity, so a threshold derived from a fracture energy must refer to the energy that bond actually stores, not to whatever fraction of it a particular way of writing the total happens to attribute to one term of a sum. And under a spatially varying horizon the threshold depends on a horizon while the two points joined by a bond no longer share one; consistency requires the same choice used to parameterise the bond's micro-potential, so that the calibration underlying the threshold and the calibration underlying the force refer to the same neighbourhood.

Returns
-------
np.ndarray, a (B,) float array of squared critical stretch per directed bond, in the bond ordering
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def bond_critical_stretch_squared(lattice: np.ndarray, bonds: np.ndarray,
                                  micromodulus: np.ndarray, fracture_energy: float,
                                  influence_exponent: float) -> np.ndarray:
    '''Squared critical stretch of each directed bond under the critical energy density criterion.

    Parameters
    ----------
    lattice : np.ndarray
        (N, 5) float array with columns [x, y, spacing, horizon, volume].
    bonds : np.ndarray
        (B, 2) integer array with columns [owner, neighbour].
    micromodulus : np.ndarray
        (N,) float array of micro-potential constants, one per point.
    fracture_energy : float
        Critical energy release rate G_c of the material.
    influence_exponent : float
        Exponent a in the influence function omega(r) = r ** (-a). Must be < 3.

    Returns
    -------
    critical_stretch_squared : np.ndarray
        (B,) float array giving the squared critical stretch of each directed
        bond, in the bond ordering.
    '''
    critical_stretch_squared = np.zeros(len(np.asarray(bonds)), dtype=float)
    return critical_stretch_squared  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_bond_critical_stretch_squared(lattice: np.ndarray, bonds: np.ndarray,
                                          micromodulus: np.ndarray, fracture_energy: float,
                                          influence_exponent: float) -> np.ndarray:
    """Reference implementation."""
    lat = np.asarray(lattice, dtype=float)
    bd = np.asarray(bonds)
    cm = np.asarray(micromodulus, dtype=float)
    G = float(fracture_energy)
    a = float(influence_exponent)
    if lat.ndim != 2 or lat.shape[1] != 5 or lat.shape[0] < 2:
        raise ValueError("lattice must have shape (N, 5) with N >= 2")
    if not np.all(np.isfinite(lat)):
        raise ValueError("lattice must be finite")
    if bd.ndim != 2 or bd.shape[1] != 2:
        raise ValueError("bonds must have shape (B, 2)")
    if bd.shape[0] > 0 and (bd.min() < 0 or bd.max() >= lat.shape[0]):
        raise ValueError("bond indices out of range for lattice")
    if cm.ndim != 1 or cm.shape[0] != lat.shape[0]:
        raise ValueError("micromodulus must have shape (N,) matching lattice")
    if not np.all(np.isfinite(cm)) or np.any(cm <= 0.0):
        raise ValueError("micromodulus must be finite and > 0")
    if not (np.isfinite(G) and G > 0.0):
        raise ValueError("fracture_energy must be finite and > 0")
    if not np.isfinite(a):
        raise ValueError("influence_exponent must be finite")
    if a >= 3.0:
        raise ValueError("influence_exponent must be < 3")
    if bd.shape[0] == 0:
        return np.zeros(0, dtype=float)

    own = bd[:, 0].astype(np.int64)
    nbr = bd[:, 1].astype(np.int64)
    R = np.hypot(lat[nbr, 0] - lat[own, 0], lat[nbr, 1] - lat[own, 1])
    if np.any(R <= 0.0):
        raise ValueError("bonds must have strictly positive reference length")

    horizon_n = lat[nbr, 3]
    c_n = cm[nbr]
    return 3.0 * G / (2.0 * c_n * horizon_n ** 3 * R ** (1.0 - a))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: full production system, singular influence function ---
        {
            "setup": """import numpy as np
def _mk(xb, sp, h, mr):
    rows = []
    for r in range(len(sp)):
        d = sp[r]; lo = xb[r][0]
        nx = int(round((xb[r][1] - lo) / d)); ny = int(round(h / d))
        for ix in range(nx):
            for iy in range(ny):
                rows.append((lo + (ix + 0.5) * d, (iy + 0.5) * d, d, mr * d, d * d))
    return np.asarray(rows, dtype=float)
def _bonds(lat, cy, cxm):
    X, Y, dl = lat[:, 0], lat[:, 1], lat[:, 3]
    o, n = [], []
    for i in range(len(lat)):
        r = np.hypot(X - X[i], Y - Y[i])
        for j in np.where((r > 0.0) & (r <= dl[i] * (1.0 + 1e-12)))[0]:
            if (Y[i] - cy) * (Y[j] - cy) < 0.0:
                t = (cy - Y[i]) / (Y[j] - Y[i])
                if X[i] + t * (X[j] - X[i]) <= cxm:
                    continue
            o.append(i); n.append(j)
    return np.asarray([o, n], dtype=np.int64).T.reshape(-1, 2)
def _cm(E, d, a):
    return 8.0 * E / (5.0 * np.pi * (d ** (3.0 - a) / (3.0 - a)))
lattice = _mk([[0.0, 0.02], [0.02, 0.04]], [0.002, 0.001], 0.02, 3.015)
bonds = _bonds(lattice, 0.0101, 0.0151)
micromodulus = _cm(72e9, lattice[:, 3], 2.0)
fracture_energy = 135.0
influence_exponent = 2.0
""",
            "call": "bond_critical_stretch_squared(lattice, bonds, micromodulus, fracture_energy, influence_exponent)",
            "gold_call": "_oracle_bond_critical_stretch_squared(lattice, bonds, micromodulus, fracture_energy, influence_exponent)",
        },
        # --- Boundary: uniform horizon, constant influence function ---
        {
            "setup": """import numpy as np
def _mk(xb, sp, h, mr):
    rows = []
    for r in range(len(sp)):
        d = sp[r]; lo = xb[r][0]
        nx = int(round((xb[r][1] - lo) / d)); ny = int(round(h / d))
        for ix in range(nx):
            for iy in range(ny):
                rows.append((lo + (ix + 0.5) * d, (iy + 0.5) * d, d, mr * d, d * d))
    return np.asarray(rows, dtype=float)
def _bonds(lat, cy, cxm):
    X, Y, dl = lat[:, 0], lat[:, 1], lat[:, 3]
    o, n = [], []
    for i in range(len(lat)):
        r = np.hypot(X - X[i], Y - Y[i])
        for j in np.where((r > 0.0) & (r <= dl[i] * (1.0 + 1e-12)))[0]:
            o.append(i); n.append(j)
    return np.asarray([o, n], dtype=np.int64).T.reshape(-1, 2)
def _cm(E, d, a):
    return 8.0 * E / (5.0 * np.pi * (d ** (3.0 - a) / (3.0 - a)))
lattice = _mk([[0.0, 4.0]], [1.0], 4.0, 1.5)
bonds = _bonds(lattice, -10.0, 0.0)
micromodulus = _cm(1.0, lattice[:, 3], 0.0)
fracture_energy = 2.0
influence_exponent = 0.0
""",
            "call": "bond_critical_stretch_squared(lattice, bonds, micromodulus, fracture_energy, influence_exponent)",
            "gold_call": "_oracle_bond_critical_stretch_squared(lattice, bonds, micromodulus, fracture_energy, influence_exponent)",
        },
        # --- Edge: two regions with asymmetric family membership across the interface ---
        {
            "setup": """import numpy as np
def _mk(xb, sp, h, mr):
    rows = []
    for r in range(len(sp)):
        d = sp[r]; lo = xb[r][0]
        nx = int(round((xb[r][1] - lo) / d)); ny = int(round(h / d))
        for ix in range(nx):
            for iy in range(ny):
                rows.append((lo + (ix + 0.5) * d, (iy + 0.5) * d, d, mr * d, d * d))
    return np.asarray(rows, dtype=float)
def _bonds(lat, cy, cxm):
    X, Y, dl = lat[:, 0], lat[:, 1], lat[:, 3]
    o, n = [], []
    for i in range(len(lat)):
        r = np.hypot(X - X[i], Y - Y[i])
        for j in np.where((r > 0.0) & (r <= dl[i] * (1.0 + 1e-12)))[0]:
            o.append(i); n.append(j)
    return np.asarray([o, n], dtype=np.int64).T.reshape(-1, 2)
def _cm(E, d, a):
    return 8.0 * E / (5.0 * np.pi * (d ** (3.0 - a) / (3.0 - a)))
lattice = _mk([[0.0, 0.5], [0.5, 1.0]], [0.5, 0.25], 0.5, 2.0)
bonds = _bonds(lattice, -10.0, 0.0)
micromodulus = _cm(72e9, lattice[:, 3], 2.0)
fracture_energy = 135.0
influence_exponent = 2.0
""",
            "call": "bond_critical_stretch_squared(lattice, bonds, micromodulus, fracture_energy, influence_exponent)",
            "gold_call": "_oracle_bond_critical_stretch_squared(lattice, bonds, micromodulus, fracture_energy, influence_exponent)",
        },
        # --- Edge: reciprocal influence function on a three-region lattice with a partial crack ---
        {
            "setup": """import numpy as np
def _mk(xb, sp, h, mr):
    rows = []
    for r in range(len(sp)):
        d = sp[r]; lo = xb[r][0]
        nx = int(round((xb[r][1] - lo) / d)); ny = int(round(h / d))
        for ix in range(nx):
            for iy in range(ny):
                rows.append((lo + (ix + 0.5) * d, (iy + 0.5) * d, d, mr * d, d * d))
    return np.asarray(rows, dtype=float)
def _bonds(lat, cy, cxm):
    X, Y, dl = lat[:, 0], lat[:, 1], lat[:, 3]
    o, n = [], []
    for i in range(len(lat)):
        r = np.hypot(X - X[i], Y - Y[i])
        for j in np.where((r > 0.0) & (r <= dl[i] * (1.0 + 1e-12)))[0]:
            if (Y[i] - cy) * (Y[j] - cy) < 0.0:
                t = (cy - Y[i]) / (Y[j] - Y[i])
                if X[i] + t * (X[j] - X[i]) <= cxm:
                    continue
            o.append(i); n.append(j)
    return np.asarray([o, n], dtype=np.int64).T.reshape(-1, 2)
def _cm(E, d, a):
    return 8.0 * E / (5.0 * np.pi * (d ** (3.0 - a) / (3.0 - a)))
lattice = _mk([[0.0, 0.25], [0.25, 0.75], [0.75, 1.0]], [0.25, 0.125, 0.25], 0.25, 1.5)
bonds = _bonds(lattice, 0.13, 0.4)
micromodulus = _cm(190e9, lattice[:, 3], 1.0)
fracture_energy = 22170.0
influence_exponent = 1.0
""",
            "call": "bond_critical_stretch_squared(lattice, bonds, micromodulus, fracture_energy, influence_exponent)",
            "gold_call": "_oracle_bond_critical_stretch_squared(lattice, bonds, micromodulus, fracture_energy, influence_exponent)",
        },
        # --- Invalid: bond index outside the lattice ---
        {
            "setup": """import numpy as np
def _mk(xb, sp, h, mr):
    rows = []
    for r in range(len(sp)):
        d = sp[r]; lo = xb[r][0]
        nx = int(round((xb[r][1] - lo) / d)); ny = int(round(h / d))
        for ix in range(nx):
            for iy in range(ny):
                rows.append((lo + (ix + 0.5) * d, (iy + 0.5) * d, d, mr * d, d * d))
    return np.asarray(rows, dtype=float)
def _cm(E, d, a):
    return 8.0 * E / (5.0 * np.pi * (d ** (3.0 - a) / (3.0 - a)))
lattice = _mk([[0.0, 4.0]], [1.0], 4.0, 1.5)
micromodulus = _cm(1.0, lattice[:, 3], 0.0)
bonds = np.array([[0, 99]], dtype=np.int64)
def run_model():
    try:
        bond_critical_stretch_squared(lattice, bonds, micromodulus, 2.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_bond_critical_stretch_squared(lattice, bonds, micromodulus, 2.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive fracture energy ---
        {
            "setup": """import numpy as np
def _mk(xb, sp, h, mr):
    rows = []
    for r in range(len(sp)):
        d = sp[r]; lo = xb[r][0]
        nx = int(round((xb[r][1] - lo) / d)); ny = int(round(h / d))
        for ix in range(nx):
            for iy in range(ny):
                rows.append((lo + (ix + 0.5) * d, (iy + 0.5) * d, d, mr * d, d * d))
    return np.asarray(rows, dtype=float)
def _bonds(lat, cy, cxm):
    X, Y, dl = lat[:, 0], lat[:, 1], lat[:, 3]
    o, n = [], []
    for i in range(len(lat)):
        r = np.hypot(X - X[i], Y - Y[i])
        for j in np.where((r > 0.0) & (r <= dl[i] * (1.0 + 1e-12)))[0]:
            o.append(i); n.append(j)
    return np.asarray([o, n], dtype=np.int64).T.reshape(-1, 2)
def _cm(E, d, a):
    return 8.0 * E / (5.0 * np.pi * (d ** (3.0 - a) / (3.0 - a)))
lattice = _mk([[0.0, 4.0]], [1.0], 4.0, 1.5)
bonds = _bonds(lattice, -10.0, 0.0)
micromodulus = _cm(1.0, lattice[:, 3], 0.0)
def run_model():
    try:
        bond_critical_stretch_squared(lattice, bonds, micromodulus, 0.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_bond_critical_stretch_squared(lattice, bonds, micromodulus, 0.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: micromodulus length does not match the lattice ---
        {
            "setup": """import numpy as np
def _mk(xb, sp, h, mr):
    rows = []
    for r in range(len(sp)):
        d = sp[r]; lo = xb[r][0]
        nx = int(round((xb[r][1] - lo) / d)); ny = int(round(h / d))
        for ix in range(nx):
            for iy in range(ny):
                rows.append((lo + (ix + 0.5) * d, (iy + 0.5) * d, d, mr * d, d * d))
    return np.asarray(rows, dtype=float)
def _bonds(lat, cy, cxm):
    X, Y, dl = lat[:, 0], lat[:, 1], lat[:, 3]
    o, n = [], []
    for i in range(len(lat)):
        r = np.hypot(X - X[i], Y - Y[i])
        for j in np.where((r > 0.0) & (r <= dl[i] * (1.0 + 1e-12)))[0]:
            o.append(i); n.append(j)
    return np.asarray([o, n], dtype=np.int64).T.reshape(-1, 2)
lattice = _mk([[0.0, 4.0]], [1.0], 4.0, 1.5)
bonds = _bonds(lattice, -10.0, 0.0)
micromodulus = np.ones(3, dtype=float)
def run_model():
    try:
        bond_critical_stretch_squared(lattice, bonds, micromodulus, 2.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_bond_critical_stretch_squared(lattice, bonds, micromodulus, 2.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
