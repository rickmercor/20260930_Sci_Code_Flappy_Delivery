"""
Estimate the critical explicit time step of each region from the discrete peridynamic stiffness.

Von Neumann stability analysis of the explicit update bounds the step size at a material point by the square root of twice the mass density divided by the total stiffness accumulated at that point. The stiffness assembled here is that of the equations of motion the integrator actually advances, so every directed bond contributes to both of the points it joins, using the same coefficient that multiplies the stretch in the pairwise force and the same volume weighting: the contribution deposited on a bond's owner carries the neighbour's volume, and the contribution deposited on its neighbour carries the owner's volume. Under a spatially varying horizon these two contributions differ, so a point's stiffness cannot be obtained from its own family alone.

This convention differs from the form in which the bound is usually tabulated, where a single summation over each point's own family is taken; that form is equivalent only when neighbourhood membership is reciprocal.

The critical step of a region is the smallest point-wise bound among the points assigned to that region.

Points are assigned to regions by region_ids. The returned array has one entry per distinct identifier present, ordered by increasing identifier.

The influence function is the power law omega(r) = r ** (-influence_exponent), and micromodulus is the constant appearing in the micro-potential, as returned by the calibration step and used unchanged.

Raises ValueError if: lattice does not have shape (N, 5) with N >= 2; lattice contains non-finite values; bonds does not have shape (B, 2) with B >= 1; any bond index lies outside the range of lattice; micromodulus does not have shape (N,) matching lattice; micromodulus contains non-finite or non-positive values; region_ids does not have shape (N,) matching lattice; region_ids is not an integer array; density is not finite and strictly positive; influence_exponent is not finite or is greater than or equal to 3; any bond has non-positive reference length; any point accumulates no stiffness contribution.

Explicit time integration of peridynamic dynamics is conditionally stable. The admissible step follows from a von Neumann analysis of the linearised discrete equations, in which the point-wise stiffness is assembled by summing, over all interactions touching a point, the pairwise force coefficient divided by the bond length and weighted by the quadrature volume. The resulting bound scales with the square root of the mass density over that stiffness, so refining the discretisation tightens it twice over: the horizon shrinks, which raises the calibrated micro-modulus, and the neighbourhood volume shrinks, which changes how the interactions are weighted.

For a uniform discretisation this yields a single global bound, and a safety factor below one is applied to accommodate the nonlinearity, the loading, and the progressive loss of stiffness as bonds break. For a locally refined discretisation the bound is governed everywhere by the smallest horizon in the domain, which is precisely what makes globally synchronous explicit integration inefficient: the coarse region is forced to advance at the pace set by the fine region, and most of its internal-force evaluations produce no additional accuracy. Estimating the bound region by region instead is what makes multirate integration possible, since each region can then be advanced at its own admissible rate with a common safety factor.

The bound is only as good as its agreement with the dynamics it is meant to constrain. Standard statements of the criterion are written for reciprocal neighbourhoods, where a single sum over a point's own family already captures every interaction touching it. When neighbourhood relations are one-sided that equivalence fails: a point receives stiffness both from the bonds it owns and from the bonds that reach it from elsewhere, and the two carry different quadrature volumes because the interacting points occupy voxels of different size. Applying the tabulated single-sum form unchanged then understates the stiffness of points near a refinement interface and returns a bound that is not conservative, so the convention in use has to be stated rather than inherited.

Returns
-------
np.ndarray, a (R,) float array of critical time steps, one per distinct region identifier, ordered by increasing identifier
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def regional_critical_time_step(lattice: np.ndarray, bonds: np.ndarray,
                                micromodulus: np.ndarray, region_ids: np.ndarray,
                                density: float, influence_exponent: float) -> np.ndarray:
    '''Critical explicit time step of each region.

    Parameters
    ----------
    lattice : np.ndarray
        (N, 5) float array with columns [x, y, spacing, horizon, volume].
    bonds : np.ndarray
        (B, 2) integer array with columns [owner, neighbour].
    micromodulus : np.ndarray
        (N,) float array of micro-potential constants, one per point.
    region_ids : np.ndarray
        (N,) integer array assigning each point to a region.
    density : float
        Mass density of the material.
    influence_exponent : float
        Exponent a in the influence function omega(r) = r ** (-a). Must be < 3.

    Returns
    -------
    critical_step : np.ndarray
        Array with one entry per distinct region identifier, ordered by
        increasing identifier.
    '''
    critical_step = np.zeros(len(np.unique(np.asarray(region_ids))), dtype=float)
    return critical_step  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_regional_critical_time_step(lattice: np.ndarray, bonds: np.ndarray,
                                        micromodulus: np.ndarray, region_ids: np.ndarray,
                                        density: float, influence_exponent: float) -> np.ndarray:
    """Reference implementation."""
    lat = np.asarray(lattice, dtype=float)
    bd = np.asarray(bonds)
    cm = np.asarray(micromodulus, dtype=float)
    rid = np.asarray(region_ids)
    rho = float(density)
    a = float(influence_exponent)
    if lat.ndim != 2 or lat.shape[1] != 5 or lat.shape[0] < 2:
        raise ValueError("lattice must have shape (N, 5) with N >= 2")
    if not np.all(np.isfinite(lat)):
        raise ValueError("lattice must be finite")
    if bd.ndim != 2 or bd.shape[1] != 2 or bd.shape[0] < 1:
        raise ValueError("bonds must have shape (B, 2) with B >= 1")
    if bd.min() < 0 or bd.max() >= lat.shape[0]:
        raise ValueError("bond indices out of range for lattice")
    if cm.ndim != 1 or cm.shape[0] != lat.shape[0]:
        raise ValueError("micromodulus must have shape (N,) matching lattice")
    if not np.all(np.isfinite(cm)) or np.any(cm <= 0.0):
        raise ValueError("micromodulus must be finite and > 0")
    if rid.ndim != 1 or rid.shape[0] != lat.shape[0]:
        raise ValueError("region_ids must have shape (N,) matching lattice")
    if not np.issubdtype(rid.dtype, np.integer):
        raise ValueError("region_ids must be an integer array")
    if not (np.isfinite(rho) and rho > 0.0):
        raise ValueError("density must be finite and > 0")
    if not np.isfinite(a) or a >= 3.0:
        raise ValueError("influence_exponent must be finite and < 3")

    own = bd[:, 0].astype(np.int64)
    nbr = bd[:, 1].astype(np.int64)
    R = np.hypot(lat[nbr, 0] - lat[own, 0], lat[nbr, 1] - lat[own, 1])
    if np.any(R <= 0.0):
        raise ValueError("bonds must have strictly positive reference length")

    coef = cm[nbr] * R ** (-a) / R
    stiffness = np.zeros(lat.shape[0], dtype=float)
    np.add.at(stiffness, own, coef * lat[nbr, 4])
    np.add.at(stiffness, nbr, coef * lat[own, 4])
    if np.any(stiffness <= 0.0):
        raise ValueError("every point must carry at least one bond contribution")

    point_step = np.sqrt(2.0 * rho / stiffness)
    return np.array([point_step[rid == r].min() for r in np.unique(rid)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: two production regions ---
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
region_ids = (lattice[:, 0] > 0.02).astype(np.int64)
density = 2440.0
influence_exponent = 2.0
""",
            "call": "regional_critical_time_step(lattice, bonds, micromodulus, region_ids, density, influence_exponent)",
            "gold_call": "_oracle_regional_critical_time_step(lattice, bonds, micromodulus, region_ids, density, influence_exponent)",
        },
        # --- Boundary: single uniform region, constant influence function ---
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
region_ids = np.zeros(len(lattice), dtype=np.int64)
density = 1.0
influence_exponent = 0.0
""",
            "call": "regional_critical_time_step(lattice, bonds, micromodulus, region_ids, density, influence_exponent)",
            "gold_call": "_oracle_regional_critical_time_step(lattice, bonds, micromodulus, region_ids, density, influence_exponent)",
        },
        # --- Edge: three regions, coarse-fine-coarse, with a partial crack ---
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
region_ids = np.zeros(len(lattice), dtype=np.int64)
region_ids[lattice[:, 0] > 0.25] = 1
region_ids[lattice[:, 0] > 0.75] = 2
density = 8000.0
influence_exponent = 1.0
""",
            "call": "regional_critical_time_step(lattice, bonds, micromodulus, region_ids, density, influence_exponent)",
            "gold_call": "_oracle_regional_critical_time_step(lattice, bonds, micromodulus, region_ids, density, influence_exponent)",
        },
        # --- Invalid: an isolated point carrying no bonds ---
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
lattice = np.vstack([lattice, np.array([[100.0, 100.0, 1.0, 1.5, 1.0]])])
bonds = _bonds(lattice, -10.0, 0.0)
micromodulus = _cm(1.0, lattice[:, 3], 0.0)
region_ids = np.zeros(len(lattice), dtype=np.int64)
def run_model():
    try:
        regional_critical_time_step(lattice, bonds, micromodulus, region_ids, 1.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_regional_critical_time_step(lattice, bonds, micromodulus, region_ids, 1.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive density ---
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
region_ids = np.zeros(len(lattice), dtype=np.int64)
def run_model():
    try:
        regional_critical_time_step(lattice, bonds, micromodulus, region_ids, 0.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_regional_critical_time_step(lattice, bonds, micromodulus, region_ids, 0.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-integer region identifiers ---
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
region_ids = np.zeros(len(lattice), dtype=float)
def run_model():
    try:
        regional_critical_time_step(lattice, bonds, micromodulus, region_ids, 1.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_regional_critical_time_step(lattice, bonds, micromodulus, region_ids, 1.0, 0.0)
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
