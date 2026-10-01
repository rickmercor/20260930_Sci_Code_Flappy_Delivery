"""
Constructs the geometric substrate that every later step consumes: the uniform Cartesian point cloud, the reference-configuration bond families within the horizon, the benchmark's normalized cubic B-spline influence function on every bond, the per-point discrete kernel integral, the method's truncation-corrected averaged bond kernel used in every family average, and the benchmark's pre-notch and no-fail bond masks. Returns one dictionary of arrays (points, bond pairs, kernel weights, masks) that steps 3, 4, 7, 8 and 9 take as input. Validates grid placement, the family radius rule, the kernel, the treatment of truncated families, and the bond bookkeeping. Deliberately excluded: damage, deformation, constitutive response, and dynamics.

Points sit at ((i+1/2)dx, (j+1/2)dx, (k+1/2)dx), i/j/k 0-based along x/y/z, each with volume V = dx^3. The family of a point holds every other point with 0 < |DX| <= delta in the reference configuration; each unordered pair is stored once with endpoints pk, pn and bond vector dX = X[pn] - X[pk]. The influence function is the benchmark's normalized cubic B-spline omega = 8/(pi delta^3) * f(q), f(q) = 1 - 6q^2 + 6q^3 for q <= 1/2, f(q) = 2(1-q)^3 for 1/2 < q <= 1, f(q) = 0 beyond, q = |DX|/delta. The discrete kernel integral omega0(X_k) = sum_n omega(|DX_kn|) V_n is close to one deep in the bulk and drops near free surfaces, and in this thin slab every point is boundary-affected. The method does not use the raw kernel in its family averages: it uses a truncation-corrected averaged bond kernel that accounts for the kernel content of both endpoint families, whose form must be taken from the source. Pre-notch pairs cross the row mid-plane (j <= ny/2-1 versus j >= ny/2) with both endpoint columns i <= nx/2-1; no-fail pairs have an endpoint in row j = 0 or j = ny-1; no bond is ever removed from a family.

Returns
-------
dict: point cloud, bond pairs, kernel omega, discrete kernel integral omega0, truncation-corrected omega_b, notch/nofail masks
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_families(nx: int, ny: int, nz: int, dx: float, delta: float) -> dict:
    """Point cloud, bond families, kernel weights and bond masks of the benchmark.

    Builds the uniform Cartesian point cloud with points at
    ((i+1/2)*dx, (j+1/2)*dx, (k+1/2)*dx) for 0 <= i < nx, 0 <= j < ny,
    0 <= k < nz (0-based indices i/j/k along x/y/z), each carrying volume
    V = dx**3, forms every bond family {X' : 0 < |X'-X| <= delta} in the
    reference configuration, evaluates the benchmark's normalized cubic
    B-spline influence function on every bond, accumulates the per-point
    discrete kernel integral omega0(X_k) = sum_n omega(|DX_kn|) * V_n,
    forms the method's truncation-corrected averaged bond kernel omega_b
    of every bond pair (the weighting the source uses in every family
    average; its form must be taken from the source), and tags the
    benchmark's pre-notch bond pairs (endpoints in rows j <= ny/2-1 and
    j >= ny/2 with both endpoint columns i <= nx/2-1) and no-fail bond
    pairs (at least one endpoint in row j = 0 or j = ny-1). This
    dictionary is the single geometric substrate consumed by every later
    step.

    Parameters
    ----------
    nx : int
        Number of grid points along x (>= 1).
    ny : int
        Number of grid points along y (>= 1).
    nz : int
        Number of grid points along z (>= 1).
    dx : float
        Grid spacing in m (> 0).
    delta : float
        Horizon radius in m (> dx).

    Returns
    -------
    dict
        Keys (N = nx*ny*nz points, P = number of unordered bond pairs, each
        pair listed exactly once with endpoints pk[p] and pn[p]):
        'nx', 'ny', 'nz' (int), 'dx', 'delta', 'V' (float, V = dx**3);
        'ijk' : (N, 3) int array of grid indices (i, j, k) of every point;
        'X' : (N, 3) float array of reference positions in m;
        'pk', 'pn' : (P,) int arrays of the two endpoint indices of every pair;
        'dX' : (P, 3) float array X[pn] - X[pk] in m;
        'r' : (P,) float array of bond lengths |dX| in m;
        'omega' : (P,) float array of the cubic B-spline kernel omega(|dX|) in 1/m^3;
        'omega0' : (N,) float array of the discrete kernel integral (dimensionless);
        'omega_b' : (P,) float array of the truncation-corrected averaged bond kernel in 1/m^3;
        'notch' : (P,) bool array, True for pre-notch bond pairs;
        'nofail' : (P,) bool array, True for no-fail bond pairs.

    Raises
    ------
    ValueError
        If a grid dimension is not a positive integer, dx <= 0, delta <= dx,
        or some point has an empty family.
    """
    return {}

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _s01_kernel(r, delta):
    q = np.asarray(r, dtype=float) / delta
    f = np.where(q <= 0.5, 1.0 - 6.0*q**2 + 6.0*q**3, 2.0*(1.0 - q)**3)
    f = np.where(q <= 1.0, np.maximum(f, 0.0), 0.0)
    return (8.0/np.pi) / delta**3 * f


def _oracle_build_families(nx, ny, nz, dx, delta):
    if not all(isinstance(v, (int, np.integer)) and v >= 1 for v in (nx, ny, nz)):
        raise ValueError("grid dimensions must be positive integers")
    if not (dx > 0.0 and delta > dx):
        raise ValueError("require dx > 0 and delta > dx")
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    ijk = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
    X = (ijk + 0.5) * dx
    N = nx * ny * nz
    V = dx**3
    m = int(np.floor(delta/dx)) + 1
    offs = []
    for a in range(-m, m+1):
        for b in range(-m, m+1):
            for c in range(-m, m+1):
                if (a, b, c) == (0, 0, 0):
                    continue
                if dx*np.sqrt(a*a + b*b + c*c) <= delta:
                    offs.append((a, b, c))
    offs = np.array(offs, dtype=int).reshape(-1, 3)
    pos = offs[(offs[:, 0] > 0) | ((offs[:, 0] == 0) & (offs[:, 1] > 0)) |
               ((offs[:, 0] == 0) & (offs[:, 1] == 0) & (offs[:, 2] > 0))]
    lut = -np.ones((nx, ny, nz), dtype=int)
    lut[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = np.arange(N)
    K, L = [], []
    for (a, b, c) in pos:
        i2 = ijk[:, 0] + a; j2 = ijk[:, 1] + b; k2 = ijk[:, 2] + c
        ok = (i2 >= 0) & (i2 < nx) & (j2 >= 0) & (j2 < ny) & (k2 >= 0) & (k2 < nz)
        K.append(np.arange(N)[ok]); L.append(lut[i2[ok], j2[ok], k2[ok]])
    pk = np.concatenate(K) if K else np.zeros(0, dtype=int)
    pn = np.concatenate(L) if L else np.zeros(0, dtype=int)
    if pk.size == 0:
        raise ValueError("no bond lies within the horizon: every point is isolated")
    dX = X[pn] - X[pk]
    r = np.linalg.norm(dX, axis=1)
    om = _s01_kernel(r, delta)
    om0 = np.zeros(N)
    np.add.at(om0, pk, om*V)
    np.add.at(om0, pn, om*V)
    if np.any(om0 <= 0.0):
        raise ValueError("a point has an empty family (zero discrete kernel integral)")
    omb = om*0.5*(1.0/om0[pk] + 1.0/om0[pn])
    jk = ijk[pk, 1]; jn = ijk[pn, 1]
    ik = ijk[pk, 0]; iN = ijk[pn, 0]
    half = ny // 2
    cross = ((jk <= half-1) & (jn >= half)) | ((jn <= half-1) & (jk >= half))
    notch = cross & (ik <= nx//2 - 1) & (iN <= nx//2 - 1)
    nofail = (jk == 0) | (jn == 0) | (jk == ny-1) | (jn == ny-1)
    return {'nx': int(nx), 'ny': int(ny), 'nz': int(nz), 'dx': float(dx),
            'delta': float(delta), 'V': float(V), 'ijk': ijk, 'X': X,
            'pk': pk, 'pn': pn, 'dX': dX, 'r': r, 'omega': om, 'omega0': om0,
            'omega_b': omb, 'notch': notch, 'nofail': nofail}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"name": "normal_frozen_configuration_mean_kernel_integral",
         "setup": "import numpy as np",
         "call": "float(np.mean(build_families(16, 10, 2, 1.0e-3, 2.015e-3)['omega0']))",
         "gold_call": "float(np.mean(_oracle_build_families(16, 10, 2, 1.0e-3, 2.015e-3)['omega0']))"},
        {"name": "normal_frozen_configuration_bond_pair_count",
         "setup": "import numpy as np",
         "call": "float(len(build_families(16, 10, 2, 1.0e-3, 2.015e-3)['pk']))",
         "gold_call": "float(len(_oracle_build_families(16, 10, 2, 1.0e-3, 2.015e-3)['pk']))"},
        {"name": "normal_frozen_configuration_prenotch_and_nofail_counts",
         "setup": "import numpy as np",
         "call": "float(1000*np.sum(build_families(16, 10, 2, 1.0e-3, 2.015e-3)['notch']) + np.sum(build_families(16, 10, 2, 1.0e-3, 2.015e-3)['nofail']))",
         "gold_call": "float(1000*np.sum(_oracle_build_families(16, 10, 2, 1.0e-3, 2.015e-3)['notch']) + np.sum(_oracle_build_families(16, 10, 2, 1.0e-3, 2.015e-3)['nofail']))"},
        {"name": "boundary_small_anisotropic_grid_finer_spacing",
         "setup": "import numpy as np",
         "call": "float(np.mean(build_families(6, 4, 2, 5.0e-4, 1.0075e-3)['omega0']))",
         "gold_call": "float(np.mean(_oracle_build_families(6, 4, 2, 5.0e-4, 1.0075e-3)['omega0']))"},
        {"name": "edge_axial_neighbors_closed_form",
         "setup": "import numpy as np",
         "call": "float(np.mean(build_families(3, 3, 3, 1.0e-3, 1.015e-3)['omega0']))",
         "gold_call": "float(np.mean(_oracle_build_families(3, 3, 3, 1.0e-3, 1.015e-3)['omega0']))"},
        {"name": "edge_truncation_corrected_weight_sum",
         "setup": "import numpy as np",
         "call": "float(np.sum(build_families(16, 10, 2, 1.0e-3, 2.015e-3)['omega_b'])*1.0e-9)",
         "gold_call": "float(np.sum(_oracle_build_families(16, 10, 2, 1.0e-3, 2.015e-3)['omega_b'])*1.0e-9)"},
    ]
