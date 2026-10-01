"""
Assemble the whole chain and return the local hybrid's exact-exchange admixture energy for a spherical, spin-polarised atom, summed over both spin channels. The majority channel holds one electron in the inner orbital and outer_occupation of an electron in the outer one; the minority channel holds one electron in the inner orbital alone. A channel is evaluated only at the radii where its own density exceeds 1e-14, and each channel is integrated over its own surviving radii. Both channels' effective hole normalisations must be formed before either channel's exchange-correlation hole normalisation is taken, and at a radius a channel did not survive its effective hole normalisation is taken to be one. Return the total as a plain float.

The admixture energy is the whole difference between the local hybrid and the plain semilocal functional it is built on, so every convention in the chain reaches it undiluted. It is a small difference of two much larger exchange energies, which is why the quadrature has to be converged rather than merely reasonable, and why masking the far tail matters at all.

Returns
-------
float: the exact-exchange admixture energy of the atom, in hartree.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def lhnz_exchange_correction(inner_exponents: "np.ndarray", inner_coefficients: "np.ndarray", outer_exponent: float, outer_occupation: float, c_param: float, b_param: float, n_nodes: int, scale: float) -> float:
    '''Assemble the whole chain and return the local hybrid's exact-exchange admixture energy for a spherical, spin-polarised atom, summed over both spin channels. The majority channel holds one electron in the inner orbital and outer_occupation of an electron in the outer one; the minority channel holds one electron in the inner orbital alone. A channel is evaluated only at the radii where its own density exceeds 1e-14, and each channel is integrated over its own surviving radii. Both channels' effective hole normalisations must be formed before either channel's exchange-correlation hole normalisation is taken, and at a radius a channel did not survive its effective hole normalisation is taken to be one. Return the total as a plain float.

    Parameters
    ----------
    inner_exponents : np.ndarray
        Positive Slater exponents of the inner orbital, in inverse bohr.
    inner_coefficients : np.ndarray
        Weights of those primitives, same length as inner_exponents.
    outer_exponent : float
        Positive Slater exponent of the outer orbital, in inverse bohr.
    outer_occupation : float
        Occupation of the outer orbital in the majority channel, between zero and one.
    c_param : float
        Positive coefficient multiplying the length inside the mixing function.
    b_param : float
        Non-negative constant scaling the hole-normalisation factor.
    n_nodes : int
        Number of radial quadrature nodes, at least one.
    scale : float
        Positive quadrature length scale in bohr.

    Returns
    -------
    admixture : float
        float: the exact-exchange admixture energy of the atom, in hartree.

    Raises
    ------
    ValueError
        if outer_occupation lies outside zero to one, or if any value it forwards to an earlier step is invalid.
    '''
    return admixture  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _gl01(n):
    """Gauss-Legendre nodes and weights mapped from [-1, 1] onto (0, 1)."""
    t, w = np.polynomial.legendre.leggauss(int(n))
    return 0.5 * (t + 1.0), 0.5 * w


def _sto_norm(principal, exponent):
    """Norm of the radial Slater primitive r**(principal-1) * exp(-exponent r)."""
    if principal == 1:
        return math.sqrt(exponent ** 3 / math.pi)
    return math.sqrt(exponent ** 5 / (3.0 * math.pi))


def _sto_eval(principal, exponent, r, order):
    """Value (order 0), d/dr (order 1) or d2/dr2 (order 2) of a normalised primitive."""
    c = _sto_norm(principal, exponent)
    e = np.exp(-exponent * r)
    if principal == 1:
        if order == 0:
            return c * e
        if order == 1:
            return -exponent * c * e
        return exponent * exponent * c * e
    if order == 0:
        return c * r * e
    if order == 1:
        return c * (1.0 - exponent * r) * e
    return c * (exponent * exponent * r - 2.0 * exponent) * e


def _expand(orbital, r, order):
    """Evaluate one padded (nterm, 3) orbital table on r."""
    out = np.zeros_like(r, dtype=float)
    for principal, exponent, coef in orbital:
        if coef == 0.0:
            continue
        out = out + coef * _sto_eval(int(round(principal)), float(exponent), r, order)
    return out


def _pair_terms(a, b):
    """Collect the product of two orbital tables as {(power, decay): coefficient}."""
    out = {}
    for pa, za, ca in a:
        if ca == 0.0:
            continue
        for pb, zb, cb in b:
            if cb == 0.0:
                continue
            key = (int(round(pa)) + int(round(pb)) - 2, float(za) + float(zb))
            w = ca * cb * _sto_norm(int(round(pa)), float(za)) * _sto_norm(int(round(pb)), float(zb))
            out[key] = out.get(key, 0.0) + w
    return out


def _overlap(a, b):
    """Overlap of two orbital tables, integrated over all space."""
    s = 0.0
    for (power, decay), w in _pair_terms(a, b).items():
        s += w * 4.0 * math.pi * math.factorial(power + 2) / decay ** (power + 3)
    return s


def _gamma_lo(n, x):
    """Lower incomplete gamma P(n+1, x) for integer n, by its finite exponential sum."""
    s = np.zeros_like(x)
    term = np.ones_like(x)
    for m in range(n + 1):
        s = s + term
        term = term * x / (m + 1)
    return 1.0 - np.exp(-x) * s


def _gamma_hi(n, x):
    s = np.zeros_like(x)
    term = np.ones_like(x)
    for m in range(n + 1):
        s = s + term
        term = term * x / (m + 1)
    return np.exp(-x) * s


def _radial_coulomb(power, decay, r):
    """Potential at radius r of the spherical density s**power * exp(-decay s)."""
    x = decay * r
    lo = math.factorial(power + 2) / decay ** (power + 3) * _gamma_lo(power + 2, x)
    hi = math.factorial(power + 1) / decay ** (power + 2) * _gamma_hi(power + 1, x)
    return 4.0 * math.pi * (lo / r + hi)


def _brx(x, rhs):
    """Residual and derivative of the Becke-Roussel defining equation."""
    e = np.exp(-2.0 * x / 3.0)
    return x * e / (x - 2.0) - rhs, 2.0 / 3.0 * (2.0 * x - x * x - 3.0) / (x - 2.0) ** 2 * e


def _bisect(fun, lo, hi, iters=200):
    """Vectorised bisection on a monotone residual, bracketed by lo and hi."""
    lo = np.array(lo, dtype=float)
    hi = np.array(hi, dtype=float)
    flo = fun(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = fun(mid)
        same = np.sign(fm) == np.sign(flo)
        lo = np.where(same, mid, lo)
        flo = np.where(same, fm, flo)
        hi = np.where(same, hi, mid)
    return 0.5 * (lo + hi)


def _br_potential(density, curvature, hole_normalisation):
    """Becke-Roussel model exchange potential at the reference point."""
    rho = np.asarray(density, dtype=float).ravel()
    q = np.asarray(curvature, dtype=float).ravel()
    n = np.asarray(hole_normalisation, dtype=float).ravel()
    if rho.size != q.size:
        raise ValueError("density and curvature must have the same length")
    if n.size == 1:
        n = np.full(rho.size, float(n[0]))
    if n.size != rho.size:
        raise ValueError("hole_normalisation must be a scalar or match the density length")
    if np.any(rho <= 0.0):
        raise ValueError("every density value must be positive")
    if np.any(n <= 0.0):
        raise ValueError("every hole normalisation must be positive")
    flat = q == 0.0
    safe = np.where(flat, 1.0, q)
    rhs = (2.0 / 3.0) * (math.pi * rho / n) ** (2.0 / 3.0) * rho / safe
    lo = np.where(rhs < 0.0, 1e-12, 2.0 + 1e-13)
    hi = np.where(rhs < 0.0, 2.0 - 1e-13, 400.0)
    x = _bisect(lambda v: _brx(v, rhs)[0], lo, hi)
    for _ in range(80):
        f, df = _brx(x, rhs)
        x = x - f / df
    # A vanishing curvature sends the right-hand side to infinity from either side, and
    # the left-hand side diverges only at x = 2, so x = 2 IS the limit -- not an error.
    x = np.where(flat, 2.0, x)
    e = np.exp(-x)
    alpha = (8.0 * math.pi * rho / e / n) ** (1.0 / 3.0)
    b = x / alpha
    return -n * (1.0 - e - 0.5 * x * e) / b


def _erf_array(values):
    """Error function evaluated elementwise, using only the standard library."""
    arr = np.asarray(values, dtype=float)
    flat = np.array([math.erf(float(v)) for v in arr.ravel()], dtype=float)
    return flat.reshape(arr.shape)


def _oracle_lhnz_exchange_correction(inner_exponents: "np.ndarray", inner_coefficients: "np.ndarray", outer_exponent: float, outer_occupation: float, c_param: float, b_param: float, n_nodes: int, scale: float) -> float:
    """Local-hybrid exchange correction of the LHnz functional for the atom."""
    if not 0.0 <= float(outer_occupation) <= 1.0:
        raise ValueError("outer_occupation must lie between zero and one")
    grid = _oracle_radial_quadrature(n_nodes, scale)
    r, w = grid[0], grid[1]
    orb = _oracle_orthonormal_orbitals(inner_exponents, inner_coefficients, outer_exponent)
    occ = (np.array([1.0, float(outer_occupation)]), np.array([1.0]))
    sets = (orb, orb[:1])
    keep = []
    fields = []
    epsx = []
    epsb = []
    for k in range(2):
        f = _oracle_spin_channel_fields(sets[k], occ[k], r)
        m = f[0] > 1e-14
        keep.append(m)
        fields.append(f[:, m])
        epsx.append(_oracle_exact_exchange_density(sets[k], occ[k], r[m]))
        epsb.append(_oracle_b86b_exchange_density(f[0][m], f[1][m]))
    norms = [_oracle_effective_hole_normalisation(fields[k][0], fields[k][4], epsx[k])
             for k in range(2)]
    full = []
    for k in range(2):
        v = np.ones_like(r)
        v[keep[k]] = norms[k]
        full.append(v)
    total = 0.0
    for k in range(2):
        m = keep[k]
        nxc = _oracle_xc_hole_normalisation(full[k][m], full[1 - k][m])
        g = _oracle_local_mixing_function(fields[k][0], epsb[k], nxc, c_param, b_param)
        total += _oracle_channel_admixture(g, epsx[k], epsb[k], r[m], w[m])
    return total

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "lhnz_exchange_correction([4.70,2.45],[0.15,0.90],0.66,0.5,0.10,4.6,400,1.0)",
         "gold_call": "_oracle_lhnz_exchange_correction([4.70,2.45],[0.15,0.90],0.66,0.5,0.10,4.6,400,1.0)"},   # normal
        {"setup": "import numpy as np",
         "call": "lhnz_exchange_correction([4.70,2.45],[0.15,0.90],0.66,1.0,0.10,4.6,300,1.0)",
         "gold_call": "_oracle_lhnz_exchange_correction([4.70,2.45],[0.15,0.90],0.66,1.0,0.10,4.6,300,1.0)"},   # boundary
        {"setup": "import numpy as np",
         "call": "lhnz_exchange_correction([3.0],[1.0],0.9,0.25,0.10,4.6,300,2.0)",
         "gold_call": "_oracle_lhnz_exchange_correction([3.0],[1.0],0.9,0.25,0.10,4.6,300,2.0)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        lhnz_exchange_correction([4.70,2.45],[0.15,0.90],0.66,1.5,0.10,4.6,200,1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_lhnz_exchange_correction([4.70,2.45],[0.15,0.90],0.66,1.5,0.10,4.6,200,1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
