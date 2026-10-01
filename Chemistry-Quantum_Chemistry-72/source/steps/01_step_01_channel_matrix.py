"""
Step 01: Coupled-channel matrix of an atom plus rigid rotor for given J and parity. Coupled-channel interaction matrix of an atom and a rigid linear rotor for a given total angular momentum and parity.

A weakly bound complex of a rare-gas atom and a diatomic molecule is described by the atom-molecule distance R and the orientation of the molecule. In the coupled-channel picture only R is treated on a grid; the angular motion is expanded in a basis, and each basis function defines a channel. In a space-fixed description the rotor angular momentum j and the end-over-end angular momentum l of the complex are coupled to a total J that is conserved, and the total parity, which inversion of all coordinates multiplies each coupled function by, splits the problem into two independent blocks. Every channel carries a rotor energy and a centrifugal barrier, and the anisotropic interaction, written as a Legendre expansion in the angle between the molecular axis and the intermolecular vector, couples channels with different j and l within one block. The resulting matrix, scaled by twice the reduced mass over hbar squared, is the object that the radial second-order equations propagate. Channels whose rotor and centrifugal energy plus the diagonal interaction exceed the total energy at a given distance are locally closed and decay rather than oscillate.

Returns
-------
numpy.ndarray of shape (N, N): real symmetric coupled-channel matrix W(R) in inverse square angstrom over the space-fixed channels of the requested J and parity
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def channel_matrix(R: float, params: dict) -> "np.ndarray":
    '''Coupled-channel matrix W(R) of an atom plus rigid linear rotor complex, in inverse square angstrom.

    Parameters
    ----------
    R : float
        Intermolecular distance in angstrom, finite and > 0.
    params : dict
        Model parameters with keys
        'mu'     : reduced mass in amu (> 0);
        'B'      : rotational constant in cm^-1 (>= 0);
        'eps'    : well-depth scale in cm^-1 (> 0);
        'Rm'     : distance scale in angstrom (> 0);
        'a'      : sequence of three floats (a_0, a_1, a_2);
        'b'      : sequence of three floats (b_0, b_1, b_2);
        'jmax'   : largest rotor quantum number, integer >= 0;
        'J'      : total angular momentum, integer >= 0;
        'parity' : total parity, +1 or -1;
        'scale'  : dimensionless factor lambda multiplying the whole interaction potential (> 0).

    Returns
    -------
    W : np.ndarray
        Real symmetric N x N array. The channels are all space-fixed coupled states |(j l) J M> built from rotor
        functions of the molecular axis with j = 0, ..., jmax and partial waves of the intermolecular vector with any l
        allowed by the triangle rule for (j, l, J), restricted to total parity (-1)^(j + l) equal to 'parity'; N is
        their number. With the interaction V(R, theta) = scale * sum_{L=0,1,2} V_L(R) P_L(cos theta), where theta is
        the angle between the molecular axis and the intermolecular vector and
        V_L(R) = eps [ a_L (Rm / R)^12 - 2 b_L (Rm / R)^6 ] in cm^-1, and with C = 16.8576292 cm^-1 = hbar^2 / (2 amu
        angstrom^2), W = (mu / C) [ H_rot + C l(l + 1) / (mu R^2) + V ] represented in that basis, where H_rot has
        eigenvalue B j(j + 1). The radial equations are d^2 psi / dR^2 = [W(R) - (mu / C) E I] psi for energy E in
        cm^-1. The order of the channels and the sign of each basis function are free: results are compared only
        through quantities that do not depend on them, such as the eigenvalues of W.

    Raises
    ------
    ValueError
        If R is not finite or <= 0, a required parameter is missing or out of range, or no channel exists for the
        requested J, parity and jmax.
    '''
    return W

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from math import factorial

def _threej_zero(j1, j2, j3):
    from math import factorial
    total = j1 + j2 + j3
    if total % 2 or j3 > j1 + j2 or j3 < abs(j1 - j2):
        return 0.0
    g = total // 2
    return ((-1) ** g) * (factorial(total - 2 * j1) * factorial(total - 2 * j2) * factorial(total - 2 * j3)
                          / factorial(total + 1)) ** 0.5 * factorial(g) / (
        factorial(g - j1) * factorial(g - j2) * factorial(g - j3))


def _triangle_coefficient(a, b, c):
    from math import factorial
    return (factorial(a + b - c) * factorial(a - b + c) * factorial(-a + b + c) / factorial(a + b + c + 1)) ** 0.5


def _sixj(j1, j2, j3, j4, j5, j6):
    from math import factorial
    for a, b, c in ((j1, j2, j3), (j1, j5, j6), (j4, j2, j6), (j4, j5, j3)):
        if c > a + b or c < abs(a - b):
            return 0.0
    lo = max(j1 + j2 + j3, j1 + j5 + j6, j4 + j2 + j6, j4 + j5 + j3)
    hi = min(j1 + j2 + j4 + j5, j2 + j3 + j5 + j6, j3 + j1 + j6 + j4)
    total = 0.0
    for t in range(lo, hi + 1):
        total += (-1) ** t * factorial(t + 1) / (
            factorial(t - j1 - j2 - j3) * factorial(t - j1 - j5 - j6) * factorial(t - j4 - j2 - j6)
            * factorial(t - j4 - j5 - j3) * factorial(j1 + j2 + j4 + j5 - t) * factorial(j2 + j3 + j5 + j6 - t)
            * factorial(j3 + j1 + j6 + j4 - t))
    return (_triangle_coefficient(j1, j2, j3) * _triangle_coefficient(j1, j5, j6)
            * _triangle_coefficient(j4, j2, j6) * _triangle_coefficient(j4, j5, j3) * total)


def _check_params(params):
    keys = ("mu", "B", "eps", "Rm", "a", "b", "jmax", "J", "parity", "scale")
    if not isinstance(params, dict) or any(k not in params for k in keys):
        raise ValueError("params must contain mu, B, eps, Rm, a, b, jmax, J, parity, scale")
    a, b, jmax, J, parity = params["a"], params["b"], params["jmax"], params["J"], params["parity"]
    if len(a) != 3 or len(b) != 3:
        raise ValueError("a and b need three entries")
    for n in (jmax, J):
        if isinstance(n, bool) or int(n) != n or n < 0:
            raise ValueError("jmax and J must be non-negative integers")
    if parity not in (1, -1):
        raise ValueError("parity must be +1 or -1")
    if not (params["mu"] > 0 and params["B"] >= 0 and params["eps"] > 0 and params["Rm"] > 0 and params["scale"] > 0):
        raise ValueError("model parameter out of range")
    tables = _coupling_tables(int(jmax), int(J), int(parity))
    if tables[0].size == 0:
        raise ValueError("no channel for this J, parity and jmax")
    return [float(v) for v in a], [float(v) for v in b], tables


def _coupling_tables(jmax, J, parity):
    """Channel quantum numbers (j, l), j(j+1), l(l+1) and the Percival-Seaton coefficients for L = 0, 1, 2."""
    import numpy as np
    cache = _coupling_tables.__dict__.setdefault("cache", {})
    key = (jmax, J, parity)
    if key not in cache:
        chans = [(j, l) for j in range(jmax + 1) for l in range(abs(J - j), J + j + 1) if (-1) ** (j + l) == parity]
        N = len(chans)
        f = np.zeros((3, N, N))
        for L in range(3):
            for p, (j, l) in enumerate(chans):
                for q, (jp, lp) in enumerate(chans):
                    f[L, p, q] = ((-1) ** (j + jp - J) * ((2 * j + 1) * (2 * jp + 1) * (2 * l + 1) * (2 * lp + 1)) ** 0.5
                                  * _threej_zero(j, L, jp) * _threej_zero(l, L, lp) * _sixj(j, l, J, lp, jp, L))
        jj = np.array([j * (j + 1.0) for j, _ in chans])
        ll = np.array([l * (l + 1.0) for _, l in chans])
        cache[key] = (np.array(chans, dtype=int).reshape(N, 2), jj, ll, f)
    return cache[key]


def _oracle_channel_matrix(R: float, params: dict) -> "np.ndarray":
    import numpy as np
    if not np.isfinite(R) or R <= 0.0:
        raise ValueError("R must be finite and positive")
    a, b, (chans, jj, ll, f) = _check_params(params)
    C = 16.8576292
    mu = float(params["mu"])
    x6 = (params["Rm"] / R) ** 6
    lam_eps = params["scale"] * params["eps"]
    M = (lam_eps * (a[0] * x6 * x6 - 2.0 * b[0] * x6)) * f[0]
    M += (lam_eps * (a[1] * x6 * x6 - 2.0 * b[1] * x6)) * f[1]
    M += (lam_eps * (a[2] * x6 * x6 - 2.0 * b[2] * x6)) * f[2]
    M[np.diag_indices(jj.size)] += params["B"] * jj + C * ll / (mu * R * R)
    M *= mu / C
    return M

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = ("import numpy as np\n"
            "P = {'mu': 19.2, 'B': 10.4, 'eps': 180.0, 'Rm': 3.8, 'a': (1.0, 0.25, 0.35), 'b': (1.0, 0.15, 0.30),"
            " 'jmax': 6, 'J': 0, 'parity': 1, 'scale': 1.0}\n"
            "def _spec(W):\n    W = np.asarray(W, dtype=float)\n    return np.sort(np.linalg.eigvalsh(0.5 * (W + W.T)))\n")
    return [
        # --- Normal: benchmark complex at J = 0 near the potential minimum ---
        {"setup": base, "call": "_spec(channel_matrix(3.8, dict(P)))",
         "gold_call": "_spec(_oracle_channel_matrix(3.8, dict(P)))", "tol": 1e-9},
        # --- Normal: J = 3, even parity, scaled potential on the repulsive wall ---
        {"setup": base, "call": "_spec(channel_matrix(2.9, dict(P, J=3, scale=1.07)))",
         "gold_call": "_spec(_oracle_channel_matrix(2.9, dict(P, J=3, scale=1.07)))", "tol": 1e-9},
        # --- Normal: J = 2, odd parity, where every channel has l different from j ---
        {"setup": base, "call": "_spec(channel_matrix(4.1, dict(P, J=2, parity=-1)))",
         "gold_call": "_spec(_oracle_channel_matrix(4.1, dict(P, J=2, parity=-1)))", "tol": 1e-9},
        # --- Boundary: J larger than jmax, so every rotor state carries the full set of partial waves ---
        {"setup": base, "call": "_spec(channel_matrix(5.2, dict(P, jmax=4, J=9, parity=-1)))",
         "gold_call": "_spec(_oracle_channel_matrix(5.2, dict(P, jmax=4, J=9, parity=-1)))", "tol": 1e-9},
        # --- Boundary: heavy, strongly anisotropic complex with sign-changing anisotropy at long range ---
        {"setup": base + "Q = dict(P, mu=33.0, B=0.9, a=(1.0, -0.4, 0.6), b=(1.0, 0.3, -0.2), jmax=8, J=5, parity=1)\n",
         "call": "_spec(channel_matrix(11.0, dict(Q)))",
         "gold_call": "_spec(_oracle_channel_matrix(11.0, dict(Q)))", "tol": 1e-9},
        # --- Edge: J = 1 odd parity with jmax = 0, a single s-rotor channel with l = 1 ---
        {"setup": base, "call": "_spec(channel_matrix(3.5, dict(P, jmax=0, J=1, parity=-1)))",
         "gold_call": "_spec(_oracle_channel_matrix(3.5, dict(P, jmax=0, J=1, parity=-1)))", "tol": 1e-9},
        # --- Error: J = 0 with odd parity has no channel ---
        {"setup": base + "def _probe(fn):\n    try:\n        fn(3.8, dict(P, parity=-1))\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(channel_matrix)", "gold_call": "_probe(_oracle_channel_matrix)"},
    ]
