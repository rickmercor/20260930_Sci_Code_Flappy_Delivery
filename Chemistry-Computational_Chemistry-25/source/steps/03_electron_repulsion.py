"""
Assemble the two-electron repulsion tensor over the contracted basis. SPECIFICATION: table and coords are as in the previous step. Element (m, n, l, s) is the integral over both electron coordinates of the product of orbitals m and n evaluated at the first electron, times the product of orbitals l and s evaluated at the second, divided by the distance between the two electrons. The result is a real array of shape (n, n, n, n).

Storing the repulsion integrals as a rank-four tensor rather than contracting them against a density straight away is what makes the rest of the problem possible: the search needs individual slices of this tensor as raw material, and it needs them before any density exists. The tensor is invariant under exchanging the first pair, under exchanging the second pair, and under exchanging the two pairs with each other, which is worth exploiting because the cost otherwise grows as the fourth power of the basis size at every one of the geometries.

Returns
-------
numpy.ndarray of shape (n, n, n, n) and real dtype.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def electron_repulsion(table: "np.ndarray", coords: "np.ndarray") -> "np.ndarray":
    """Assemble the two-electron repulsion tensor over the contracted basis. SPECIFICATION: table and coords are as in the previous step. Element (m, n, l, s) is the integral over both electron coordinates of the product of orbitals m and n evaluated at the first electron, times the product of orbitals l and s evaluated at the second, divided by the distance between the two electrons. The result is a real array of shape (n, n, n, n).

    Parameters
    ----------
    table : numpy.ndarray of shape (n, 10) and real dtype
        The contracted-orbital table, one row per orbital.
    coords : numpy.ndarray of shape (2, 3) and real dtype
        Cartesian nuclear positions in bohr, one row per atom.

    Returns
    -------
    numpy.ndarray of shape (n, n, n, n) and real dtype.

    Raises
    ------
    ValueError: if the basis table is empty or does not have ten columns, or if coords does not have shape (2, 3).
    """
    return [[[[0.0]]]]  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import eigh
from scipy.special import gammainc, gamma

BOHR_PER_ANGSTROM = 1.0/0.52917721092
HARTREE2KCAL = 627.5094740631


def _boys(nmax, T):
    T = np.asarray(T, dtype=float)
    out = np.empty((nmax + 1,) + T.shape)
    small = T < 1e-8
    Ts = np.where(small, 1.0, T)
    for n in range(nmax + 1):
        a = n + 0.5
        big = 0.5*gamma(a)*gammainc(a, Ts)*np.power(Ts, -a)
        s = np.zeros_like(T)
        fac = 1.0
        for k in range(8):
            s = s + ((-T)**k)/(fac*(2*n + 2*k + 1))
            fac *= (k + 1)
        out[n] = np.where(small, s, big)
    return out


def _hermite_E(i, j, t, Qx, a, b):
    p = a + b
    q = a*b/p
    if t < 0 or t > i + j:
        return 0.0
    if i == 0 and j == 0:
        return np.exp(-q*Qx*Qx) if t == 0 else 0.0
    if j == 0:
        return (1.0/(2*p))*_hermite_E(i-1, j, t-1, Qx, a, b) \
               - (q*Qx/a)*_hermite_E(i-1, j, t, Qx, a, b) \
               + (t+1)*_hermite_E(i-1, j, t+1, Qx, a, b)
    return (1.0/(2*p))*_hermite_E(i, j-1, t-1, Qx, a, b) \
           + (q*Qx/b)*_hermite_E(i, j-1, t, Qx, a, b) \
           + (t+1)*_hermite_E(i, j-1, t+1, Qx, a, b)


def _hermite_R(tmax, umax, vmax, p, PQ):
    T = p*np.sum(PQ*PQ, axis=-1)
    nmax = tmax + umax + vmax
    F = _boys(nmax, T)
    R = {(0, 0, 0, n): ((-2.0*p)**n)*F[n] for n in range(nmax + 1)}
    for tot in range(1, nmax + 1):
        for t in range(tot + 1):
            for u in range(tot - t + 1):
                v = tot - t - u
                if t > tmax or u > umax or v > vmax:
                    continue
                for n in range(nmax - tot + 1):
                    if t > 0:
                        val = PQ[..., 0]*R[(t-1, u, v, n+1)]
                        if t > 1:
                            val = val + (t-1)*R[(t-2, u, v, n+1)]
                    elif u > 0:
                        val = PQ[..., 1]*R[(t, u-1, v, n+1)]
                        if u > 1:
                            val = val + (u-1)*R[(t, u-2, v, n+1)]
                    else:
                        val = PQ[..., 2]*R[(t, u, v-1, n+1)]
                        if v > 1:
                            val = val + (v-1)*R[(t, u, v-2, n+1)]
                    R[(t, u, v, n)] = val
    return R


def _unpack(table):
    table = np.asarray(table, dtype=float)
    if table.ndim != 2 or table.shape[1] != 10:
        raise ValueError("basis table must have ten columns")
    return (table[:, 0].astype(int), table[:, 1:4].astype(int),
            table[:, 4:7], table[:, 7:10])


def _pair_data(centers, lxyz, exps, coefs, coords, i, j):
    A = coords[centers[i]]
    B = coords[centers[j]]
    ei, ej = exps[i], exps[j]
    a = ei[:, None]
    b = ej[None, :]
    p = (a + b).ravel()
    P = ((a[..., None]*A + b[..., None]*B)/(a + b)[..., None]).reshape(-1, 3)
    c = (coefs[i][:, None]*coefs[j][None, :]).ravel()
    ab = [(x, y) for x in ei for y in ej]
    E = []
    for d in range(3):
        tmax = int(lxyz[i][d] + lxyz[j][d])
        tab = np.zeros((tmax + 1, p.size))
        for k, (aa, bb) in enumerate(ab):
            for t in range(tmax + 1):
                tab[t, k] = _hermite_E(int(lxyz[i][d]), int(lxyz[j][d]), t, A[d]-B[d], aa, bb)
        E.append(tab)
    return dict(p=p, P=P, c=c, E=E, ab=ab, A=A, B=B,
                lsum=tuple(int(lxyz[i][d] + lxyz[j][d]) for d in range(3)))


def _oracle_electron_repulsion(table: "np.ndarray", coords: "np.ndarray") -> "np.ndarray":
    centers, lxyz, exps, coefs = _unpack(table)
    coords = np.asarray(coords, dtype=float)
    n = len(centers)
    if n == 0:
        raise ValueError("empty basis")
    if coords.shape != (2, 3):
        raise ValueError("coords must have shape (2, 3)")
    keys = [(i, j) for i in range(n) for j in range(i + 1)]
    pr = {k: _pair_data(centers, lxyz, exps, coefs, coords, k[0], k[1]) for k in keys}
    out = np.zeros((n, n, n, n))
    for xa, (i, j) in enumerate(keys):
        d1 = pr[(i, j)]
        for (k, l) in keys[:xa + 1]:
            d2 = pr[(k, l)]
            p, q = d1["p"], d2["p"]
            alpha = p[:, None]*q[None, :]/(p[:, None] + q[None, :])
            PQ = d1["P"][:, None, :] - d2["P"][None, :, :]
            l1, l2 = d1["lsum"], d2["lsum"]
            R = _hermite_R(l1[0]+l2[0], l1[1]+l2[1], l1[2]+l2[2], alpha, PQ)
            acc = np.zeros_like(alpha)
            for t in range(l1[0] + 1):
                for u in range(l1[1] + 1):
                    for v in range(l1[2] + 1):
                        e1 = (d1["E"][0][t]*d1["E"][1][u]*d1["E"][2][v])[:, None]
                        for tt in range(l2[0] + 1):
                            for uu in range(l2[1] + 1):
                                for vv in range(l2[2] + 1):
                                    e2 = (d2["E"][0][tt]*d2["E"][1][uu]
                                          * d2["E"][2][vv])[None, :]
                                    acc = acc + ((-1)**(tt+uu+vv))*e1*e2 \
                                        * R[(t+tt, u+uu, v+vv, 0)]
            pref = 2.0*np.pi**2.5/(p[:, None]*q[None, :]*np.sqrt(p[:, None] + q[None, :]))
            val = float(np.sum(d1["c"][:, None]*d2["c"][None, :]*pref*acc))
            for a, b in ((i, j), (j, i)):
                for cq, dq in ((k, l), (l, k)):
                    out[a, b, cq, dq] = val
                    out[cq, dq, a, b] = val
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base_0 = """from copy import deepcopy as _dc
import numpy as np
BD = {'Li': [(0, [16.119575, 2.9362007, 0.7946505], [0.15432897, 0.53532814, 0.44463454]), (0, [0.6362897, 0.1478601, 0.0480887], [-0.09996723, 0.39951283, 0.70011547]), (1, [0.6362897, 0.1478601, 0.0480887], [0.15591627, 0.60768372, 0.39195739])], 'F': [(0, [166.67913, 30.360812, 8.2168207], [0.15432897, 0.53532814, 0.44463454]), (0, [6.4648032, 1.5022812, 0.4885885], [-0.09996723, 0.39951283, 0.70011547]), (1, [6.4648032, 1.5022812, 0.4885885], [0.15591627, 0.60768372, 0.39195739])]}
BD_G = _dc(BD)
"""
    base_1 = """from copy import deepcopy as _dc
import numpy as np
BD2 = {'Na': [(0, [250.77243, 45.678511, 12.362388], [0.1543289673, 0.5353281423, 0.4446345422]), (0, [12.040193, 2.7978819, 0.909958], [-0.09996722919, 0.3995128261, 0.7001154689]), (0, [1.4787406, 0.4125649, 0.1614751], [-0.219620369, 0.2255954336, 0.900398426]), (1, [12.040193, 2.7978819, 0.909958], [0.155916275, 0.6076837186, 0.3919573931]), (1, [1.4787406, 0.4125649, 0.1614751], [0.01058760429, 0.5951670053, 0.462001012])], 'Cl': [(0, [601.3456136, 109.5358542, 29.64467686], [0.1543289673, 0.5353281423, 0.4446345422]), (0, [38.96041889, 9.053563477, 2.944499834], [-0.09996722919, 0.3995128261, 0.7001154689]), (0, [2.129386495, 0.5940934274, 0.232524141], [-0.219620369, 0.2255954336, 0.900398426]), (1, [38.96041889, 9.053563477, 2.944499834], [0.155916275, 0.6076837186, 0.3919573931]), (1, [2.129386495, 0.5940934274, 0.232524141], [0.01058760429, 0.5951670053, 0.462001012])]}
BD2_G = _dc(BD2)
CO2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.4 / 0.52917721092]])
CO2_G = _dc(CO2)
TB2 = basis_table(['Na', 'Cl'], BD2)
TB2_G = _oracle_basis_table(['Na', 'Cl'], BD2_G)"""
    return [
        {
            "setup": base_0 + """CO = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.6 / 0.52917721092]])
CO_G = _dc(CO)
TB = basis_table(['Li', 'F'], BD)
TB_G = _oracle_basis_table(['Li', 'F'], BD_G)""",
            'call': 'electron_repulsion(TB,CO)',
            'gold_call': '_oracle_electron_repulsion(TB_G,CO_G)',
        },
        {
            "setup": base_0 + """CO2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.4 / 0.52917721092]])
CO2_G = _dc(CO2)
TB = basis_table(['Li', 'F'], BD)
TB_G = _oracle_basis_table(['Li', 'F'], BD_G)""",
            'call': 'electron_repulsion(TB,CO2)',
            'gold_call': '_oracle_electron_repulsion(TB_G,CO2_G)',
        },
        {
            "setup": base_1,
            'call': 'electron_repulsion(TB2,CO2)',
            'gold_call': '_oracle_electron_repulsion(TB2_G,CO2_G)',
        },
        {
            "setup": base_0 + """CO = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.6 / 0.52917721092]])
CO_G = _dc(CO)
TB = basis_table(['Li', 'F'], BD)
TB_G = _oracle_basis_table(['Li', 'F'], BD_G)

def _c():
    try:
        electron_repulsion(TB, CO[:1])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _g():
    try:
        _oracle_electron_repulsion(TB_G, CO_G[:1])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            'call': '_c()',
            'gold_call': '_g()',
        },
    ]
