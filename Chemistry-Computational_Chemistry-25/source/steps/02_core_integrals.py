"""
Assemble the one-electron matrices over the contracted basis. SPECIFICATION: table is the array returned by the previous step, coords holds the two nuclear positions in bohr with shape (2, 3), and charges holds their two nuclear charges in the same order. Element (i, j) of the overlap matrix is the integral of the product of orbitals i and j; of the kinetic matrix, the integral of orbital i against minus one half of the Laplacian of orbital j; and of the nuclear attraction matrix, the integral of the product of orbitals i and j against the sum over both nuclei of minus the charge divided by the distance to that nucleus. All three are real and symmetric. The result is a real array of shape (3, n, n) holding, in this order, the overlap matrix, the kinetic matrix and the nuclear attraction matrix.

These three matrices are the entire one-electron content of the problem and they are what the later search is allowed to combine. The nuclear attraction integral is the only one of the three that cannot be written in closed form with elementary functions; it needs the Boys function, whose naive power series loses all its digits once its argument is large and whose closed form loses them when the argument is near zero, so both regimes have to be handled or a handful of matrix elements come back as noise.

Returns
-------
numpy.ndarray of shape (3, n, n) and real dtype.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def core_integrals(table: "np.ndarray", coords: "np.ndarray", charges: "list") -> "np.ndarray":
    """Assemble the one-electron matrices over the contracted basis. SPECIFICATION: table is the array returned by the previous step, coords holds the two nuclear positions in bohr with shape (2, 3), and charges holds their two nuclear charges in the same order. Element (i, j) of the overlap matrix is the integral of the product of orbitals i and j; of the kinetic matrix, the integral of orbital i against minus one half of the Laplacian of orbital j; and of the nuclear attraction matrix, the integral of the product of orbitals i and j against the sum over both nuclei of minus the charge divided by the distance to that nucleus. All three are real and symmetric. The result is a real array of shape (3, n, n) holding, in this order, the overlap matrix, the kinetic matrix and the nuclear attraction matrix.

    Parameters
    ----------
    table : numpy.ndarray of shape (n, 10) and real dtype
        The contracted-orbital table, one row per orbital.
    coords : numpy.ndarray of shape (2, 3) and real dtype
        Cartesian nuclear positions in bohr, one row per atom.
    charges : list of float, length 2
        Nuclear charges, in the same atom order as coords.

    Returns
    -------
    numpy.ndarray of shape (3, n, n) and real dtype.

    Raises
    ------
    ValueError: if the basis table is empty or does not have ten columns, or if coords does not have shape (2, 3), or if charges does not have length two.
    """
    return [[[0.0]]]  # placeholder

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


def _oracle_core_integrals(table: "np.ndarray", coords: "np.ndarray", charges: "list") -> "np.ndarray":
    centers, lxyz, exps, coefs = _unpack(table)
    coords = np.asarray(coords, dtype=float)
    Z = np.asarray(charges, dtype=float)
    n = len(centers)
    if n == 0:
        raise ValueError("empty basis")
    if coords.shape != (2, 3) or Z.size != 2:
        raise ValueError("coords must have shape (2, 3) and charges length two")
    S = np.zeros((n, n)); T = np.zeros((n, n)); V = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            d = _pair_data(centers, lxyz, exps, coefs, coords, i, j)
            p, c, E, ab = d["p"], d["c"], d["E"], d["ab"]
            Q = d["A"] - d["B"]
            bj = np.array([y for _, y in ab])

            def _ov(shift):
                tot = np.ones_like(p)
                for dax in range(3):
                    jj = int(lxyz[j][dax]) + shift[dax]
                    if jj < 0:
                        return np.zeros_like(p)
                    tot = tot*np.array([_hermite_E(int(lxyz[i][dax]), jj, 0, Q[dax], aa, bb)
                                        for aa, bb in ab])
                return tot*np.power(np.pi/p, 1.5)

            s0 = _ov((0, 0, 0))
            S[i, j] = S[j, i] = float(np.sum(c*s0))
            t = bj*(2*int(np.sum(lxyz[j])) + 3)*s0
            for dax in range(3):
                up = [0, 0, 0]; up[dax] = 2
                dn = [0, 0, 0]; dn[dax] = -2
                t = t - 2.0*bj*bj*_ov(tuple(up))
                if int(lxyz[j][dax]) >= 2:
                    t = t - 0.5*lxyz[j][dax]*(lxyz[j][dax]-1)*_ov(tuple(dn))
            T[i, j] = T[j, i] = float(np.sum(c*t))
            ls = d["lsum"]
            acc = np.zeros_like(p)
            for C, Zc in zip(coords, Z):
                R = _hermite_R(ls[0], ls[1], ls[2], p, d["P"] - C[None, :])
                sm = np.zeros_like(p)
                for t1 in range(ls[0] + 1):
                    for u1 in range(ls[1] + 1):
                        for v1 in range(ls[2] + 1):
                            sm = sm + E[0][t1]*E[1][u1]*E[2][v1]*R[(t1, u1, v1, 0)]
                acc = acc - Zc*sm
            V[i, j] = V[j, i] = float(np.sum(c*2.0*np.pi/p*acc))
    return np.stack([S, T, V])

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
            'call': 'core_integrals(TB,CO,[3.,9.])',
            'gold_call': '_oracle_core_integrals(TB_G,CO_G,[3.,9.])',
        },
        {
            "setup": base_0 + """CO2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.4 / 0.52917721092]])
CO2_G = _dc(CO2)
TB = basis_table(['Li', 'F'], BD)
TB_G = _oracle_basis_table(['Li', 'F'], BD_G)""",
            'call': 'core_integrals(TB,CO2,[3.,9.])',
            'gold_call': '_oracle_core_integrals(TB_G,CO2_G,[3.,9.])',
        },
        {
            "setup": base_1,
            'call': 'core_integrals(TB2,CO2,[11.,17.])',
            'gold_call': '_oracle_core_integrals(TB2_G,CO2_G,[11.,17.])',
        },
        {
            "setup": base_0 + """CO = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.6 / 0.52917721092]])
CO_G = _dc(CO)
TB = basis_table(['Li', 'F'], BD)
TB_G = _oracle_basis_table(['Li', 'F'], BD_G)

def _c():
    try:
        core_integrals(TB, CO, [3.0])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _g():
    try:
        _oracle_core_integrals(TB_G, CO_G, [3.0])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            'call': '_c()',
            'gold_call': '_g()',
        },
    ]
