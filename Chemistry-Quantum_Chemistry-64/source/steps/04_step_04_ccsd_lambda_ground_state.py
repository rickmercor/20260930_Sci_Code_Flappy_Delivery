"""
Step 04 - Ground-state CCSD cluster and Lambda amplitudes in spin orbitals.

Ground-state coupled-cluster singles and doubles with its de-excitation (Lambda)
amplitudes, in a spin-orbital basis.

A coupled-cluster state is described by two independent objects: the ket
|Psi> = exp(T)|Phi0> and the bra <~Psi| = <Phi0|(1 + Lambda) exp(-T). The bra
is not the Hermitian conjugate of the ket; it is determined by its own
equations, and both are needed later, because every expectation value and
every population of the state is a bra-ket pair. The real-time propagation of
this task starts from the converged pair.

Work in spin orbitals. The reference determinant |Phi0> occupies spin orbitals
0 .. n_electrons - 1; the others are virtual. Indices i, j label occupied and
a, b virtual spin orbitals, and the amplitude arrays use local indices (a runs
from 0 over the virtual block). With the excitation operators
X_i^a = a_a^+ a_i and X_ij^ab = a_a^+ a_b^+ a_j a_i,

T = sum_ia t1[i,a] X_i^a + (1/4) sum_ijab t2[i,j,a,b] X_ij^ab
Lambda = sum_ia l1[i,a] (X_i^a)^+ + (1/4) sum_ijab l2[i,j,a,b] (X_ij^ab)^+

where t2 and l2 are antisymmetric in (i, j) and in (a, b). The Hamiltonian is
given by the Fock matrix f of the reference determinant and the antisymmetrised
integrals g[p,q,r,s] = <pq||rs>; the constant reference energy plays no role.
The Fock matrix is not assumed diagonal: its occupied-occupied and
virtual-virtual blocks may have off-diagonal elements and its
occupied-virtual block need not vanish, and the equations must hold as written
for such a matrix.

The amplitudes solve the projected equations
<Phi_i^a| exp(-T) H exp(T) |Phi0> = 0 and <Phi_ij^ab| exp(-T) H exp(T) |Phi0> = 0,
and then, with T fixed, <Phi0| (1 + Lambda) [exp(-T) H exp(T), X_mu] |Phi0> = 0
for every single and double excitation X_mu. Both sets are solved by Jacobi
iteration with denominators from the Fock diagonal,
D_i^a = f_ii - f_aa and D_ij^ab = f_ii + f_jj - f_aa - f_bb: each sweep adds the
current residual divided by the denominator. The cluster amplitudes start from
t1 = 0 and t2 = g_ijab / D_ij^ab and the de-excitation amplitudes start from
the converged cluster amplitudes. A set is converged when the largest residual
magnitude falls below 1e-11; 500 sweeps are allowed for each.

The four arrays are returned flattened in C order and concatenated as
[t1, t2, l1, l2], with the full antisymmetric (o, o, v, v) arrays for the
doubles, so the vector has length 2 (o v + o^2 v^2) with o occupied and v
virtual spin orbitals.

Returns
-------
numpy.ndarray of length 2 (o v + o^2 v^2): the converged ground-state amplitudes packed as [t1, t2, l1, l2]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def ccsd_lambda_ground_state(fock: np.ndarray, eri_as: np.ndarray, n_electrons: int) -> np.ndarray:
    '''Converged ground-state CCSD cluster and de-excitation amplitudes, packed in one vector.

    Parameters
    ----------
    fock : np.ndarray
        Spin-orbital Fock matrix of the reference determinant, shape (n, n),
        real and symmetric, not necessarily diagonal, hartree.
    eri_as : np.ndarray
        Antisymmetrised two-electron integrals g[p,q,r,s] = <pq||rs>, shape
        (n, n, n, n), hartree.
    n_electrons : int
        Number of electrons; the reference occupies spin orbitals
        0 .. n_electrons - 1.

    Returns
    -------
    amplitudes : np.ndarray
        Real vector [t1, t2, l1, l2] (each flattened in C order, doubles as full
        antisymmetric (o, o, v, v) arrays) of length 2 (o v + o^2 v^2), where
        o = n_electrons and v = n - o.

    Raises
    ------
    ValueError
        If fock is not a finite symmetric square matrix of size at least 2
        (tolerance 1e-10), if eri_as is not a finite (n, n, n, n) array that is
        antisymmetric within each index pair and symmetric under exchange of
        the pairs (tolerance 1e-10), if n_electrons is not an integer in
        [1, n - 1], if a Jacobi denominator D_i^a, or D_ij^ab with i != j and
        a != b, is smaller than 1e-8 in magnitude, or if either set of
        equations fails to converge within 500 sweeps.
    '''
    return amplitudes

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _cc_blocks(f, g, o):
    O, V = slice(0, o), slice(o, f.shape[0])
    return dict(foo=f[O, O], fov=f[O, V], fvv=f[V, V],
                oooo=g[O, O, O, O], ooov=g[O, O, O, V], oovo=g[O, O, V, O], oovv=g[O, O, V, V],
                ovov=g[O, V, O, V], ovvo=g[O, V, V, O], ovoo=g[O, V, O, O], ovvv=g[O, V, V, V],
                vovv=g[V, O, V, V], vvvo=g[V, V, V, O], vvvv=g[V, V, V, V])


def _cc_ein(s, *a):
    return np.einsum(s, *a, optimize=True)


def _cc_taus(t1, t2):
    tt = _cc_ein('ia,jb->ijab', t1, t1)
    tt = tt - tt.transpose(0, 1, 3, 2)
    return t2 + tt, t2 + 0.5 * tt


def _cc_pij(x):
    return x - x.transpose(1, 0, 2, 3)


def _cc_pab(x):
    return x - x.transpose(0, 1, 3, 2)


def _cc_t_residuals(f, g, o, t1, t2):
    """<Phi_mu| exp(-T) H exp(T) |Phi0> for singles and doubles (general, non-canonical f)."""
    B = _cc_blocks(f, g, o)
    e = _cc_ein
    tau, taut = _cc_taus(t1, t2)
    Fae = B['fvv'] - 0.5 * e('me,ma->ae', B['fov'], t1) + e('mf,mafe->ae', t1, B['ovvv']) - 0.5 * e('mnaf,mnef->ae', taut, B['oovv'])
    Fmi = B['foo'] + 0.5 * e('ie,me->mi', t1, B['fov']) + e('ne,mnie->mi', t1, B['ooov']) + 0.5 * e('inef,mnef->mi', taut, B['oovv'])
    Fme = B['fov'] + e('nf,mnef->me', t1, B['oovv'])
    x = e('je,mnie->mnij', t1, B['ooov'])
    Wmnij = B['oooo'] + x - x.transpose(0, 1, 3, 2) + 0.25 * e('ijef,mnef->mnij', tau, B['oovv'])
    x = e('mb,amef->abef', t1, B['vovv'])
    Wabef = B['vvvv'] - x + x.transpose(1, 0, 2, 3) + 0.25 * e('mnab,mnef->abef', tau, B['oovv'])
    Wmbej = (B['ovvo'] + e('jf,mbef->mbej', t1, B['ovvv']) - e('nb,mnej->mbej', t1, B['oovo'])
             - e('jnfb,mnef->mbej', 0.5 * t2 + e('jf,nb->jnfb', t1, t1), B['oovv']))
    R1 = (B['fov'] + e('ie,ae->ia', t1, Fae) - e('ma,mi->ia', t1, Fmi) + e('imae,me->ia', t2, Fme)
          - e('nf,naif->ia', t1, B['ovov']) - 0.5 * e('imef,maef->ia', t2, B['ovvv'])
          - 0.5 * e('mnae,nmei->ia', t2, B['oovo']))
    R2 = B['oovv'] + _cc_pab(e('ijae,be->ijab', t2, Fae - 0.5 * e('mb,me->be', t1, Fme)))
    R2 = R2 - _cc_pij(e('imab,mj->ijab', t2, Fmi + 0.5 * e('je,me->mj', t1, Fme)))
    R2 = R2 + 0.5 * e('mnab,mnij->ijab', tau, Wmnij) + 0.5 * e('ijef,abef->ijab', tau, Wabef)
    R2 = R2 + _cc_pij(_cc_pab(e('imae,mbej->ijab', t2, Wmbej) - e('ie,ma,mbej->ijab', t1, t1, B['ovvo'])))
    R2 = R2 + _cc_pij(e('ie,abej->ijab', t1, B['vvvo'])) - _cc_pab(e('ma,mbij->ijab', t1, B['ovoo']))
    return R1, R2


def _cc_l_residuals(f, g, o, t1, t2, l1, l2):
    """<Phi0|(1 + Lambda)[exp(-T) H exp(T), X_mu]|Phi0> for singles and doubles, exact for any T."""
    B = _cc_blocks(f, g, o)
    e = _cc_ein
    tau, taut = _cc_taus(t1, t2)
    Fme = B['fov'] + e('nf,mnef->me', t1, B['oovv'])
    Fae = B['fvv'] - 0.5 * e('me,ma->ae', B['fov'], t1) + e('mf,mafe->ae', t1, B['ovvv']) - 0.5 * e('mnaf,mnef->ae', taut, B['oovv'])
    Fmi = B['foo'] + 0.5 * e('ie,me->mi', t1, B['fov']) + e('ne,mnie->mi', t1, B['ooov']) + 0.5 * e('inef,mnef->mi', taut, B['oovv'])
    Hvv = Fae - 0.5 * e('ma,me->ae', t1, Fme)
    Hoo = Fmi + 0.5 * e('ie,me->mi', t1, Fme)
    x = e('je,mnie->mnij', t1, B['ooov'])
    Hoooo = B['oooo'] + x - x.transpose(0, 1, 3, 2) + 0.5 * e('ijef,mnef->mnij', tau, B['oovv'])
    x = e('mb,amef->abef', t1, B['vovv'])
    Hvvvv = B['vvvv'] - x + x.transpose(1, 0, 2, 3) + 0.5 * e('mnab,mnef->abef', tau, B['oovv'])
    Hooov = B['ooov'] + e('if,mnfe->mnie', t1, B['oovv'])
    Hvovv = B['vovv'] - e('na,nmef->amef', t1, B['oovv'])
    Hovvo = (B['ovvo'] + e('jf,mbef->mbej', t1, B['ovvv']) - e('nb,mnej->mbej', t1, B['oovo'])
             - e('jnfb,mnef->mbej', t2 + e('jf,nb->jnfb', t1, t1), B['oovv']))
    Z = B['ovvo'] - e('njbf,mnef->mbej', t2, B['oovv'])
    x = e('mnie,jnbe->mbij', B['ooov'], t2) + e('ie,mbej->mbij', t1, Z)
    Hovoo = (B['ovoo'] - e('me,ijbe->mbij', Fme, t2) - e('nb,mnij->mbij', t1, Hoooo)
             + 0.5 * e('mbef,ijef->mbij', B['ovvv'], tau) + x - x.transpose(0, 1, 3, 2))
    Z = B['ovvo'] - e('nibf,mnef->mbei', t2, B['oovv'])
    x = e('mbef,miaf->abei', B['ovvv'], t2) + e('ma,mbei->abei', t1, Z)
    Hvvvo = (B['vvvo'] - e('me,miab->abei', Fme, t2) + e('if,abef->abei', t1, Hvvvv)
             + 0.5 * e('mnei,mnab->abei', B['oovo'], tau) - x + x.transpose(1, 0, 2, 3))
    Gvv = -0.5 * e('mnef,mnaf->ae', t2, l2)
    Goo = 0.5 * e('mnef,inef->mi', t2, l2)
    G1 = (Fme + e('ie,ea->ia', l1, Hvv) - e('ma,im->ia', l1, Hoo) + e('me,ieam->ia', l1, Hovvo)
          + 0.5 * e('imef,efam->ia', l2, Hvvvo) - 0.5 * e('mnae,iemn->ia', l2, Hovoo)
          - e('ef,eifa->ia', Gvv, Hvovv) - e('mn,mina->ia', Goo, Hooov))
    G2 = B['oovv'] + _cc_pab(e('ijae,eb->ijab', l2, Hvv)) - _cc_pij(e('imab,jm->ijab', l2, Hoo))
    G2 = G2 + 0.5 * e('mnab,ijmn->ijab', l2, Hoooo) + 0.5 * e('ijef,efab->ijab', l2, Hvvvv)
    G2 = G2 + _cc_pij(e('ie,ejab->ijab', l1, Hvovv)) - _cc_pab(e('ma,ijmb->ijab', l1, Hooov))
    G2 = G2 + _cc_pij(_cc_pab(e('ia,jb->ijab', l1, Fme))) + _cc_pij(_cc_pab(e('imae,jebm->ijab', l2, Hovvo)))
    G2 = G2 + _cc_pab(e('ijae,be->ijab', B['oovv'], Gvv)) - _cc_pij(e('imab,mj->ijab', B['oovv'], Goo))
    return G1, G2


def _cc_check_hamiltonian(fock, eri_as, n_electrons):
    """Validate the spin-orbital Hamiltonian; return float arrays and (o, v)."""
    f = np.asarray(fock, dtype=float)
    g = np.asarray(eri_as, dtype=float)
    if f.ndim != 2 or f.shape[0] != f.shape[1] or f.shape[0] < 2:
        raise ValueError("fock must be a square matrix of size at least 2")
    n = f.shape[0]
    if not np.all(np.isfinite(f)) or not np.allclose(f, f.T, atol=1e-10):
        raise ValueError("fock must be finite and symmetric")
    if g.shape != (n, n, n, n) or not np.all(np.isfinite(g)):
        raise ValueError("eri_as must be a finite (n, n, n, n) array")
    if (not np.allclose(g, -g.transpose(1, 0, 2, 3), atol=1e-10) or not np.allclose(g, -g.transpose(0, 1, 3, 2), atol=1e-10)
            or not np.allclose(g, g.transpose(2, 3, 0, 1), atol=1e-10)):
        raise ValueError("eri_as must be antisymmetric in each index pair and symmetric under pair exchange")
    if isinstance(n_electrons, bool) or not isinstance(n_electrons, (int, np.integer)) or not 1 <= int(n_electrons) <= n - 1:
        raise ValueError("n_electrons must be an integer between 1 and n - 1")
    o = int(n_electrons)
    return f, g, o, n - o


def _cc_denominators(f, o):
    d = np.diag(f)
    D1 = d[:o, None] - d[None, o:]
    D2 = d[:o, None, None, None] + d[None, :o, None, None] - d[None, None, o:, None] - d[None, None, None, o:]
    return D1, D2


def _cc_pack(t1, t2, l1, l2):
    return np.concatenate([t1.ravel(), t2.ravel(), l1.ravel(), l2.ravel()])


def _cc_unpack(y, o, v):
    n1, n2 = o * v, o * o * v * v
    y = np.asarray(y)
    if y.ndim != 1 or y.size != 2 * (n1 + n2):
        raise ValueError("amplitude vector has the wrong length for this spin-orbital space")
    return (y[:n1].reshape(o, v), y[n1:n1 + n2].reshape(o, o, v, v),
            y[n1 + n2:2 * n1 + n2].reshape(o, v), y[2 * n1 + n2:].reshape(o, o, v, v))


def _oracle_ccsd_lambda_ground_state(fock: np.ndarray, eri_as: np.ndarray, n_electrons: int) -> np.ndarray:
    f, g, o, v = _cc_check_hamiltonian(fock, eri_as, n_electrons)
    D1, D2 = _cc_denominators(f, o)
    pairs = (~np.eye(o, dtype=bool))[:, :, None, None] & (~np.eye(v, dtype=bool))[None, None, :, :]
    if np.any(np.abs(D1) < 1e-8) or np.any(np.abs(D2[pairs]) < 1e-8):
        raise ValueError("a Jacobi denominator vanishes")
    D2s = np.where(np.abs(D2) < 1e-8, 1.0, D2)
    t1 = np.zeros((o, v))
    t2 = _cc_blocks(f, g, o)['oovv'] / D2s
    for _ in range(500):
        R1, R2 = _cc_t_residuals(f, g, o, t1, t2)
        if max(np.max(np.abs(R1)), np.max(np.abs(R2))) < 1e-11:
            break
        t1 = t1 + R1 / D1
        t2 = t2 + R2 / D2s
    else:
        raise ValueError("the cluster amplitude equations did not converge")
    l1, l2 = t1.copy(), t2.copy()
    for _ in range(500):
        G1, G2 = _cc_l_residuals(f, g, o, t1, t2, l1, l2)
        if max(np.max(np.abs(G1)), np.max(np.abs(G2))) < 1e-11:
            break
        l1 = l1 + G1 / D1
        l2 = l2 + G2 / D2s
    else:
        raise ValueError("the de-excitation amplitude equations did not converge")
    return _cc_pack(t1, t2, l1, l2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: four electrons in eight spin orbitals, non-canonical Fock matrix ---
        {
            "setup": """import numpy as np
import copy
rng = np.random.default_rng(401)
n, ne = 8, 4
A = rng.normal(scale=0.04, size=(n, n))
fock = 0.5 * (A + A.T) + np.diag([-1.2, -1.1, -0.9, -0.85, 0.45, 0.6, 0.8, 1.1])
W = rng.normal(scale=0.03, size=(n, n, n, n))
W = W + W.transpose(1, 0, 3, 2)
W = W + W.transpose(2, 3, 0, 1)
eri_as = W - W.transpose(0, 1, 3, 2)
""",
            "call": "ccsd_lambda_ground_state(copy.deepcopy(fock), copy.deepcopy(eri_as), ne)",
            "gold_call": "_oracle_ccsd_lambda_ground_state(copy.deepcopy(fock), copy.deepcopy(eri_as), ne)",
            "tol": 1e-8,
        },
        # --- Boundary: two electrons in four spin orbitals, the smallest space with double excitations ---
        {
            "setup": """import numpy as np
import copy
rng = np.random.default_rng(402)
n, ne = 4, 2
A = rng.normal(scale=0.05, size=(n, n))
fock = 0.5 * (A + A.T) + np.diag([-0.7, -0.6, 0.35, 0.5])
W = rng.normal(scale=0.06, size=(n, n, n, n))
W = W + W.transpose(1, 0, 3, 2)
W = W + W.transpose(2, 3, 0, 1)
eri_as = W - W.transpose(0, 1, 3, 2)
""",
            "call": "ccsd_lambda_ground_state(copy.deepcopy(fock), copy.deepcopy(eri_as), ne)",
            "gold_call": "_oracle_ccsd_lambda_ground_state(copy.deepcopy(fock), copy.deepcopy(eri_as), ne)",
            "tol": 1e-8,
        },
        # --- Edge: three electrons in nine spin orbitals, large occupied-virtual Fock block and small gap ---
        {
            "setup": """import numpy as np
import copy
rng = np.random.default_rng(403)
n, ne = 9, 3
A = rng.normal(scale=0.1, size=(n, n))
fock = 0.5 * (A + A.T) + np.diag([-0.8, -0.55, -0.35, 0.05, 0.3, 0.55, 0.7, 0.95, 1.3])
W = rng.normal(scale=0.02, size=(n, n, n, n))
W = W + W.transpose(1, 0, 3, 2)
W = W + W.transpose(2, 3, 0, 1)
eri_as = W - W.transpose(0, 1, 3, 2)
""",
            "call": "ccsd_lambda_ground_state(copy.deepcopy(fock), copy.deepcopy(eri_as), ne)",
            "gold_call": "_oracle_ccsd_lambda_ground_state(copy.deepcopy(fock), copy.deepcopy(eri_as), ne)",
            "tol": 1e-8,
        },
        # --- Invalid: integrals that are not antisymmetric in the first index pair ---
        {
            "setup": """import numpy as np
import copy
n, ne = 4, 2
fock = np.diag([-0.7, -0.6, 0.35, 0.5])
eri_as = np.zeros((n, n, n, n))
eri_as[0, 1, 2, 3] = 0.1
def run_model():
    try:
        ccsd_lambda_ground_state(copy.deepcopy(fock), copy.deepcopy(eri_as), ne)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_ccsd_lambda_ground_state(copy.deepcopy(fock), copy.deepcopy(eri_as), ne)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
