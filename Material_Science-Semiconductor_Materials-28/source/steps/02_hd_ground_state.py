"""
Computes the exact ground state of the dimer and the reduced quantities of the source's hierarchy evaluated in it.

The exact ground state fixes both the reference energy and the correlated initial state from which the approximate hierarchy is propagated.

Returns
-------
A float64 array of shape (7,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hd_ground_state(lam: float, gam: float, nph: int) -> "np.ndarray":
    r"""Parameters as in the Hamiltonian step.

    Returns a numpy float64 array of shape $(7,)$: the exact ground-state energy, the gap to the first
    excited state, and the reduced quantities of the ground state that the source's hierarchy
    propagates, for one spin: the off-diagonal element $\rho_{12} = \langle \hat c^\dagger_2 \hat c_1\rangle$
    of the one-body reduced density matrix, the phonon number $\langle \hat b^\dagger \hat b\rangle$, the
    anomalous phonon correlator $\langle \hat b \hat b\rangle$, and the electron-phonon correlators
    $\rho^q_{11} = \langle \hat c^\dagger_1 \hat c_1 \hat b\rangle$ and $\rho^q_{12} = \langle \hat c^\dagger_2
    \hat c_1 \hat b\rangle$, all real in the ground state.

    Raises:
        ValueError: on invalid lam, gam or nph.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh

def _pos(x, name):
    v = float(x)
    if not np.isfinite(v) or v <= 0.0: raise ValueError("bad " + name)
    return v
def _nonneg(x, name):
    v = float(x)
    if not np.isfinite(v) or v < 0.0: raise ValueError("bad " + name)
    return v
def _posint(x, name):
    v = float(x)
    if not np.isfinite(v) or v <= 0 or abs(v - round(v)) > 0: raise ValueError("bad " + name)
    return int(round(v))
def _flag(x, name):
    v = float(x)
    if v not in (0.0, 1.0): raise ValueError("bad " + name)
    return int(v)
def _params(lam, gam):
    return gam, np.sqrt(lam*gam/2.0)
def _pulse(t, v):
    return v*np.sin(np.pi*t/4.0)*np.exp(-0.5*t*t)
def _ham(lam, gam, nph):
    om, g = _params(lam, gam); N = nph+1
    b = np.diag(np.sqrt(np.arange(1, N, dtype=float)), 1); Iph = np.eye(N)
    sx = np.array([[0., 1.], [1., 0.]]); I2 = np.eye(2)
    hop = -(np.kron(sx, I2) + np.kron(I2, sx))
    n1 = np.kron(np.diag([1., 0.]), I2) + np.kron(I2, np.diag([1., 0.])); dz = 2.0*n1 - 2.0*np.eye(4)
    H = np.kron(hop, Iph) + om*np.kron(np.eye(4), b.T@b) + (g/np.sqrt(2.0))*np.kron(dz, b + b.T)
    return H, np.kron(dz, Iph), b
def _reduced(psi, nph):
    N = nph+1; P = psi.reshape(2, 2, N); b = np.diag(np.sqrt(np.arange(1, N, dtype=float)), 1)
    rho = np.einsum('isn,jsn->ij', P, P.conj())
    bP = np.einsum('nm,ijm->ijn', b, P)
    B = complex(np.sum(P.conj()*bP)); nb = float(np.sum(np.abs(P)**2*np.arange(N)[None, None, :]).real)
    bb = complex(np.sum(P.conj()*np.einsum('nm,ijm->ijn', b@b, P)))
    rhoq = np.einsum('isn,jsn->ij', bP, P.conj())
    return rho, B, nb, bb, rhoq
def _gs(lam, gam, nph):
    H, Vz, b = _ham(lam, gam, nph); w, U = eigh(H)
    return float(w[0]), float(w[1]-w[0]), U[:, 0].astype(complex)
def _unpack(z):
    z = np.asarray(z, dtype=float)
    rho = np.array([[z[0], z[2]+1j*z[3]], [z[2]-1j*z[3], z[1]]]); dn = z[4]; dbar = z[5]+1j*z[6]
    dq = (z[7:11] + 1j*z[11:15]).reshape(2, 2); B = z[15]+1j*z[16]
    return rho, dn, dbar, dq, B
def _pack(rho, dn, dbar, dq, B):
    return np.array([rho[0,0].real, rho[1,1].real, rho[0,1].real, rho[0,1].imag, float(np.real(dn)), np.real(dbar), np.imag(dbar),
                     *np.real(dq).ravel(), *np.imag(dq).ravel(), np.real(B), np.imag(B)], dtype=float)
def _rhs(t, z, lam, gam, v, full):
    om, g = _params(lam, gam); G = (g/np.sqrt(2.0))*np.diag([1.0, -1.0]); h0 = -np.array([[0., 1.], [1., 0.]]); sz = np.diag([1.0, -1.0])
    rho, dn, dbar, dq, B = _unpack(z)
    h = h0 + G*(B + np.conj(B)) + _pulse(t, v)*sz
    if full:
        X = -1j*(G@dq + (dq@G).conj().T)
        drho = -1j*(h@rho - rho@h) + X + X.conj().T
        ddn = 2.0*(-1j)*np.sum(G*(-dq.T + dq.T.conj()))
        ddbar = -2j*om*dbar - 2.0*2j*np.sum(G*dq)
        Pm = np.eye(2) - rho
        ddq = -1j*(h@dq - dq@h + om*dq) - 1j*(1.0+dn)*(Pm@G@rho) + 1j*dn*(rho@G@Pm) - 1j*dbar*(Pm@G@rho) + 1j*dbar*(rho@G@Pm)
    else:
        drho = -1j*(h@rho - rho@h); ddn = 0.0; ddbar = 0.0; ddq = np.zeros((2, 2), complex)
    dB = -1j*(om*B + 2.0*np.sum(np.diag(G)*np.diag(rho)))
    return _pack(drho, ddn, ddbar, ddq, dB)
def _energy(z, lam, gam):
    om, g = _params(lam, gam); G = (g/np.sqrt(2.0))*np.diag([1.0, -1.0]); h0 = -np.array([[0., 1.], [1., 0.]])
    rho, dn, dbar, dq, B = _unpack(z)
    Ee = 2.0*float(np.trace(h0@rho).real); Eph = float(om*(abs(B)**2 + dn))
    Eint = 4.0*float(np.real(np.sum(np.diag(G)*(B*np.diag(rho) + np.diag(dq)))))
    return Ee, Eph, Eint, Ee+Eph+Eint
def _init(lam, gam, nph, full):
    E0, gap, psi = _gs(lam, gam, nph); rho, B, nb, bb, rhoq = _reduced(psi, nph)
    dn = nb - abs(B)**2; dbar = bb - B*B; dq = rhoq - rho*B
    if not full: dn = 0.0; dbar = 0.0; dq = np.zeros((2, 2), complex)
    return _pack(rho, dn, dbar, dq, B)
def _exact_run(lam, gam, nph, v, tf):
    H, Vz, b = _ham(lam, gam, nph); E0, gap, psi0 = _gs(lam, gam, nph)
    if tf == 0.0: return psi0, float((psi0.conj() @ (H @ psi0)).real)
    sol = solve_ivp(lambda t, y: -1j*((H + _pulse(t, v)*Vz) @ y), (0.0, tf), psi0, method='DOP853', rtol=1e-12, atol=1e-14)
    psi = sol.y[:, -1]; return psi, float((psi.conj() @ (H @ psi)).real)
def _eom_run(lam, gam, nph, v, tf, full):
    z0 = _init(lam, gam, nph, full)
    if tf == 0.0: return z0
    sol = solve_ivp(lambda t, y: _rhs(t, y, lam, gam, v, full), (0.0, tf), z0, method='DOP853', rtol=1e-12, atol=1e-14)
    return sol.y[:, -1]

def _oracle_hd_ground_state(lam: float, gam: float, nph: int) -> "np.ndarray":
    if not np.isfinite(float(gam)) or float(gam) <= 0.0: raise ValueError("gam must be positive")
    lam = _nonneg(lam, "lam"); gam = _pos(gam, "gam"); nph = _posint(nph, "nph")
    E0, gap, psi = _gs(lam, gam, nph); rho, B, nb, bb, rhoq = _reduced(psi, nph)
    return np.array([E0, gap, rho[0,1].real, nb, bb.real, rhoq[0,0].real, rhoq[0,1].real], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\n', 'call': 'hd_ground_state(0.5,0.5,30)', 'gold_call': '_oracle_hd_ground_state(0.5,0.5,30)'},
        {'setup': 'import numpy as np\n', 'call': 'hd_ground_state(0.3,0.8,30)', 'gold_call': '_oracle_hd_ground_state(0.3,0.8,30)'},
        {'setup': 'import numpy as np\n', 'call': 'hd_ground_state(1.0,0.5,40)', 'gold_call': '_oracle_hd_ground_state(1.0,0.5,40)'},
        {'setup': 'import numpy as np\n# boundary: no coupling, where the ground state is the doubly occupied bonding orbital with an empty phonon\n', 'call': 'hd_ground_state(0.0,0.5,10)', 'gold_call': '_oracle_hd_ground_state(0.0,0.5,10)'},
        {'setup': 'import numpy as np\n# invalid input: a non-integral phonon cutoff must raise ValueError\ndef _probe(func, *args):\n    try:\n        func(*args)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_probe(hd_ground_state, 0.5, 0.5, 30.5)', 'gold_call': '_probe(_oracle_hd_ground_state, 0.5, 0.5, 30.5)'},
    ]
