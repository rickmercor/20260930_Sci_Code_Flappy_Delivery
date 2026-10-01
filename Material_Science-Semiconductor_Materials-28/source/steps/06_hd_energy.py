"""
Evaluates the source's energy functional of the hierarchy, split into electronic, phonon and electron-phonon parts.

An energy functional that depends only on the propagated quantities makes energy conservation a test of the closure itself: the truncated hierarchy must conserve it once the drive is over.

Returns
-------
A float64 array of shape (4,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hd_energy(state: "np.ndarray", lam: float, gam: float) -> "np.ndarray":
    r"""state: array-like of shape $(17,)$, the packed state. lam, gam: as before.

    Returns a numpy float64 array of shape $(4,)$: the source's energy functional of the hierarchy,
    the expectation value of the time-independent Hamiltonian written in the propagated quantities:
    the electronic part $2\,\mathrm{Tr}(h_0\rho)$ with $h_0 = -t_{hop}\sigma_x$, the phonon part
    $\omega(|B|^2 + \delta\rho_{qq})$, the electron-phonon part $4\,\mathrm{Re}\sum_i G_{ii}(B\rho_{ii} +
    \delta\rho^q_{ii})$, and their sum, the factors of two again counting the two spins.

    Raises:
        ValueError: on a state of the wrong shape or with non-finite entries, or invalid lam or gam.
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

def _oracle_hd_energy(state: "np.ndarray", lam: float, gam: float) -> "np.ndarray":
    z = np.asarray(state, dtype=float)
    if z.shape != (17,) or not np.all(np.isfinite(z)): raise ValueError("state must be 17 finite numbers")
    lam = _nonneg(lam, "lam"); gam = _pos(gam, "gam")
    return np.array(_energy(z, lam, gam), dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nz=np.array([0.52,0.48,0.47,0.03,0.04,0.13,-0.01,-0.065,-0.05,0.05,0.065,0.002,0.001,-0.001,-0.002,-0.2,0.05])\n', 'call': 'hd_energy(z,0.5,0.5)', 'gold_call': '_oracle_hd_energy(z,0.5,0.5)'},
        {'setup': 'import numpy as np\nz=np.array([0.3,0.7,-0.2,0.35,0.08,-0.05,0.02,0.01,-0.03,0.03,-0.01,0.004,-0.002,0.002,-0.004,0.1,-0.3])\n', 'call': 'hd_energy(z,0.8,0.6)', 'gold_call': '_oracle_hd_energy(z,0.8,0.6)'},
        {'setup': 'import numpy as np\nz=np.array([0.5,0.5,0.48674558970257115,0.0,0.04215622934600437,0.13102229592343673,0.0,-0.0655111479617183,-0.04960665148469338,0.04960665148469338,0.0655111479617183,0.0,0.0,0.0,0.0,0.0,0.0])\n', 'call': 'hd_energy(z,0.5,0.5)', 'gold_call': '_oracle_hd_energy(z,0.5,0.5)'},
        {'setup': 'import numpy as np\n# boundary: the uncorrelated bonding state, whose energy is minus two hopping energies\nz=np.array([0.5,0.5,0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])\n', 'call': 'hd_energy(z,0.5,0.5)', 'gold_call': '_oracle_hd_energy(z,0.5,0.5)'},
        {'setup': 'import numpy as np\n# invalid input: a state of the wrong length must raise ValueError\ndef _probe(func, *args):\n    try:\n        func(*args)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_probe(hd_energy, np.zeros(16), 0.5, 0.5)', 'gold_call': '_probe(_oracle_hd_energy, np.zeros(16), 0.5, 0.5)'},
    ]
