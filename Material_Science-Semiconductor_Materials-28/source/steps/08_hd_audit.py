"""
Runs the whole chain from the exact ground state through the exact, coherent and full-hierarchy dynamics and reports the observables side by side.

Reporting the exact, the coherent and the correlated propagation of the same driven dimer next to each other shows what the electron-phonon correlation equation adds and what it costs in accuracy.

Returns
-------
A float64 array of shape (10,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hd_audit(lam: float, gam: float, nph: int, v: float, tf: float) -> "np.ndarray":
    r"""Parameters as before.

    The orchestrator. It must call the earlier functions rather than reimplementing them.
    Returns a numpy float64 array of shape $(10,)$: the exact ground-state energy and gap; the
    site-1 occupation of one spin at tf and the energy after the drive from the exact dynamics; the
    site-1 occupation at tf from the coherent level; the site-1 occupation, the energy after the
    drive, the excess phonon number and the real part of the coherent amplitude at tf from the full
    hierarchy; and the site-1 occupation at tf from the full hierarchy minus that of the exact
    dynamics.

    Raises:
        ValueError: whenever any of the functions it calls would raise; when the energy functional of
            the full hierarchy evaluated in its initial state differs from the exact ground-state energy
            by more than 1e-9; when the trace of the propagated density matrix of either level departs
            from one by more than 1e-9; and when the energy of the full hierarchy at tf differs from its
            value at time 8, after the drive has decayed, by more than 1e-8 (tf at least 8).
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

def _oracle_hd_audit(lam: float, gam: float, nph: int, v: float, tf: float) -> "np.ndarray":
    lam = _nonneg(lam, "lam"); gam = _pos(gam, "gam"); nph = _posint(nph, "nph"); v = _nonneg(v, "v"); tf = _nonneg(tf, "tf")
    if tf < 8.0: raise ValueError("tf must be at least 8")
    HV = _oracle_hd_hamiltonian(lam, gam, nph)
    gs = _oracle_hd_ground_state(lam, gam, nph); E0, gap = float(gs[0]), float(gs[1])
    D = 4*(nph+1)
    if HV.shape != (2, D, D) or not np.all(np.isfinite(HV)): raise ValueError("invalid Hamiltonian or drive operator")
    if not np.allclose(HV[0], HV[0].T, atol=1e-12, rtol=0.0) or not np.allclose(HV[1], HV[1].T, atol=1e-12, rtol=0.0):
        raise ValueError("Hamiltonian and drive operator must be Hermitian")
    if abs(float(eigh(HV[0], subset_by_index=[0, 0], eigvals_only=True)[0]) - E0) > 1e-9*max(1.0, abs(E0)):
        raise ValueError("Hamiltonian does not reproduce the ground-state energy")
    ex = _oracle_hd_exact_dynamics(lam, gam, nph, v, tf)
    z0 = _oracle_hd_initial_state(lam, gam, nph, 1)
    if abs(_oracle_hd_energy(z0, lam, gam)[3] - E0) > 1e-9*max(1.0, abs(E0)): raise ValueError("the energy functional does not reproduce the ground-state energy")
    r0 = _oracle_hd_eom_rhs(0.0, z0, lam, gam, v, 1)
    if abs(r0[0] + r0[1]) > 1e-12: raise ValueError("the hierarchy does not conserve the electron number")
    zc = _oracle_hd_eom_dynamics(lam, gam, nph, v, tf, 0); zf = _oracle_hd_eom_dynamics(lam, gam, nph, v, tf, 1)
    for z in (zc, zf):
        if abs(z[0] + z[1] - 1.0) > 1e-9: raise ValueError("the trace of the density matrix is not conserved")
    z8 = _oracle_hd_eom_dynamics(lam, gam, nph, v, 8.0, 1)
    Ef = _oracle_hd_energy(zf, lam, gam)[3]; E8 = _oracle_hd_energy(z8, lam, gam)[3]
    if abs(Ef - E8) > 1e-8*max(1.0, abs(E8)): raise ValueError("the hierarchy does not conserve the energy after the drive")
    return np.array([E0, gap, ex[0], ex[6], zc[0], zf[0], Ef, zf[4], zf[15], zf[0] - ex[0]], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\n', 'call': 'hd_audit(0.5,0.5,30,1.0,30.0)', 'gold_call': '_oracle_hd_audit(0.5,0.5,30,1.0,30.0)'},
        {'setup': 'import numpy as np\n', 'call': 'hd_audit(0.3,0.8,30,1.0,12.0)', 'gold_call': '_oracle_hd_audit(0.3,0.8,30,1.0,12.0)'},
        {'setup': 'import numpy as np\n', 'call': 'hd_audit(1.0,0.5,40,0.5,20.0)', 'gold_call': '_oracle_hd_audit(1.0,0.5,40,0.5,20.0)'},
        {'setup': 'import numpy as np\n# boundary: no drive, where every level keeps the occupation at one half\n', 'call': 'hd_audit(0.5,0.5,30,0.0,10.0)', 'gold_call': '_oracle_hd_audit(0.5,0.5,30,0.0,10.0)'},
        {'setup': 'import numpy as np\n# invalid input: a final time before the drive has decayed must raise ValueError\ndef _probe(func, *args):\n    try:\n        func(*args)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_probe(hd_audit, 0.5, 0.5, 30, 1.0, 5.0)', 'gold_call': '_probe(_oracle_hd_audit, 0.5, 0.5, 30, 1.0, 5.0)'},
    ]
