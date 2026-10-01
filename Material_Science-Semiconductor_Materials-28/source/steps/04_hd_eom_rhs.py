"""
Evaluates one time derivative of the source's second-order equal-time electron-phonon hierarchy for the driven spin-symmetric Holstein dimer, or of its coherent restriction.

The hierarchy propagates only equal-time quantities: the one-body density matrix, the excess and anomalous phonon correlators, the electron-phonon correlation and the coherent phonon amplitude; how the correlation feeds back into the density matrix and how the phonon fluctuations enter its own equation are the source's closure.

Returns
-------
A float64 array of shape (17,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hd_eom_rhs(t: float, state: "np.ndarray", lam: float, gam: float, v: float, full: int) -> "np.ndarray":
    r"""Evaluate the source's five coupled EOMs for the one-mode spin-symmetric Holstein dimer.

    Parameters
    ----------
    t : float
        Finite time in units of $t_{hop}^{-1}$.
    state : "np.ndarray"
        Shape (17,), finite. The packing is
        $\rho_{11}, \rho_{22}, \mathrm{Re}\,\rho_{12}, \mathrm{Im}\,\rho_{12},
        \delta\rho_{qq}, \mathrm{Re}\,\delta\bar\rho, \mathrm{Im}\,\delta\bar\rho$,
        then the four real parts and four imaginary parts of $\delta\rho^q_{ij}$ in row-major
        $(i,j)$ order, followed by $\mathrm{Re}\,B, \mathrm{Im}\,B$.
    lam : float
        Dimensionless electron-phonon coupling $\lambda \ge 0$.
    gam : float
        Positive phonon-frequency ratio $\gamma$.
    v : float
        Non-negative drive amplitude.
    full : int
        1 for the full hierarchy of source Eqs. (6a)-(6e); 0 for the coherent restriction
        containing only the electronic 1-RDM and coherent phonon amplitude.

    Notes
    -----
    Use $\omega=\gamma$, $g=\sqrt{\lambda\gamma/2}$ and
    $G=(g/\sqrt{2})\,\mathrm{diag}(1,-1)$. Apply the source's Eq. (7) for the effective
    one-body Hamiltonian with the stated oscillatory Gaussian drive. Reduce Eqs. (6a)-(6e)
    exactly to this single-mode, two-identical-spin dimer. Preserve the source's index order,
    complex conjugations, spontaneous/stimulated terms and spin-degeneracy factors. When
    ``full == 0``, the connected quantities $\delta\rho_{qq}$, $\delta\bar\rho$ and
    $\delta\rho^q$ are held fixed at zero rather than evolved.

    Returns
    -------
    "np.ndarray"
        Float64 array of shape (17,), the derivative in exactly the same packing as ``state``.

    Raises
    ------
    ValueError
        If ``t`` is non-finite; ``state`` does not contain exactly 17 finite values; ``lam`` is
        negative; ``gam`` is non-positive; ``v`` is negative; or ``full`` is not 0 or 1.
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

def _oracle_hd_eom_rhs(t: float, state: "np.ndarray", lam: float, gam: float, v: float, full: int) -> "np.ndarray":
    t = float(t)
    if not np.isfinite(t): raise ValueError("bad t")
    z = np.asarray(state, dtype=float)
    if z.shape != (17,) or not np.all(np.isfinite(z)): raise ValueError("state must be 17 finite numbers")
    lam = _nonneg(lam, "lam"); gam = _pos(gam, "gam"); v = _nonneg(v, "v"); full = _flag(full, "full")
    return _rhs(t, z, lam, gam, v, full).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\nz=np.array([0.52,0.48,0.47,0.03,0.04,0.13,-0.01,-0.065,-0.05,0.05,0.065,0.002,0.001,-0.001,-0.002,-0.2,0.05])\n',
            'call': 'hd_eom_rhs(1.3,z,0.5,0.5,1.0,1)',
            'gold_call': '_oracle_hd_eom_rhs(1.3,z,0.5,0.5,1.0,1)'
        },
        {
            'setup': 'import numpy as np\nz=np.array([0.52,0.48,0.47,0.03,0.04,0.13,-0.01,-0.065,-0.05,0.05,0.065,0.002,0.001,-0.001,-0.002,-0.2,0.05])\n',
            'call': 'hd_eom_rhs(1.3,z,0.5,0.5,1.0,0)',
            'gold_call': '_oracle_hd_eom_rhs(1.3,z,0.5,0.5,1.0,0)'
        },
        {
            'setup': 'import numpy as np\nz=np.array([0.3,0.7,-0.2,0.35,0.08,-0.05,0.02,0.01,-0.03,0.03,-0.01,0.004,-0.002,0.002,-0.004,0.1,-0.3])\n',
            'call': 'hd_eom_rhs(0.4,z,0.8,0.6,0.7,1)',
            'gold_call': '_oracle_hd_eom_rhs(0.4,z,0.8,0.6,0.7,1)'
        },
        {
            'setup': 'import numpy as np\n# adversarial complex state: exercises index ordering, conjugations, anomalous terms and spin factors\nz=np.array([0.61,0.39,-0.27,0.31,0.11,-0.08,0.17,0.04,-0.07,0.09,-0.02,0.03,-0.05,0.06,0.08,-0.14,0.22])\n',
            'call': 'hd_eom_rhs(2.35,z,0.9,0.37,1.25,1)',
            'gold_call': '_oracle_hd_eom_rhs(2.35,z,0.9,0.37,1.25,1)'
        },
        {
            'setup': "import numpy as np\n# boundary: the exact ground state's reduced quantities at zero field, where only the correlation equation has a non-zero rate\nz=np.array([0.5,0.5,0.48674558970257115,0.0,0.04215622934600437,0.13102229592343673,0.0,-0.0655111479617183,-0.04960665148469338,0.04960665148469338,0.0655111479617183,0.0,0.0,0.0,0.0,0.0,0.0])\n",
            'call': 'hd_eom_rhs(0.0,z,0.5,0.5,0.0,1)',
            'gold_call': '_oracle_hd_eom_rhs(0.0,z,0.5,0.5,0.0,1)'
        },
        {
            'setup': 'import numpy as np\nz=np.zeros(17)\n# invalid input: a flag other than 0 or 1 must raise ValueError\ndef _probe(func, *args):\n    try:\n        func(*args)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            'call': '_probe(hd_eom_rhs, 0.0, z, 0.5, 0.5, 1.0, 2)',
            'gold_call': '_probe(_oracle_hd_eom_rhs, 0.0, z, 0.5, 0.5, 1.0, 2)'
        },
    ]
