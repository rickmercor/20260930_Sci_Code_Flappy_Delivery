#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np

def hex8_element_matrices(node_coords: 'np.ndarray', E: float, nu: float,
                                  rho: float) -> 'np.ndarray':
    import numpy as np
    CORNERS = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.shape != (8, 3):
        raise ValueError("node_coords must have shape (8, 3)")
    if not E > 0:
        raise ValueError("E must be strictly positive")
    if not rho > 0:
        raise ValueError("rho must be strictly positive")
    if not -1.0 < nu < 0.5:
        raise ValueError("nu must lie strictly between -1.0 and 0.5")
    sgn = np.array([[-1.0 if c == 0 else 1.0 for c in cc] for cc in CORNERS])
    lam = E*nu/((1.0+nu)*(1.0-2.0*nu))
    mu = E/(2.0*(1.0+nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0*mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    Ke = np.zeros((24, 24))
    Me = np.zeros((24, 24))
    g = 1.0/np.sqrt(3.0)
    for xi in (-g, g):
        for eta in (-g, g):
            for ze in (-g, g):
                N = 0.125*(1+sgn[:, 0]*xi)*(1+sgn[:, 1]*eta)*(1+sgn[:, 2]*ze)
                dN = np.empty((8, 3))
                dN[:, 0] = 0.125*sgn[:, 0]*(1+sgn[:, 1]*eta)*(1+sgn[:, 2]*ze)
                dN[:, 1] = 0.125*(1+sgn[:, 0]*xi)*sgn[:, 1]*(1+sgn[:, 2]*ze)
                dN[:, 2] = 0.125*(1+sgn[:, 0]*xi)*(1+sgn[:, 1]*eta)*sgn[:, 2]
                J = dN.T @ node_coords
                detJ = np.linalg.det(J)
                if not detJ > 0:
                    raise ValueError("non-positive Jacobian determinant at an integration point")
                dNx = np.linalg.solve(J, dN.T).T
                B = np.zeros((6, 24))
                Nm = np.zeros((3, 24))
                for a in range(8):
                    B[0, 3*a+0] = dNx[a, 0]
                    B[1, 3*a+1] = dNx[a, 1]
                    B[2, 3*a+2] = dNx[a, 2]
                    B[3, 3*a+0] = dNx[a, 1]; B[3, 3*a+1] = dNx[a, 0]
                    B[4, 3*a+1] = dNx[a, 2]; B[4, 3*a+2] = dNx[a, 1]
                    B[5, 3*a+0] = dNx[a, 2]; B[5, 3*a+2] = dNx[a, 0]
                    Nm[0, 3*a+0] = N[a]; Nm[1, 3*a+1] = N[a]; Nm[2, 3*a+2] = N[a]
                Ke += (B.T @ D @ B)*detJ
                Me += rho*(Nm.T @ Nm)*detJ
    return np.stack([Ke, Me])

import numpy as np

def assemble_partitioned_system(nx: int, ny: int, nz: int, Lx: float,
                                        Ly: float, Lz: float, E: float, nu: float,
                                        rho: float) -> 'np.ndarray':
    import numpy as np
    CORNERS = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
    if nx % 3 != 0:
        raise ValueError("nx must be divisible by three")
    if nx < 6 or ny < 1 or nz < 1:
        raise ValueError("element counts are too small; each interior needs at least one node layer")
    if min(Lx, Ly, Lz) <= 0:
        raise ValueError("edge lengths must be strictly positive")

    def _nid(i, j, k):
        return i*((ny+1)*(nz+1)) + j*(nz+1) + k

    nn = (nx+1)*(ny+1)*(nz+1)
    X = np.zeros((nn, 3))
    for i in range(nx+1):
        for j in range(ny+1):
            for k in range(nz+1):
                X[_nid(i, j, k)] = [i*Lx/nx, j*Ly/ny, k*Lz/nz]
    K = np.zeros((3*nn, 3*nn))
    M = np.zeros((3*nn, 3*nn))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                conn = [_nid(i+c[0], j+c[1], k+c[2]) for c in CORNERS]
                EM = hex8_element_matrices(X[conn], E, nu, rho)
                d = np.array([3*n+q for n in conn for q in range(3)])
                K[np.ix_(d, d)] += EM[0]
                M[np.ix_(d, d)] += EM[1]
    cl = [_nid(0, j, k) for j in range(ny+1) for k in range(nz+1)]
    cdof = np.array([3*n+q for n in cl for q in range(3)])
    free = np.setdiff1d(np.arange(3*nn), cdof)
    pos = -np.ones(3*nn, int)
    pos[free] = np.arange(free.size)

    def _blk(ii):
        return np.array([pos[3*_nid(i, j, k)+d] for i in ii
                         for j in range(ny+1) for k in range(nz+1) for d in range(3)])

    c = nx//3
    order = np.concatenate([_blk(range(1, c)), _blk(range(c+1, 2*c)),
                            _blk(range(2*c+1, nx+1)), _blk([c, 2*c])])
    Kf = K[np.ix_(free, free)]
    Mf = M[np.ix_(free, free)]
    return np.stack([Kf[np.ix_(order, order)], Mf[np.ix_(order, order)]])

import numpy as np

def fixed_interface_modes(Kp: 'np.ndarray', Mp: 'np.ndarray', counts: 'np.ndarray',
                                  k: int, nd: int) -> 'np.ndarray':
    import numpy as np
    from scipy.linalg import eigh
    counts = np.asarray(counts, dtype=int)
    if counts.size != 4:
        raise ValueError("counts must have four entries")
    if int(counts.sum()) != Kp.shape[0]:
        raise ValueError("counts do not sum to the system size")
    if k not in (0, 1, 2):
        raise ValueError("k must be 0, 1 or 2")
    o = np.concatenate([[0], np.cumsum(counts)])
    if not 1 <= nd <= counts[k]:
        raise ValueError("nd out of range")
    ii = slice(o[k], o[k+1])
    w, V = eigh(Kp[ii, ii], Mp[ii, ii], subset_by_index=[0, nd-1])
    for c in range(nd):
        if V[np.argmax(np.abs(V[:, c])), c] < 0:
            V[:, c] *= -1.0
    out = np.zeros((Kp.shape[0], nd))
    out[ii, :] = V
    return out

import numpy as np

def constraint_modes(Kp: 'np.ndarray', counts: 'np.ndarray', k: int) -> 'np.ndarray':
    import numpy as np
    counts = np.asarray(counts, dtype=int)
    if counts.size != 4:
        raise ValueError("counts must have four entries")
    if int(counts.sum()) != Kp.shape[0]:
        raise ValueError("counts do not sum to the system size")
    if k not in (0, 1, 2):
        raise ValueError("k must be 0, 1 or 2")
    o = np.concatenate([[0], np.cumsum(counts)])
    ii = slice(o[k], o[k+1])
    bb = slice(o[3], o[4])
    out = np.zeros((Kp.shape[0], counts[3]))
    out[ii, :] = -np.linalg.solve(Kp[ii, ii], Kp[ii, bb])
    return out

import numpy as np

def residual_modes(Kp: 'np.ndarray', Mp: 'np.ndarray', counts: 'np.ndarray', k: int,
                           Phi: 'np.ndarray') -> 'np.ndarray':
    import numpy as np
    counts = np.asarray(counts, dtype=int)
    if counts.size != 4:
        raise ValueError("counts must have four entries")
    if int(counts.sum()) != Kp.shape[0]:
        raise ValueError("counts do not sum to the system size")
    if k not in (0, 1, 2):
        raise ValueError("k must be 0, 1 or 2")
    Phi = np.asarray(Phi, dtype=float)
    if Phi.ndim != 2 or Phi.shape[0] != Kp.shape[0]:
        raise ValueError("Phi must have one row per equation of the system")
    o = np.concatenate([[0], np.cumsum(counts)])
    ii = slice(o[k], o[k+1])
    bb = slice(o[3], o[4])
    Kss = Kp[ii, ii]; Mss = Mp[ii, ii]; Ksb = Kp[ii, bb]; Msb = Mp[ii, bb]
    P = Phi[ii, :]
    lam = np.diag(P.T @ Kss @ P)
    if np.any(lam <= 0):
        raise ValueError("non-positive retained eigenvalue recovered from Phi")
    Psi = -np.linalg.solve(Kss, Ksb)
    Mc = Mss @ Psi + Msb
    F1 = np.linalg.inv(Kss) - P @ np.diag(1.0/lam) @ P.T
    R = F1 @ Mc
    nrm = np.linalg.norm(R, axis=0)
    good = nrm > 0
    R[:, good] = R[:, good]/nrm[good]
    out = np.zeros((Kp.shape[0], counts[3]))
    out[ii, :] = R
    return out

import numpy as np

def reduced_eigenvalues(Kp: 'np.ndarray', Mp: 'np.ndarray', T: 'np.ndarray',
                                nCB: int) -> 'np.ndarray':
    import numpy as np
    from scipy.linalg import eigh
    T = np.asarray(T, dtype=float)
    if T.ndim != 2 or T.shape[0] != Kp.shape[0]:
        raise ValueError("T must have one row per equation of the system")
    if not 1 <= nCB <= T.shape[1]:
        raise ValueError("nCB out of range")
    Ka = T.T @ Kp @ T
    Ma = T.T @ Mp @ T
    w, V = eigh(Ka, Ma, subset_by_index=[0, nCB-1])
    Ts = V[:, :nCB]
    lb, _ = eigh(Ts.T @ Ka @ Ts, Ts.T @ Ma @ Ts)
    return np.sort(lb[lb > 0])

import numpy as np

def base_contribution(Kp: 'np.ndarray', Mp: 'np.ndarray', T: 'np.ndarray', nCB: int,
                              mode_index: int) -> 'np.ndarray':
    import numpy as np
    from scipy.linalg import eigh
    T = np.asarray(T, dtype=float)
    if T.ndim != 2 or T.shape[0] != Kp.shape[0]:
        raise ValueError("T must have one row per equation of the system")
    if not 1 <= nCB <= T.shape[1]:
        raise ValueError("nCB out of range")
    Ka = T.T @ Kp @ T
    Ma = T.T @ Mp @ T
    w, V = eigh(Ka, Ma, subset_by_index=[0, nCB-1])
    Ts = V[:, :nCB]
    lb, Vb = eigh(Ts.T @ Ka @ Ts, Ts.T @ Ma @ Ts)
    keep = lb > 0
    lb = lb[keep]; Vb = Vb[:, keep]
    if not 1 <= mode_index <= lb.size:
        raise ValueError("mode_index out of range")
    q = Ts @ Vb[:, mode_index-1]
    u0 = T[:, :nCB] @ q[:nCB]
    if u0[np.argmax(np.abs(u0))] < 0:
        u0 = -u0
    return u0

import numpy as np

def relative_deviation(Kp: 'np.ndarray', Mp: 'np.ndarray', u0: 'np.ndarray',
                               lam: float) -> float:
    import numpy as np
    u0 = np.asarray(u0, dtype=float)
    if u0.ndim != 1:
        raise ValueError("u0 must be one dimensional")
    if u0.size != Kp.shape[0]:
        raise ValueError("u0 has the wrong length")
    if not lam > 0:
        raise ValueError("lam must be strictly positive")
    d = float(u0 @ Mp @ u0)
    if not d > 0:
        raise ValueError("base contribution has non-positive mass")
    return float((u0 @ Kp @ u0 - lam*d)/(lam*d))

import numpy as np

def reduction_report(nx: int, ny: int, nz: int, Lx: float, Ly: float,
                             Lz: float, E: float, nu: float, rho: float, nd: tuple,
                             mode_index: int) -> 'np.ndarray':
    import numpy as np
    S = assemble_partitioned_system(nx, ny, nz, Lx, Ly, Lz, E, nu, rho)
    Kp, Mp = S[0], S[1]
    nd = np.asarray(nd, dtype=int)
    if nd.size != 3:
        raise ValueError("nd must have three entries")
    per = (ny+1)*(nz+1)*3
    c = nx//3
    counts = np.array([(c-1)*per, (c-1)*per, (nx-2*c)*per, 2*per])
    o = np.concatenate([[0], np.cumsum(counts)])
    nCB = int(nd.sum() + counts[3])
    cols = []
    psi = np.zeros((Kp.shape[0], counts[3]))
    res = np.zeros((Kp.shape[0], counts[3]))
    flags = 0
    vanish = 0
    for k in range(3):
        Phi = fixed_interface_modes(Kp, Mp, counts, k, int(nd[k]))
        Psi = constraint_modes(Kp, counts, k)
        R = residual_modes(Kp, Mp, counts, k, Phi)
        ii = slice(o[k], o[k+1])
        if np.allclose(Phi[ii, :].T @ Mp[ii, ii] @ Phi[ii, :], np.eye(int(nd[k])), atol=1e-9):
            flags += 1
        scale = np.abs(Kp[ii, ii]).max()
        if np.abs(Kp[ii, ii] @ Psi[ii, :] + Kp[ii, o[3]:o[4]]).max() < 1e-6*scale:
            flags += 1
        vanish += int((np.linalg.norm(R[ii, :], axis=0) == 0).sum())
        cols.append(Phi)
        psi += Psi
        res += R
    psi[o[3]:o[4], :] = np.eye(counts[3])
    T = np.hstack(cols + [psi, res])
    lam = reduced_eigenvalues(Kp, Mp, T, nCB)
    if not 1 <= mode_index <= lam.size:
        raise ValueError("mode_index out of range")
    u0 = base_contribution(Kp, Mp, T, nCB, mode_index)
    val = relative_deviation(Kp, Mp, u0, lam[mode_index-1])
    return np.array([val, lam[mode_index-1], float(u0 @ Kp @ u0), float(u0 @ Mp @ u0),
                     float(Kp.shape[0]), float(nCB), float(T.shape[1]),
                     float(vanish), float(flags)])
SCICODE_GOLD_EOF
