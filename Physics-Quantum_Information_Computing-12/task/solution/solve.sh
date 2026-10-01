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

import numpy as np

def _block_gate(angles):
    a = [float(v) for v in angles]
    x1, x2 = a[-2], a[-1]
    cp, sp = np.cos(np.pi * (x1 + x2) / 2.0), np.sin(np.pi * (x1 + x2) / 2.0)
    cm, sm = np.cos(np.pi * (x2 - x1) / 2.0), np.sin(np.pi * (x2 - x1) / 2.0)
    g = np.array([[cp, 0.0, 0.0, -sp],
                  [0.0, cm, sm, 0.0],
                  [0.0, -sm, cm, 0.0],
                  [sp, 0.0, 0.0, cp]])
    if len(a) == 4:
        c1, s1 = np.cos(np.pi * a[0] / 2.0), np.sin(np.pi * a[0] / 2.0)
        c2, s2 = np.cos(np.pi * a[1] / 2.0), np.sin(np.pi * a[1] / 2.0)
        g = np.kron(np.array([[c1, -s1], [s1, c1]]), np.array([[c2, -s2], [s2, c2]])) @ g
    return g


def block_gate(angles: list) -> "np.ndarray":
    return _block_gate(angles)

import numpy as np

def _layer_tensors(gate_u, gate_w):
    gu = np.asarray(gate_u, dtype=float)
    gw = np.asarray(gate_w, dtype=float)
    return gu.reshape(2, 2, 2, 2), gw[:, [0, 2]].reshape(2, 2, 2)


def layer_tensors(gate_u: "np.ndarray", gate_w: "np.ndarray") -> tuple:
    return _layer_tensors(gate_u, gate_w)

import numpy as np

def _doubled_layer_map(Q, U, W):
    # copy-1 ket W[i,j,A] W[k,l,B]   copy-1 bra W[i,n,C] W[o,t,D]
    # copy-2 ket W[q,r,E] W[s,t,F]   copy-2 bra W[q,v,G] W[x,l,H]
    # i, q: site -1 traced in each copy; l, t: site 2 contracted across the copies
    Wc, Uc = np.conj(W), np.conj(U)
    return np.einsum(
        "ABCDEFGH,ijA,klB,inC,otD,qrE,stF,qvG,xlH,abjk,cdno,efrs,ghvx->abcdefgh",
        Q, W, W, Wc, Wc, W, W, Wc, Wc, U, Uc, U, Uc, optimize=True)


def doubled_layer_map(Q: "np.ndarray", U: "np.ndarray", W: "np.ndarray") -> "np.ndarray":
    return _doubled_layer_map(Q, U, W)

import numpy as np

def _renyi2_from_doubled(Q):
    return float(-np.log2(np.einsum("ababcdcd->", np.asarray(Q, dtype=float))))


def renyi2_from_doubled(Q: "np.ndarray") -> float:
    return _renyi2_from_doubled(Q)

import numpy as np

def _edge_layer_map(R, U, W):
    # ket W[i,j,A] W[k,l,B], bra W[i,n,C] W[o,p,D]; i: site -1 traced; l, p: new edge qubit
    d = R.shape[2]
    Wc, Uc = np.conj(W), np.conj(U)
    X = np.einsum("ABeCDf,ijA,klB,inC,opD,abjk,cdno->abelcdfp",
                  R, W, W, Wc, Wc, U, Uc, optimize=True)
    return X.reshape(2, 2, 2 * d, 2, 2, 2 * d)


def edge_layer_map(R: "np.ndarray", U: "np.ndarray", W: "np.ndarray") -> "np.ndarray":
    return _edge_layer_map(R, U, W)

import numpy as np

def _edge_reduced_state(R):
    return np.einsum("abeabf->ef", np.asarray(R, dtype=float))


def edge_reduced_state(R: "np.ndarray") -> "np.ndarray":
    return _edge_reduced_state(R)

import numpy as np

def _entanglement_spectrum(rho):
    m = np.asarray(rho, dtype=float)
    return np.linalg.eigvalsh(0.5 * (m + m.T))[::-1].copy()


def entanglement_spectrum(rho: "np.ndarray") -> "np.ndarray":
    return _entanglement_spectrum(rho)

import numpy as np

def _block_gate(angles):
    a = [float(v) for v in angles]
    x1, x2 = a[-2], a[-1]
    cp, sp = np.cos(np.pi * (x1 + x2) / 2.0), np.sin(np.pi * (x1 + x2) / 2.0)
    cm, sm = np.cos(np.pi * (x2 - x1) / 2.0), np.sin(np.pi * (x2 - x1) / 2.0)
    g = np.array([[cp, 0.0, 0.0, -sp],
                  [0.0, cm, sm, 0.0],
                  [0.0, -sm, cm, 0.0],
                  [sp, 0.0, 0.0, cp]])
    if len(a) == 4:
        c1, s1 = np.cos(np.pi * a[0] / 2.0), np.sin(np.pi * a[0] / 2.0)
        c2, s2 = np.cos(np.pi * a[1] / 2.0), np.sin(np.pi * a[1] / 2.0)
        g = np.kron(np.array([[c1, -s1], [s1, c1]]), np.array([[c2, -s2], [s2, c2]])) @ g
    return g

def _layer_tensors(gate_u, gate_w):
    gu = np.asarray(gate_u, dtype=float)
    gw = np.asarray(gate_w, dtype=float)
    return gu.reshape(2, 2, 2, 2), gw[:, [0, 2]].reshape(2, 2, 2)

def _doubled_layer_map(Q, U, W):
    # copy-1 ket W[i,j,A] W[k,l,B]   copy-1 bra W[i,n,C] W[o,t,D]
    # copy-2 ket W[q,r,E] W[s,t,F]   copy-2 bra W[q,v,G] W[x,l,H]
    # i, q: site -1 traced in each copy; l, t: site 2 contracted across the copies
    Wc, Uc = np.conj(W), np.conj(U)
    return np.einsum(
        "ABCDEFGH,ijA,klB,inC,otD,qrE,stF,qvG,xlH,abjk,cdno,efrs,ghvx->abcdefgh",
        Q, W, W, Wc, Wc, W, W, Wc, Wc, U, Uc, U, Uc, optimize=True)

def _renyi2_from_doubled(Q):
    return float(-np.log2(np.einsum("ababcdcd->", np.asarray(Q, dtype=float))))

def _edge_layer_map(R, U, W):
    # ket W[i,j,A] W[k,l,B], bra W[i,n,C] W[o,p,D]; i: site -1 traced; l, p: new edge qubit
    d = R.shape[2]
    Wc, Uc = np.conj(W), np.conj(U)
    X = np.einsum("ABeCDf,ijA,klB,inC,opD,abjk,cdno->abelcdfp",
                  R, W, W, Wc, Wc, U, Uc, optimize=True)
    return X.reshape(2, 2, 2 * d, 2, 2, 2 * d)

def _edge_reduced_state(R):
    return np.einsum("abeabf->ef", np.asarray(R, dtype=float))

def _entanglement_spectrum(rho):
    m = np.asarray(rho, dtype=float)
    return np.linalg.eigvalsh(0.5 * (m + m.T))[::-1].copy()

def _bind():
    fallbacks = (("block_gate", _block_gate),
                 ("layer_tensors", _layer_tensors),
                 ("doubled_layer_map", _doubled_layer_map),
                 ("renyi2_from_doubled", _renyi2_from_doubled),
                 ("edge_layer_map", _edge_layer_map),
                 ("edge_reduced_state", _edge_reduced_state),
                 ("entanglement_spectrum", _entanglement_spectrum))
    g = globals()
    for nm, fb in fallbacks:
        g.setdefault(nm, fb)
    return dict(fallbacks)


def schmidt_gap(u_angles: list, w_angles: list) -> float:
    _bind()
    layers = [layer_tensors(block_gate(ua), block_gate(wa))
              for ua, wa in zip(u_angles, w_angles)]
    Q = np.zeros([2] * 8)
    Q[(0,) * 8] = 1.0
    R = np.zeros((2, 2, 1, 2, 2, 1))
    R[0, 0, 0, 0, 0, 0] = 1.0
    for U, W in reversed(layers):
        Q = doubled_layer_map(Q, U, W)
        R = edge_layer_map(R, U, W)
    s2 = renyi2_from_doubled(Q)
    lam = entanglement_spectrum(edge_reduced_state(R))
    if abs(s2 + np.log2(np.sum(lam ** 2))) > 1e-8:
        raise ValueError("purity of the doubled map disagrees with the edge spectrum")
    return float(lam[0] - lam[1])
SCICODE_GOLD_EOF
