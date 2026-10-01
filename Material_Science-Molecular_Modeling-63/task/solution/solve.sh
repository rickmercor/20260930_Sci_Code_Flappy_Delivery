#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def _mj_validate(positions, box, species, pair_weights, cutoff):
    import numpy as np
    r = np.asarray(positions, dtype=float)
    L = np.asarray(box, dtype=float)
    z0 = np.asarray(species)
    c = np.asarray(pair_weights, dtype=float)
    if r.ndim != 2 or r.shape[1:] != (3,) or len(r) < 1 or (L.shape != (3,)):
        raise ValueError('geometry shape')
    if c.ndim != 2 or c.shape[0] != c.shape[1] or len(c) < 1 or (z0.shape != (len(r),)):
        raise ValueError('type shape')
    if not all((np.all(np.isfinite(a)) for a in (r, L, z0, c))) or not np.isfinite(cutoff):
        raise ValueError('nonfinite input')
    if np.any(L <= 0) or cutoff <= 0 or np.any(z0 != np.floor(z0)) or np.any(z0 < 0) or np.any(z0 >= len(c)):
        raise ValueError('input domain')
    z = z0.astype(int)
    r = np.mod(r, L)
    for i in range(len(r)):
        for j in range(i):
            d = r[i] - r[j]
            d -= L * np.rint(d / L)
            if np.linalg.norm(d) < 1e-10:
                raise ValueError('coincident sites')
    return (r, L, z, c)

def moment_jets(positions, box, species, pair_weights, cutoff):
    """Reference implementation."""
    import numpy as np
    import itertools
    r, L, z, c = _mj_validate(positions, box, species, pair_weights, cutoff)
    n, D = (len(r), 3 * len(r))
    m = np.zeros((n, 13))
    g = np.zeros((n, 13, D))
    h = np.zeros((n, 13, D, D))
    bounds = np.ceil(cutoff / L).astype(int)
    eye = np.eye(3)
    for i in range(n):
        for j in range(n):
            B = np.zeros((3, D))
            B[:, 3 * j:3 * j + 3] += eye
            B[:, 3 * i:3 * i + 3] -= eye
            for image in itertools.product(*(range(-v, v + 1) for v in bounds)):
                if i == j and image == (0, 0, 0):
                    continue
                d = r[j] + L * np.array(image) - r[i]
                distance = np.linalg.norm(d)
                if distance >= cutoff:
                    continue
                e = d / distance
                t = 1 - distance / cutoff
                weight = c[z[i], z[j]]
                w = weight * t ** 4
                wp = -4 * weight * t ** 3 / cutoff
                wpp = 12 * weight * t ** 2 / cutoff ** 2
                dw = wp * e
                hw = wpp * np.outer(e, e) + wp / distance * (eye - np.outer(e, e))
                vals = np.r_[1.0, d, np.outer(d, d).ravel()]
                dv = np.zeros((13, 3))
                dv[1:4] = eye
                hv = np.zeros((13, 3, 3))
                for a in range(3):
                    for b in range(3):
                        k = 4 + 3 * a + b
                        dv[k] = eye[a] * d[b] + eye[b] * d[a]
                        hv[k] = np.outer(eye[a], eye[b]) + np.outer(eye[b], eye[a])
                local_g = vals[:, None] * dw + w * dv
                local_h = vals[:, None, None] * hw + w * hv + np.einsum('ka,b->kab', dv, dw) + np.einsum('a,kb->kab', dw, dv)
                m[i] += w * vals
                g[i] += local_g @ B
                h[i] += np.einsum('ad,kab,be->kde', B, local_h, B)
    return (m, g, h)

def _ij_mul(a, b):
    import numpy as np
    av, ag, ah = a
    bv, bg, bh = b
    return (av * bv, ag * bv + av * bg, ah * bv + av * bh + np.outer(ag, bg) + np.outer(bg, ag))

def _ij_sum(items):
    return tuple((sum((x[k] for x in items)) for k in range(3)))

def invariant_jets(moments, first, second):
    """Reference implementation."""
    import numpy as np
    m = np.asarray(moments, float)
    g = np.asarray(first, float)
    h = np.asarray(second, float)
    if m.ndim != 2 or m.shape[1] != 13 or len(m) < 1 or (g.ndim != 3) or (g.shape[:2] != m.shape) or (g.shape[2] < 1) or (h.shape != g.shape + (g.shape[-1],)):
        raise ValueError('jet shape')
    if not all((np.all(np.isfinite(a)) for a in (m, g, h))):
        raise ValueError('nonfinite jet')
    n, D = (g.shape[0], g.shape[2])
    f = np.zeros((n, 5))
    df = np.zeros((n, 5, D))
    ddf = np.zeros((n, 5, D, D))
    for i in range(n):
        jets = [(m[i, k], g[i, k], h[i, k]) for k in range(13)]
        out = [jets[0], _ij_mul(jets[0], jets[0])]
        out.append(_ij_sum([_ij_mul(jets[k], jets[k]) for k in range(1, 4)]))
        out.append(_ij_sum([_ij_mul(jets[k], jets[k]) for k in range(4, 13)]))
        out.append(_ij_sum([_ij_mul(_ij_mul(jets[1 + a], jets[4 + 3 * a + b]), jets[1 + b]) for a in range(3) for b in range(3)]))
        for k, (v, dv, hv) in enumerate(out):
            f[i, k] = v
            df[i, k] = dv
            ddf[i, k] = hv
    return (f, df, ddf)

def redistributed_charge_jets(features, first, second, species, bias, coefficients, softness, total_charge):
    """Reference implementation."""
    import numpy as np
    f = np.asarray(features, float)
    g = np.asarray(first, float)
    h = np.asarray(second, float)
    z0 = np.asarray(species)
    b = np.asarray(bias, float)
    c = np.asarray(coefficients, float)
    s = np.asarray(softness, float)
    if f.ndim != 2 or f.shape[1] != 5 or len(f) < 1 or (g.ndim != 3) or (g.shape[:2] != f.shape) or (g.shape[-1] < 1) or (h.shape != g.shape + (g.shape[-1],)):
        raise ValueError('feature shape')
    if b.ndim != 1 or len(b) < 1 or c.shape != (len(b), 5) or (s.shape != b.shape) or (z0.shape != (len(f),)):
        raise ValueError('parameter shape')
    if not all((np.all(np.isfinite(a)) for a in (f, g, h, z0, b, c, s))) or not np.isfinite(total_charge):
        raise ValueError('nonfinite input')
    if np.any(s <= 0) or np.any(z0 != np.floor(z0)) or np.any(z0 < 0) or np.any(z0 >= len(b)):
        raise ValueError('parameter domain')
    z = z0.astype(int)
    w = s[z] / s[z].sum()
    y = b[z] + np.einsum('nk,nk->n', c[z], f)
    dy = np.einsum('nk,nkd->nd', c[z], g)
    hy = np.einsum('nk,nkde->nde', c[z], h)
    return (y + w * (total_charge - y.sum()), dy - w[:, None] * dy.sum(axis=0), hy - w[:, None, None] * hy.sum(axis=0))

def ewald_kernel_jets(positions, box, alpha, real_extent, reciprocal_extent):
    """Reference implementation."""
    import numpy as np
    import itertools
    from scipy.special import erfc
    r = np.asarray(positions, float)
    L = np.asarray(box, float)
    M0 = np.asarray(reciprocal_extent)
    if r.ndim != 2 or r.shape[1:] != (3,) or len(r) < 1 or (L.shape != (3,)) or (M0.shape != (3,)):
        raise ValueError('shape')
    if not all((np.all(np.isfinite(a)) for a in (r, L, M0))) or not np.isfinite(alpha) or (not np.isfinite(real_extent)):
        raise ValueError('nonfinite')
    if np.any(L <= 0) or alpha <= 0 or real_extent < 0 or (real_extent != int(real_extent)) or np.any(M0 < 0) or np.any(M0 != np.floor(M0)):
        raise ValueError('domain')
    r = np.mod(r, L)
    M = M0.astype(int)
    R = int(real_extent)
    n = len(r)
    D = 3 * n
    V = L.prod()
    eye = np.eye(3)
    for i in range(n):
        for j in range(i):
            d = r[i] - r[j]
            d -= L * np.rint(d / L)
            if np.linalg.norm(d) < 1e-10:
                raise ValueError('coincident sites')
    modes = np.array([v for v in itertools.product(*(range(-v, v + 1) for v in M)) if v != (0, 0, 0)], float).reshape(-1, 3)
    kv = 2 * np.pi * modes / L
    k2 = np.einsum('ka,ka->k', kv, kv)
    weights = 4 * np.pi / V * np.exp(-k2 / (4 * alpha ** 2)) / k2
    K = np.zeros((n, n))
    G = np.zeros((n, n, D))
    H = np.zeros((n, n, D, D))
    images = list(itertools.product(range(-R, R + 1), repeat=3))
    for i in range(n):
        for j in range(n):
            B = np.zeros((3, D))
            B[:, 3 * i:3 * i + 3] += eye
            B[:, 3 * j:3 * j + 3] -= eye
            delta = r[i] - r[j]
            phase = kv @ delta
            value = np.dot(weights, np.cos(phase)) - np.pi / (alpha ** 2 * V)
            grad = -(weights * np.sin(phase)) @ kv
            hes = -np.einsum('k,ka,kb->ab', weights * np.cos(phase), kv, kv)
            if i == j:
                value -= 2 * alpha / np.sqrt(np.pi)
            for image in images:
                if i == j and image == (0, 0, 0):
                    continue
                d = delta + L * np.array(image)
                rho = np.linalg.norm(d)
                e = d / rho
                f = erfc(alpha * rho)
                fp = -2 * alpha / np.sqrt(np.pi) * np.exp(-(alpha * rho) ** 2)
                fpp = -2 * alpha ** 2 * rho * fp
                v = f / rho
                vp = fp / rho - f / rho ** 2
                vpp = fpp / rho - 2 * fp / rho ** 2 + 2 * f / rho ** 3
                value += v
                grad += vp * e
                hes += vpp * np.outer(e, e) + vp / rho * (eye - np.outer(e, e))
            K[i, j] = value
            G[i, j] = grad @ B
            H[i, j] = B.T @ hes @ B
    return (K, G, H)

def electrostatic_jet(charges, charge_first, charge_second, kernel, kernel_first, kernel_second):
    """Reference implementation."""
    import numpy as np
    q = np.asarray(charges, float)
    J = np.asarray(charge_first, float)
    Q = np.asarray(charge_second, float)
    K = np.asarray(kernel, float)
    G = np.asarray(kernel_first, float)
    H = np.asarray(kernel_second, float)
    if q.ndim != 1 or len(q) < 1 or J.ndim != 2 or (J.shape[0] != len(q)) or (J.shape[1] < 1):
        raise ValueError('charge shape')
    n, D = J.shape
    if Q.shape != (n, D, D) or K.shape != (n, n) or G.shape != (n, n, D) or (H.shape != (n, n, D, D)):
        raise ValueError('derivative shape')
    if not all((np.all(np.isfinite(a)) for a in (q, J, Q, K, G, H))):
        raise ValueError('nonfinite')
    Kq = K @ q
    energy = 0.5 * q @ Kq
    grad = J.T @ Kq + 0.5 * np.einsum('i,ija,j->a', q, G, q)
    mixed = np.einsum('ia,ijb,j->ab', J, G, q)
    hess = np.einsum('iab,i->ab', Q, Kq) + J.T @ K @ J + mixed + mixed.T + 0.5 * np.einsum('i,ijab,j->ab', q, H, q)
    return (float(energy), -grad, hess)

def periodic_born_tensors(positions, box, charges, charge_first):
    """Reference implementation."""
    import numpy as np
    r = np.asarray(positions, float)
    L = np.asarray(box, float)
    q = np.asarray(charges, float)
    J = np.asarray(charge_first, float)
    if r.ndim != 2 or r.shape[1:] != (3,) or len(r) < 1 or (L.shape != (3,)) or (q.shape != (len(r),)) or (J.shape != (len(r), 3 * len(r))):
        raise ValueError('shape')
    if not all((np.all(np.isfinite(a)) for a in (r, L, q, J))) or np.any(L <= 0):
        raise ValueError('domain')
    n = len(r)
    Z = np.zeros((n, 3, 3))
    phase = 2 * np.pi * r / L
    for l in range(n):
        for a in range(3):
            Z[l, a] = L[a] / (2 * np.pi) * np.sin(phase[:, a] - phase[l, a]) @ J[:, 3 * l:3 * l + 3]
            Z[l, a, a] += q[l]
    return Z

def longitudinal_shift(analytic_hessian, born_scaled, masses, box, direction):
    """Reference implementation."""
    import numpy as np
    H = np.asarray(analytic_hessian, float)
    Z = np.asarray(born_scaled, float)
    m = np.asarray(masses, float)
    L = np.asarray(box, float)
    vdir = np.asarray(direction, float)
    if m.ndim != 1 or len(m) < 2 or H.shape != (3 * len(m), 3 * len(m)) or (Z.shape != (len(m), 3, 3)) or (L.shape != (3,)) or (vdir.shape != (3,)):
        raise ValueError('shape')
    if not all((np.all(np.isfinite(a)) for a in (H, Z, m, L, vdir))) or np.any(m <= 0) or np.any(L <= 0) or (np.linalg.norm(vdir) == 0):
        raise ValueError('domain')
    n = len(m)
    T = np.tile(np.eye(3), (n, 1))
    P = np.eye(3 * n) - T @ T.T / n
    Hc = P @ ((H + H.T) / 2) @ P
    Z = Z - Z.mean(axis=0)
    scale = np.repeat(np.sqrt(m), 3)
    D = Hc / np.outer(scale, scale)
    normal = vdir / np.linalg.norm(vdir)
    vector = np.einsum('a,lab->lb', normal, Z) / np.sqrt(m)[:, None]
    nac = 4 * np.pi / L.prod() * np.outer(vector.ravel(), vector.ravel())
    Q, _ = np.linalg.qr(scale[:, None] * T, mode='complete')
    optical = Q[:, 3:]
    lo0 = np.linalg.eigvalsh(optical.T @ D @ optical)[-1]
    lo1 = np.linalg.eigvalsh(optical.T @ (D + nac) @ optical)[-1]
    if lo0 <= 0 or lo1 <= 0:
        raise ValueError('nonpositive top optical curvature')
    return float(np.sqrt(lo1) - np.sqrt(lo0))

def solve_phonon_shift(positions, box, species, pair_weights, cutoff, bias, coefficients, softness, alpha, real_extent, reciprocal_extent, short_hessian, masses, direction):
    """Reference implementation of the complete neutral pipeline."""
    import numpy as np
    r = np.asarray(positions, dtype=float)
    short = np.asarray(short_hessian, dtype=float)
    if r.ndim != 2 or r.shape[1:] != (3,) or len(r) < 2:
        raise ValueError('positions must have shape (N, 3), N >= 2')
    size = 3 * len(r)
    if short.shape != (size, size) or not np.all(np.isfinite(short)):
        raise ValueError('short_hessian must be finite with shape (3*N, 3*N)')
    moments, moment_first, moment_second = moment_jets(positions, box, species, pair_weights, cutoff)
    features, feature_first, feature_second = invariant_jets(moments, moment_first, moment_second)
    charges, charge_first, charge_second = redistributed_charge_jets(features, feature_first, feature_second, species, bias, coefficients, softness, 0.0)
    kernel, kernel_first, kernel_second = ewald_kernel_jets(positions, box, alpha, real_extent, reciprocal_extent)
    _, _, electrostatic_hessian = electrostatic_jet(charges, charge_first, charge_second, kernel, kernel_first, kernel_second)
    born_scaled = periodic_born_tensors(positions, box, charges, charge_first)
    shift = longitudinal_shift(electrostatic_hessian + short, born_scaled, masses, box, direction)
    return float(shift)
SCICODE_GOLD_EOF
