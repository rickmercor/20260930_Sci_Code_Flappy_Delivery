#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def environmental_response(v: "np.ndarray", eps: "np.ndarray", occupied: "np.ndarray", active: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
    import numpy as np
    _, pairs, gaps, k, act, _, _ = _qc38_active_reference(v, eps, occupied, active)
    keep = np.array([not (i in act and a in act) for i, a in pairs], dtype=bool)
    retained = pairs[keep]
    _, response, _ = _qc38_rpa_response(gaps[keep], k[np.ix_(keep, keep)])
    return retained.copy(), response

def _qc38_reference_data(v: "np.ndarray", eps: "np.ndarray", occupied: "np.ndarray") -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
    import numpy as np
    try:
        if np.iscomplexobj(v) or np.iscomplexobj(eps):
            raise ValueError('Real inputs required.')
        v = np.asarray(v, dtype=float)
        eps = np.asarray(eps, dtype=float)
        oi = np.asarray(occupied)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid numeric input.') from exc
    if eps.ndim != 1 or len(eps) == 0 or not np.all(np.isfinite(eps)):
        raise ValueError('Invalid orbital energies.')
    n = len(eps)
    if v.shape != (n, n, n, n) or not np.all(np.isfinite(v)):
        raise ValueError('Invalid integral tensor.')
    if any(not np.allclose(v, w, atol=1e-12, rtol=0) for w in
           (v.swapaxes(0, 1), v.swapaxes(2, 3), v.transpose(2, 3, 0, 1))):
        raise ValueError('Invalid Coulomb symmetries.')
    if oi.ndim != 1 or (oi.size and oi.dtype.kind not in 'iu'):
        raise ValueError('Occupied indices must be integers.')
    if np.any(oi < 0) or np.any(oi >= n) or len(np.unique(oi)) != len(oi):
        raise ValueError('Invalid occupied indices.')
    occ = [int(i) for i in oi]
    vir = [a for a in range(n) if a not in occ]
    pairs = np.array([(i, a) for i in occ for a in vir], dtype=np.int64).reshape(-1, 2)
    gaps = np.array([eps[a]-eps[i] for i, a in pairs])
    if np.any(gaps <= 0):
        raise ValueError('Occupied-to-virtual gaps must be positive.')
    h = np.diag(eps).copy()
    for i in occ:
        h -= 2*v[:, :, i, i]-v[:, i, i, :]
    k = np.array([[v[i, a, j, b] for j, b in pairs] for i, a in pairs], dtype=float)
    k = k.reshape(len(pairs), len(pairs))
    if not all(np.all(np.isfinite(a)) for a in (h, gaps, k)):
        raise ValueError('Nonfinite reference result.')
    return h, pairs, gaps, k

def _qc38_rpa_response(gaps: "np.ndarray", coupling: "np.ndarray") -> tuple["np.ndarray", "np.ndarray", float]:
    import numpy as np
    try:
        if np.iscomplexobj(gaps) or np.iscomplexobj(coupling):
            raise ValueError('Real inputs required.')
        d = np.asarray(gaps, dtype=float)
        k = np.asarray(coupling, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid numeric input.') from exc
    if d.ndim != 1 or not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise ValueError('Invalid gaps.')
    m = len(d)
    if k.shape != (m, m) or not np.all(np.isfinite(k)):
        raise ValueError('Invalid coupling.')
    if not np.allclose(k, k.T, atol=1e-12, rtol=0):
        raise ValueError('Coupling must be symmetric.')
    if m == 0:
        return np.empty(0), np.empty((0, 0)), 0.0
    root = np.sqrt(d)
    c = root[:, None]*(np.diag(d)+4*k)*root[None, :]
    try:
        lam, z = np.linalg.eigh(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError('RPA eigensolver failed.') from exc
    if not np.all(np.isfinite(lam)) or np.any(lam <= 0):
        raise ValueError('RPA is not strictly stable.')
    omega = np.sqrt(lam)
    u = root[:, None]*z/np.sqrt(omega)[None, :]
    response = -4*(u/omega[None, :])@u.T
    ec = float(0.5*(omega.sum()-d.sum())-np.trace(k))
    if not np.all(np.isfinite(response)) or not np.isfinite(ec):
        raise ValueError('Nonfinite RPA result.')
    return omega, response, ec

def _qc38_active_reference(v, eps, occupied, active):
    import numpy as np
    h, pairs, gaps, coupling = _qc38_reference_data(v, eps, occupied)
    try:
        ax = np.asarray(active)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid active indices.') from exc
    n = len(eps)
    if (ax.ndim != 1 or ax.size == 0 or ax.dtype.kind not in 'iu'
            or np.any(ax < 0) or np.any(ax >= n)
            or len(np.unique(ax)) != len(ax)):
        raise ValueError('Invalid active indices.')
    act = [int(i) for i in ax]
    occ = [int(i) for i in occupied]
    ia = [act.index(i) for i in occ if i in act]
    ie = [i for i in occ if i not in act]
    return h, pairs, gaps, coupling, act, ia, ie

def _qc38_screened_input(w, m):
    import numpy as np
    try:
        if np.iscomplexobj(w):
            raise ValueError('Real screened tensor required.')
        w = np.asarray(w, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid screened tensor.') from exc
    if w.shape != (m,m,m,m) or not np.all(np.isfinite(w)):
        raise ValueError('Invalid screened tensor.')
    if any(not np.allclose(w, other, atol=1e-12, rtol=0) for other in
           (w.swapaxes(0,1), w.swapaxes(2,3), w.transpose(2,3,0,1))):
        raise ValueError('Invalid screened-tensor symmetry.')
    return w

def screened_interaction(v: "np.ndarray", eps: "np.ndarray", occupied: "np.ndarray", active: "np.ndarray") -> "np.ndarray":
    import numpy as np
    pairs, response = environmental_response(v, eps, occupied, active)
    act = np.asarray(active, dtype=int)
    v = np.asarray(v, dtype=float)
    m = len(act)
    bare = v[np.ix_(act, act, act, act)]
    c = np.array([v[:,:,i,a][np.ix_(act,act)].reshape(-1)
                  for i,a in pairs], dtype=float).reshape(len(pairs), m*m)
    w = (bare.reshape(m*m,m*m) + c.T @ response @ c).reshape(bare.shape)
    return _qc38_screened_input(w, m).copy()

def corrected_one_body(v: "np.ndarray", eps: "np.ndarray", occupied: "np.ndarray", active: "np.ndarray", w: "np.ndarray") -> "np.ndarray":
    import numpy as np
    h, _, _, _, act, ia, ie = _qc38_active_reference(v, eps, occupied, active)
    v = np.asarray(v, dtype=float)
    w = _qc38_screened_input(w, len(act))
    delta = w - v[np.ix_(act,act,act,act)]
    t = h[np.ix_(act,act)].copy()
    for i in ie:
        t += (2*v[:,:,i,i] - v[:,i,i,:])[np.ix_(act,act)]
    for i in ia:
        t -= 2*delta[:,:,i,i] - delta[:,i,i,:]
    if not np.all(np.isfinite(t)):
        raise ValueError('Nonfinite one-electron term.')
    return t

def reference_offset(v: "np.ndarray", eps: "np.ndarray", occupied: "np.ndarray", active: "np.ndarray", w: "np.ndarray") -> float:
    import numpy as np
    h, _, _, _, act, ia, ie = _qc38_active_reference(v, eps, occupied, active)
    v = np.asarray(v, dtype=float)
    w = _qc38_screened_input(w, len(act))
    delta = w - v[np.ix_(act,act,act,act)]
    value = 2*sum(h[i,i] for i in ie)
    value += sum(2*v[i,i,j,j]-v[i,j,j,i] for i in ie for j in ie)
    value += sum(2*delta[i,i,j,j]-delta[i,j,j,i] for i in ia for j in ia)
    if not np.isfinite(value):
        raise ValueError('Nonfinite reference offset.')
    return float(value)

def correlation_offset(v: "np.ndarray", eps: "np.ndarray", occupied: "np.ndarray", active: "np.ndarray", w: "np.ndarray") -> float:
    import numpy as np
    _, _, gaps, k, act, ia, _ = _qc38_active_reference(v, eps, occupied, active)
    w = _qc38_screened_input(w, len(act))
    ec_full = _qc38_rpa_response(gaps, k)[2]
    _, _, ga, ka = _qc38_reference_data(w, np.asarray(eps)[act], np.array(ia, dtype=int))
    ec_active = _qc38_rpa_response(ga, ka)[2]
    value = float(ec_full - ec_active)
    if not np.isfinite(value):
        raise ValueError('Nonfinite correlation offset.')
    return value

def selected_spaces(n_a: int, n_b: int, n_e_a: int, samples: "np.ndarray") -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    import numpy as np
    for n in (n_a, n_b):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError('Fragment orbital counts must be positive integers.')
    n_a, n_b = int(n_a), int(n_b)
    completed = _qc38_complete_spins(n_a+n_b, samples)
    if isinstance(n_e_a, (bool, np.bool_)) or not isinstance(n_e_a, (int, np.integer)):
        raise ValueError('Fragment electron count must be an integer.')
    n = n_a+n_b
    a0, b0 = np.asarray(samples)[0]
    na, nb = int(a0).bit_count(), int(b0).bit_count()
    n_e_b = na+nb-int(n_e_a)
    if not 0 <= n_e_a <= 2*n_a or not 0 <= n_e_b <= 2*n_b:
        raise ValueError('Infeasible fragment electron counts.')
    fa, fb, ct = set(), set(), set()
    for packed in completed:
        a, b = int(packed) & ((1 << n)-1), int(packed) >> n
        aa, ba = a & ((1 << n_a)-1), b & ((1 << n_a)-1)
        ab, bb = a >> n_a, b >> n_a
        if aa.bit_count()+ba.bit_count() == n_e_a:
            fa.add(aa | (ba << n_a)); fb.add(ab | (bb << n_b))
        else:
            ct.add(int(packed))
    if not fa or not fb:
        raise ValueError('No neutral fragment configurations.')
    dimer = set(ct)
    for da in fa:
        aa, ba = da & ((1 << n_a)-1), da >> n_a
        for db in fb:
            ab, bb = db & ((1 << n_b)-1), db >> n_b
            a, b = aa | (ab << n_a), ba | (bb << n_a)
            if a.bit_count() == na and b.bit_count() == nb:
                dimer.add(a | (b << n))
    return tuple(np.array(sorted(s), dtype=np.int64) for s in (fa, fb, dimer))

def _qc38_complete_spins(norb: int, samples: "np.ndarray") -> "np.ndarray":
    import numpy as np
    from itertools import combinations
    if isinstance(norb, (bool, np.bool_)) or not isinstance(norb, (int, np.integer)) or not 1 <= norb <= 10:
        raise ValueError('norb must be an integer in [1,10].')
    try:
        s = np.asarray(samples)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid samples.') from exc
    if s.ndim != 2 or s.shape[1] != 2 or s.shape[0] == 0 or s.dtype.kind not in 'iu':
        raise ValueError('Samples must be a nonempty integer (k,2) array.')
    if np.any(s < 0) or np.any(s >= 1 << int(norb)):
        raise ValueError('Mask out of range.')
    counts = {(int(a).bit_count(), int(b).bit_count()) for a, b in s}
    if len(counts) != 1:
        raise ValueError('Inconsistent spin-resolved particle counts.')
    na, nb = next(iter(counts))
    out = set()
    for a, b in s:
        a, b = int(a), int(b)
        double = a & b
        singles = a ^ b
        positions = [p for p in range(norb) if (singles >> p) & 1]
        for selected in combinations(positions, na-double.bit_count()):
            alpha_single = sum(1 << p for p in selected)
            aa = double | alpha_single
            bb = double | (singles ^ alpha_single)
            out.add(aa | (bb << int(norb)))
    return np.array(sorted(out), dtype=np.int64)

def selected_energy(v: "np.ndarray", eps: "np.ndarray", occupied: "np.ndarray", active: "np.ndarray", determinants: "np.ndarray") -> float:
    import numpy as np
    w = screened_interaction(v, eps, occupied, active)
    t = corrected_one_body(v, eps, occupied, active, w)
    e_ref = reference_offset(v, eps, occupied, active, w)
    e_corr = correlation_offset(v, eps, occupied, active, w)
    ham = _qc38_selected_hamiltonian(t, w, determinants)
    required = 2*len(set(int(i) for i in occupied) & set(int(i) for i in active))
    if any(int(d).bit_count() != required for d in determinants):
        raise ValueError('Incorrect active electron count.')
    try:
        value = float(np.linalg.eigvalsh(ham)[0] + (e_ref + e_corr))
    except np.linalg.LinAlgError as exc:
        raise ValueError('Selected eigenproblem failed.') from exc
    if not np.isfinite(value):
        raise ValueError('Nonfinite energy.')
    return value

def _qc38_selected_hamiltonian(h: "np.ndarray", v: "np.ndarray", determinants: "np.ndarray") -> "np.ndarray":
    import numpy as np
    try:
        if np.iscomplexobj(h) or np.iscomplexobj(v):
            raise ValueError('Real inputs required.')
        h = np.asarray(h, dtype=float)
        v = np.asarray(v, dtype=float)
        ds = np.asarray(determinants)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid numeric input.') from exc
    if h.ndim != 2 or h.shape[0] != h.shape[1] or not 1 <= h.shape[0] <= 8:
        raise ValueError('h must be square with 1 to 8 orbitals.')
    n = len(h)
    if not np.all(np.isfinite(h)) or not np.allclose(h, h.T, atol=1e-12, rtol=0):
        raise ValueError('h must be finite and symmetric.')
    if v.shape != (n, n, n, n) or not np.all(np.isfinite(v)):
        raise ValueError('Invalid interaction tensor.')
    if any(not np.allclose(v, w, atol=1e-12, rtol=0) for w in
           (v.swapaxes(0, 1), v.swapaxes(2, 3), v.transpose(2, 3, 0, 1))):
        raise ValueError('Invalid Coulomb symmetries.')
    if ds.ndim != 1 or ds.size == 0 or ds.dtype.kind not in 'iu':
        raise ValueError('Determinants must be a nonempty integer vector.')
    if np.any(ds < 0) or np.any(ds >= 1 << (2*n)) or len(np.unique(ds)) != len(ds):
        raise ValueError('Invalid or repeated determinants.')
    if len({int(d).bit_count() for d in ds}) != 1:
        raise ValueError('Determinants must have the same total electron count.')
    index = {int(d): i for i, d in enumerate(ds)}
    out = np.zeros((len(ds), len(ds)))
    for col, det in enumerate(ds):
        for p, q in np.ndindex(n, n):
            for spin in (0, 1):
                bits, sign = int(det), 1
                for pos, create in ((q+spin*n, False), (p+spin*n, True)):
                    if bool((bits >> pos) & 1) == create:
                        sign = 0; break
                    sign *= -1 if (bits & ((1 << pos)-1)).bit_count() % 2 else 1
                    bits ^= 1 << pos
                if sign and bits in index:
                    out[index[bits], col] += sign*h[p, q]
        for p, q, r, s in np.ndindex(n, n, n, n):
            for spin in (0, 1):
                for tau in (0, 1):
                    bits, sign = int(det), 1
                    for pos, create in ((q+spin*n, False), (s+tau*n, False),
                                        (r+tau*n, True), (p+spin*n, True)):
                        if bool((bits >> pos) & 1) == create:
                            sign = 0; break
                        sign *= -1 if (bits & ((1 << pos)-1)).bit_count() % 2 else 1
                        bits ^= 1 << pos
                    if sign and bits in index:
                        out[index[bits], col] += 0.5*sign*v[p, q, r, s]
    if not np.all(np.isfinite(out)):
        raise ValueError('Nonfinite Hamiltonian.')
    return out

def interaction_energy(factors: "np.ndarray", eps: "np.ndarray", split: int, occupied: "np.ndarray", active: "np.ndarray", samples: "np.ndarray") -> float:
    import numpy as np
    try:
        if np.iscomplexobj(factors):
            raise ValueError('Real factors required.')
        factors = np.asarray(factors, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid factors.') from exc
    if factors.ndim != 3 or factors.shape[1] != factors.shape[2] or not 2 <= factors.shape[1] <= 12:
        raise ValueError('Factors require shape (r,n,n), 2<=n<=12.')
    if not np.all(np.isfinite(factors)) or not np.allclose(factors, factors.swapaxes(1, 2), atol=1e-12, rtol=0):
        raise ValueError('Factors must be finite and symmetric.')
    n = factors.shape[1]
    if isinstance(split, (bool, np.bool_)) or not isinstance(split, (int, np.integer)) or not 1 <= split < n:
        raise ValueError('Invalid fragment split.')
    v = np.einsum('lpq,lrs->pqrs', factors, factors)
    _qc38_reference_data(v, eps, occupied)
    try:
        ax = np.asarray(active)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid active indices.') from exc
    if ax.ndim != 1 or not 2 <= ax.size <= 8 or ax.dtype.kind not in 'iu':
        raise ValueError('Active indices require 2 to 8 integer entries.')
    if np.any(ax < 0) or np.any(ax >= n) or len(np.unique(ax)) != len(ax):
        raise ValueError('Invalid active indices.')
    aa = [int(i) for i in ax if i < split]
    ab = [int(i) for i in ax if i >= split]
    if not aa or not ab or list(ax) != aa+ab:
        raise ValueError('Active A orbitals must precede active B orbitals.')
    occ = [int(i) for i in occupied]
    n_a_e = 2*len(set(aa) & set(occ))
    n_active_occ = len(set(int(i) for i in ax) & set(occ))
    sa, sb, sab = selected_spaces(len(aa), len(ab), n_a_e, samples)
    for alpha, beta in np.asarray(samples):
        if int(alpha).bit_count() != n_active_occ or int(beta).bit_count() != n_active_occ:
            raise ValueError('Samples disagree with active reference spin counts.')
    energies = []
    for inds, act, dets in ((list(range(split)), aa, sa),
                            (list(range(split, n)), ab, sb),
                            (list(range(n)), list(ax), sab)):
        local_occ = np.array([inds.index(i) for i in occ if i in inds], dtype=int)
        local_act = np.array([inds.index(i) for i in act], dtype=int)
        energies.append(selected_energy(v[np.ix_(inds, inds, inds, inds)],
                        np.asarray(eps)[inds], local_occ, local_act, dets))
    return float(energies[2]-energies[0]-energies[1])
SCICODE_GOLD_EOF
