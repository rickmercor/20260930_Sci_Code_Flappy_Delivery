"""
Compute the constrained environmental static density response.

Use the molecular cRPA response prescription in the fixed direct-Coulomb RPA convention of the problem. A spatial-transition representation is used for the spin-summed physical response.

Returns
-------
tuple (pairs, response): int64 transition labels of shape (m,2) and the real spin-summed static response of shape (m,m).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def environmental_response(v: "np.ndarray", eps: "np.ndarray", occupied: "np.ndarray", active: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
    """Compute the constrained environmental static density response.
    
    Parameters and conventions
    --------------------------
    v is a real finite chemists' tensor of shape (n,n,n,n), n>=1,
    with p/q, r/s and pair-exchange symmetry (atol=1e-12, rtol=0).
    eps is a real finite vector of length n. occupied is a vector of
    distinct integer indices in [0,n), each doubly occupied; empty is
    allowed. Every occupied-to-virtual gap must be positive. active is
    a nonempty vector of distinct integer indices in [0,n); its order
    defines the local active orbital order. Booleans are not indices.
    The reference orbitals, energies and occupations remain fixed.
    Complex dtype numerical inputs are rejected even if their imaginary
    parts are zero. Raise ValueError for violations, failed numerical
    solves, or nonfinite computed results.
    
    Return retained spatial transition labels (i,a) in occupied input
    order and increasing virtual label within each occupied index.
    The response is the retarded, zero-frequency, zero-broadening physical
    density response, including BOTH spin copies, expressed in this
    spatial transition basis; coupling entries carry no added spin factor.
    An empty constrained transition space returns shapes (0,2) and (0,0).
    Only the constrained RPA problem is solved here; it must be strictly
    stable. In the supplied convention this means that the symmetric
    charge-channel RPA squared-frequency matrix is positive definite.
    
    Returns
    -------
    tuple (pairs, response): int64 transition labels of shape (m,2) and the real spin-summed static response of shape (m,m).
    """
    return pairs, response

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_environmental_response(v: "np.ndarray", eps: "np.ndarray", occupied: "np.ndarray", active: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base = "import numpy as np\nfactors=np.array([[[.50,.16,.30],[.16,.36,.09],[.30,.09,.30]],\n                  [[.20,-.08,.22],[-.08,.28,.07],[.22,.07,.24]]])\nv=np.einsum('lpq,lrs->pqrs',factors,factors)\neps=np.array([-.8,-.35,1.1])\noccupied=np.array([0],dtype=int)\nactive=np.array([0,1],dtype=int)\n"
    variants = [
        "",
        "active=np.arange(3,dtype=int)\n",
        "occupied=np.array([0,1],dtype=int)\nactive=np.array([1,2],dtype=int)\n",
        "active=active[::-1].copy()\n",
        "occupied=np.array([],dtype=int)\n",
    ]
    model = 'environmental_response(v.copy(), eps.copy(), occupied.copy(), active.copy())'
    gold = '_oracle_environmental_response(v.copy(), eps.copy(), occupied.copy(), active.copy())'
    cases = [
        {"setup": base + variants[0], "call": model, "gold_call": gold, "tol": 1e-10},
        {"setup": base + variants[1], "call": model, "gold_call": gold, "tol": 1e-10},
        {"setup": base + variants[2], "call": model, "gold_call": gold, "tol": 1e-10},
        {"setup": base + variants[3], "call": model, "gold_call": gold, "tol": 1e-10},
        {"setup": base + variants[4], "call": model, "gold_call": gold, "tol": 1e-10},
    ]
    cases.append({"setup":base+'v=v.astype(complex)\n\ndef run_model():\n    try:\n        environmental_response(v.copy(), eps.copy(), occupied.copy(), active.copy())\n    except ValueError:\n        return True\n    return False\n\ndef run_gold():\n    try:\n        _oracle_environmental_response(v.copy(), eps.copy(), occupied.copy(), active.copy())\n    except ValueError:\n        return True\n    return False\n', "call":"run_model()", "gold_call":"run_gold()"})
    return cases
