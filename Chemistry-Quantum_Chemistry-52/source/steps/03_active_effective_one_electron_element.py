"""
Active-space effective one-electron operator of the Dyall zeroth-order Hamiltonian.

This step returns one element ``heff[x, y]`` of the effective one-electron operator that

acts inside the active space of the Dyall partition, built from the orthonormal one- and

two-electron integrals and the list of occupied inactive (core) spin orbitals.



Scientific background.  The multi-reference construction splits the Hamiltonian as

H = H0 + V, where the zeroth-order part keeps the *full* two-electron interaction inside a

chosen active space and treats the inactive orbitals only through one-body terms,



H0 = eps_P P^+ P + heff_xy x^+ y + (1/2) <xy|zw> x^+ y^+ w z ,



with P running over the occupied and unoccupied inactive spin orbitals and x, y, z, w over

the active ones.  The active one-electron operator is not the bare integral and not the

full Fock operator either.  It is the bare integral dressed by the mean field of the

*occupied inactive orbitals only*,



heff_xy = h_xy + sum_{k in core} <xk||yk> ,



where <pq||rs> = <pq|rs> - <pq|sr> is the antisymmetrised two-electron integral in

physicists' notation and the sum runs over the occupied inactive spin orbitals alone.

Adding the active mean field as well would double count: the active-active interaction is

already carried explicitly by the two-electron term of H0, so including it again in the

one-body part counts it twice and changes every quantity downstream.  Equally, leaving the

core mean field out would leave the active electrons unscreened by the inactive ones.

The operator is symmetric, heff_xy = heff_yx, for the real orbitals used here.



Both indices must be active spin orbitals; the element is not defined for an inactive

index, because for those the zeroth-order Hamiltonian carries an orbital energy instead.



Raises ValueError when h is not a non-empty square array, when g does not have shape

(n, n, n, n) matching h, when either contains a non-finite entry, when core and act are

not disjoint sequences of integers drawn from 0 to n - 1, and when x or y is not one of

the active spin orbitals listed in act.

Returns
-------
float, the element heff[x, y] of the active-space effective one-electron operator of the Dyall zeroth-order Hamiltonian, in hartree, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def active_effective_one_electron_element(h: np.ndarray, g: np.ndarray, core: list[int], act: list[int],
        x: int, y: int) -> float:
    """One element of the active-space effective one-electron operator heff.

    Parameters
    ----------
    h : array_like, shape (n, n)
        One-electron Hamiltonian in an orthonormal spin-orbital basis, in hartree.
    g : array_like, shape (n, n, n, n)
        Two-electron integrals in physicists' notation, g[p, q, r, s] = <pq|rs>.
    core : sequence of int
        Occupied inactive (core) spin-orbital indices.
    act : sequence of int
        Active spin-orbital indices; must be disjoint from core.
    x : int
        Row spin orbital of the requested element; must appear in act.
    y : int
        Column spin orbital of the requested element; must appear in act.

    Returns
    -------
    float
        heff[x, y] = h[x, y] + sum over the core spin orbitals k of <xk||yk>, in hartree.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _check_int(name, value, low, high):
    """Return int(value) after checking it is an integer with low <= value <= high."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must lie between {low} and {high}")
    return value

def _validate_integrals(h, g):
    """Check h and g are finite real arrays of matching shape; return them as floats."""
    try:
        h_arr = np.asarray(h, dtype=float)
        g_arr = np.asarray(g, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("h and g must be real numeric arrays") from exc
    if h_arr.ndim != 2 or h_arr.shape[0] != h_arr.shape[1] or h_arr.shape[0] < 1:
        raise ValueError("h must be a non-empty square two-dimensional array")
    n_orb = h_arr.shape[0]
    if g_arr.shape != (n_orb, n_orb, n_orb, n_orb):
        raise ValueError("g must have shape (n, n, n, n) matching h")
    if not (np.all(np.isfinite(h_arr)) and np.all(np.isfinite(g_arr))):
        raise ValueError("h and g must contain only finite numbers")
    return h_arr, g_arr

def _validate_two_sets(core, act, n_orb):
    """Check core and act are disjoint integer index sets inside range(n_orb)."""
    try:
        core_l = [int(i) for i in core]
        act_l = [int(i) for i in act]
    except (TypeError, ValueError) as exc:
        raise ValueError("core and act must be sequences of integers") from exc
    joined = core_l + act_l
    if len(set(joined)) != len(joined):
        raise ValueError("core and act must be disjoint and free of repeats")
    if any(not 0 <= i < n_orb for i in joined):
        raise ValueError("core and act indices must lie inside the spin-orbital range")
    return core_l, act_l

def _oracle_active_effective_one_electron_element(h: np.ndarray, g: np.ndarray, core: list[int], act: list[int],
        x: int, y: int) -> float:
    """Reference implementation of the core-only active mean field."""
    h_arr, g_arr = _validate_integrals(h, g)
    n_orb = h_arr.shape[0]
    core_l, act_l = _validate_two_sets(core, act, n_orb)
    x = _check_int("x", x, 0, n_orb - 1)
    y = _check_int("y", y, 0, n_orb - 1)
    if x not in act_l or y not in act_l:
        raise ValueError("x and y must both be active spin orbitals")
    gaa = g_arr - np.transpose(g_arr, (0, 1, 3, 2))
    return float(h_arr[x, y] + sum(gaa[x, k, y, k] for k in core_l))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\nrng = np.random.default_rng(20260913)\ndef _toy(n):\n    a = rng.standard_normal((n, n))\n    h = 0.5 * (a + a.T)\n    b = rng.standard_normal((n, n, n, n))\n    c = 0.25 * (b + np.transpose(b, (1, 0, 3, 2)) + np.transpose(b, (2, 3, 0, 1))\n                + np.transpose(b, (3, 2, 1, 0)))\n    return h, c\nH2, G2 = _toy(2)\nH4, G4 = _toy(4)\nH6, G6 = _toy(6)\n',
            'call': 'active_effective_one_electron_element(H2.copy(), G2.copy(), [], [0, 1], 0, 1)',
            'gold_call': '_oracle_active_effective_one_electron_element(H2.copy(), G2.copy(), [], [0, 1], 0, 1)',
        },
        {
            'setup': 'import numpy as np\nrng = np.random.default_rng(20260913)\ndef _toy(n):\n    a = rng.standard_normal((n, n))\n    h = 0.5 * (a + a.T)\n    b = rng.standard_normal((n, n, n, n))\n    c = 0.25 * (b + np.transpose(b, (1, 0, 3, 2)) + np.transpose(b, (2, 3, 0, 1))\n                + np.transpose(b, (3, 2, 1, 0)))\n    return h, c\nH2, G2 = _toy(2)\nH4, G4 = _toy(4)\nH6, G6 = _toy(6)\n',
            'call': 'active_effective_one_electron_element(H4, G4, [3], [0, 1, 2], 1, 1)',
            'gold_call': '_oracle_active_effective_one_electron_element(H4, G4, [3], [0, 1, 2], 1, 1)',
        },
        {
            'setup': 'import numpy as np\nrng = np.random.default_rng(20260913)\ndef _toy(n):\n    a = rng.standard_normal((n, n))\n    h = 0.5 * (a + a.T)\n    b = rng.standard_normal((n, n, n, n))\n    c = 0.25 * (b + np.transpose(b, (1, 0, 3, 2)) + np.transpose(b, (2, 3, 0, 1))\n                + np.transpose(b, (3, 2, 1, 0)))\n    return h, c\nH2, G2 = _toy(2)\nH4, G4 = _toy(4)\nH6, G6 = _toy(6)\n',
            'call': 'active_effective_one_electron_element(H6.copy(), G6.copy(), [4, 5], [0, 1, 2, 3], 0, 3)',
            'gold_call': '_oracle_active_effective_one_electron_element(H6.copy(), G6.copy(), [4, 5], [0, 1, 2, 3], 0, 3)',
        },
        {
            'setup': 'import numpy as np\nrng = np.random.default_rng(20260913)\ndef _toy(n):\n    a = rng.standard_normal((n, n))\n    h = 0.5 * (a + a.T)\n    b = rng.standard_normal((n, n, n, n))\n    c = 0.25 * (b + np.transpose(b, (1, 0, 3, 2)) + np.transpose(b, (2, 3, 0, 1))\n                + np.transpose(b, (3, 2, 1, 0)))\n    return h, c\nH2, G2 = _toy(2)\nH4, G4 = _toy(4)\nH6, G6 = _toy(6)\n\ndef run_model():\n    return sum(active_effective_one_electron_element(H6, G6, [4, 5], [0, 1, 2, 3], x, x)\n               for x in [0, 1, 2, 3])\ndef run_gold():\n    return sum(_oracle_active_effective_one_electron_element(H6, G6, [4, 5], [0, 1, 2, 3], x, x)\n               for x in [0, 1, 2, 3])\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH12, G12 = _fx_system(12)\nCORE12 = [4, 5]\nACT12 = [1, 2, 3, 6, 7, 8]\nVIR12 = [0, 9, 10, 11]\n",
            'call': 'active_effective_one_electron_element(H12.copy(), G12.copy(), CORE12.copy(), ACT12.copy(), ACT12[-1], ACT12[-1])',
            'gold_call': '_oracle_active_effective_one_electron_element(H12.copy(), G12.copy(), CORE12.copy(), ACT12.copy(), ACT12[-1], ACT12[-1])',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH12, G12 = _fx_system(12)\nCORE12 = [4, 5]\nACT12 = [1, 2, 3, 6, 7, 8]\nVIR12 = [0, 9, 10, 11]\n",
            'call': 'active_effective_one_electron_element(H12.copy(), G12.copy(), CORE12.copy(), ACT12.copy(), ACT12[0], ACT12[-1])',
            'gold_call': '_oracle_active_effective_one_electron_element(H12.copy(), G12.copy(), CORE12.copy(), ACT12.copy(), ACT12[0], ACT12[-1])',
        },
        {
            'setup': 'import numpy as np\nrng = np.random.default_rng(20260913)\ndef _toy(n):\n    a = rng.standard_normal((n, n))\n    h = 0.5 * (a + a.T)\n    b = rng.standard_normal((n, n, n, n))\n    c = 0.25 * (b + np.transpose(b, (1, 0, 3, 2)) + np.transpose(b, (2, 3, 0, 1))\n                + np.transpose(b, (3, 2, 1, 0)))\n    return h, c\nH2, G2 = _toy(2)\nH4, G4 = _toy(4)\nH6, G6 = _toy(6)\n\ndef run_model():\n    try:    active_effective_one_electron_element(H4, G4, [3], [0, 1, 2], 3, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_active_effective_one_electron_element(H4, G4, [3], [0, 1, 2], 3, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'import numpy as np\nrng = np.random.default_rng(20260913)\ndef _toy(n):\n    a = rng.standard_normal((n, n))\n    h = 0.5 * (a + a.T)\n    b = rng.standard_normal((n, n, n, n))\n    c = 0.25 * (b + np.transpose(b, (1, 0, 3, 2)) + np.transpose(b, (2, 3, 0, 1))\n                + np.transpose(b, (3, 2, 1, 0)))\n    return h, c\nH2, G2 = _toy(2)\nH4, G4 = _toy(4)\nH6, G6 = _toy(6)\n\ndef run_model():\n    try:    active_effective_one_electron_element(H4, G4, [2], [0, 1, 2], 0, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_active_effective_one_electron_element(H4, G4, [2], [0, 1, 2], 0, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'import numpy as np\nrng = np.random.default_rng(20260913)\ndef _toy(n):\n    a = rng.standard_normal((n, n))\n    h = 0.5 * (a + a.T)\n    b = rng.standard_normal((n, n, n, n))\n    c = 0.25 * (b + np.transpose(b, (1, 0, 3, 2)) + np.transpose(b, (2, 3, 0, 1))\n                + np.transpose(b, (3, 2, 1, 0)))\n    return h, c\nH2, G2 = _toy(2)\nH4, G4 = _toy(4)\nH6, G6 = _toy(6)\n\ndef run_model():\n    try:    active_effective_one_electron_element(H4, G6, [3], [0, 1, 2], 0, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_active_effective_one_electron_element(H4, G6, [3], [0, 1, 2], 0, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'import numpy as np\nrng = np.random.default_rng(20260913)\ndef _toy(n):\n    a = rng.standard_normal((n, n))\n    h = 0.5 * (a + a.T)\n    b = rng.standard_normal((n, n, n, n))\n    c = 0.25 * (b + np.transpose(b, (1, 0, 3, 2)) + np.transpose(b, (2, 3, 0, 1))\n                + np.transpose(b, (3, 2, 1, 0)))\n    return h, c\nH2, G2 = _toy(2)\nH4, G4 = _toy(4)\nH6, G6 = _toy(6)\n\nbad = H4.copy(); bad[0, 0] = np.nan\ndef run_model():\n    try:    active_effective_one_electron_element(bad, G4, [3], [0, 1, 2], 0, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_active_effective_one_electron_element(bad, G4, [3], [0, 1, 2], 0, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
    ]
