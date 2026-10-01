"""
Step 02: Basis-free self-consistent LSD spin densities of an s-electron atom by a radial solver. Basis-free self-consistent spin densities of a spherical atom or ion in the local spin density approximation.

For an atom or ion whose occupied orbitals are all s orbitals, such as H, He, Li, Be+ and B2+, each spin density is spherical, and the Kohn-Sham equations reduce to one radial equation per spin channel for the functions u(r) = r R(r), with u(0) = 0 and u decaying at infinity. Solving these equations numerically, without any basis set, gives the complete-basis-set limit of the local spin density approximation; the NIST atomic reference data for LSD calculations were produced this way and quote total energies and orbital energies to a microhartree. The effective potential of each spin combines the attraction to the nucleus, the Hartree potential of the total density, which follows from the radial Poisson equation with the total charge fixing its long-range behaviour, and the exchange-correlation potential of that spin. The equations are nonlinear and are iterated to self-consistency, and the occupied orbitals of each spin are the lowest states of that spin's radial equation. Near the nucleus the orbitals vary on a length scale of 1/Z, while the valence density of a neutral atom decays exponentially with a rate set by its highest orbital energy and remains significant out to many bohr, so a numerical solution must resolve both regions at once.

Returns
-------
numpy.ndarray of shape (2, len(radii)): basis-free self-consistent LSD up-spin and down-spin densities at the given radii in bohr^-3
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def radial_lsd_spin_densities(Z: float, n_up: int, n_down: int, radii: "np.ndarray") -> "np.ndarray":
    '''Converged basis-free LSD spin densities of an atom or ion with only s electrons, at given distances.

    Parameters
    ----------
    Z : float
        Nuclear charge in units of e, finite and > 0.
    n_up : int
        Number of occupied up-spin s orbitals, 1 or 2.
    n_down : int
        Number of occupied down-spin s orbitals, integer with 0 <= n_down <= n_up; the system has n_up + n_down <= Z
        electrons.
    radii : np.ndarray
        1-D array of finite distances r > 0 from the nucleus in bohr.

    Returns
    -------
    result : np.ndarray
        Array of shape (2, len(radii)) in bohr^-3: row 0 is the up-spin and row 1 the down-spin density of the self-
        consistent, spin-unrestricted Kohn-Sham solution in the complete-basis-set limit. Each spin channel has the
        nonrelativistic kinetic energy, the attraction -Z/r to a point nucleus, the Hartree potential of the total
        density and its own exchange-correlation potential from lsd_exchange_correlation (no density cutoff), and its
        n_sigma lowest s states are singly occupied. The densities are converged to a relative accuracy of 1e-8 wherever
        they exceed 1e-10 bohr^-3.

    Raises
    ------
    ValueError
        If Z is not finite or not positive, n_up or n_down is not an integer in the ranges above, or radii contains a
        value that is not finite and positive.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _radial_orbitals(L_int, wr, V, n):
    import numpy as np
    if n == 0:
        return np.zeros((0, V.size)), np.zeros(0)
    e, U = np.linalg.eig(-0.5 * L_int + np.diag(V))
    e = e.real
    U = U.real
    idx = np.argsort(e)[:n]
    U = U[:, idx]
    U = U / np.sqrt((wr[:, None] * U * U).sum(0))
    return U.T, e[idx]

def _radial_densities(L_int, wr, r, V_up, V_down, n_up, n_down):
    import numpy as np
    Uu, eu = _radial_orbitals(L_int, wr, V_up, n_up)
    Ud, ed = _radial_orbitals(L_int, wr, V_down, n_down)
    ru = (Uu ** 2).sum(0) / (4.0 * np.pi * r * r)
    rd = (Ud ** 2).sum(0) / (4.0 * np.pi * r * r) if n_down else np.zeros_like(r)
    return ru, rd, Uu, Ud, eu, ed

def _radial_potentials(L_int, L_inf, r, Z, n_el, ru, rd):
    import numpy as np
    U = np.linalg.solve(L_int, -4.0 * np.pi * r * (ru + rd) - L_inf * n_el)
    xc = _oracle_lsd_exchange_correlation(np.maximum(ru, 0.0), np.maximum(rd, 0.0))
    return -Z / r + U / r + xc[1], -Z / r + U / r + xc[2]

def _radial_lsd_atom(Z, n_up, n_down, N=140):
    import numpy as np
    cache = _radial_lsd_atom.__dict__.setdefault("cache", {})
    key = (float(Z), int(n_up), int(n_down), int(N))
    if key in cache:
        return cache[key]
    k = np.arange(N + 1)
    t = np.cos(np.pi * k / N)
    c = np.hstack([2.0, np.ones(N - 1), 2.0]) * (-1.0) ** k
    T = np.tile(t, (N + 1, 1)).T
    D = np.outer(c, 1.0 / c) / (T - T.T + np.eye(N + 1))
    D -= np.diag(D.sum(1))
    a = 2.0 / np.sqrt(Z)
    ti = t[1:N]
    r = a * (1.0 + ti) / (1.0 - ti)
    rt = 2.0 * a / (1.0 - ti) ** 2
    rtt = 4.0 * a / (1.0 - ti) ** 3
    L = (1.0 / rt ** 2)[:, None] * (D @ D)[1:N, :] - (rtt / rt ** 3)[:, None] * D[1:N, :]
    L_int = L[:, 1:N]
    L_inf = L[:, 0]
    th = np.pi * k / N
    wcc = np.zeros(N + 1)
    v = np.ones(N - 1)
    for j in range(1, N // 2):
        v -= 2.0 * np.cos(2.0 * j * th[1:N]) / (4.0 * j * j - 1.0)
    v -= np.cos(N * th[1:N]) / (N * N - 1.0)
    wcc[1:N] = 2.0 * v / N
    wr = wcc[1:N] * rt
    n_el = n_up + n_down
    V_up = -Z / r
    V_down = -Z / r
    xs, fs = [], []
    best = None
    for it in range(400):
        ru, rd, _, _, _, _ = _radial_densities(L_int, wr, r, V_up, V_down, n_up, n_down)
        new_up, new_down = _radial_potentials(L_int, L_inf, r, Z, n_el, ru, rd)
        x = np.concatenate([V_up, V_down])
        f = np.concatenate([new_up, new_down]) - x
        res = np.max(np.abs(f))
        if best is None or res < best[0]:
            best = (res, V_up.copy(), V_down.copy())
        if res < 1e-11:
            break
        xs.append(x)
        fs.append(f)
        xs, fs = xs[-8:], fs[-8:]
        if len(xs) > 1:
            dX = np.array([xs[i + 1] - xs[i] for i in range(len(xs) - 1)]).T
            dF = np.array([fs[i + 1] - fs[i] for i in range(len(fs) - 1)]).T
            g = np.linalg.lstsq(dF, f, rcond=None)[0]
            x = x + 0.5 * f - (dX + 0.5 * dF) @ g
        else:
            x = x + 0.3 * f
        V_up, V_down = x[:N - 1], x[N - 1:]
    ru, rd, Uu, Ud, eu, ed = _radial_densities(L_int, wr, r, best[1], best[2], n_up, n_down)
    uu = np.zeros((n_up, N + 1))
    uu[:, 1:N] = Uu
    ud = np.zeros((n_down, N + 1))
    ud[:, 1:N] = Ud
    bw = np.hstack([0.5, np.ones(N - 1), 0.5]) * (-1.0) ** k
    atom = {"t": t, "a": a, "bw": bw, "uu": uu, "ud": ud, "eu": eu, "ed": ed}
    cache[key] = atom
    return atom

def _oracle_radial_lsd_spin_densities(Z: float, n_up: int, n_down: int, radii: "np.ndarray") -> "np.ndarray":
    import numpy as np
    if not np.isfinite(Z) or Z <= 0.0:
        raise ValueError("Z must be finite and positive")
    for n in (n_up, n_down):
        if isinstance(n, bool) or int(n) != n:
            raise ValueError("occupation numbers must be integers")
    n_up, n_down = int(n_up), int(n_down)
    if n_up < 1 or n_up > 2 or n_down < 0 or n_down > n_up or n_up + n_down > Z + 1e-12:
        raise ValueError("need 1 <= n_up <= 2, 0 <= n_down <= n_up and n_up + n_down <= Z")
    r = np.asarray(radii, dtype=float)
    if r.ndim != 1 or not np.all(np.isfinite(r)) or np.any(r <= 0.0):
        raise ValueError("radii must be a 1-D array of finite positive distances")
    atom = _radial_lsd_atom(float(Z), n_up, n_down)
    tt = (r - atom["a"]) / (r + atom["a"])
    diff = tt[:, None] - atom["t"][None, :]
    hit = np.abs(diff) < 1e-15
    diff[hit] = 1.0
    C = atom["bw"][None, :] / diff
    out = np.zeros((2, r.size))
    for s, U in enumerate((atom["uu"], atom["ud"])):
        if U.shape[0] == 0:
            continue
        val = (U @ C.T) / C.sum(1)
        for i in np.nonzero(hit.any(1))[0]:
            val[:, i] = U[:, np.nonzero(hit[i])[0][0]]
        out[s] = (val ** 2).sum(0) / (4.0 * np.pi * r * r)
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: lithium, 1s2 2s1, from the core to the valence tail ---
        {"setup": "import numpy as np\nrr = np.array([0.01, 0.1, 0.5, 1.0, 2.5, 5.0, 9.0])\n",
         "call": "radial_lsd_spin_densities(3.0, 2, 1, rr.copy())",
         "gold_call": "_oracle_radial_lsd_spin_densities(3.0, 2, 1, rr.copy())", "tol": 1e-7},
        # --- Normal: the Be+ cation ---
        {"setup": "import numpy as np\nrr = np.array([0.02, 0.2, 0.7, 1.5, 3.0, 6.0])\n",
         "call": "radial_lsd_spin_densities(4.0, 2, 1, rr.copy())",
         "gold_call": "_oracle_radial_lsd_spin_densities(4.0, 2, 1, rr.copy())", "tol": 1e-7},
        # --- Boundary: closed-shell helium, equal spin densities ---
        {"setup": "import numpy as np\nrr = np.array([0.05, 0.4, 1.2, 3.0, 5.0])\n",
         "call": "radial_lsd_spin_densities(2.0, 1, 1, rr.copy())",
         "gold_call": "_oracle_radial_lsd_spin_densities(2.0, 1, 1, rr.copy())", "tol": 1e-7},
        # --- Boundary: the hydrogen atom, fully polarized with an empty down-spin channel ---
        {"setup": "import numpy as np\nrr = np.array([0.1, 1.0, 3.0, 6.0, 10.0])\n",
         "call": "radial_lsd_spin_densities(1.0, 1, 0, rr.copy())",
         "gold_call": "_oracle_radial_lsd_spin_densities(1.0, 1, 0, rr.copy())", "tol": 1e-7},
        # --- Edge: the Li+ cation with its two electrons at points very close to the nucleus ---
        {"setup": "import numpy as np\nrr = np.array([1e-4, 1e-3, 0.03])\n",
         "call": "radial_lsd_spin_densities(3.0, 1, 1, rr.copy())",
         "gold_call": "_oracle_radial_lsd_spin_densities(3.0, 1, 1, rr.copy())", "tol": 1e-7},
        # --- Error: more down-spin than up-spin electrons ---
        {"setup": "import numpy as np\ndef _probe(fn):\n    try:\n        fn(3.0, 1, 2, np.array([1.0]))\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(radial_lsd_spin_densities)", "gold_call": "_probe(_oracle_radial_lsd_spin_densities)"},
    ]
