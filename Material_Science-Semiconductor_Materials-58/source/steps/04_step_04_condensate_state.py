"""
Step 04 - Self-consistent condensate: occupation, pairing amplitude and quasiparticle bands.

Self-consistent mean-field condensate: occupation, pairing amplitude and quasiparticle bands.

Below the transition (E_G < E_b) the condensate is the Slater determinant prod_k (u_k c^dag_{k,v} + v_k c^dag_{k,c})
|vac>, with real u_k, v_k, u_k^2 + v_k^2 = 1, obtained by Hartree-Fock decoupling of the two-band Hamiltonian at
charge neutrality. Its mean-field Hamiltonian is a 2x2 matrix in band space at every k whose diagonal entries are
the bare band energies epsilon^c_k = (m/m_c) k^2 + E_G/2 and epsilon^v_k = -(m/m_v) k^2 - E_G/2 (reduced mass
m = m_c m_v/(m_c + m_v), so m/m_c = r/(1 + r) and m/m_v = 1/(1 + r) with r = m_v/m_c) shifted by the intraband Fock
self-energies (built with the intralayer kernel V_0 on v_k^2) and, for a bilayer, by the capacitive charging
energy +- n_ex/(2 C_G) with 1/C_G = 8 pi d, while its off-diagonal entry is the pairing self-energy built with the
interlayer kernel V_d on u_k v_k. The self-consistency condition that this determinant minimises the energy is the
Hartree-Fock gap equation; its solution fixes v_k^2, u_k v_k and the exciton density n_ex = int d^2k/(2 pi)^2 v_k^2, and
Koopmans' theorem gives the two quasiparticle bands E_{k,+-} as the eigenvalues of the same 2x2 matrix. Above the
transition (E_G >= E_b) the only solution is the trivial one, v_k = 0, u_k = 1, with the bare bands. The exact
form of the self-energies and of the self-consistency loop is not restated here.

Converge the loop until every element of v_k^2 and u_k v_k changes by less than 1e-12 between iterations; the
bare bands are E_{k,+} = epsilon^c_k, E_{k,-} = epsilon^v_k when v_k = 0.

Inputs: E_G (Ry*, any finite value), d >= 0 (a_B*), r = m_v/m_c > 0, k_out a one-dimensional array of finite
non-negative magnitudes (0 allowed). Output: array of shape (4, len(k_out)) with rows v_k^2, u_k v_k, E_{k,+},
E_{k,-} (energies in Ry*, measured from the middle of the bare gap), each converged to 1e-9 relative accuracy
(absolute 1e-9 where the quantity vanishes). Raises ValueError if E_G is not finite, d is negative, r is not
positive, k_out is invalid, or the self-consistent iteration does not converge within 400 iterations.

Returns
-------
numpy.ndarray of shape (4, len(k_out)): rows [v_k^2, u_k v_k, E_{k,+} (Ry*), E_{k,-} (Ry*)] at k_out
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def condensate_state(EG: float, d: float, r: float, k_out: np.ndarray) -> np.ndarray:
    '''Self-consistent condensate: occupation, pairing amplitude and quasiparticle bands at given magnitudes.

    Parameters
    ----------
    EG : float
        Bare band gap E_G in Ry*, finite.
    d : float
        Interlayer distance in units of a_B*, >= 0.
    r : float
        Mass ratio m_v/m_c, > 0.
    k_out : np.ndarray
        One-dimensional array of finite non-negative wavevector magnitudes (0 allowed).

    Returns
    -------
    result : np.ndarray
        Shape (4, len(k_out)): rows [v_k^2, u_k v_k, E_{k,+}, E_{k,-}] at k_out, energies in Ry* measured from
        the middle of the bare gap. Trivial state (zeros and bare bands) when E_G >= E_b. Each value converged
        to 1e-9 relative accuracy (1e-9 absolute where it vanishes).

    Raises
    ------
    ValueError
        If E_G is not finite, d is negative or not finite, r is not a positive finite number, k_out is not a
        one-dimensional array of finite non-negative numbers, or the self-consistent iteration does not converge
        within 400 iterations.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _hf(EG, d, r, start=None, tol=1e-13, maxit=400, m_hist=6, beta=0.5, _HF={}):
    """Anderson-accelerated fixed-point solution of Eq. (20) on the trusted nodes for the pair (v^2, uv)."""
    key = (round(float(EG), 12), round(float(d), 12), round(float(r), 12))
    if key in _HF and start is None:
        return _HF[key]
    op = _operator(d)
    op0 = _operator(0.0)
    k, mu = op["k"], op["mu"]
    N = len(k)
    Eb, phi = _exciton(d)
    dmu = _oracle_inverse_compressibility(d)
    hs = 0.5 * k * k * (r - 1) / (r + 1)
    if EG >= Eb:
        z = np.zeros(N)
        Eq = 0.5 * (k * k + EG)
        out = dict(k=k, v2=z, uv=z, F=z, D=z, xi=Eq, Eq=Eq, n=0.0, Ep=hs + Eq, Em=hs - Eq, EG=EG, d=d, r=r, Eb=Eb, dmu=dmu, trivial=True)
        _HF[key] = out
        return out
    if start is None:
        lam = np.sqrt(max((Eb - EG) / dmu, 1e-6))
        x = np.concatenate([np.minimum(lam ** 2 * phi ** 2, 0.5), lam * phi])
    else:
        x = np.concatenate([start["v2"], start["uv"]])

    def _step(x):
        v2, uv = x[:N], x[N:]
        n = np.sum(mu * v2)
        F = 2 * (op0["M"] @ v2)
        D = op["M"] @ uv
        xi = 0.5 * (k * k + EG + 8 * np.pi * d * n - F)
        Eq = np.sqrt(xi * xi + D * D)
        return np.concatenate([0.5 * (1 - xi / Eq), D / (2 * Eq)])

    X, Fh = [], []
    err = np.inf
    for it in range(maxit):
        gx = _step(x)
        f = gx - x
        err = np.max(np.abs(f))
        if err < tol:
            x = gx
            break
        X.append(x.copy())
        Fh.append(f.copy())
        if len(X) > m_hist:
            X.pop(0)
            Fh.pop(0)
        if len(X) >= 2:
            dF = np.array([Fh[j + 1] - Fh[j] for j in range(len(Fh) - 1)]).T
            dX = np.array([X[j + 1] - X[j] for j in range(len(X) - 1)]).T
            gamma, *_ = np.linalg.lstsq(dF, f, rcond=None)
            x = x + beta * f - (dX + beta * dF) @ gamma
        else:
            x = x + beta * f
    if not err < tol:
        raise ValueError("the self-consistent iteration did not converge")
    v2, uv = x[:N], x[N:]
    n = np.sum(mu * v2)
    F = 2 * (op0["M"] @ v2)
    D = op["M"] @ uv
    xi = 0.5 * (k * k + EG + 8 * np.pi * d * n - F)
    Eq = np.sqrt(xi * xi + D * D)
    out = dict(k=k, v2=v2, uv=uv, F=F, D=D, xi=xi, Eq=Eq, n=n, Ep=hs + Eq, Em=hs - Eq, EG=EG, d=d, r=r, Eb=Eb, dmu=dmu, trivial=False)
    if start is None:
        _HF[key] = out
    return out


def _state_at(s, k_out):
    """v^2, uv, E+, E- of a converged state at arbitrary magnitudes, from the self-energies evaluated there."""
    d, r, EG = s["d"], s["r"], s["EG"]
    hs = 0.5 * k_out * k_out * (r - 1) / (r + 1)
    if s["trivial"]:
        Eq = 0.5 * (k_out * k_out + EG)
        z = np.zeros_like(k_out)
        return np.vstack([z, z, hs + Eq, hs - Eq])
    op = _operator(d)
    op0 = _operator(0.0)
    F = 2 * _apply_to_grid_values(_full_values(op0, s["v2"]), k_out, 0.0)
    D = _apply_to_grid_values(_full_values(op, s["uv"]), k_out, d)
    xi = 0.5 * (k_out * k_out + EG + 8 * np.pi * d * s["n"] - F)
    Eq = np.sqrt(xi * xi + D * D)
    return np.vstack([0.5 * (1 - xi / Eq), D / (2 * Eq), hs + Eq, hs - Eq])


def _oracle_condensate_state(EG: float, d: float, r: float, k_out: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    EG = _check_scalar(EG, "EG")
    d = _check_scalar(d, "d", nonneg=True)
    r = _check_scalar(r, "r", positive=True)
    k_out = _check_k_array(k_out, "k_out")
    s = _hf(EG, d, r)
    return _state_at(s, k_out)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: monolayer, equal masses, bare gap set for a target density near 0.025 (the source's figure regime) ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "k_out = np.array([0.0, 0.5, 1.0, 2.0, 5.0])\n",
            "call": "condensate_state(3.848585, 0.0, 1.0, k_out.copy())",
            "gold_call": "_oracle_condensate_state(3.848585, 0.0, 1.0, k_out.copy())",
            "tol": 1e-6,
        },
        # --- Normal: the heterobilayer of the task with a heavier valence band ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "k_out = np.array([0.0, 0.2, 0.8, 1.6, 3.2, 10.0])\n",
            "call": "condensate_state(1.7815, 0.25, 2.0, k_out.copy())",
            "gold_call": "_oracle_condensate_state(1.7815, 0.25, 2.0, k_out.copy())",
            "tol": 1e-6,
        },
        # --- Boundary: a very dilute monolayer condensate (gap just below E_b), light valence band ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "k_out = np.array([0.0, 1.0, 3.0])\n",
            "call": "condensate_state(3.985, 0.0, 0.5, k_out.copy())",
            "gold_call": "_oracle_condensate_state(3.985, 0.0, 0.5, k_out.copy())",
            "tol": 1e-6,
        },
        # --- Boundary: half a Bohr radius, three-to-one masses, the source's unequal-mass regime ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "k_out = np.array([0.0, 0.3, 1.2, 4.0])\n",
            "call": "condensate_state(1.15, 0.5, 3.0, k_out.copy())",
            "gold_call": "_oracle_condensate_state(1.15, 0.5, 3.0, k_out.copy())",
            "tol": 1e-6,
        },
        # --- Edge: gap above the binding energy, only the trivial state exists (zeros and bare bands) ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "k_out = np.array([0.0, 1.0, 2.5])\n",
            "call": "condensate_state(2.3, 0.25, 2.0, k_out.copy())",
            "gold_call": "_oracle_condensate_state(2.3, 0.25, 2.0, k_out.copy())",
            "tol": 1e-6,
        },
        # --- Invalid: a non-positive mass ratio must raise ValueError (0 returned, 1 raised) ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(1.7815, 0.25, 0.0, np.array([0.0, 1.0]))\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    return 0\n",
            "call": "_probe(condensate_state)",
            "gold_call": "_probe(_oracle_condensate_state)",
        },
    ]
