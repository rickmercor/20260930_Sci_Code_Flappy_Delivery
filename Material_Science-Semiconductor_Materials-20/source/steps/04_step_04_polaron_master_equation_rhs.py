"""
Step 04 - Polaron master equation of the dot-cavity system.

Polaron-frame master equation of a biexciton-cascade quantum dot in a bimodal cavity.

The dot has a ground state G, two linearly polarised excitons XH and XV and a biexciton
XX. A cavity with two degenerate modes, H and V (annihilation operators a_H, a_V), is
tuned to half the biexciton transition frequency. In the frame rotating at that
frequency the Hamiltonian is

    H_S = (E_B/2 + delta/2) |XH><XH| + (E_B/2 - delta/2) |XV><XV|
          + g (sigma_H a_H^dag + sigma_V a_V^dag) + h.c.,     sigma_k = |G><X_k| + |X_k><XX|,

with G and XX at zero energy, E_B the biexciton binding energy, delta the exciton
fine-structure splitting and g the bare dot-cavity coupling. Each cavity mode loses
photons into the environment at the rate gamma, through the Lindblad term
gamma (a rho a^dag - {a^dag a, rho}/2) of that mode. The lattice couples to the dot through
sum_k (g_k b_k^dag + g_k^* b_k) O, where O = |XH><XH| + |XV><XV| + 2 |XX><XX| counts excitons,
with the spectral density of step 01.

The lattice is treated in the polaron frame of steps 02 and 03: after the transformation
the residual dot-lattice coupling is linear in the bath operators B_x and B_y of step 03.
The energies E_B/2 +- delta/2 given here are the phonon-renormalised (polaron-frame)
values, so no polaron shift is added to them. The dissipative part is the time-local
second-order (Born-Markov) expansion in the residual coupling, with the bath correlation
functions of step 03, the memory integrals extended to infinity and the dot-cavity
operators evolved under the full phonon-dressed Hamiltonian, and no approximation beyond
that: the dissipator is not secularised and its energy-shift parts are kept. The cavity
losses are added as above.

Neither H_S, the lattice coupling nor the losses can raise the number of excitations
(excitons plus photons), so states |dot, n_H, n_V> with at most two excitations form a
closed space. The density matrix is represented on these 13 states in the fixed order

     0: |G,0,0>
     1: |XH,0,0>   2: |XV,0,0>   3: |G,1,0>    4: |G,0,1>
     5: |XX,0,0>   6: |XH,1,0>   7: |XV,0,1>   8: |G,2,0>    9: |G,0,2>
    10: |XH,0,1>  11: |XV,1,0>  12: |G,1,1>

and this step returns d rho / dt. The map is linear and must be applied to any complex
13 x 13 matrix, Hermitian or not, and to stacks of such matrices.

Parameters are grouped as dot = [E_B (meV), delta (meV), g (meV), gamma (1/ps)] and the
lattice array of step 01. Use hbar = 0.6582119569 meV ps; energies enter the equation of
motion as angular frequencies E / hbar in rad/ps. Results must be converged to at least
eight significant figures.

Returns
-------
numpy.ndarray of shape (2, 13, 13) (or (m, 2, 13, 13) for a stack): real and imaginary parts of d rho/dt in 1/ps
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def polaron_master_equation_rhs(rho: np.ndarray, T: float, dot: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    '''Time derivative of the dot-cavity density matrix under the polaron master equation.

    Parameters
    ----------
    rho : np.ndarray
        Complex (or real) array of shape (13, 13), or a stack of shape (m, 13, 13), in the
        basis order given in the step background.
    T : float
        Lattice temperature in K, > 0.
    dot : np.ndarray
        Shape (4,): [E_B (meV), delta (meV), g (meV) >= 0, gamma (1/ps) >= 0].
    lattice : np.ndarray
        Shape (6,): [D_e (eV), D_h (eV), mass density (kg/m^3), c_s (m/s), a_e (nm), a_h (nm)].

    Returns
    -------
    drho_dt : np.ndarray
        For a (13, 13) input, shape (2, 13, 13): real and imaginary parts of d rho / dt
        in 1/ps. For an (m, 13, 13) input, shape (m, 2, 13, 13).

    Raises
    ------
    ValueError
        If rho is not a numeric array of shape (13, 13) or (m, 13, 13) with finite
        entries, if T is not finite and positive, if dot is not a numpy array of four
        finite real numbers with g >= 0 and gamma >= 0, or if lattice is invalid (as in
        step 01).
    '''
    return drho_dt

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_dot(dot, need_loss):
    """Validate dot = [E_B, delta, g, gamma]; return the four floats."""
    arr = np.asarray(dot)
    if not isinstance(dot, np.ndarray) or arr.shape != (4,) or arr.dtype.kind not in "iuf":
        raise ValueError("dot must be a numpy array of four real numbers")
    arr = arr.astype(float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("dot entries must be finite")
    eb, dfs, g, gamma = (float(x) for x in arr)
    if g < 0.0 or gamma < 0.0 or (need_loss and (gamma <= 0.0 or g <= 0.0)):
        raise ValueError("g and gamma must be >= 0, and both > 0 where the emission must complete")
    return eb, dfs, g, gamma


def _cavity_space():
    """States (dot, n_H, n_V) of the 13-state space, the coupled pairs and the photon operators."""
    states = [("G", 0, 0), ("XH", 0, 0), ("XV", 0, 0), ("G", 1, 0), ("G", 0, 1),
              ("XX", 0, 0), ("XH", 1, 0), ("XV", 0, 1), ("G", 2, 0), ("G", 0, 2),
              ("XH", 0, 1), ("XV", 1, 0), ("G", 1, 1)]
    index = {s: i for i, s in enumerate(states)}
    pairs = []                      # (upper index, lower index, sqrt(photon number after emission))
    for s, i in index.items():
        dot, nh, nv = s
        for mode, lower in (("H", {"XX": "XH", "XH": "G"}), ("V", {"XX": "XV", "XV": "G"})):
            if dot not in lower:
                continue
            nh2, nv2 = nh + (mode == "H"), nv + (mode == "V")
            j = index.get((lower[dot], nh2, nv2))
            if j is not None:
                pairs.append((i, j, np.sqrt(nh2 if mode == "H" else nv2)))
    ah = np.zeros((13, 13))
    av = np.zeros((13, 13))
    for s, i in index.items():
        dot, nh, nv = s
        if nh:
            ah[index[(dot, nh - 1, nv)], i] = np.sqrt(nh)
        if nv:
            av[index[(dot, nh, nv - 1)], i] = np.sqrt(nv)
    return states, pairs, ah, av


def _oracle_polaron_master_equation_rhs(rho: np.ndarray, T: float, dot: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    r = np.asarray(rho)
    if r.dtype.kind not in "iufc" or r.ndim not in (2, 3) or r.shape[-2:] != (13, 13):
        raise ValueError("rho must be a numeric array of shape (13, 13) or (m, 13, 13)")
    if not np.all(np.isfinite(r)):
        raise ValueError("rho entries must be finite")
    single = r.ndim == 2
    r = r.astype(complex).reshape((-1, 13, 13))
    T = _check_temperature(T)
    eb, dfs, g, gamma = _check_dot(dot, need_loss=False)
    hbar = 0.6582119569
    bmean = float(np.exp(-0.5 * _oracle_phonon_propagator(np.array([0.0]), T, lattice)[0, 0]))
    states, pairs, ah, av = _cavity_space()
    level = {"G": 0.0, "XX": 0.0, "XH": eb / 2 + dfs / 2, "XV": eb / 2 - dfs / 2}
    ham = np.diag([level[s[0]] / hbar for s in states]).astype(complex)
    xx = np.zeros((13, 13), dtype=complex)
    xy = np.zeros((13, 13), dtype=complex)
    for i, j, amp in pairs:
        c = g * amp / hbar
        ham[i, j] += bmean * c
        ham[j, i] += bmean * c
        xx[i, j] += c
        xx[j, i] += c
        xy[i, j] += 1j * c
        xy[j, i] -= 1j * c
    evals, vecs = np.linalg.eigh(ham)
    dw = evals[None, :] - evals[:, None]          # element (i, j): omega_j - omega_i
    kap = _oracle_polaron_response_functions(dw.ravel(), T, lattice)
    kx = (kap[0] + 1j * kap[1]).reshape(13, 13)
    ky = (kap[2] + 1j * kap[3]).reshape(13, 13)
    out = -1j * (ham @ r - r @ ham)
    for xop, k in ((xx, kx), (xy, ky)):
        xe = vecs.conj().T @ xop @ vecs
        dop = vecs @ (2.0 * k.real * xe) @ vecs.conj().T
        sop = vecs @ (k.imag * xe) @ vecs.conj().T
        a = dop @ r - r @ dop.conj().T
        b = sop @ r + r @ sop.conj().T
        out = out - 0.5 * (xop @ a - a @ xop) - 1j * (xop @ b - b @ xop)
    for op in (ah, av):
        num = op.T @ op
        out = out + gamma * (op @ r @ op.T - 0.5 * (num @ r + r @ num))
    res = np.stack([out.real, out.imag], axis=1)
    return res[0] if single else res

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: benchmark dot and cavity at 20 K, a random density matrix of unit trace ---
        {
            "setup": "import numpy as np\n"
                     "rng = np.random.default_rng(11)\n"
                     "A = rng.normal(size=(13, 13)) + 1j * rng.normal(size=(13, 13))\n"
                     "R = A @ A.conj().T\n"
                     "R = R / np.trace(R)\n"
                     "dot = np.array([1.2, 0.15, 0.2, 0.3])\n"
                     "lat = np.array([7.0, -3.5, 5370.0, 5110.0, 3.5, 3.5 / 1.15])\n",
            "call": "polaron_master_equation_rhs(R.copy(), 20.0, dot.copy(), lat.copy())",
            "gold_call": "_oracle_polaron_master_equation_rhs(R.copy(), 20.0, dot.copy(), lat.copy())",
            "tol": 1e-7,
        },
        # --- Boundary: cold lattice, no fine-structure splitting, a non-Hermitian input ---
        {
            "setup": "import numpy as np\n"
                     "rng = np.random.default_rng(5)\n"
                     "R = rng.normal(size=(13, 13)) + 1j * rng.normal(size=(13, 13))\n"
                     "dot = np.array([1.5, 0.0, 0.1, 0.25])\n"
                     "lat = np.array([7.0, -3.5, 5370.0, 5110.0, 3.0, 3.0 / 1.15])\n",
            "call": "polaron_master_equation_rhs(R.copy(), 2.0, dot.copy(), lat.copy())",
            "gold_call": "_oracle_polaron_master_equation_rhs(R.copy(), 2.0, dot.copy(), lat.copy())",
            "tol": 1e-7,
        },
        # --- Edge: strong dot-cavity mixing in a hot lattice, a stack of three matrices ---
        {
            "setup": "import numpy as np\n"
                     "rng = np.random.default_rng(23)\n"
                     "S = rng.normal(size=(3, 13, 13)) + 1j * rng.normal(size=(3, 13, 13))\n"
                     "dot = np.array([0.6, 0.05, 0.35, 0.2])\n"
                     "lat = np.array([7.0, -3.5, 5370.0, 5110.0, 2.5, 2.5 / 1.15])\n",
            "call": "polaron_master_equation_rhs(S.copy(), 45.0, dot.copy(), lat.copy())",
            "gold_call": "_oracle_polaron_master_equation_rhs(S.copy(), 45.0, dot.copy(), lat.copy())",
            "tol": 1e-7,
        },
        # --- Edge: lossless cavity (gamma = 0), large binding energy and splitting, real symmetric input ---
        {
            "setup": "import numpy as np\n"
                     "rng = np.random.default_rng(2)\n"
                     "R = rng.normal(size=(13, 13))\n"
                     "R = R + R.T\n"
                     "dot = np.array([2.0, 0.3, 0.12, 0.0])\n"
                     "lat = np.array([7.0, -3.5, 5370.0, 5110.0, 4.5, 4.5 / 1.15])\n",
            "call": "polaron_master_equation_rhs(R.copy(), 10.0, dot.copy(), lat.copy())",
            "gold_call": "_oracle_polaron_master_equation_rhs(R.copy(), 10.0, dot.copy(), lat.copy())",
            "tol": 1e-7,
        },
        # --- Invalid: wrong matrix size ---
        {
            "setup": "import numpy as np\n"
                     "dot = np.array([1.2, 0.15, 0.2, 0.3])\n"
                     "lat = np.array([7.0, -3.5, 5370.0, 5110.0, 3.5, 3.0])\n"
                     "def run_model():\n"
                     "    try:\n"
                     "        polaron_master_equation_rhs(np.eye(5, dtype=complex), 10.0, dot.copy(), lat.copy())\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "def run_gold():\n"
                     "    try:\n"
                     "        _oracle_polaron_master_equation_rhs(np.eye(5, dtype=complex), 10.0, dot.copy(), lat.copy())\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
