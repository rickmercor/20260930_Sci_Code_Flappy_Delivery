"""
Solve, for a bipartite state, the homogeneous semidefinite program that directly certifies the level-k symmetric-extension entanglement measure, in an unnormalized extension variable, by consensus ADMM.



States admitting a symmetric, partial-transpose-positive extension form a hierarchy of outer approximations to the separable states, refined by the number of extension copies. Writing the measure's own defining domination constraint rho <= (1+t)*sigma in terms of the unnormalized variable omega' := (1+t)*omega removes the bilinear coupling between the scalar t and the extension variable, so the whole problem becomes a single linear-objective semidefinite program in omega' alone, with no outer search over t needed: minimizing Tr(omega') directly returns 1+t at the optimum. Consensus ADMM solves this by alternately projecting a shared candidate onto every constraint set in turn (positive semidefiniteness, every partial-transpose positivity, permutation symmetry, and the domination constraint against the given state) and tracking the constraint-wise disagreement at every cycle, giving a real, per-constraint convergence record rather than a single residual.

Returns
-------
return omega
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def symmetric_extension_sdp_solve(rho: "np.ndarray", dA: int, dB: int, k: int, iters: int) -> "np.ndarray":
    """Solve the homogeneous k-symmetric-extension SDP for a state by consensus ADMM.

    Parameters
    ----------
    rho : np.ndarray
        (dA*dB, dA*dB) complex Hermitian, trace-1 density matrix.
    dA : int
        Dimension of the first subsystem, dA >= 1.
    dB : int
        Dimension of each of the k extended copies of the second subsystem, dB >= 1.
    k : int
        Number of symmetric extension copies, k >= 1.
    iters : int
        Number of ADMM cycles to run, iters >= 1.

    Returns
    -------
    omega : np.ndarray
        (dA*dB**k, dA*dB**k) complex array, the ADMM-converged unnormalized extension
        operator omega'; Tr(omega') - 1 equals the level-k symmetric-extension measure.

    Raises
    ------
    ValueError
        If k < 1 or iters < 1.
    """
    return omega

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from itertools import permutations, combinations
import string


def _partial_transpose_full(rho: "np.ndarray", dims: "list", sys_list: "tuple") -> "np.ndarray":
    n = len(dims)
    shape = dims + dims
    arr = rho.reshape(shape)
    axes = list(range(2 * n))
    for sys in sys_list:
        axes[sys], axes[n + sys] = axes[n + sys], axes[sys]
    arr = np.transpose(arr, axes)
    return arr.reshape(rho.shape)


def _proj_psd(H: "np.ndarray") -> "np.ndarray":
    H = (H + H.conj().T) / 2
    w, v = np.linalg.eigh(H)
    w_clipped = np.clip(w, 0, None)
    return (v * w_clipped) @ v.conj().T


def _proj_psd_on_subset(omega: "np.ndarray", dims: "list", J: "tuple") -> "np.ndarray":
    om2 = _partial_transpose_full(omega, dims, J)
    om2p = _proj_psd(om2)
    return _partial_transpose_full(om2p, dims, J)


def _sym_project_k(omega: "np.ndarray", dims: "list") -> "np.ndarray":
    n = len(dims)
    total = np.zeros_like(omega)
    perms = list(permutations(range(1, n)))
    for perm in perms:
        axes_ket = [0] + list(perm)
        full_perm = axes_ket + [n + a for a in axes_ket]
        shape = dims + dims
        om = omega.reshape(shape)
        om_p = np.transpose(om, full_perm).reshape(omega.shape)
        total = total + om_p
    return total / len(perms)


def _partial_trace_to_AB1(om: "np.ndarray", dims: "list") -> "np.ndarray":
    n = len(dims)
    shape = dims + dims
    arr = om.reshape(shape)
    ket_labels = list(string.ascii_lowercase[:n])
    bra_labels = list(string.ascii_uppercase[:n])
    for i in range(2, n):
        bra_labels[i] = ket_labels[i]
    in_sub = "".join(ket_labels) + "".join(bra_labels)
    out_sub = ket_labels[0] + ket_labels[1] + bra_labels[0] + bra_labels[1]
    result = np.einsum(f"{in_sub}->{out_sub}", arr)
    dA, dB = dims[0], dims[1]
    return result.reshape(dA * dB, dA * dB)


def _proj_domination(omega: "np.ndarray", dims: "list", rho: "np.ndarray") -> "np.ndarray":
    k = len(dims) - 1
    extra = int(np.prod(dims[2:])) if k > 1 else 1
    sigma = _partial_trace_to_AB1(omega, dims)
    M = sigma - rho
    Mp = _proj_psd(M)
    diff = Mp - M
    I_extra = np.eye(extra)
    delta_omega = np.kron(diff, I_extra) / extra
    return omega + delta_omega


def _oracle_symmetric_extension_sdp_solve(rho: "np.ndarray", dA: int, dB: int, k: int, iters: int) -> "np.ndarray":
    if k < 1:
        raise ValueError("k must be at least 1")
    if iters < 1:
        raise ValueError("iters must be at least 1")

    dims = [dA] + [dB] * k
    dtot = int(np.prod(dims))
    subsets = []
    for r in range(1, k + 1):
        for J in combinations(range(1, k + 1), r):
            subsets.append(J)

    projs = [lambda y: _proj_psd(y)]
    for J in subsets:
        projs.append(lambda y, J=J: _proj_psd_on_subset(y, dims, J))
    projs.append(lambda y: _sym_project_k(y, dims))
    projs.append(lambda y: _proj_domination(y, dims, rho))
    m = len(projs)

    x = np.eye(dtot, dtype=complex) / dtot
    z = [x.copy() for _ in range(m)]
    u = [np.zeros((dtot, dtot), dtype=complex) for _ in range(m)]
    grad_c = np.eye(dtot, dtype=complex)
    rho_penalty = 1.0

    for _ in range(iters):
        avg_zu = sum(z[i] - u[i] for i in range(m)) / m
        x = avg_zu - grad_c / (m * rho_penalty)
        x = (x + x.conj().T) / 2
        for i in range(m):
            y = x + u[i]
            z[i] = projs[i](y)
            u[i] = u[i] + x - z[i]

    return x

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: 2-qubit Werner state at p=0.5, k=1; known E_1=0.25, so Tr(omega')~1.25.
        #     The optimal extension is not unique for this state, so the measure
        #     Tr(omega') - 1 and the four constraint flags are compared, not the matrix. ---
        {
            "setup": """
import numpy as np
psi_minus = np.array([0,1,-1,0], dtype=complex)/np.sqrt(2)
proj = np.outer(psi_minus, psi_minus.conj())
rho = 0.5*proj + 0.5*np.eye(4)/4
dA, dB, k = 2, 2, 1
iters = 800
def summarize(fn):
    omega = fn(rho.copy(), dA, dB, k, iters)
    dims = [dA] + [dB] * k
    from itertools import combinations
    subsets = [J for r in range(1, k + 1) for J in combinations(range(1, k + 1), r)]
    tol = 1e-3
    psd_ok = float(np.allclose(_proj_psd(omega), omega, atol=tol))
    pt_ok = float(all(np.allclose(_proj_psd_on_subset(omega, dims, J), omega, atol=tol) for J in subsets))
    sym_ok = float(np.allclose(_sym_project_k(omega, dims), omega, atol=tol))
    dom_ok = float(np.allclose(_proj_domination(omega, dims, rho), omega, atol=tol))
    flags = np.array([psd_ok, pt_ok, sym_ok, dom_ok])
    return np.round(np.concatenate([[np.trace(omega).real - 1.0], flags]), 5)
""",
            "call": "summarize(symmetric_extension_sdp_solve)",
            "gold_call": "summarize(_oracle_symmetric_extension_sdp_solve)",
        },
        # --- Boundary: a product qubit state, k=1; expect Tr(omega')~1 (E_1~0) ---
        {
            "setup": """
import numpy as np
psi0 = np.array([1,0,0,0], dtype=complex)
rho = np.outer(psi0, psi0.conj())
dA, dB, k = 2, 2, 1
iters = 800
def summarize(fn):
    omega = fn(rho.copy(), dA, dB, k, iters)
    dims = [dA] + [dB] * k
    from itertools import combinations
    subsets = [J for r in range(1, k + 1) for J in combinations(range(1, k + 1), r)]
    tol = 1e-3
    psd_ok = float(np.allclose(_proj_psd(omega), omega, atol=tol))
    pt_ok = float(all(np.allclose(_proj_psd_on_subset(omega, dims, J), omega, atol=tol) for J in subsets))
    sym_ok = float(np.allclose(_sym_project_k(omega, dims), omega, atol=tol))
    dom_ok = float(np.allclose(_proj_domination(omega, dims, rho), omega, atol=tol))
    flags = np.array([psd_ok, pt_ok, sym_ok, dom_ok])
    # For this fixture the SDP optimum is unique, so the converged matrix is a
    # well-defined target and comparing it whole, not just its trace, catches
    # a candidate that gets the objective right while violating a constraint.
    return np.round(np.concatenate([omega.ravel(), flags]), 5)
""",
            "call": "summarize(symmetric_extension_sdp_solve)",
            "gold_call": "summarize(_oracle_symmetric_extension_sdp_solve)",
        },
        # --- Edge: a two-qutrit separable product state, k=2; expect Tr(omega')~1 (E_2~0) ---
        {
            "setup": """
import numpy as np
psi0 = np.zeros(9, dtype=complex); psi0[0] = 1.0
rho = np.outer(psi0, psi0.conj())
dA, dB, k = 3, 3, 2
iters = 1500
def summarize(fn):
    omega = fn(rho.copy(), dA, dB, k, iters)
    dims = [dA] + [dB] * k
    from itertools import combinations
    subsets = [J for r in range(1, k + 1) for J in combinations(range(1, k + 1), r)]
    tol = 1e-3
    psd_ok = float(np.allclose(_proj_psd(omega), omega, atol=tol))
    pt_ok = float(all(np.allclose(_proj_psd_on_subset(omega, dims, J), omega, atol=tol) for J in subsets))
    sym_ok = float(np.allclose(_sym_project_k(omega, dims), omega, atol=tol))
    dom_ok = float(np.allclose(_proj_domination(omega, dims, rho), omega, atol=tol))
    flags = np.array([psd_ok, pt_ok, sym_ok, dom_ok])
    # For this fixture the SDP optimum is unique, so the converged matrix is a
    # well-defined target and comparing it whole, not just its trace, catches
    # a candidate that gets the objective right while violating a constraint.
    return np.round(np.concatenate([omega.ravel(), flags]), 5)
""",
            "call": "summarize(symmetric_extension_sdp_solve)",
            "gold_call": "summarize(_oracle_symmetric_extension_sdp_solve)",
        },
        # --- Edge: the 3x3 Horodecki bound-entangled state at a=0.3, k=2. This state is PPT
        #     (so its level-1 measure is exactly zero) yet entangled, and the level-2
        #     measure is strictly positive; it is the only fixture on which the level-2
        #     extension constraints change the answer. The optimum need not be unique, so
        #     the matrix is not compared whole: the returned extension must be a fixed point
        #     of every constraint projection to 3e-4 and its trace must exceed 1.005. ---
        {
            "setup": """
import numpy as np
a = 0.3
rho = np.zeros((9, 9), dtype=complex)
for i in (0, 4, 8):
    for j in (0, 4, 8):
        rho[i, j] = a
for i in (1, 2, 3, 5, 7):
    rho[i, i] = a
rho[6, 6] = (1 + a) / 2
rho[8, 8] = (1 + a) / 2
rho[6, 8] = np.sqrt(1 - a * a) / 2
rho[8, 6] = np.sqrt(1 - a * a) / 2
rho = rho / (8 * a + 1)
dA, dB, k = 3, 3, 2
iters = 2500
def summarize(fn):
    omega = fn(rho.copy(), dA, dB, k, iters)
    dims = [dA] + [dB] * k
    from itertools import combinations
    subsets = [J for r in range(1, k + 1) for J in combinations(range(1, k + 1), r)]
    tol = 3e-4
    psd_ok = float(np.allclose(_proj_psd(omega), omega, atol=tol))
    pt_ok = float(all(np.allclose(_proj_psd_on_subset(omega, dims, J), omega, atol=tol) for J in subsets))
    sym_ok = float(np.allclose(_sym_project_k(omega, dims), omega, atol=tol))
    dom_ok = float(np.allclose(_proj_domination(omega, dims, rho), omega, atol=tol))
    detected = float(np.trace(omega).real - 1.0 > 0.005)
    return np.array([psd_ok, pt_ok, sym_ok, dom_ok, detected])
""",
            "call": "summarize(symmetric_extension_sdp_solve)",
            "gold_call": "summarize(_oracle_symmetric_extension_sdp_solve)",
        },
        # --- Edge: 2-qubit Werner state at p=0.7, k=3; known E_3=E_1=(3p-1)/2=0.55, so
        #     Tr(omega')~1.55. Exercises the general-k paths (six copy permutations, seven
        #     partial-transpose subsets, a partial trace over two copies). The optimal
        #     extension is not unique for this state, so the measure Tr(omega') - 1 and
        #     the four constraint flags are compared, not the matrix. ---
        {
            "setup": """
import numpy as np
psi_minus = np.array([0,1,-1,0], dtype=complex)/np.sqrt(2)
proj = np.outer(psi_minus, psi_minus.conj())
rho = 0.7*proj + 0.3*np.eye(4)/4
dA, dB, k = 2, 2, 3
iters = 3000
def summarize(fn):
    omega = fn(rho.copy(), dA, dB, k, iters)
    dims = [dA] + [dB] * k
    from itertools import combinations
    subsets = [J for r in range(1, k + 1) for J in combinations(range(1, k + 1), r)]
    tol = 1e-3
    psd_ok = float(np.allclose(_proj_psd(omega), omega, atol=tol))
    pt_ok = float(all(np.allclose(_proj_psd_on_subset(omega, dims, J), omega, atol=tol) for J in subsets))
    sym_ok = float(np.allclose(_sym_project_k(omega, dims), omega, atol=tol))
    dom_ok = float(np.allclose(_proj_domination(omega, dims, rho), omega, atol=tol))
    flags = np.array([psd_ok, pt_ok, sym_ok, dom_ok])
    return np.round(np.concatenate([[np.trace(omega).real - 1.0], flags]), 5)
""",
            "call": "summarize(symmetric_extension_sdp_solve)",
            "gold_call": "summarize(_oracle_symmetric_extension_sdp_solve)",
        },
    ]
