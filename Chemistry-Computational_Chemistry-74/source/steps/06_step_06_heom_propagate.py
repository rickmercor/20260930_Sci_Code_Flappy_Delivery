"""
Hierarchical equations of motion for a system coupled to several independent harmonic baths whose correlation functions are finite sums of exponentials with real or complex rates: propagation of the reduced density matrix with fixed-step fourth-order Runge-Kutta.

The system, with Liouvillian L from step 05, couples to B independent harmonic baths through sum over b of Q_b V_b, with Hermitian system operators V_b from step 04. Each bath b has the correlation function C_b(t) = <Q_b(t) Q_b(0)> / hbar^2 = sum over its terms k of c_k exp(-gamma_k t), with the amplitudes and rates of steps 01 and 02; the rates of each bath may be complex, and the set of rates of each bath is closed under complex conjugation. All baths start in equilibrium and uncorrelated with the system.

In the hierarchical equations of motion the reduced density matrix is the root of a hierarchy of auxiliary density operators labelled by vectors of non-negative integers, one entry per exponential term of every bath; each operator is coupled to the operators whose label differs by one in a single entry. Build the hierarchy so that, with no label discarded, it reproduces the influence of each bath exactly while using only the exponentials exp(-gamma_k t) of that bath. The system Liouvillian, including the photon loss, acts on every auxiliary operator in the same way as on the reduced density matrix. Keep every label whose entries sum to at most the given depth and set all deeper operators to zero, with no terminator correction. All auxiliary operators start at zero.

Integrate the full hierarchy with the classical fourth-order Runge-Kutta scheme at the fixed step dt, and record the reduced density matrix at step numbers 0, stride, 2 stride, ..., n_steps.

Returns
-------
complex numpy.ndarray of shape (n_steps // stride + 1, d, d): the reduced density matrix at the sampled times
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def heom_propagate(liouvillian: "np.ndarray", coupling_ops: "np.ndarray", exponents: "np.ndarray", bath_index: "np.ndarray", depth: int, rho0: "np.ndarray", dt_ps: float, n_steps: int, stride: int) -> "np.ndarray":
    '''Reduced density matrix from the hierarchical equations of motion with several baths.

    Parameters
    ----------
    liouvillian : numpy.ndarray
        (d^2, d^2) system superoperator in ps^-1 acting on row-major vectorised
        d x d matrices (step 05).
    coupling_ops : numpy.ndarray
        (B, d, d) array of Hermitian system operators V_b, one per bath.
    exponents : numpy.ndarray
        (K, 4) array of rows (Re c_k, Im c_k, Re gamma_k, Im gamma_k), in ps^-2
        and ps^-1, for all exponential terms of all baths.
    bath_index : numpy.ndarray
        (K,) integer array; bath_index[k] is the bath b to which term k belongs,
        so C_b(t) = sum over k with bath_index[k] = b of c_k exp(-gamma_k t).
    depth : int
        Hierarchy depth: auxiliary operators with index sum <= depth are kept.
    rho0 : numpy.ndarray
        (d, d) initial reduced density matrix.
    dt_ps : float
        Runge-Kutta time step in ps.
    n_steps : int
        Number of steps, a non-negative multiple of stride.
    stride : int
        Output every stride steps.

    Returns
    -------
    frames : numpy.ndarray
        Complex array of shape (n_steps // stride + 1, d, d): the reduced density
        matrix at times 0, stride * dt, ..., n_steps * dt.

    Raises
    ------
    ValueError
        If the shapes of liouvillian, coupling_ops and rho0 are inconsistent, a
        coupling operator is not Hermitian (tolerance 1e-10), exponents is not a
        finite array of shape (K, 4) with K >= 1, bath_index does not assign each
        term to an existing bath, the rates of some bath are not closed under
        complex conjugation (to within 1e-9 max(1, |gamma|)), depth is not a
        non-negative integer, dt_ps is not positive and finite, or n_steps is not
        a non-negative integer multiple of the positive integer stride.
    '''
    return frames

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_heom_propagate(liouvillian: "np.ndarray", coupling_ops: "np.ndarray", exponents: "np.ndarray",
                           bath_index: "np.ndarray", depth: int, rho0: "np.ndarray", dt_ps: float,
                           n_steps: int, stride: int) -> "np.ndarray":
    import numpy as np
    lv = np.asarray(liouvillian, dtype=complex)
    ops = np.asarray(coupling_ops, dtype=complex)
    ex = np.asarray(exponents, dtype=float)
    bi = np.asarray(bath_index)
    r0 = np.asarray(rho0, dtype=complex)
    if ops.ndim != 3 or ops.shape[1] != ops.shape[2] or ops.shape[0] < 1:
        raise ValueError("coupling_ops must have shape (B, d, d)")
    n_bath, d = ops.shape[0], ops.shape[1]
    if lv.shape != (d * d, d * d) or r0.shape != (d, d):
        raise ValueError("liouvillian, coupling operators and rho0 dimensions do not match")
    for v in ops:
        if not np.allclose(v, v.conj().T, rtol=0.0, atol=1e-10):
            raise ValueError("coupling operators must be Hermitian")
    if ex.ndim != 2 or ex.shape[1] != 4 or ex.shape[0] < 1 or not np.all(np.isfinite(ex)):
        raise ValueError("exponents must be a finite array of shape (K, 4)")
    if bi.shape != (ex.shape[0],) or not np.all(np.equal(np.mod(bi, 1), 0)) or bi.min() < 0 or bi.max() >= n_bath:
        raise ValueError("bath_index must hold one valid bath number per exponential term")
    bi = bi.astype(int)
    amp = ex[:, 0] + 1j * ex[:, 1]
    rate = ex[:, 2] + 1j * ex[:, 3]
    conj_amp = np.zeros_like(amp)
    for k in range(ex.shape[0]):
        same = np.where(bi == bi[k])[0]
        gap = np.abs(rate[same] - np.conj(rate[k]))
        if gap.min() > 1e-9 * max(1.0, abs(rate[k])):
            raise ValueError("the rates of each bath must be closed under complex conjugation")
        conj_amp[k] = np.conj(amp[same[int(np.argmin(gap))]])
    if int(depth) != depth or depth < 0:
        raise ValueError("depth must be a non-negative integer")
    if not np.isfinite(dt_ps) or dt_ps <= 0.0:
        raise ValueError("time step must be positive")
    if int(n_steps) != n_steps or n_steps < 0 or int(stride) != stride or stride < 1 or n_steps % stride:
        raise ValueError("n_steps must be a non-negative multiple of the positive integer stride")
    depth, n_steps, stride = int(depth), int(n_steps), int(stride)
    import scipy.sparse as sp
    n_mode = ex.shape[0]
    index = [()]
    for _ in range(n_mode):
        index = [t + (j,) for t in index for j in range(depth + 1)]
    index = sorted([t for t in index if sum(t) <= depth], key=lambda t: (sum(t), t))
    where = {t: i for i, t in enumerate(index)}
    n_ado = len(index)
    dd = d * d
    eye = np.eye(d)
    left = [sp.csr_matrix(np.kron(ops[bi[k]], eye)) for k in range(n_mode)]
    right = [sp.csr_matrix(np.kron(eye, ops[bi[k]].T)) for k in range(n_mode)]
    ident = sp.identity(dd, format='csr', dtype=complex)
    l_sys = sp.csr_matrix(lv)
    blocks = {}
    for i, t in enumerate(index):
        blocks[(i, i)] = l_sys - complex(sum(t[k] * rate[k] for k in range(n_mode))) * ident
        for k in range(n_mode):
            j = where.get(t[:k] + (t[k] + 1,) + t[k + 1:])
            if j is not None:
                blocks[(i, j)] = blocks.get((i, j), 0) - 1j * (left[k] - right[k])
            if t[k] > 0:
                j = where[t[:k] + (t[k] - 1,) + t[k + 1:]]
                blocks[(i, j)] = blocks.get((i, j), 0) - 1j * t[k] * (amp[k] * left[k] - conj_amp[k] * right[k])
    gen = sp.bmat([[blocks.get((i, j)) for j in range(n_ado)] for i in range(n_ado)], format='csr')
    vec = np.zeros(n_ado * dd, dtype=complex)
    vec[:dd] = r0.reshape(-1)
    frames = [r0.copy()]
    for s in range(1, n_steps + 1):
        k1 = gen @ vec
        k2 = gen @ (vec + 0.5 * dt_ps * k1)
        k3 = gen @ (vec + 0.5 * dt_ps * k2)
        k4 = gen @ (vec + dt_ps * k3)
        vec = vec + dt_ps / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if s % stride == 0:
            frames.append(vec[:dd].reshape(d, d).copy())
    return np.array(frames)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: a donor-acceptor pair coupled at the acceptor to one bath with two real-rate terms, depth 4 ---
        {
            "setup": """import numpy as np
W = 2.0 * np.pi * 2.99792458e-2
H = np.array([[0.0, 30.0], [30.0, -100.0]])
L = -1j * W * (np.kron(H, np.eye(2)) - np.kron(np.eye(2), H.T))
V = np.diag([0.0, 1.0]).astype(complex)[None, :, :]
rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
ex = np.array([[700.0, -90.0, 10.0, 0.0], [60.0, 0.0, 247.0, 0.0]])
bi = np.array([0, 0])
depth = 4
dt = 0.001
n_steps = 400
stride = 100
""",
            "call": "heom_propagate(L.copy(), V.copy(), ex.copy(), bi.copy(), depth, rho0.copy(), dt, n_steps, stride)",
            "gold_call": "_oracle_heom_propagate(L.copy(), V.copy(), ex.copy(), bi.copy(), depth, rho0.copy(), dt, n_steps, stride)",
            "tol": 1e-09,
        },
        # --- Normal: electron-photon system with loss and two baths, the second with a complex-rate pair, depth 3 ---
        {
            "setup": """import numpy as np
W = 2.0 * np.pi * 2.99792458e-2
H = np.array([[0.0, 0.0, 25.0, 8.0], [0.0, 90.0, 8.0, 25.0], [25.0, 8.0, -80.0, 0.0], [8.0, 25.0, 0.0, 10.0]], dtype=complex)
a = np.kron(np.eye(2), np.array([[0.0, 1.0], [0.0, 0.0]]))
I = np.eye(4)
k = 15.0 * W
L = -1j * W * (np.kron(H, I) - np.kron(I, H.T)) + k * (np.kron(a, a) - 0.5 * np.kron(a.T @ a, I) - 0.5 * np.kron(I, (a.T @ a).T))
V1 = np.diag([0.0, 0.0, 1.0, 1.0]).astype(complex)
V2 = np.array([[0.5, 0.3j, 0.0, 0.2], [-0.3j, 0.0, 0.1, 0.0], [0.0, 0.1, -0.4, 0.3j], [0.2, 0.0, -0.3j, 0.0]])
V = np.array([V1, V2])
ex = np.array([[420.0, -55.0, 14.0, 0.0], [35.0, 0.0, 180.0, 0.0], [30.0, 2.0, 6.0, -80.0], [5.0, -2.0, 6.0, 80.0]])
bi = np.array([0, 0, 1, 1])
rho0 = np.zeros((4, 4), dtype=complex)
rho0[0, 0] = 1.0
depth = 3
dt = 0.001
n_steps = 300
stride = 50
""",
            "call": "heom_propagate(L.copy(), V.copy(), ex.copy(), bi.copy(), depth, rho0.copy(), dt, n_steps, stride)",
            "gold_call": "_oracle_heom_propagate(L.copy(), V.copy(), ex.copy(), bi.copy(), depth, rho0.copy(), dt, n_steps, stride)",
            "tol": 1e-09,
        },
        # --- Boundary: depth 0, only the system Liouvillian acts ---
        {
            "setup": """import numpy as np
W = 2.0 * np.pi * 2.99792458e-2
H = np.array([[0.0, 30.0], [30.0, -100.0]])
L = -1j * W * (np.kron(H, np.eye(2)) - np.kron(np.eye(2), H.T))
V = np.diag([0.0, 1.0]).astype(complex)[None, :, :]
rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
ex = np.array([[700.0, -90.0, 10.0, 0.0], [60.0, 0.0, 247.0, 0.0]])
bi = np.array([0, 0])
depth = 0
dt = 0.001
n_steps = 200
stride = 40
""",
            "call": "heom_propagate(L.copy(), V.copy(), ex.copy(), bi.copy(), depth, rho0.copy(), dt, n_steps, stride)",
            "gold_call": "_oracle_heom_propagate(L.copy(), V.copy(), ex.copy(), bi.copy(), depth, rho0.copy(), dt, n_steps, stride)",
            "tol": 1e-09,
        },
        # --- Edge: one bath made only of a complex-rate pair, with a coherent superposition as initial state ---
        {
            "setup": """import numpy as np
W = 2.0 * np.pi * 2.99792458e-2
H = np.array([[0.0, 30.0], [30.0, -100.0]])
L = -1j * W * (np.kron(H, np.eye(2)) - np.kron(np.eye(2), H.T))
V = np.diag([0.0, 1.0]).astype(complex)[None, :, :]
ex = np.array([[40.0, 3.0, 8.0, -120.0], [4.0, -3.0, 8.0, 120.0]])
bi = np.array([0, 0])
rho0 = 0.5 * np.array([[1.0, 1.0j], [-1.0j, 1.0]])
depth = 4
dt = 0.0005
n_steps = 400
stride = 80
""",
            "call": "heom_propagate(L.copy(), V.copy(), ex.copy(), bi.copy(), depth, rho0.copy(), dt, n_steps, stride)",
            "gold_call": "_oracle_heom_propagate(L.copy(), V.copy(), ex.copy(), bi.copy(), depth, rho0.copy(), dt, n_steps, stride)",
            "tol": 1e-09,
        },
        # --- Edge: two baths whose oscillating pairs share the same complex rates, with the terms of the two baths interleaved ---
        {
            "setup": """import numpy as np
W = 2.0 * np.pi * 2.99792458e-2
H = np.array([[0.0, 0.0, 28.0, 6.0], [0.0, 95.0, 6.0, 28.0], [28.0, 6.0, -90.0, 0.0], [6.0, 28.0, 0.0, 5.0]], dtype=complex)
a = np.kron(np.eye(2), np.array([[0.0, 1.0], [0.0, 0.0]]))
I = np.eye(4)
k = 12.0 * W
L = -1j * W * (np.kron(H, I) - np.kron(I, H.T)) + k * (np.kron(a, a) - 0.5 * np.kron(a.T @ a, I) - 0.5 * np.kron(I, (a.T @ a).T))
V1 = np.diag([0.0, 0.0, 1.0, 1.0]).astype(complex)
V2 = np.array([[0.0, 0.4j, 0.0, 0.25], [-0.4j, 0.0, 0.25, 0.0], [0.0, 0.25, 0.0, 0.4j], [0.25, 0.0, -0.4j, 0.0]])
V = np.array([V1, V2])
ex = np.array([[22.0, 1.5, 6.0, -90.0], [9.0, -4.0, 6.0, 90.0], [510.0, -60.0, 12.0, 0.0], [3.0, 4.0, 6.0, -90.0], [2.0, -1.5, 6.0, 90.0], [14.0, 0.0, 150.0, 0.0]])
bi = np.array([0, 1, 0, 1, 0, 0])
rho0 = np.zeros((4, 4), dtype=complex)
rho0[0, 0] = 1.0
depth = 3
dt = 0.0005
n_steps = 400
stride = 100
""",
            "call": "heom_propagate(L.copy(), V.copy(), ex.copy(), bi.copy(), depth, rho0.copy(), dt, n_steps, stride)",
            "gold_call": "_oracle_heom_propagate(L.copy(), V.copy(), ex.copy(), bi.copy(), depth, rho0.copy(), dt, n_steps, stride)",
            "tol": 1e-09,
        },
        # --- Invalid: a complex rate without its conjugate partner ---
        {
            "setup": """import numpy as np
W = 2.0 * np.pi * 2.99792458e-2
H = np.array([[0.0, 30.0], [30.0, -100.0]])
L = -1j * W * (np.kron(H, np.eye(2)) - np.kron(np.eye(2), H.T))
V = np.diag([0.0, 1.0]).astype(complex)[None, :, :]
rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
ex = np.array([[40.0, 3.0, 8.0, -120.0]])
bi = np.array([0])
depth = 2
dt = 0.001
n_steps = 10
stride = 5
def run_model():
    try:
        heom_propagate(L.copy(), V.copy(), ex.copy(), bi.copy(), depth, rho0.copy(), dt, n_steps, stride)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_heom_propagate(L.copy(), V.copy(), ex.copy(), bi.copy(), depth, rho0.copy(), dt, n_steps, stride)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
