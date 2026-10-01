"""
Advance the classical envelope and both quantum Bogoliubov maps in one coupled QSSF step, using the prescribed frozen field and the distinct annihilation and creation Fourier bases.

Advance the classical envelope and both quantum maps in one QSSF step.



Implement the prescribed coupled discrete evolution, returning the updated

envelope together with the normal and anomalous Bogoliubov matrices. Both

quantum blocks must use the same incoming state and frozen classical field.



The envelope A is in the time basis. U and V are in the unshifted frequency

basis, with a(z) = U a(0) + V a(0)^dagger. P is a spectral linear half-step.

Use the unitary forward DFT F[j,k] = exp(-2*pi*i*j*k/N)/sqrt(N). Apply a

linear half-step, the exact local Kerr phase exp(i*gamma*|A_half|^2*dz),

then a second linear half-step to A. Freeze A_half, the field immediately

after the FIRST linear half-step, during the quantum nonlinear substep.



The local fluctuation pair (a, a^dagger) at each time sample evolves by

exp(-i*H*dz), where H = [[-alpha,-mu],[conj(mu),alpha]],

alpha = 2*gamma*|A_half|^2 and mu = gamma*A_half^2. This is the prescribed

frozen-generator map, including its continuous zero-coupling limit; it is

not the tangent map of the exact classical Kerr phase. Enclose this local

map in the same two spectral half-steps. Spectral propagation acts on the

output rows of both frequency-domain maps. When representing their input

and output modes in time, use U_t = F^dagger U_omega F and

V_t = F^dagger V_omega conj(F). Creation modes use the conjugate basis.

Compose the local map with the incoming pair before transforming back.



This algebraic operation also accepts finite noncanonical U,V and finite

nonunit-modulus P for diagnostics; do not impose extra canonical or modulus

tests. A and P must be finite nonempty vectors of the same length N, and U

and V finite matrices of shape (N,N). Odd N is supported. dz must be finite,

real and nonnegative; gamma finite and real, including zero or negative.

Invalid inputs raise ValueError. No caller inputs may be modified, including

read-only arrays. A supplied P is used as given, even when dz=0.



Return a complex array of shape (2*N+1,N): row 0 is A_new, rows 1 through N

are U_new, and the final N rows are V_new. There is no intermediate-field

output and no requirement on a particular FFT or exponential algorithm.

Returns
-------
complex ndarray (2*N+1,N): row 0 A_new; rows 1:N+1 U_new; rows N+1: V_new.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def qssf_coupled_step(
    A: "np.ndarray",
    U: "np.ndarray",
    V: "np.ndarray",
    P: "np.ndarray",
    dz: float,
    gamma: float = 1.0,
) -> "np.ndarray":
    """Return the jointly propagated classical and quantum state.

    Parameters
    ----------
    A, P : "np.ndarray"
        Time envelope and spectral half-step, each of shape (N,).
    U, V : "np.ndarray"
        Frequency-basis Bogoliubov matrices, each of shape (N,N).
    dz : float
        Finite nonnegative propagation distance.
    gamma : float, default 1.0
        Finite real Kerr coefficient.

    Returns
    -------
    result : "np.ndarray"
        Complex (2*N+1,N) array: A_new row, U_new block, V_new block.

    Raises
    ------
    ValueError
        For invalid shape, nonfinite data or invalid scalar parameters.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_qssf_coupled_step(
    A: "np.ndarray",
    U: "np.ndarray",
    V: "np.ndarray",
    P: "np.ndarray",
    dz: float,
    gamma: float = 1.0,
) -> "np.ndarray":
    A = _qssf_array(A, 1)
    P = _qssf_array(P, 1)
    U, V = _qssf_pair(U, V)
    dz = _qssf_scalar(dz, nonnegative=True)
    gamma = _qssf_scalar(gamma)
    if A.shape != P.shape or U.shape != (A.size, A.size):
        raise ValueError("Envelope, propagator and maps must match")

    A_half = np.fft.ifft(P * np.fft.fft(A))
    intensity = np.abs(A_half) ** 2
    A_new = np.fft.ifft(
        P * np.fft.fft(A_half * np.exp(1j * gamma * intensity * dz))
    )
    alpha = 2.0 * gamma * intensity
    mu = gamma * A_half**2
    kappa = np.sqrt(3.0) * abs(gamma) * intensity
    scale = dz * np.sinc(kappa * dz / np.pi)
    u = np.cos(kappa * dz) + 1j * alpha * scale
    v = 1j * mu * scale

    U_half = P[:, None] * U
    V_half = P[:, None] * V
    Ut = np.fft.ifft(
        np.fft.fft(U_half, axis=1, norm="ortho"), axis=0, norm="ortho"
    )
    Vt = np.fft.ifft(
        np.fft.ifft(V_half, axis=1, norm="ortho"), axis=0, norm="ortho"
    )
    Ut_new = u[:, None] * Ut + v[:, None] * np.conj(Vt)
    Vt_new = u[:, None] * Vt + v[:, None] * np.conj(Ut)
    U_freq = np.fft.fft(
        np.fft.ifft(Ut_new, axis=1, norm="ortho"), axis=0, norm="ortho"
    )
    V_freq = np.fft.fft(
        np.fft.fft(Vt_new, axis=1, norm="ortho"), axis=0, norm="ortho"
    )
    return np.vstack((A_new, P[:, None] * U_freq, P[:, None] * V_freq))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return normal, boundary and edge-case specifications."""
    return [
        {
            "setup": """
import numpy as np

N = 3
rng = np.random.default_rng(41)
A = 0.6 * (rng.normal(size=N) + 1j * rng.normal(size=N))
P = np.exp(0.3j * rng.normal(size=N))
U = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
V = 0.3 * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
dz, gamma = 0.0, 1.0
P = np.ones(N, dtype=complex)
""",
            "call": """
qssf_coupled_step(A.copy(), U.copy(), V.copy(), P.copy(), dz, gamma)
""",
            "gold_call": """
_oracle_qssf_coupled_step(A.copy(), U.copy(), V.copy(), P.copy(), dz, gamma)
""",
            "tol": 1e-10,
        },
        {
            "setup": """
import numpy as np

N = 5
rng = np.random.default_rng(42)
A = 0.6 * (rng.normal(size=N) + 1j * rng.normal(size=N))
P = np.exp(0.3j * rng.normal(size=N))
U = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
V = 0.3 * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
dz, gamma = 0.0, 1.0
""",
            "call": """
qssf_coupled_step(A.copy(), U.copy(), V.copy(), P.copy(), dz, gamma)
""",
            "gold_call": """
_oracle_qssf_coupled_step(A.copy(), U.copy(), V.copy(), P.copy(), dz, gamma)
""",
            "tol": 1e-10,
        },
        {
            "setup": """
import numpy as np

N = 7
rng = np.random.default_rng(43)
A = 0.6 * (rng.normal(size=N) + 1j * rng.normal(size=N))
P = np.exp(0.3j * rng.normal(size=N))
U = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
V = 0.3 * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
dz, gamma = 0.08, 0.0
""",
            "call": """
qssf_coupled_step(A.copy(), U.copy(), V.copy(), P.copy(), dz, gamma)
""",
            "gold_call": """
_oracle_qssf_coupled_step(A.copy(), U.copy(), V.copy(), P.copy(), dz, gamma)
""",
            "tol": 1e-10,
        },
        {
            "setup": """
import numpy as np

N = 4
rng = np.random.default_rng(44)
A = 0.6 * (rng.normal(size=N) + 1j * rng.normal(size=N))
P = np.exp(0.3j * rng.normal(size=N))
U = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
V = 0.3 * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
dz, gamma = 0.05, 1.0
A = np.zeros(N, dtype=complex)
U = np.eye(N, dtype=complex)
V = np.zeros((N, N), dtype=complex)
""",
            "call": """
qssf_coupled_step(A.copy(), U.copy(), V.copy(), P.copy(), dz, gamma)
""",
            "gold_call": """
_oracle_qssf_coupled_step(A.copy(), U.copy(), V.copy(), P.copy(), dz, gamma)
""",
            "tol": 1e-10,
        },
        {
            "setup": """
import numpy as np

N = 5
rng = np.random.default_rng(45)
A = 0.6 * (rng.normal(size=N) + 1j * rng.normal(size=N))
P = np.exp(0.3j * rng.normal(size=N))
U = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
V = 0.3 * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
dz, gamma = 0.09, -0.7
""",
            "call": """
qssf_coupled_step(A.copy(), U.copy(), V.copy(), P.copy(), dz, gamma)
""",
            "gold_call": """
_oracle_qssf_coupled_step(A.copy(), U.copy(), V.copy(), P.copy(), dz, gamma)
""",
            "tol": 1e-10,
        },
        {
            "setup": """
import numpy as np

N = 6
rng = np.random.default_rng(46)
A = 0.6 * (rng.normal(size=N) + 1j * rng.normal(size=N))
P = np.exp(0.3j * rng.normal(size=N))
U = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
V = 0.3 * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
dz, gamma = 0.04, 1.3
left = np.linalg.qr(U)[0]
right = np.linalg.qr(V)[0]
r = np.linspace(0.03, 0.3, N)
U = (left * np.cosh(r)) @ right.conj().T
V = (left * np.sinh(r)) @ right.T
""",
            "call": """
qssf_coupled_step(A.copy(), U.copy(), V.copy(), P.copy(), dz, gamma)
""",
            "gold_call": """
_oracle_qssf_coupled_step(A.copy(), U.copy(), V.copy(), P.copy(), dz, gamma)
""",
            "tol": 1e-10,
        },
        {
            "setup": """
import numpy as np

N = 7
rng = np.random.default_rng(47)
A = 0.6 * (rng.normal(size=N) + 1j * rng.normal(size=N))
P = np.exp(0.3j * rng.normal(size=N))
U = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
V = 0.3 * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
dz, gamma = 0.11, 0.9
""",
            "call": """
qssf_coupled_step(A.copy(), U.copy(), V.copy(), P.copy(), dz, gamma)
""",
            "gold_call": """
_oracle_qssf_coupled_step(A.copy(), U.copy(), V.copy(), P.copy(), dz, gamma)
""",
            "tol": 1e-10,
        },
        {
            "setup": """
import numpy as np

N = 1
rng = np.random.default_rng(48)
A = 0.6 * (rng.normal(size=N) + 1j * rng.normal(size=N))
P = np.exp(0.3j * rng.normal(size=N))
U = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
V = 0.3 * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
dz, gamma = 0.07, -1.1
""",
            "call": """
qssf_coupled_step(A.copy(), U.copy(), V.copy(), P.copy(), dz, gamma)
""",
            "gold_call": """
_oracle_qssf_coupled_step(A.copy(), U.copy(), V.copy(), P.copy(), dz, gamma)
""",
            "tol": 1e-10,
        },
        {
            "setup": """
import numpy as np

N = 8
rng = np.random.default_rng(49)
A = 0.6 * (rng.normal(size=N) + 1j * rng.normal(size=N))
P = np.exp(0.3j * rng.normal(size=N))
U = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
V = 0.3 * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
dz, gamma = 0.03, 0.8
P = P * np.linspace(0.85, 1.05, N)
""",
            "call": """
qssf_coupled_step(A.copy(), U.copy(), V.copy(), P.copy(), dz, gamma)
""",
            "gold_call": """
_oracle_qssf_coupled_step(A.copy(), U.copy(), V.copy(), P.copy(), dz, gamma)
""",
            "tol": 1e-10,
        },
        {
            "setup": """
import numpy as np

N = 6
rng = np.random.default_rng(46)
A = 0.6 * (rng.normal(size=N) + 1j * rng.normal(size=N))
P = np.exp(0.3j * rng.normal(size=N))
U = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
V = 0.3 * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
dz, gamma = 0.04, 1.3
left = np.linalg.qr(U)[0]
right = np.linalg.qr(V)[0]
r = np.linspace(0.03, 0.3, N)
U = (left * np.cosh(r)) @ right.conj().T
V = (left * np.sinh(r)) @ right.T


def immutable(fn):
    arrays = [A.copy(), U.copy(), V.copy(), P.copy()]
    before = [array.copy() for array in arrays]
    for array in arrays:
        array.setflags(write=False)
    result = np.asarray(fn(*arrays, dz, gamma))
    assert result.shape == (2 * N + 1, N)
    assert np.iscomplexobj(result) and np.isfinite(result).all()
    assert all(np.array_equal(a, b) for a, b in zip(arrays, before))
    return result
""",
            "call": """
immutable(qssf_coupled_step)
""",
            "gold_call": """
immutable(_oracle_qssf_coupled_step)
""",
            "tol": 1e-10,
        },
        {
            "setup": """
import numpy as np

A = np.ones(3, dtype=complex)
P = np.ones(3, dtype=complex)
U = np.eye(3, dtype=complex)
V = np.zeros((3, 3), dtype=complex)
dz, gamma = 0.1, 1.0
V = np.zeros((2, 3), dtype=complex)


def invalid(fn):
    try:
        fn(A.copy(), U.copy(), V.copy(), P.copy(), dz, gamma)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": """
invalid(qssf_coupled_step)
""",
            "gold_call": """
invalid(_oracle_qssf_coupled_step)
""",
            "tol": 0.0,
        },
        {
            "setup": """
import numpy as np

A = np.ones(3, dtype=complex)
P = np.ones(3, dtype=complex)
U = np.eye(3, dtype=complex)
V = np.zeros((3, 3), dtype=complex)
dz, gamma = 0.1, 1.0
A[1] = np.nan


def invalid(fn):
    try:
        fn(A.copy(), U.copy(), V.copy(), P.copy(), dz, gamma)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": """
invalid(qssf_coupled_step)
""",
            "gold_call": """
invalid(_oracle_qssf_coupled_step)
""",
            "tol": 0.0,
        },
        {
            "setup": """
import numpy as np

A = np.ones(3, dtype=complex)
P = np.ones(3, dtype=complex)
U = np.eye(3, dtype=complex)
V = np.zeros((3, 3), dtype=complex)
dz, gamma = 0.1, 1.0
dz = -0.1


def invalid(fn):
    try:
        fn(A.copy(), U.copy(), V.copy(), P.copy(), dz, gamma)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": """
invalid(qssf_coupled_step)
""",
            "gold_call": """
invalid(_oracle_qssf_coupled_step)
""",
            "tol": 0.0,
        },
        {
            "setup": """
import numpy as np

A = np.ones(3, dtype=complex)
P = np.ones(3, dtype=complex)
U = np.eye(3, dtype=complex)
V = np.zeros((3, 3), dtype=complex)
dz, gamma = 0.1, 1.0
gamma = 1.0 + 0.1j


def invalid(fn):
    try:
        fn(A.copy(), U.copy(), V.copy(), P.copy(), dz, gamma)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": """
invalid(qssf_coupled_step)
""",
            "gold_call": """
invalid(_oracle_qssf_coupled_step)
""",
            "tol": 0.0,
        },
    ]
