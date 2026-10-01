"""
Step 05 - Time derivative of the bivariational TD-CCSD amplitudes in a field.

Time derivative of the bivariational time-dependent CCSD state in an electric
field.

In real time the cluster amplitudes t and the de-excitation amplitudes l of the
previous step become complex functions of time, and both are propagated: the
ket |Psi(t)> = exp(T(t))|Phi0> and the bra <~Psi(t)| = <Phi0|(1 + Lambda(t))
exp(-T(t)) evolve independently under the time-dependent bivariational
principle. With the operator conventions of the ground-state step, the
equations of motion are

i dt_mu/dt = <Phi_mu| exp(-T) H(t) exp(T) |Phi0>
-i dlambda_mu/dt = <Phi0| (1 + Lambda) [exp(-T) H(t) exp(T), X_mu] |Phi0>

for every single and double excitation mu, the derivative of each doubles
amplitude array being the antisymmetric array of these values. The
coefficient of the unit operator in Lambda is kept equal to one, and the phase
amplitude of the unit operator in T is not propagated: it multiplies the ket
by a number and the bra by its inverse, so it cancels from every bra-ket
product this task uses. Every amplitude is complex and no quantity is complex
conjugated anywhere in these equations.

The molecule couples to a z-polarised field E(t) in the length gauge: the
electrons carry charge -1, so the interaction -d.E adds E(t) times the electronic
z coordinate to the one-electron operator. The reference occupations do not
change, so with the field present the Fock matrix of the reference is simply
fock + E(t) * dipole, where dipole is the spin-orbital matrix of the electronic
z coordinate, and the two-electron integrals are unchanged. This field-dressed
Fock matrix in general has a non-zero occupied-virtual block, and the amplitudes
at which the right-hand sides are evaluated are complex and do not satisfy any
stationarity condition.

The amplitude vector has the packed layout of the ground-state step,
[t1, t2, l1, l2] flattened in C order with full antisymmetric doubles arrays,
of length L = 2 (o v + o^2 v^2), and may be real or complex. The derivative is
returned as a real vector of length 2 L: the real parts of the L derivative
components followed by their imaginary parts.

Returns
-------
numpy.ndarray of length 2 L: real parts then imaginary parts of the time derivative of the packed amplitudes
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def tdccsd_time_derivative(fock: np.ndarray, dipole: np.ndarray, eri_as: np.ndarray, n_electrons: int, amplitudes: np.ndarray, field: float) -> np.ndarray:
    '''Time derivative of the packed TD-CCSD amplitudes [t1, t2, l1, l2] in a z-polarised field.

    Parameters
    ----------
    fock : np.ndarray
        Field-free spin-orbital Fock matrix of the reference, shape (n, n),
        real and symmetric, hartree.
    dipole : np.ndarray
        Spin-orbital matrix of the electronic z coordinate, shape (n, n), real
        and symmetric, bohr.
    eri_as : np.ndarray
        Antisymmetrised two-electron integrals <pq||rs>, shape (n, n, n, n).
    n_electrons : int
        Number of electrons; the reference occupies spin orbitals
        0 .. n_electrons - 1.
    amplitudes : np.ndarray
        Packed amplitude vector [t1, t2, l1, l2], real or complex, length
        L = 2 (o v + o^2 v^2) with o = n_electrons and v = n - o.
    field : float
        Instantaneous electric field strength E(t) in atomic units.

    Returns
    -------
    derivative : np.ndarray
        Real vector of length 2 L: [Re(dy/dt), Im(dy/dt)] for the packed
        amplitude vector y, in atomic units of inverse time.

    Raises
    ------
    ValueError
        If fock, eri_as or n_electrons violate the conditions of the
        ground-state step, if dipole is not a finite symmetric (n, n) matrix
        (tolerance 1e-10), if amplitudes is not a finite 1D vector of length L,
        or if field is not a finite real number.
    '''
    return derivative

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _td_check_inputs(fock, dipole, eri_as, n_electrons, amplitudes):
    f, g, o, v = _cc_check_hamiltonian(fock, eri_as, n_electrons)
    z = np.asarray(dipole, dtype=float)
    if z.shape != f.shape or not np.all(np.isfinite(z)) or not np.allclose(z, z.T, atol=1e-10):
        raise ValueError("dipole must be a finite symmetric matrix of the Fock shape")
    y = np.asarray(amplitudes, dtype=complex)
    if y.ndim != 1 or y.size != 2 * (o * v + o * o * v * v) or not np.all(np.isfinite(y)):
        raise ValueError("amplitudes must be a finite 1D vector of length 2 (o v + o^2 v^2)")
    return f, z, g, o, v, y


def _td_derivative(f, z, g, o, v, y, field):
    t1, t2, l1, l2 = _cc_unpack(y, o, v)
    ft = f + field * z
    R1, R2 = _cc_t_residuals(ft, g, o, t1, t2)
    G1, G2 = _cc_l_residuals(ft, g, o, t1, t2, l1, l2)
    return _cc_pack(-1j * R1, -1j * R2, 1j * G1, 1j * G2)


def _oracle_tdccsd_time_derivative(fock: np.ndarray, dipole: np.ndarray, eri_as: np.ndarray, n_electrons: int, amplitudes: np.ndarray, field: float) -> np.ndarray:
    f, z, g, o, v, y = _td_check_inputs(fock, dipole, eri_as, n_electrons, amplitudes)
    if isinstance(field, bool) or not isinstance(field, (int, float, np.integer, np.floating)) or not np.isfinite(field):
        raise ValueError("field must be a finite real number")
    dy = _td_derivative(f, z, g, o, v, y, float(field))
    return np.concatenate([dy.real, dy.imag])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: four electrons in eight spin orbitals, random complex amplitudes, weak field ---
        {
            "setup": """import numpy as np
import copy
rng = np.random.default_rng(501)
n, ne = 8, 4
A = rng.normal(scale=0.04, size=(n, n))
fock = 0.5 * (A + A.T) + np.diag([-1.2, -1.1, -0.9, -0.85, 0.45, 0.6, 0.8, 1.1])
B = rng.normal(scale=0.5, size=(n, n))
dipole = 0.5 * (B + B.T)
W = rng.normal(scale=0.03, size=(n, n, n, n))
W = W + W.transpose(1, 0, 3, 2)
W = W + W.transpose(2, 3, 0, 1)
eri_as = W - W.transpose(0, 1, 3, 2)
o, v = ne, n - ne
def amp(shape, s):
    return rng.normal(scale=s, size=shape) + 1j * rng.normal(scale=s, size=shape)
def asym(x):
    x = x - x.transpose(1, 0, 2, 3)
    return x - x.transpose(0, 1, 3, 2)
amplitudes = np.concatenate([amp((o, v), 0.05).ravel(), asym(amp((o, o, v, v), 0.03)).ravel(),
                             amp((o, v), 0.05).ravel(), asym(amp((o, o, v, v), 0.03)).ravel()])
""",
            "call": "tdccsd_time_derivative(copy.deepcopy(fock), copy.deepcopy(dipole), copy.deepcopy(eri_as), ne, copy.deepcopy(amplitudes), 0.013)",
            "gold_call": "_oracle_tdccsd_time_derivative(copy.deepcopy(fock), copy.deepcopy(dipole), copy.deepcopy(eri_as), ne, copy.deepcopy(amplitudes), 0.013)",
            "tol": 1e-10,
        },
        # --- Boundary: the converged ground state with no field is stationary ---
        {
            "setup": """import numpy as np
import copy
rng = np.random.default_rng(502)
n, ne = 6, 2
A = rng.normal(scale=0.05, size=(n, n))
fock = 0.5 * (A + A.T) + np.diag([-0.9, -0.75, 0.3, 0.45, 0.7, 1.0])
dipole = np.diag([0.4, 0.4, -0.3, -0.3, 0.1, 0.1]) + 0.2
W = rng.normal(scale=0.04, size=(n, n, n, n))
W = W + W.transpose(1, 0, 3, 2)
W = W + W.transpose(2, 3, 0, 1)
eri_as = W - W.transpose(0, 1, 3, 2)
amplitudes = _oracle_ccsd_lambda_ground_state(fock, eri_as, ne)
""",
            "call": "tdccsd_time_derivative(copy.deepcopy(fock), copy.deepcopy(dipole), copy.deepcopy(eri_as), ne, copy.deepcopy(amplitudes), 0.0)",
            "gold_call": "_oracle_tdccsd_time_derivative(copy.deepcopy(fock), copy.deepcopy(dipole), copy.deepcopy(eri_as), ne, copy.deepcopy(amplitudes), 0.0)",
            "tol": 1e-9,
        },
        # --- Edge: strong field and large complex amplitudes, three electrons in nine spin orbitals ---
        {
            "setup": """import numpy as np
import copy
rng = np.random.default_rng(503)
n, ne = 9, 3
A = rng.normal(scale=0.08, size=(n, n))
fock = 0.5 * (A + A.T) + np.diag([-0.8, -0.55, -0.35, 0.05, 0.3, 0.55, 0.7, 0.95, 1.3])
B = rng.normal(scale=1.0, size=(n, n))
dipole = 0.5 * (B + B.T)
W = rng.normal(scale=0.05, size=(n, n, n, n))
W = W + W.transpose(1, 0, 3, 2)
W = W + W.transpose(2, 3, 0, 1)
eri_as = W - W.transpose(0, 1, 3, 2)
o, v = ne, n - ne
def amp(shape, s):
    return rng.normal(scale=s, size=shape) + 1j * rng.normal(scale=s, size=shape)
def asym(x):
    x = x - x.transpose(1, 0, 2, 3)
    return x - x.transpose(0, 1, 3, 2)
amplitudes = np.concatenate([amp((o, v), 0.4).ravel(), asym(amp((o, o, v, v), 0.3)).ravel(),
                             amp((o, v), 0.4).ravel(), asym(amp((o, o, v, v), 0.3)).ravel()])
""",
            "call": "tdccsd_time_derivative(copy.deepcopy(fock), copy.deepcopy(dipole), copy.deepcopy(eri_as), ne, copy.deepcopy(amplitudes), -0.7)",
            "gold_call": "_oracle_tdccsd_time_derivative(copy.deepcopy(fock), copy.deepcopy(dipole), copy.deepcopy(eri_as), ne, copy.deepcopy(amplitudes), -0.7)",
            "tol": 1e-10,
        },
        # --- Invalid: an amplitude vector of the wrong length ---
        {
            "setup": """import numpy as np
import copy
n, ne = 4, 2
fock = np.diag([-0.7, -0.6, 0.35, 0.5])
dipole = np.eye(4)
eri_as = np.zeros((n, n, n, n))
amplitudes = np.zeros(11, dtype=complex)
def run_model():
    try:
        tdccsd_time_derivative(copy.deepcopy(fock), copy.deepcopy(dipole), copy.deepcopy(eri_as), ne, copy.deepcopy(amplitudes), 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_tdccsd_time_derivative(copy.deepcopy(fock), copy.deepcopy(dipole), copy.deepcopy(eri_as), ne, copy.deepcopy(amplitudes), 0.1)
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
