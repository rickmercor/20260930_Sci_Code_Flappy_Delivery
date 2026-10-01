"""
Step 06 - Runge-Kutta propagation of TD-CCSD through a trigonometric-envelope pulse.

Real-time propagation of the TD-CCSD amplitudes through a laser pulse.

The state is advanced from an initial packed amplitude vector y(t_start) (for
the molecule, the converged ground state) under the equations of motion of the
previous step. The driving field is a z-polarised pulse with a carrier of
angular frequency omega under a trigonometric envelope of strictly finite
support, the form used for short intense pulses in real-time simulations:

E(t) = e0 cos(omega (t - t_center)) g(t),
g(t) = cos(pi (t - t_center) / t_foot) ** n_env if |t - t_center| <= t_foot / 2,
g(t) = 0 otherwise,

where t_foot is the foot-to-foot duration and n_env a positive integer. The
envelope and its first n_env - 1 derivatives are continuous, so the field
switches on and off smoothly and is exactly zero outside the window.

Integrate with the classical fourth-order Runge-Kutta scheme and a fixed step
dt. With F(t, y) the derivative of the previous step evaluated with the field
E(t), and t_k = t_start + k dt,

k1 = F(t_k, y_k)
k2 = F(t_k + dt/2, y_k + (dt/2) k1)
k3 = F(t_k + dt/2, y_k + (dt/2) k2)
k4 = F(t_k + dt, y_k + dt k3)
y_{k+1} = y_k + (dt/6) (k1 + 2 k2 + 2 k3 + k4),

for k = 0 .. n_steps - 1, with every stage using the field at its own time.
The amplitudes are complex throughout. The state after n_steps steps is
returned as a real vector of length 2 L, the real parts of the packed vector
[t1, t2, l1, l2] followed by its imaginary parts, L being the packed length of
the previous steps; with n_steps = 0 the initial vector is returned in that
form.

Returns
-------
numpy.ndarray of length 2 L: real parts then imaginary parts of the packed amplitudes after the last Runge-Kutta step
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def tdccsd_propagate(fock: np.ndarray, dipole: np.ndarray, eri_as: np.ndarray, n_electrons: int, amplitudes: np.ndarray, e0: float, omega: float, t_center: float, t_foot: float, n_env: int, t_start: float, dt: float, n_steps: int) -> np.ndarray:
    '''Propagate the packed TD-CCSD amplitudes with fourth-order Runge-Kutta through a trigonometric-envelope pulse.

    Parameters
    ----------
    fock : np.ndarray
        Field-free spin-orbital Fock matrix of the reference, shape (n, n).
    dipole : np.ndarray
        Spin-orbital matrix of the electronic z coordinate, shape (n, n).
    eri_as : np.ndarray
        Antisymmetrised two-electron integrals <pq||rs>, shape (n, n, n, n).
    n_electrons : int
        Number of electrons (occupied spin orbitals 0 .. n_electrons - 1).
    amplitudes : np.ndarray
        Initial packed vector [t1, t2, l1, l2], real or complex, length L.
    e0 : float
        Peak field amplitude in atomic units (any finite real number).
    omega : float
        Carrier angular frequency in hartree (finite, non-negative).
    t_center : float
        Centre of the envelope in atomic units of time.
    t_foot : float
        Foot-to-foot duration of the envelope, strictly positive.
    n_env : int
        Positive integer exponent of the cosine envelope.
    t_start : float
        Time of the initial vector.
    dt : float
        Strictly positive Runge-Kutta step.
    n_steps : int
        Non-negative number of steps.

    Returns
    -------
    final : np.ndarray
        Real vector of length 2 L: real parts of the packed amplitudes at
        t_start + n_steps dt followed by their imaginary parts.

    Raises
    ------
    ValueError
        If the Hamiltonian, dipole or amplitude inputs violate the conditions
        of the previous step, if e0, omega, t_center or t_start is not finite
        or omega is negative, if t_foot or dt is not a finite positive number,
        if n_env is not a positive integer, or if n_steps is not a non-negative
        integer.
    '''
    return final

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _pulse_field(t, e0, omega, t_center, t_foot, n_env):
    x = t - t_center
    if abs(x) > 0.5 * t_foot:
        return 0.0
    return e0 * np.cos(omega * x) * np.cos(np.pi * x / t_foot) ** n_env


def _oracle_tdccsd_propagate(fock: np.ndarray, dipole: np.ndarray, eri_as: np.ndarray, n_electrons: int, amplitudes: np.ndarray, e0: float, omega: float, t_center: float, t_foot: float, n_env: int, t_start: float, dt: float, n_steps: int) -> np.ndarray:
    f, z, g, o, v, y = _td_check_inputs(fock, dipole, eri_as, n_electrons, amplitudes)
    for name, val in (('e0', e0), ('omega', omega), ('t_center', t_center), ('t_start', t_start), ('t_foot', t_foot), ('dt', dt)):
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)) or not np.isfinite(val):
            raise ValueError("%s must be a finite real number" % name)
    if omega < 0.0:
        raise ValueError("omega must be non-negative")
    if t_foot <= 0.0 or dt <= 0.0:
        raise ValueError("t_foot and dt must be positive")
    if isinstance(n_env, bool) or not isinstance(n_env, (int, np.integer)) or n_env < 1:
        raise ValueError("n_env must be a positive integer")
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer")
    E = lambda t: _pulse_field(t, float(e0), float(omega), float(t_center), float(t_foot), int(n_env))
    F = lambda t, yy: _td_derivative(f, z, g, o, v, yy, E(t))
    t_start = float(t_start)
    dt = float(dt)
    for k in range(int(n_steps)):
        tk = t_start + k * dt
        k1 = F(tk, y)
        k2 = F(tk + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = F(tk + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = F(tk + dt, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.concatenate([y.real, y.imag])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: four electrons in eight spin orbitals, from the ground state through a full pulse ---
        {
            "setup": """import numpy as np
import copy
rng = np.random.default_rng(601)
n, ne = 8, 4
A = rng.normal(scale=0.04, size=(n, n))
fock = 0.5 * (A + A.T) + np.diag([-1.2, -1.1, -0.9, -0.85, 0.45, 0.6, 0.8, 1.1])
B = rng.normal(scale=0.6, size=(n, n))
dipole = 0.5 * (B + B.T)
W = rng.normal(scale=0.03, size=(n, n, n, n))
W = W + W.transpose(1, 0, 3, 2)
W = W + W.transpose(2, 3, 0, 1)
eri_as = W - W.transpose(0, 1, 3, 2)
y0 = _oracle_ccsd_lambda_ground_state(fock, eri_as, ne)
""",
            "call": "tdccsd_propagate(copy.deepcopy(fock), copy.deepcopy(dipole), copy.deepcopy(eri_as), ne, copy.deepcopy(y0), 0.08, 1.4, 0.0, 6.0, 4, -3.0, 0.1, 60)",
            "gold_call": "_oracle_tdccsd_propagate(copy.deepcopy(fock), copy.deepcopy(dipole), copy.deepcopy(eri_as), ne, copy.deepcopy(y0), 0.08, 1.4, 0.0, 6.0, 4, -3.0, 0.1, 60)",
            "tol": 1e-9,
        },
        # --- Boundary: zero steps returns the initial vector as [real parts, imaginary parts] ---
        {
            "setup": """import numpy as np
import copy
rng = np.random.default_rng(602)
n, ne = 4, 2
fock = np.diag([-0.7, -0.6, 0.35, 0.5]) + 0.01
dipole = np.eye(4) * 0.3
W = rng.normal(scale=0.05, size=(n, n, n, n))
W = W + W.transpose(1, 0, 3, 2)
W = W + W.transpose(2, 3, 0, 1)
eri_as = W - W.transpose(0, 1, 3, 2)
y0 = _oracle_ccsd_lambda_ground_state(fock, eri_as, ne) * (1.0 + 0.5j)
""",
            "call": "tdccsd_propagate(copy.deepcopy(fock), copy.deepcopy(dipole), copy.deepcopy(eri_as), ne, copy.deepcopy(y0), 0.2, 0.9, 1.0, 4.0, 2, 0.0, 0.05, 0)",
            "gold_call": "_oracle_tdccsd_propagate(copy.deepcopy(fock), copy.deepcopy(dipole), copy.deepcopy(eri_as), ne, copy.deepcopy(y0), 0.2, 0.9, 1.0, 4.0, 2, 0.0, 0.05, 0)",
            "tol": 1e-12,
        },
        # --- Edge: odd envelope exponent, propagation starting inside the pulse and running past its end ---
        {
            "setup": """import numpy as np
import copy
rng = np.random.default_rng(603)
n, ne = 6, 2
A = rng.normal(scale=0.06, size=(n, n))
fock = 0.5 * (A + A.T) + np.diag([-0.9, -0.75, 0.3, 0.45, 0.7, 1.0])
B = rng.normal(scale=0.8, size=(n, n))
dipole = 0.5 * (B + B.T)
W = rng.normal(scale=0.05, size=(n, n, n, n))
W = W + W.transpose(1, 0, 3, 2)
W = W + W.transpose(2, 3, 0, 1)
eri_as = W - W.transpose(0, 1, 3, 2)
y0 = _oracle_ccsd_lambda_ground_state(fock, eri_as, ne)
""",
            "call": "tdccsd_propagate(copy.deepcopy(fock), copy.deepcopy(dipole), copy.deepcopy(eri_as), ne, copy.deepcopy(y0), -0.3, 1.1, 2.0, 5.0, 3, 1.0, 0.04, 100)",
            "gold_call": "_oracle_tdccsd_propagate(copy.deepcopy(fock), copy.deepcopy(dipole), copy.deepcopy(eri_as), ne, copy.deepcopy(y0), -0.3, 1.1, 2.0, 5.0, 3, 1.0, 0.04, 100)",
            "tol": 1e-9,
        },
        # --- Invalid: a zero envelope exponent ---
        {
            "setup": """import numpy as np
import copy
n, ne = 4, 2
fock = np.diag([-0.7, -0.6, 0.35, 0.5])
dipole = np.eye(4)
eri_as = np.zeros((n, n, n, n))
y0 = np.zeros(2 * (4 + 16))
def run_model():
    try:
        tdccsd_propagate(copy.deepcopy(fock), copy.deepcopy(dipole), copy.deepcopy(eri_as), ne, copy.deepcopy(y0), 0.1, 0.5, 0.0, 4.0, 0, -2.0, 0.1, 10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_tdccsd_propagate(copy.deepcopy(fock), copy.deepcopy(dipole), copy.deepcopy(eri_as), ne, copy.deepcopy(y0), 0.1, 0.5, 0.0, 4.0, 0, -2.0, 0.1, 10)
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
