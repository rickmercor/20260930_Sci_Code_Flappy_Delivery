"""
Compose all six preceding functions to propagate the coupled state, validate its commutators and Williamson coordinates, and return the selected spectral-band diagnostic.

Compose all six preceding functions into the complete QSSF simulation.



Propagate the coherent pulse and vacuum fluctuations, check the canonical

relations, construct the prescribed spectral-band covariance, obtain its

Williamson coordinates and validate those coordinates before returning

the selected scalar diagnostic.



Use t_j=(j-Nt/2)*T/Nt, unshifted angular FFT frequencies,

D(w)=d2*w^2/2+beta3*w^3 and P=exp(i*D*dz/2). Initialize

A=A0*sech(A0*t), U=I and V=0. Advance A,U,V together through

qssf_coupled_step. In each step the field after the first classical linear

half-step remains frozen throughout the local quantum substep. Select

|w-w_RR| <= delta_w/2 using the root defined by the dispersion step.



n_steps is a nonnegative integer and Nt an even integer >=2; booleans are

not integers for these parameters. T,delta_w,d2,A0,gamma are finite positive

real scalars. dz is finite, real and nonnegative; beta3 is finite, real and

nonzero. quantity must be "entropy", "purity", "K_eff" or "photons".

Invalid inputs raise ValueError. RuntimeError is raised if either

normalized commutator error exceeds 1e-9 after any step. Check the initial

map even for n_steps=0. No caller inputs or global state are modified.



Pass the band covariance from qssf_second_moments to

qssf_williamson_decomposition, then pass both covariance and decomposition

to qssf_williamson_entropy. Entropy uses natural logarithms. An empty band

gives entropy=0, purity=1, K_eff=1 and photons=0. Photon count is Tr(N_W),

equivalently (Tr(Vq)-len(window))/2, not the sum of Williamson occupations.



Return the selected diagnostic as a float at z=n_steps*dz. Defaults are

the benchmark specified by the main problem; all exposed parameters

must affect the prescribed computation.

Returns
-------
float: selected band diagnostic at z=n_steps*dz.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def qssf_full_simulation(
    n_steps: int = 40,
    dz: float = 0.01,
    Nt: int = 128,
    T: float = 40.0,
    A0: float = 1.5,
    beta3: float = 0.08,
    d2: float = 1.0,
    gamma: float = 1.0,
    delta_w: float = 3.0,
    quantity: str = "entropy",
) -> float:
    """
    Run the complete QSSF simulation pipeline and return the requested
    observable.

    Parameters
    ----------
    n_steps : int
        Number of propagation steps.
    dz : float
        Spatial step size.
    Nt : int
        Grid size.
    T : float
        Temporal window duration.
    A0 : float
        Sech-pulse amplitude.
    beta3 : float
        Third-order dispersion coefficient.
    d2 : float
        Second-order dispersion coefficient.
    gamma : float
        Kerr nonlinear coefficient.
    delta_w : float
        Prescribed analysis-band width.
    quantity : str
        Target observable name.

    Returns
    -------
    float
        Computed observable value.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_qssf_full_simulation(
    n_steps: int = 40,
    dz: float = 0.01,
    Nt: int = 128,
    T: float = 40.0,
    A0: float = 1.5,
    beta3: float = 0.08,
    d2: float = 1.0,
    gamma: float = 1.0,
    delta_w: float = 3.0,
    quantity: str = "entropy",
) -> float:
    if (
        not isinstance(n_steps, (int, np.integer))
        or isinstance(n_steps, (bool, np.bool_))
        or n_steps < 0
    ):
        raise ValueError("n_steps must be a nonnegative integer")
    if (
        not isinstance(Nt, (int, np.integer))
        or isinstance(Nt, (bool, np.bool_))
        or Nt < 2
        or Nt % 2
    ):
        raise ValueError("Nt must be an even integer >= 2")
    if quantity not in ("entropy", "purity", "K_eff", "photons"):
        raise ValueError("Unknown diagnostic quantity")
    T = _qssf_scalar(T, positive=True)
    delta_w = _qssf_scalar(delta_w, positive=True)
    dz = _qssf_scalar(dz, nonnegative=True)
    dt = T / Nt
    t = (np.arange(Nt) - Nt / 2) * dt
    w = 2.0 * np.pi * np.fft.fftfreq(Nt, d=dt)

    res1 = _oracle_qssf_dispersion_propagator(
        w, dz, d2=d2, beta3=beta3, A0=A0, gamma=gamma
    )
    P = res1[0]
    w_RR = float(res1[2, 0].real)

    window = np.where(np.abs(w - w_RR) <= delta_w / 2.0)[0]

    A = A0 / np.cosh(A0 * t)
    U = np.eye(Nt, dtype=complex)
    V = np.zeros((Nt, Nt), dtype=complex)

    errors = _oracle_qssf_symplecticity_check(U, V)
    if not np.isfinite(errors).all() or np.max(errors) > 1e-9:
        raise RuntimeError("Initial commutator check failed")

    for _ in range(n_steps):
        state = _oracle_qssf_coupled_step(A, U, V, P, dz, gamma)
        A = state[0]
        boundary = Nt + 1
        U = state[1:boundary]
        V = state[boundary:]
        errors = _oracle_qssf_symplecticity_check(U, V)
        if not np.isfinite(errors).all() or np.max(errors) > 1e-9:
            raise RuntimeError(
                "Quantum propagation lost canonical commutators"
            )

    Vq = _oracle_qssf_second_moments(U, V, window)
    decomposition = _oracle_qssf_williamson_decomposition(Vq)
    res7 = _oracle_qssf_williamson_entropy(Vq, decomposition)

    SvN = float(res7[0])
    purity = float(res7[2])
    K_eff = float(res7[3])

    if quantity == "purity":
        return purity
    elif quantity == "K_eff":
        return K_eff
    elif quantity == "photons":
        return float(max((np.trace(Vq) - len(window)) / 2.0, 0.0))
    return SvN

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return normal, boundary and edge-case specifications."""
    return [
        {
            "setup": """
import numpy as np
""",
            "call": """
qssf_full_simulation(n_steps=0)
""",
            "gold_call": """
_oracle_qssf_full_simulation(n_steps=0)
""",
            "tol": 1e-10,
        },
        {
            "setup": """
import numpy as np
""",
            "call": """
qssf_full_simulation(n_steps=1, Nt=32, T=100.0)
""",
            "gold_call": """
_oracle_qssf_full_simulation(n_steps=1, Nt=32, T=100.0)
""",
            "tol": 1e-10,
        },
        {
            "setup": """
import numpy as np


def invalid(fn):
    try:
        fn(n_steps=-1)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": """
invalid(qssf_full_simulation)
""",
            "gold_call": """
invalid(_oracle_qssf_full_simulation)
""",
            "tol": 0.0,
        },
        {
            "setup": """
import numpy as np


def invalid(fn):
    try:
        fn(n_steps=0, quantity="invalid")
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": """
invalid(qssf_full_simulation)
""",
            "gold_call": """
invalid(_oracle_qssf_full_simulation)
""",
            "tol": 0.0,
        },
        {
            "setup": """
import numpy as np
""",
            "call": """
qssf_full_simulation(
    n_steps=10,
    dz=0.01,
    Nt=64,
    T=30.0,
    A0=1.5,
    beta3=0.08,
    d2=1.0,
    gamma=1.0,
    delta_w=3.0,
    quantity="entropy",
)
""",
            "gold_call": """
_oracle_qssf_full_simulation(
    n_steps=10,
    dz=0.01,
    Nt=64,
    T=30.0,
    A0=1.5,
    beta3=0.08,
    d2=1.0,
    gamma=1.0,
    delta_w=3.0,
    quantity="entropy",
)
""",
            "tol": 1e-08,
        },
        {
            "setup": """
import numpy as np
""",
            "call": """
qssf_full_simulation(
    n_steps=5,
    dz=0.02,
    Nt=64,
    T=30.0,
    A0=1.5,
    beta3=0.08,
    d2=1.0,
    gamma=1.0,
    delta_w=3.0,
    quantity="photons",
)
""",
            "gold_call": """
_oracle_qssf_full_simulation(
    n_steps=5,
    dz=0.02,
    Nt=64,
    T=30.0,
    A0=1.5,
    beta3=0.08,
    d2=1.0,
    gamma=1.0,
    delta_w=3.0,
    quantity="photons",
)
""",
            "tol": 1e-08,
        },
        {
            "setup": """
import numpy as np
""",
            "call": """
qssf_full_simulation(
    n_steps=8,
    dz=0.01,
    Nt=64,
    T=30.0,
    A0=1.5,
    beta3=0.08,
    d2=1.0,
    gamma=1.0,
    delta_w=3.0,
    quantity="K_eff",
)
""",
            "gold_call": """
_oracle_qssf_full_simulation(
    n_steps=8,
    dz=0.01,
    Nt=64,
    T=30.0,
    A0=1.5,
    beta3=0.08,
    d2=1.0,
    gamma=1.0,
    delta_w=3.0,
    quantity="K_eff",
)
""",
            "tol": 1e-08,
        },
        {
            "setup": """
import numpy as np
""",
            "call": """
qssf_full_simulation(
    n_steps=12,
    dz=0.01,
    Nt=64,
    T=30.0,
    A0=1.5,
    beta3=0.08,
    d2=1.0,
    gamma=1.0,
    delta_w=3.0,
    quantity="purity",
)
""",
            "gold_call": """
_oracle_qssf_full_simulation(
    n_steps=12,
    dz=0.01,
    Nt=64,
    T=30.0,
    A0=1.5,
    beta3=0.08,
    d2=1.0,
    gamma=1.0,
    delta_w=3.0,
    quantity="purity",
)
""",
            "tol": 1e-08,
        },
        {
            "setup": """
import numpy as np
""",
            "call": """
qssf_full_simulation(
    n_steps=9,
    dz=0.025,
    Nt=48,
    T=20.0,
    A0=1.2,
    beta3=-0.13,
    d2=0.9,
    gamma=0.7,
    delta_w=2.3,
    quantity="entropy",
)
""",
            "gold_call": """
_oracle_qssf_full_simulation(
    n_steps=9,
    dz=0.025,
    Nt=48,
    T=20.0,
    A0=1.2,
    beta3=-0.13,
    d2=0.9,
    gamma=0.7,
    delta_w=2.3,
    quantity="entropy",
)
""",
            "tol": 1e-08,
        },
        {
            "setup": """
import numpy as np
""",
            "call": """
qssf_full_simulation(
    n_steps=13,
    dz=0.017,
    Nt=64,
    T=24.0,
    A0=1.1,
    beta3=0.12,
    d2=0.8,
    gamma=1.3,
    delta_w=2.0,
    quantity="purity",
)
""",
            "gold_call": """
_oracle_qssf_full_simulation(
    n_steps=13,
    dz=0.017,
    Nt=64,
    T=24.0,
    A0=1.1,
    beta3=0.12,
    d2=0.8,
    gamma=1.3,
    delta_w=2.0,
    quantity="purity",
)
""",
            "tol": 1e-08,
        },
        {
            "setup": """
import numpy as np
""",
            "call": """
qssf_full_simulation(
    n_steps=13,
    dz=0.017,
    Nt=64,
    T=24.0,
    A0=1.1,
    beta3=0.12,
    d2=0.8,
    gamma=1.3,
    delta_w=2.0,
    quantity="K_eff",
)
""",
            "gold_call": """
_oracle_qssf_full_simulation(
    n_steps=13,
    dz=0.017,
    Nt=64,
    T=24.0,
    A0=1.1,
    beta3=0.12,
    d2=0.8,
    gamma=1.3,
    delta_w=2.0,
    quantity="K_eff",
)
""",
            "tol": 1e-08,
        },
    ]
