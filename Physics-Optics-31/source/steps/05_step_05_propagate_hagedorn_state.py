"""
Advance every independent Gaussian with the source drift-kick-drift map.

The released solver evaluates the local field after a half drift, applies midpoint kicks, and completes the second half drift.

Returns
-------
tuple[np.ndarray, ...]: final finite q, p, Q, P, and action arrays.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral, Real
import numpy as np
def propagate_hagedorn_state(
    q: np.ndarray, p: np.ndarray, Q: np.ndarray,
    P: np.ndarray, S: np.ndarray, dt: float, n_steps: int,
    mass: float, field_amplitude: float, radius: float,
    dielectric: complex, omega: float,
    charge: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Advance a batch of two-dimensional Hagedorn states.

    Each step uses half drifts of ``q`` and ``Q``, one midpoint field/Hessian
    evaluation, centered kicks, and a midpoint Lagrangian action. Inputs are
    copied and the final arrays are returned.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Final ``q, p, Q, P, S`` arrays with their input shapes.

    Raises
    ------
    ValueError
        If the state arrays are non-finite or misaligned, if ``n_steps`` is not
        a nonnegative integer, or if ``dt`` or ``mass`` is not positive finite.
    """
    return q, p, Q, P, S

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from numbers import Integral, Real
import numpy as np
def _oracle_propagate_hagedorn_state(
    q: np.ndarray, p: np.ndarray, Q: np.ndarray,
    P: np.ndarray, S: np.ndarray, dt: float, n_steps: int,
    mass: float, field_amplitude: float, radius: float,
    dielectric: complex, omega: float,
    charge: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference source-order propagation with local Hessian width dynamics."""
    q, p = np.array(q, float, copy=True), np.array(p, float, copy=True)
    Q, P = np.array(Q, complex, copy=True), np.array(P, complex, copy=True)
    S = np.array(S, float, copy=True)
    if q.ndim != 2 or q.shape[1] != 2 or p.shape != q.shape:
        raise ValueError("q and p must have shape (N,2)")
    if Q.shape != (len(q), 2, 2) or P.shape != Q.shape or S.shape != (len(q),):
        raise ValueError("Hagedorn arrays are not aligned")
    if not np.all(np.isfinite(np.r_[q.ravel(), p.ravel(), Q.real.ravel(), Q.imag.ravel(), P.real.ravel(), P.imag.ravel(), S])):
        raise ValueError("state must be finite")
    if isinstance(n_steps, bool) or not isinstance(n_steps, Integral) or n_steps < 0:
        raise ValueError("n_steps must be a nonnegative integer")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) for v in (dt, mass)):
        raise ValueError("dt and mass must be finite reals")
    if dt <= 0 or mass <= 0:
        raise ValueError("dt and mass must be positive")
    step = float(dt); m = float(mass)
    for k in range(int(n_steps)):
        t_mid = (k + .5) * step
        q += .5 * step * p / m
        Q += .5 * step * P / m
        value, grad, hess = _oracle_evaluate_quasistatic_dipole(
            t_mid, q, field_amplitude, radius, dielectric, omega, charge
        )
        p -= .5 * step * grad
        P -= step * np.einsum("nij,njk->nik", hess, Q)
        S += step * (np.einsum("ni,ni->n", p, p) / (2 * m) - value)
        p -= .5 * step * grad
        q += .5 * step * p / m
        Q += .5 * step * P / m
    return q, p, Q, P, S

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    projection = "float((lambda a:sum(np.dot(np.asarray(x).real.ravel(),np.arange(1,np.asarray(x).size+1))+3*np.sum(np.asarray(x).imag) for x in a))({call}))"
    base = "import numpy as np\nfrom numbers import Integral, Real\nz=np.array([[-1.2,.6,2.,0.],[-.8,.9,1.7,-.2]]);g=np.diag([3.2,16.]);q=z[:,:2].copy();p=z[:,2:].copy();Q=np.broadcast_to(np.diag(1/np.sqrt(np.diag(g))),(2,2,2)).copy();P=np.broadcast_to(1j*np.diag(np.sqrt(np.diag(g))),(2,2,2)).copy();S=np.zeros(2)"
    return [
        {"setup":base, "call":projection.format(call="propagate_hagedorn_state(q,p,Q,P,S,.02,12,1.,.12,.4,-24.061+1.5068j,1.3)"), "gold_call":projection.format(call="_oracle_propagate_hagedorn_state(q,p,Q,P,S,.02,12,1.,.12,.4,-24.061+1.5068j,1.3)")},
        {"setup":base, "call":projection.format(call="propagate_hagedorn_state(q,p,Q,P,S,.02,0,1.,.12,.4,-24.061+1.5068j,1.3)"), "gold_call":projection.format(call="_oracle_propagate_hagedorn_state(q,p,Q,P,S,.02,0,1.,.12,.4,-24.061+1.5068j,1.3)")},
        {"setup":base+"\ndef status(fn):\n    try: fn(); return 0\n    except ValueError: return 1\n    except Exception: return 2", "call":"status(lambda: propagate_hagedorn_state(q,p,Q,P,S,0.,1,1.,.12,.4,-24.061+1.5068j,1.3))", "gold_call":"status(lambda: _oracle_propagate_hagedorn_state(q,p,Q,P,S,0.,1,1.,.12,.4,-24.061+1.5068j,1.3))"},
    ]
