"""
Return potential, gradient, and Hessian for one or many two-dimensional points.

Local harmonic evolution differentiates the quasistatic scalar field that is linear inside the nanorod and dipolar outside it.

Returns
-------
tuple[np.ndarray, np.ndarray, np.ndarray]: aligned finite potential, gradient, and Hessian arrays.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Real
import numpy as np
def evaluate_quasistatic_dipole(
    t: float, positions: np.ndarray,
    field_amplitude: float, radius: float,
    dielectric: complex, omega: float,
    charge: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the oscillating nanorod potential and its first two derivatives.

    Positions end in dimension two. With
    ``c=abs((dielectric-1)/(dielectric+1))``, define the scalar field as
    ``phi=field_amplitude*c*x`` for ``x*x+y*y < radius**2`` and
    ``phi=field_amplitude*c*radius**2*x/(x*x+y*y)`` otherwise, including the
    boundary. Return the potential ``-charge*cos(omega*t)*phi`` and its spatial
    gradient and Hessian. ``dielectric`` enters only through ``c``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Potential ``(...)``, gradient ``(...,2)``, and Hessian ``(...,2,2)``.

    Raises
    ------
    ValueError
        If positions do not end in dimension two, any input is non-finite,
        ``radius`` is not positive, or the dielectric contrast is singular.
    """
    return potential, gradient, hessian

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from numbers import Real
import numpy as np
def _oracle_evaluate_quasistatic_dipole(
    t: float, positions: np.ndarray,
    field_amplitude: float, radius: float,
    dielectric: complex, omega: float,
    charge: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference implementation of Eqs. (2)-(3) and analytic derivatives."""
    r = np.asarray(positions, float)
    scalars = (t, field_amplitude, radius, omega, charge)
    if r.ndim < 1 or r.shape[-1] != 2 or not np.all(np.isfinite(r)):
        raise ValueError("positions must be finite with final dimension two")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) for v in scalars):
        raise ValueError("real scalar inputs must be finite")
    if float(radius) <= 0 or not np.isfinite(complex(dielectric)):
        raise ValueError("radius must be positive and dielectric finite")
    eps = complex(dielectric)
    if abs(eps + 1) == 0:
        raise ValueError("dielectric contrast is singular")
    x, y = r[..., 0], r[..., 1]
    s = x * x + y * y
    inside = s < float(radius) ** 2
    safe_s = np.where(inside, 1.0, s)
    contrast = abs((eps - 1) / (eps + 1))
    base = float(field_amplitude) * contrast
    k = base * float(radius) ** 2
    phi = np.where(inside, base * x, k * x / safe_s)
    grad = np.zeros(r.shape, float)
    grad[..., 0] = np.where(inside, base, k * (y * y - x * x) / safe_s ** 2)
    grad[..., 1] = np.where(inside, 0.0, -2 * k * x * y / safe_s ** 2)
    hess = np.zeros(r.shape[:-1] + (2, 2), float)
    hxx = 2 * k * x * (x * x - 3 * y * y) / safe_s ** 3
    hxy = 2 * k * y * (3 * x * x - y * y) / safe_s ** 3
    hess[..., 0, 0] = np.where(inside, 0.0, hxx)
    hess[..., 0, 1] = hess[..., 1, 0] = np.where(inside, 0.0, hxy)
    hess[..., 1, 1] = np.where(inside, 0.0, -hxx)
    factor = -float(charge) * np.cos(float(omega) * float(t))
    return factor * phi, factor * grad, factor * hess

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    projection = "(lambda a:float(np.sum(a[0])+np.dot(a[1].ravel(),np.arange(1,{g}+1))+np.dot(a[2].ravel(),np.arange(1,{h}+1))))({call})"
    return [
        {"setup":"import numpy as np\nfrom numbers import Real\nr=np.array([[1.2,.5],[-.9,.7]])", "call":projection.format(call="evaluate_quasistatic_dipole(.3,r,.12,.4,-24.061+1.5068j,1.3)",g=4,h=8), "gold_call":projection.format(call="_oracle_evaluate_quasistatic_dipole(.3,r,.12,.4,-24.061+1.5068j,1.3)",g=4,h=8)},
        {"setup":"import numpy as np\nfrom numbers import Real\nr=np.array([[.4,0.]])", "call":projection.format(call="evaluate_quasistatic_dipole(0.,r,.2,.4,-3+1j,2.)",g=2,h=4), "gold_call":projection.format(call="_oracle_evaluate_quasistatic_dipole(0.,r,.2,.4,-3+1j,2.)",g=2,h=4)},
        {"setup":"import numpy as np\nfrom numbers import Real\nr=np.array([[.1,.1]])\ndef status(fn):\n    try: fn(); return 0\n    except ValueError: return 1\n    except Exception: return 2", "call":"status(lambda: evaluate_quasistatic_dipole(0.,r,.1,0.,-3+1j,1.))", "gold_call":"status(lambda: _oracle_evaluate_quasistatic_dipole(0.,r,.1,0.,-3+1j,1.))"},
    ]
