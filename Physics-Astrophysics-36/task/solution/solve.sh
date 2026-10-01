#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def order_parameter(d: int = 250, ell: int = 2) -> float:
    """Reference implementation of the large-d scalar Hankel order."""
    import math
    from numbers import Integral

    if isinstance(d, bool) or not isinstance(d, Integral):
        raise ValueError("d must be an integer, not a boolean")
    if isinstance(ell, bool) or not isinstance(ell, Integral):
        raise ValueError("ell must be an integer, not a boolean")

    d_i = int(d)
    ell_i = int(ell)
    if d_i < 10:
        raise ValueError("d must be >= 10")
    if ell_i < 0:
        raise ValueError("ell must be >= 0")

    try:
        nu = float(ell_i) + (float(d_i) - 3.0) / 2.0
    except (OverflowError, ValueError) as error:
        raise ValueError("d and ell must produce a representable Hankel order") from error

    if not math.isfinite(nu) or nu <= 0.0:
        raise ValueError("Hankel order must be finite and positive")
    return float(nu)

def mode_basis_at_match(
    d: int = 250,
    ell: int = 2,
    z: complex = 120.0 + 9.0j,
) -> tuple[complex, complex, complex, complex]:
    """Reference interface basis built from the paper's large-d solutions."""
    import math
    from scipy import special
    
    def _coerce_finite_complex(value, name: str) -> complex:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be numeric, not a boolean")
        try:
            z = complex(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be representable as a complex scalar") from error
        if not (math.isfinite(z.real) and math.isfinite(z.imag)):
            raise ValueError(f"{name} must be finite")
        return z

    nu = order_parameter(d=d, ell=ell)
    zz = _coerce_finite_complex(z, "z")
    if zz == 0.0j:
        raise ValueError("z must be nonzero")

    h = complex(special.hankel2(nu, zz))
    h_minus = complex(special.hankel2(nu - 1.0, zz))
    h_plus = complex(special.hankel2(nu + 1.0, zz))
    if not all(math.isfinite(v.real) and math.isfinite(v.imag) for v in (h, h_minus, h_plus)):
        raise ValueError("Hankel evaluations must be finite")
    if h == 0.0j:
        raise ValueError("outer basis vanishes at the matching point")

    h_prime = 0.5 * (h_minus - h_plus)
    psi_in = 1.0 + 0.0j
    dpsi_in = -1j * zz
    psi_out = h
    dpsi_out = 0.5 * h + zz * h_prime

    values = (psi_in, dpsi_in, psi_out, dpsi_out)
    if not all(math.isfinite(v.real) and math.isfinite(v.imag) for v in values):
        raise ValueError("basis values and derivatives must be finite")
    return tuple(complex(v) for v in values)

def matching_residual(
    d: int = 250,
    ell: int = 2,
    z: complex = 120.0 + 9.0j,
) -> complex:
    """Reference scale-invariant matching residual from earlier-step basis data."""
    import math

    psi_in, dpsi_in, psi_out, dpsi_out = mode_basis_at_match(d=d, ell=ell, z=z)
    if psi_in == 0.0j or psi_out == 0.0j:
        raise ValueError("basis normalization requires nonzero interface values")

    residual = dpsi_out / psi_out - dpsi_in / psi_in
    if not (math.isfinite(residual.real) and math.isfinite(residual.imag)):
        raise ValueError("matching residual must be finite")
    return complex(residual)

def continuation_seed(d: int = 250) -> complex:
    """Reference linear continuation predictor from the paper's Table II anchors."""
    import math
    from numbers import Integral

    if isinstance(d, bool) or not isinstance(d, Integral):
        raise ValueError("d must be an integer, not a boolean")
    d_i = int(d)
    if d_i < 200 or d_i > 300:
        raise ValueError("d must lie in [200, 300] for this local continuation step")

    d_lo = 200.0
    d_hi = 300.0
    z_lo = complex(96.020, 8.251)
    z_hi = complex(145.426, 9.358)
    fraction = (float(d_i) - d_lo) / (d_hi - d_lo)
    seed = z_lo + fraction * (z_hi - z_lo)

    if not (math.isfinite(seed.real) and math.isfinite(seed.imag)):
        raise ValueError("continuation predictor must be finite")
    if seed.real <= 0.0 or seed.imag <= 0.0:
        raise ValueError("continuation predictor must lie in the positive quadrant")
    return complex(seed)

def candidate_roots(
    d: int = 250,
    ell: int = 2,
    solver_tol: float = 1e-12,
    residual_tol: float = 1e-10,
    merge_tol: float = 1e-7,
) -> tuple[complex, ...]:
    """Reference multi-start root search using only earlier-step oracles."""
    import math
    import numpy as np
    from numbers import Real
    from scipy import optimize
    
    def _validate_positive_real(value, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a real numeric scalar, not a boolean")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be representable as a float") from error
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return number

    def _same_root(a: complex, b: complex, tolerance: float) -> bool:
        return abs(a - b) <= tolerance

    xtol = _validate_positive_real(solver_tol, "solver_tol")
    acceptance_tol = _validate_positive_real(residual_tol, "residual_tol")
    duplicate_tol = _validate_positive_real(merge_tol, "merge_tol")
    seed = continuation_seed(d=d)

    offsets = (0.0j, 2.0j, 3.0j)
    accepted = []

    def _residual_xy(vector):
        if len(vector) != 2:
            return np.array([1e300, 1e300], dtype=float)
        x = float(vector[0])
        y = float(vector[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            return np.array([1e300, 1e300], dtype=float)
        try:
            value = matching_residual(d=d, ell=ell, z=complex(x, y))
        except ValueError:
            return np.array([1e300, 1e300], dtype=float)
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            return np.array([1e300, 1e300], dtype=float)
        return np.array([value.real, value.imag], dtype=float)

    for offset in offsets:
        trial = seed + offset
        try:
            solution = optimize.root(
                _residual_xy,
                np.array([trial.real, trial.imag], dtype=float),
                method="hybr",
                options={"xtol": xtol, "maxfev": 2500},
            )
        except Exception:
            continue

        if solution.x is None or len(solution.x) != 2:
            continue
        root = complex(float(solution.x[0]), float(solution.x[1]))
        if not (math.isfinite(root.real) and math.isfinite(root.imag)):
            continue
        if root.real <= 0.0 or root.imag <= 0.0:
            continue

        try:
            residual_norm = abs(matching_residual(d=d, ell=ell, z=root))
        except ValueError:
            continue
        if not math.isfinite(residual_norm) or residual_norm > acceptance_tol:
            continue

        if not any(_same_root(root, existing, duplicate_tol) for existing in accepted):
            accepted.append(root)

    if not accepted:
        raise ValueError("no acceptable positive-quadrant matching root was found")

    accepted.sort(key=lambda root: (abs(root - seed), root.real, root.imag))
    return tuple(complex(root) for root in accepted)

def select_va_branch(
    d: int = 250,
    ell: int = 2,
    solver_tol: float = 1e-12,
    residual_tol: float = 1e-10,
    merge_tol: float = 1e-7,
    max_seed_distance: float = 1.0,
) -> complex:
    """Reference branch selection using only earlier-step oracles."""
    import math
    from numbers import Real

    def _validate_positive_real(value, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a real numeric scalar, not a boolean")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be representable as a float") from error
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return number

    distance_limit = _validate_positive_real(max_seed_distance, "max_seed_distance")
    residual_limit = _validate_positive_real(residual_tol, "residual_tol")
    seed = continuation_seed(d=d)
    roots = candidate_roots(
        d=d,
        ell=ell,
        solver_tol=solver_tol,
        residual_tol=residual_tol,
        merge_tol=merge_tol,
    )

    eligible = [root for root in roots if abs(root - seed) <= distance_limit]
    if not eligible:
        raise ValueError("no accepted root lies on the requested continuation branch")

    root = min(eligible, key=lambda value: (abs(value - seed), value.real, value.imag))
    residual_norm = abs(matching_residual(d=d, ell=ell, z=root))
    if not math.isfinite(residual_norm) or residual_norm > residual_limit:
        raise ValueError("selected branch root fails the independent residual check")
    if root.real <= 0.0 or root.imag <= 0.0:
        raise ValueError("selected branch root must lie in the positive quadrant")
    return complex(root)

def final_answer(
    d: int = 250,
    ell: int = 2,
    solver_tol: float = 1e-12,
    residual_tol: float = 1e-10,
    merge_tol: float = 1e-7,
    max_seed_distance: float = 1.0,
) -> float:
    """Reference final scalar using only the selected-branch oracle."""
    import math

    root = select_va_branch(
        d=d,
        ell=ell,
        solver_tol=solver_tol,
        residual_tol=residual_tol,
        merge_tol=merge_tol,
        max_seed_distance=max_seed_distance,
    )
    if root.imag <= 0.0:
        raise ValueError("selected root must have a positive imaginary part")

    answer = root.real / root.imag
    if not math.isfinite(answer) or answer <= 0.0:
        raise ValueError("final mode ratio must be finite and positive")
    return float(answer)
SCICODE_GOLD_EOF
