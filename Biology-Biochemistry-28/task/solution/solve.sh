#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np
def solve_stress_free_scale(kappa: float, chi: float) -> float:
    """Reference implementation (bracketed bisection with a Newton polish)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    if not (_is_number(kappa) and kappa > 0.0):
        raise ValueError("kappa must be a finite positive number")
    if not (_is_number(chi) and chi > 0.0):
        raise ValueError("chi must be a finite positive number")
    kappa = float(kappa)
    chi = float(chi)
    # Perimeter of the unit-area regular hexagon; a hexagon of edge scale k has
    # area k**2 and perimeter k * chi_star.
    chi_star = np.sqrt(8.0 * np.sqrt(3.0))

    def _slope(k):
        return 2.0 * k * (k * k - 1.0) + kappa * chi_star * (k * chi_star - chi)

    # 2 k^3 + (kappa chi*^2 - 2) k - kappa chi* chi has one sign change in its
    # coefficients, hence one positive root; the slope is negative at k = 0 and
    # non-negative at max(1, chi / chi*).
    low = 0.0
    high = max(1.0, chi / chi_star)
    for _ in range(200):
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        if _slope(middle) < 0.0:
            low = middle
        else:
            high = middle
    root = 0.5 * (low + high)
    for _ in range(2):
        curvature = 6.0 * root * root - 2.0 + kappa * chi_star * chi_star
        if curvature <= 0.0:
            break
        step = _slope(root) / curvature
        if not (np.isfinite(step) and low <= root - step <= high):
            break
        root -= step
    return float(root)

import numpy as np
def compute_honeycomb_stresses(
    stretch: float,
    lateral_stretch: float,
    edge_scale: float,
    kappa: float,
    chi: float,
) -> "np.ndarray":
    """Reference implementation (relaxed hexagon at fixed extents)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    arguments = (("stretch", stretch), ("lateral_stretch", lateral_stretch),
                 ("edge_scale", edge_scale), ("kappa", kappa), ("chi", chi))
    for name, value in arguments:
        if not (_is_number(value) and value > 0.0):
            raise ValueError(f"{name} must be a finite positive number")
    lam = float(stretch)
    lat = float(lateral_stretch)
    k = float(edge_scale)
    chi_star = np.sqrt(8.0 * np.sqrt(3.0))
    edge = k * chi_star / 6.0
    period = np.sqrt(3.0) * edge * lam
    spacing = 1.5 * edge * lat
    # With both extents fixed the area is period * spacing whatever the shape,
    # and the perimeter 2 L / cos(phi) + 2 H - L tan(phi) of a hexagon whose
    # four slanted edges make the angle phi with the load is smallest at
    # phi = 30 degrees (every interior angle 120 degrees). Under tension that
    # shape minimises e.
    phi = np.pi / 6.0
    area = period * spacing
    perimeter = 2.0 * period / np.cos(phi) + 2.0 * spacing - period * np.tan(phi)
    perpendicular_edge = spacing - 0.5 * period * np.tan(phi)
    geometry_tolerance = 1e-12 * max(period, spacing)
    if perpendicular_edge < -geometry_tolerance:
        raise ValueError("the relaxed honeycomb has a negative load-perpendicular edge")
    if perimeter < float(chi) * (1.0 - 1e-12):
        raise ValueError("cell edges are not under tension at these stretches")
    pressure = area - 1.0
    tension = float(kappa) * (perimeter - float(chi))
    d_area = np.array([area / lam, area / lat])
    d_perimeter = np.array([(2.0 / np.cos(phi) - np.tan(phi)) * period / lam,
                            2.0 * spacing / lat])
    reference_area = np.sqrt(3.0) * 1.5 * edge * edge
    return (pressure * d_area + tension * d_perimeter) / reference_area

import numpy as np
def solve_uniaxial_state(
    stretch: float,
    edge_scale: float,
    kappa: float,
    chi: float,
) -> "np.ndarray":
    """Reference implementation (the across-load stress is affine in the lateral stretch)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    arguments = (("stretch", stretch), ("edge_scale", edge_scale),
                 ("kappa", kappa), ("chi", chi))
    for name, value in arguments:
        if not (_is_number(value) and value > 0.0):
            raise ValueError(f"{name} must be a finite positive number")
    lam = float(stretch)
    k = float(edge_scale)
    chi_star = np.sqrt(8.0 * np.sqrt(3.0))
    edge = k * chi_star / 6.0
    # The relaxed perimeter is 3 * edge * (stretch + lateral), so these two trial
    # lateral stretches keep the edges in tension.
    first = max(float(chi) / (3.0 * edge) - lam, 0.0) + 1.0
    second = first + 1.0
    across_first = compute_honeycomb_stresses(lam, first, k, kappa, chi)[1]
    across_second = compute_honeycomb_stresses(lam, second, k, kappa, chi)[1]
    slope = (across_second - across_first) / (second - first)
    if not (np.isfinite(slope) and slope > 0.0):
        raise ValueError("the across-load stress does not rise with the lateral stretch")
    lateral = first - across_first / slope
    if not (np.isfinite(lateral) and lateral > 0.0):
        raise ValueError("no positive lateral stretch unloads the lateral boundaries")
    along = compute_honeycomb_stresses(lam, lateral, k, kappa, chi)[0]
    return np.array([float(along), float(lateral)])

import numpy as np
def locate_rearrangement_stretch(
    threshold: float,
    edge_scale: float,
    kappa: float,
    chi: float,
) -> float:
    """Reference implementation (march then bisection on the edge length)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    arguments = (("threshold", threshold), ("edge_scale", edge_scale),
                 ("kappa", kappa), ("chi", chi))
    for name, value in arguments:
        if not (_is_number(value) and value > 0.0):
            raise ValueError(f"{name} must be a finite positive number")
    level = float(threshold)
    k = float(edge_scale)
    edge = k * np.sqrt(8.0 * np.sqrt(3.0)) / 6.0

    def _excess(lam):
        # At 120-degree junctions the slanted edges have length edge * stretch and
        # rise by half of it, so the load-perpendicular edge is the row spacing
        # 1.5 * edge * lateral minus that rise.
        lateral = float(solve_uniaxial_state(lam, k, kappa, chi)[1])
        return edge * (1.5 * lateral - 0.5 * lam) - level

    if not _excess(1.0) > 0.0:
        raise ValueError("the edge at stretch 1 is not longer than the threshold")
    low = 1.0
    high = None
    for lam in 1.0 + 0.01 * np.arange(1, 901):
        if _excess(float(lam)) <= 0.0:
            high = float(lam)
            break
        low = float(lam)
    if high is None:
        raise ValueError("the edge does not reach the threshold up to stretch 10")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        if _excess(middle) > 0.0:
            low = middle
        else:
            high = middle
    return float(0.5 * (low + high))

import numpy as np
def compute_nominal_stress(
    stretch: float,
    transition_stretch: float,
    edge_scale: float,
    kappa: float,
    chi: float,
) -> float:
    """Reference implementation (cellular stretch jump plus energy conjugacy)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    arguments = (("stretch", stretch), ("transition_stretch", transition_stretch),
                 ("edge_scale", edge_scale), ("kappa", kappa), ("chi", chi))
    for name, value in arguments:
        if not (_is_number(value) and value > 0.0):
            raise ValueError(f"{name} must be a finite positive number")
    lam = float(stretch)
    lam_t = float(transition_stretch)
    k = float(edge_scale)
    if not lam_t > 1.0:
        raise ValueError("transition_stretch must exceed 1")
    if lam <= lam_t:
        return float(solve_uniaxial_state(lam, k, kappa, chi)[0])
    edge = k * np.sqrt(8.0 * np.sqrt(3.0)) / 6.0
    period = np.sqrt(3.0) * edge
    spacing = 1.5 * edge
    # Two cells of extent lam_t * period become three cells whose extent along
    # the load is measured on the turned cell, i.e. on the row spacing.
    jump = 2.0 * period / (3.0 * spacing)
    cellular = jump * lam
    # The relaxed cell energy is symmetric in its two extents, so the turned
    # honeycomb with free sides is the uniaxial state at the cellular stretch;
    # the energy per reference area depends on stretch only through
    # cellular = jump * stretch, hence the factor jump.
    return float(jump * solve_uniaxial_state(cellular, k, kappa, chi)[0])

import numpy as np
def compute_stretching_work(
    stretch_start: float,
    stretch_end: float,
    transition_stretch: float,
    edge_scale: float,
    kappa: float,
    chi: float,
) -> float:
    """Reference implementation (exact difference of stored energy on each branch)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    arguments = (("stretch_start", stretch_start), ("stretch_end", stretch_end),
                 ("transition_stretch", transition_stretch), ("edge_scale", edge_scale),
                 ("kappa", kappa), ("chi", chi))
    for name, value in arguments:
        if not (_is_number(value) and value > 0.0):
            raise ValueError(f"{name} must be a finite positive number")
    lam_t = float(transition_stretch)
    k = float(edge_scale)
    if not lam_t > 1.0:
        raise ValueError("transition_stretch must exceed 1")
    chi_star = np.sqrt(8.0 * np.sqrt(3.0))
    jump = 2.0 * np.sqrt(3.0) / 4.5

    def _energy(cellular):
        # Stored energy per unit reference area of the free-sided honeycomb; its
        # derivative is the along-load nominal stress because the across-load
        # stress vanishes on this path.
        lateral = float(solve_uniaxial_state(cellular, k, kappa, chi)[1])
        area = cellular * lateral * k * k
        perimeter = 0.5 * k * chi_star * (cellular + lateral)
        value = 0.5 * (area - 1.0) ** 2 + 0.5 * float(kappa) * (perimeter - float(chi)) ** 2
        return value / (k * k)

    def _validate_cellular_path(start, end):
        """Raise if the free-lateral state is invalid anywhere on this interval."""
        low, high = sorted((float(start), float(end)))
        # Positivity and geometric feasibility reduce to endpoint checks along
        # this free-lateral branch. The sign of perimeter - chi is a convex
        # quadratic in cellular stretch whose vertex is chi / (k * chi_star).
        _energy(low)
        if high > low:
            _energy(high)
            critical = float(chi) / (k * chi_star)
            if low < critical < high:
                _energy(critical)

    start = float(stretch_start)
    end = float(stretch_end)
    direction = 1.0 if end >= start else -1.0
    low, high = (start, end) if direction > 0.0 else (end, start)
    work = 0.0

    if low <= lam_t:
        pre_end = min(high, lam_t)
        _validate_cellular_path(low, pre_end)
        work += _energy(pre_end) - _energy(low)

    if high > lam_t:
        post_start = max(low, lam_t)
        cellular_start = jump * post_start
        cellular_end = jump * high
        _validate_cellular_path(cellular_start, cellular_end)
        work += _energy(cellular_end) - _energy(cellular_start)

    return float(direction * work)

import numpy as np
def solve_propagation_state(
    transition_stretch: float,
    edge_scale: float,
    kappa: float,
    chi: float,
    tolerance: float,
) -> "np.ndarray":
    """Reference implementation (Brent roots nested in a bracketed work balance)."""
    import numpy as np
    from scipy.optimize import brentq

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    arguments = (("transition_stretch", transition_stretch), ("edge_scale", edge_scale),
                 ("kappa", kappa), ("chi", chi), ("tolerance", tolerance))
    for name, value in arguments:
        if not (_is_number(value) and value > 0.0):
            raise ValueError(f"{name} must be a finite positive number")
    if tolerance > 1e-6:
        raise ValueError("tolerance must not exceed 1e-6")
    lam_t = float(transition_stretch)
    k = float(edge_scale)

    def _stress(lam):
        return compute_nominal_stress(lam, lam_t, k, kappa, chi)

    peak = _stress(lam_t)
    above = float(np.nextafter(lam_t, np.inf))
    floor = max(_stress(1.0), _stress(above))
    if not floor < peak:
        raise ValueError("the stress does not drop below its value at the transition")

    # Find either a valid point reaching the pre-transition peak or the end
    # of the connected admissible necked branch, whichever comes first.
    necked_limit = above
    necked_limit_stress = _stress(above)
    trial = 1.05 * above
    for _ in range(500):
        try:
            trial_stress = _stress(trial)
        except ValueError:
            valid = necked_limit
            invalid = trial
            valid_stress = necked_limit_stress
            for _ in range(200):
                middle = 0.5 * (valid + invalid)
                if middle <= valid or middle >= invalid:
                    break
                try:
                    middle_stress = _stress(middle)
                except ValueError:
                    invalid = middle
                else:
                    valid = middle
                    valid_stress = middle_stress
            necked_limit = valid
            necked_limit_stress = valid_stress
            break
        necked_limit = trial
        necked_limit_stress = trial_stress
        if trial_stress >= peak:
            break
        trial *= 1.05
        if trial > 1e3:
            raise ValueError("the admissible necked branch could not be bounded")
    else:
        raise ValueError("the admissible necked branch could not be bounded")

    ceiling = min(peak, necked_limit_stress)
    if not floor < ceiling:
        raise ValueError("the necked branch has no common increasing stress range")

    def _unnecked(level):
        return brentq(lambda lam: _stress(lam) - level, 1.0, lam_t, xtol=1e-15, rtol=1e-15)

    def _necked(level):
        return brentq(lambda lam: _stress(lam) - level, above, necked_limit,
                      xtol=1e-15, rtol=1e-15)

    def _balance(level):
        u, m = _unnecked(level), _necked(level)
        return compute_stretching_work(u, m, lam_t, k, kappa, chi) - level * (m - u)

    if not (_balance(floor) > 0.0 > _balance(ceiling)):
        raise ValueError("no admissible stress balances the work")
    level = brentq(_balance, floor, ceiling, xtol=float(tolerance), rtol=1e-15)
    return np.array([float(level), float(_unnecked(level)), float(_necked(level))])

import numpy as np
def estimate_propagation_ratio(
    kappa: float = 0.35,
    chi: float = 2.45,
    threshold: float = 0.08,
    tolerance: float = 1e-13,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    arguments = (("kappa", kappa), ("chi", chi), ("threshold", threshold),
                 ("tolerance", tolerance))
    for name, value in arguments:
        if not (_is_number(value) and value > 0.0):
            raise ValueError(f"{name} must be a finite positive number")
    scale = solve_stress_free_scale(kappa, chi)
    transition = locate_rearrangement_stretch(threshold, scale, kappa, chi)
    bifurcation = compute_nominal_stress(transition, transition, scale, kappa, chi)
    # Necking bifurcates at the load maximum of the homogeneous response: the
    # stress must still rise into the rearrangement and drop just past it.
    below = compute_nominal_stress(transition * (1.0 - 1e-6), transition, scale, kappa, chi)
    above = compute_nominal_stress(float(np.nextafter(transition, np.inf)),
                                           transition, scale, kappa, chi)
    if not (below < bifurcation and above < bifurcation):
        raise ValueError("the rearrangement stretch is not a load maximum")
    state = solve_propagation_state(transition, scale, kappa, chi, tolerance)
    return float(state[0]) / float(bifurcation)
SCICODE_GOLD_EOF
