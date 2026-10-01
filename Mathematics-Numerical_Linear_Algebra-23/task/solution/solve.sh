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


def evaluate_balance_candidates(
    harmonic_ritz_roots,
):
    roots = np.asarray(harmonic_ritz_roots, dtype=complex)

    if roots.ndim != 1 or roots.size == 0:
        raise ValueError(
            "harmonic_ritz_roots must be a non-empty one-dimensional array."
        )

    if not np.all(
        np.isfinite(roots.real) & np.isfinite(roots.imag)
    ):
        raise ValueError(
            "harmonic_ritz_roots must contain only finite values."
        )

    tol = 1.0e-12

    if np.any(np.abs(roots) <= tol):
        raise ValueError(
            "harmonic Ritz roots must be nonzero."
        )

    reciprocal_sum = np.sum(1.0 / roots)

    scale = max(
        1.0,
        abs(reciprocal_sum),
    )

    if abs(reciprocal_sum.imag) > tol * scale:
        raise ValueError(
            "The reciprocal sum must be real for a conjugate-symmetric root set."
        )

    phi_prime_0 = float(reciprocal_sum.real)

    used = np.zeros(
        roots.size,
        dtype=bool,
    )

    candidates = []

    for i, root in enumerate(roots):
        if used[i]:
            continue

        root_scale = max(
            1.0,
            abs(root),
        )

        if abs(root.imag) <= tol * root_scale:
            contribution = float(
                (1.0 / root).real
            )

            candidates.append(
                (
                    i,
                    -1,
                    1,
                    contribution,
                )
            )

            used[i] = True
            continue

        partner = None
        target = np.conjugate(root)

        for j in range(i + 1, roots.size):
            if used[j]:
                continue

            pair_scale = max(
                1.0,
                abs(target),
                abs(roots[j]),
            )

            if abs(roots[j] - target) <= tol * pair_scale:
                partner = j
                break

        if partner is None:
            raise ValueError(
                "Every nonreal harmonic Ritz root must have a complex-conjugate partner."
            )

        pair_contribution = (
            1.0 / root
            +
            1.0 / roots[partner]
        )

        pair_scale = max(
            1.0,
            abs(pair_contribution),
        )

        if abs(pair_contribution.imag) > tol * pair_scale:
            raise ValueError(
                "A conjugate-pair reciprocal contribution must be real."
            )

        contribution = float(
            pair_contribution.real
        )

        candidates.append(
            (
                i,
                partner,
                2,
                contribution,
            )
        )

        used[i] = True
        used[partner] = True

    table = np.zeros(
        (len(candidates), 7),
        dtype=float,
    )

    for row, (
        primary,
        partner,
        multiplicity,
        contribution,
    ) in enumerate(candidates):
        difference = abs(
            phi_prime_0
            -
            contribution
        )

        table[row, 0] = primary
        table[row, 1] = partner
        table[row, 2] = multiplicity
        table[row, 3] = contribution
        table[row, 4] = difference
        table[row, 5] = phi_prime_0

    selected_row = min(
        range(table.shape[0]),
        key=lambda k: (
            table[k, 4],
            table[k, 0],
        ),
    )

    table[selected_row, 6] = 1.0

    return table

import numpy as np


def apply_balance_method2(
    harmonic_ritz_roots,
):
    roots = np.asarray(
        harmonic_ritz_roots,
        dtype=complex,
    )

    if roots.ndim != 1 or roots.size == 0:
        raise ValueError(
            "harmonic_ritz_roots must be a non-empty one-dimensional array."
        )

    if not np.all(
        np.isfinite(roots.real)
        & np.isfinite(roots.imag)
    ):
        raise ValueError(
            "harmonic_ritz_roots must contain only finite values."
        )

    candidate_table = evaluate_balance_candidates(
        roots
    )

    selected_rows = np.flatnonzero(
        candidate_table[:, 6] == 1.0
    )

    if selected_rows.size != 1:
        raise ValueError(
            "Exactly one Balance Method 2 candidate must be selected."
        )

    row = int(selected_rows[0])

    primary_index = int(
        candidate_table[row, 0]
    )

    partner_index = int(
        candidate_table[row, 1]
    )

    multiplicity = int(
        candidate_table[row, 2]
    )

    xi = float(
        candidate_table[row, 3]
    )

    selected_difference = float(
        candidate_table[row, 4]
    )

    phi_prime_0 = float(
        candidate_table[row, 5]
    )

    remove_selected = (
        selected_difference
        <
        abs(phi_prime_0)
    )

    keep = np.ones(
        roots.size,
        dtype=bool,
    )

    if remove_selected:
        keep[primary_index] = False

        if multiplicity == 2:
            if partner_index < 0:
                raise ValueError(
                    "A conjugate-pair candidate must specify its partner."
                )

            keep[partner_index] = False

        elif multiplicity != 1:
            raise ValueError(
                "Candidate multiplicity must be one or two."
            )

        retained_sum = (
            phi_prime_0
            -
            xi
        )

    else:
        retained_sum = phi_prime_0

    tol = 1.0e-12

    if abs(retained_sum) <= tol:
        raise ValueError(
            "The balancing-root denominator is numerically zero."
        )

    eta = -1.0 / retained_sum

    balanced_roots = np.concatenate((
        roots[keep],
        np.array(
            [complex(eta, 0.0)],
            dtype=complex,
        ),
    ))

    return balanced_roots

import numpy as np


def build_balanced_residual_polynomial(
    harmonic_ritz_roots,
):
    roots = np.asarray(
        harmonic_ritz_roots,
        dtype=complex,
    )

    if roots.ndim != 1 or roots.size == 0:
        raise ValueError(
            "harmonic_ritz_roots must be a non-empty one-dimensional array."
        )

    if not np.all(
        np.isfinite(roots.real)
        & np.isfinite(roots.imag)
    ):
        raise ValueError(
            "harmonic_ritz_roots must contain only finite values."
        )

    balanced_roots = apply_balance_method2(
        roots
    )

    tol = 1.0e-12

    if np.any(np.abs(balanced_roots) <= tol):
        raise ValueError(
            "Balanced polynomial roots must be nonzero."
        )

    coefficients = np.array(
        [1.0 + 0.0j],
        dtype=complex,
    )

    for root in balanced_roots:
        factor = np.array(
            [
                1.0 + 0.0j,
                -1.0 / root,
            ],
            dtype=complex,
        )

        coefficients = np.convolve(
            coefficients,
            factor,
        )

    scale = max(
        1.0,
        float(np.max(np.abs(coefficients))),
    )

    if np.max(np.abs(coefficients.imag)) > tol * scale:
        raise ValueError(
            "The balanced polynomial coefficients must be real for a conjugate-symmetric root set."
        )

    return coefficients.real.astype(float)

import numpy as np


def evaluate_real_root_derivatives(
    harmonic_ritz_roots,
):
    roots = np.asarray(
        harmonic_ritz_roots,
        dtype=complex,
    )

    if roots.ndim != 1 or roots.size == 0:
        raise ValueError(
            "harmonic_ritz_roots must be a non-empty one-dimensional array."
        )

    if not np.all(
        np.isfinite(roots.real)
        & np.isfinite(roots.imag)
    ):
        raise ValueError(
            "harmonic_ritz_roots must contain only finite values."
        )

    balanced_roots = apply_balance_method2(
        roots
    )

    tol = 1.0e-12

    real_roots = []

    for root in balanced_roots:
        scale = max(
            1.0,
            abs(root),
        )

        if abs(root.imag) > tol * scale:
            continue

        value = float(root.real)

        duplicate = any(
            abs(value - previous)
            <= tol * max(1.0, abs(value), abs(previous))
            for previous in real_roots
        )

        if not duplicate:
            real_roots.append(value)

    if len(real_roots) < 2:
        raise ValueError(
            "At least two distinct real balanced roots are required for spline screening."
        )

    real_roots = np.array(
        sorted(real_roots),
        dtype=float,
    )

    def _product_derivative(value):
        from math import fsum, log, exp
        matching = (np.abs(balanced_roots - value)
                    <= tol * np.maximum(1.0, np.maximum(abs(value), np.abs(balanced_roots))))
        if np.count_nonzero(matching) > 1:
            return 0.0
        others = balanced_roots[~matching]
        # Subtract before dividing to retain nearby-root differences. Accumulate
        # magnitudes in logarithms so intermediate products cannot overflow.
        factors = (others - value) / others
        magnitudes = np.abs(factors)
        phase = np.prod(factors / magnitudes).real
        logarithm = fsum([*(log(float(a)) for a in magnitudes), -log(abs(value))])
        return -np.sign(value) * float(phase) * exp(logarithm)

    derivatives = np.array(
        [
            _product_derivative(root)
            for root in real_roots
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(derivatives)):
        raise ValueError(
            "Derivative evaluations must be finite."
        )

    return np.column_stack((
        real_roots,
        derivatives,
    ))

import numpy as np


def construct_hermite_spline_candidates(
    real_root_derivatives,
):
    data = np.asarray(
        real_root_derivatives,
        dtype=float,
    )

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(
            "real_root_derivatives must have shape (n_real, 2)."
        )

    if data.shape[0] < 2:
        raise ValueError(
            "At least two real roots are required."
        )

    if not np.all(np.isfinite(data)):
        raise ValueError(
            "real_root_derivatives must contain only finite values."
        )

    roots = data[:, 0]
    derivatives = data[:, 1]

    if np.any(np.diff(roots) <= 0.0):
        raise ValueError(
            "Real roots must be strictly increasing."
        )

    tol = 1.0e-12
    rows = []

    for j in range(roots.size - 1):
        left = float(roots[j])
        right = float(roots[j + 1])

        m_left = float(derivatives[j])
        m_right = float(derivatives[j + 1])

        scale = max(
            1.0,
            abs(m_left),
            abs(m_right),
        )

        if m_left < -tol * scale:
            continue

        if (
            abs(m_left) <= tol * scale
            and
            abs(m_right) <= tol * scale
        ):
            continue

        h = right - left

        a = (
            6.0
            * (m_right + m_left)
            / (h * h)
        )

        b = (
            -(
                2.0 * m_right
                +
                4.0 * m_left
            )
            / h
        )

        c = m_left

        coefficient_scale = max(
            1.0,
            abs(a),
            abs(b),
            abs(c),
        )

        x_tol = tol * max(
            1.0,
            h,
        )

        if abs(a) <= tol * coefficient_scale:
            if abs(b) <= tol * coefficient_scale:
                raise ValueError(
                    "A degenerate spline must have a nonzero linear derivative coefficient."
                )

            offsets = [-c / b]

        else:
            discriminant = (
                b * b
                -
                2.0 * a * c
            )

            disc_scale = max(
                1.0,
                b * b,
                abs(2.0 * a * c),
            )

            if discriminant < -tol * disc_scale:
                raise ValueError(
                    "The eligible spline has no real critical point."
                )

            discriminant = max(
                discriminant,
                0.0,
            )

            sqrt_disc = np.sqrt(
                discriminant
            )

            offsets = [
                (-b - sqrt_disc) / a,
                (-b + sqrt_disc) / a,
            ]

        maxima = []

        for t in offsets:
            if not (
                t > x_tol
                and
                t < h - x_tol
            ):
                continue

            if a * t + b >= 0.0:
                continue

            if not any(
                abs(t - previous) <= x_tol
                for previous in maxima
            ):
                maxima.append(
                    float(t)
                )

        if len(maxima) == 0:
            continue

        if len(maxima) != 1:
            raise ValueError(
                "A cubic spline cannot have two interior local maxima."
            )

        t = maxima[0]

        x_hat = left + t

        c_value = (
            (a / 6.0) * t**3
            +
            (b / 2.0) * t**2
            +
            c * t
        )

        rows.append([
            left,
            right,
            m_left,
            m_right,
            a,
            b,
            c,
            x_hat,
            c_value,
        ])

    if not rows:
        return np.empty(
            (0, 9),
            dtype=float,
        )

    return np.asarray(
        rows,
        dtype=float,
    )

import numpy as np


def apply_ritz_interval_screening(
    spline_candidates,
    ritz_min,
    ritz_max,
):
    table = np.asarray(
        spline_candidates,
        dtype=float,
    )

    ritz_min = float(ritz_min)
    ritz_max = float(ritz_max)

    if table.ndim != 2 or table.shape[1] != 9:
        raise ValueError(
            "spline_candidates must have shape (n_examined, 9)."
        )

    if not np.all(np.isfinite(table)):
        raise ValueError(
            "spline_candidates must contain only finite values."
        )

    if not np.isfinite(ritz_min) or not np.isfinite(ritz_max):
        raise ValueError(
            "Ritz extrema must be finite."
        )

    if ritz_min > ritz_max:
        raise ValueError(
            "ritz_min must not exceed ritz_max."
        )

    rows = []

    for candidate in table:
        left = float(candidate[0])
        right = float(candidate[1])

        a = float(candidate[4])
        b = float(candidate[5])
        c = float(candidate[6])

        x_hat = float(candidate[7])
        c_hat = float(candidate[8])

        if not left < x_hat < right:
            raise ValueError(
                "Each selected critical point must lie strictly inside its interval."
            )

        above_one = (
            c_hat > 1.0
        )

        def _spline_value(x):
            t = x - left

            return (
                (a / 6.0) * t**3
                +
                (b / 2.0) * t**2
                +
                c * t
            )

        condition_1 = False

        if (
            x_hat <= ritz_min <= right
        ):
            condition_1 = (
                _spline_value(ritz_min)
                >
                1.0
            )

        condition_2 = False

        if (
            left <= ritz_max <= x_hat
        ):
            condition_2 = (
                _spline_value(ritz_max)
                >
                1.0
            )

        condition_3 = (
            not (
                x_hat
                <
                ritz_min
                <
                right
            )
            and
            not (
                left
                <
                ritz_max
                <
                x_hat
            )
        )

        flagged = (
            above_one
            and
            (
                condition_1
                or
                condition_2
                or
                condition_3
            )
        )

        margin = (
            c_hat - 1.0
            if flagged
            else 0.0
        )

        rows.append([
            left,
            right,
            x_hat,
            c_hat,
            float(condition_1),
            float(condition_2),
            float(condition_3),
            float(flagged),
            margin,
        ])

    if not rows:
        return np.empty(
            (0, 9),
            dtype=float,
        )

    return np.asarray(
        rows,
        dtype=float,
    )

import numpy as np


def evaluate_exact_interval_maxima(
    harmonic_ritz_roots,
):
    roots = np.asarray(
        harmonic_ritz_roots,
        dtype=complex,
    )

    if roots.ndim != 1 or roots.size == 0:
        raise ValueError(
            "harmonic_ritz_roots must be a non-empty one-dimensional array."
        )

    if not np.all(
        np.isfinite(roots.real)
        & np.isfinite(roots.imag)
    ):
        raise ValueError(
            "harmonic_ritz_roots must contain only finite values."
        )

    balanced_roots = apply_balance_method2(
        roots
    )

    real_root_derivatives = evaluate_real_root_derivatives(
        roots
    )

    spline_candidates = construct_hermite_spline_candidates(
        real_root_derivatives
    )

    rows = []
    for candidate in spline_candidates:
        left, right = candidate[:2]
        maximum, overshoot = _bernstein_interval_maximum(balanced_roots, left, right, candidate[8])
        rows.append([left, right, maximum, overshoot])
    return np.asarray(rows, dtype=float).reshape(-1, 4)


def _bernstein_interval_maximum(roots, left, right, spline_value):
    """Global branch-and-bound using the Bernstein convex-hull property.

    Decimal arithmetic protects the local polynomial construction; no power-
    basis root solve or sampling density assumption enters the bound.
    """
    from decimal import Decimal, localcontext
    from math import comb
    import heapq

    with localcontext() as context:
        context.prec = 80
        D = Decimal.from_float
        lo, hi = D(float(left)), D(float(right))
        coefficients = [Decimal(1)]
        used = set()
        for index, root in enumerate(roots):
            if index in used:
                continue
            a = D(float(root.real))
            if abs(root.imag) <= 1e-12 * max(1.0, abs(root)):
                factor = [1 - lo / a, 1 - hi / a]
            else:
                partner = next(j for j in range(index + 1, len(roots))
                               if j not in used and abs(roots[j] - root.conjugate())
                               <= 1e-12 * max(1.0, abs(root), abs(roots[j])))
                used.add(partner)
                b = D(float(root.imag))
                denominator = a*a + b*b
                factor = [((lo-a)**2+b*b)/denominator,
                          ((lo-a)*(hi-a)+b*b)/denominator,
                          ((hi-a)**2+b*b)/denominator]
            n, m = len(coefficients)-1, len(factor)-1
            product = [Decimal(0)] * (n+m+1)
            for i, value in enumerate(coefficients):
                for j, weight in enumerate(factor):
                    product[i+j] += (value * weight * comb(n, i) * comb(m, j)
                                     / comb(n+m, i+j))
            coefficients = product

        # Endpoints are roots by contract. Rounding of conjugate inputs can
        # leave negligible residuals; include them conservatively.
        best = max(Decimal(0), coefficients[0], coefficients[-1])
        queue = [(-max(coefficients), 0, coefficients)]
        serial = 0
        while queue:
            negative_bound, _, values = heapq.heappop(queue)
            tolerance = Decimal('1e-13') * min(max(Decimal(1), abs(best)),
                                              max(Decimal(1), abs(D(float(spline_value))-best)))
            if -negative_bound <= best + tolerance:
                break
            lower, upper = [values[0]], [values[-1]]
            work = values
            while len(work) > 1:
                work = [(a+b)/2 for a, b in zip(work, work[1:])]
                lower.append(work[0])
                upper.append(work[-1])
            upper.reverse()
            best = max(best, lower[-1])
            for child in (lower, upper):
                bound = max(child)
                if bound > best + tolerance:
                    serial += 1
                    heapq.heappush(queue, (-bound, serial, child))
        return float(best), float(D(float(spline_value)) - best)

import numpy as np


def evaluate_final_diagnostic(
    harmonic_ritz_roots,
    ritz_min,
    ritz_max,
):
    roots = np.asarray(
        harmonic_ritz_roots,
        dtype=complex,
    )

    ritz_min = float(ritz_min)
    ritz_max = float(ritz_max)

    if roots.ndim != 1 or roots.size == 0:
        raise ValueError(
            "harmonic_ritz_roots must be a non-empty one-dimensional array."
        )

    if not np.all(
        np.isfinite(roots.real)
        & np.isfinite(roots.imag)
    ):
        raise ValueError(
            "harmonic_ritz_roots must contain only finite values."
        )

    if not np.isfinite(ritz_min) or not np.isfinite(ritz_max):
        raise ValueError(
            "Ritz extrema must be finite."
        )

    if ritz_min > ritz_max:
        raise ValueError(
            "ritz_min must not exceed ritz_max."
        )

    candidate_table = evaluate_balance_candidates(
        roots
    )

    selected_count = np.count_nonzero(
        candidate_table[:, 6] == 1.0
    )

    if selected_count != 1:
        raise ValueError(
            "Exactly one Balance Method 2 candidate must be selected."
        )

    balanced_roots = apply_balance_method2(
        roots
    )

    coefficients = build_balanced_residual_polynomial(
        roots
    )

    if coefficients.size != balanced_roots.size + 1:
        raise ValueError(
            "Polynomial degree is inconsistent with the balanced root count."
        )

    real_root_derivatives = evaluate_real_root_derivatives(
        roots
    )

    spline_candidates = construct_hermite_spline_candidates(
        real_root_derivatives
    )

    screening = apply_ritz_interval_screening(
        spline_candidates,
        ritz_min,
        ritz_max,
    )

    exact_maxima = evaluate_exact_interval_maxima(
        roots
    )

    if exact_maxima.shape[0] != screening.shape[0]:
        raise ValueError(
            "The exact-maximum table must have one row per examined interval."
        )

    if screening.shape[0] == 0:
        return 0.0

    margins = screening[:, 8]

    if not np.all(np.isfinite(margins)):
        raise ValueError(
            "Screening margins must be finite."
        )

    Q = float(
        np.max(margins)
    )

    if Q < 0.0:
        raise ValueError(
            "Exceedance margins must be nonnegative."
        )

    return Q
SCICODE_GOLD_EOF
