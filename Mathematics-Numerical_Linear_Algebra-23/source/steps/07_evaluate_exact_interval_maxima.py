"""
Compute the exact maximum of the balanced residual polynomial on every examined real-root interval and the amount by which the Hermite-spline estimate overshoots it.

The source describes the Hermite cubic $C_j$ as an estimate of the balanced residual polynomial $\pi$ between consecutive real roots, and mentions the more expensive alternative of locating the true maximum of $\pi$ on each interval from $\pi$ and $\pi'$. This step implements that alternative for the intervals that the spline construction examines.



Use apply_balance_method2 to obtain the balanced root set $\{r_1, \ldots, r_m\}$ (complex roots included), evaluate_real_root_derivatives to obtain the sorted distinct real roots with $\pi'$, and construct_hermite_spline_candidates to obtain the examined intervals $(r_j, r_{j+1})$ together with $C_j(\widehat{x}_j)$.



For each examined interval, compute the global maximum of the normalized residual polynomial $\pi(x)=\prod_k(1-x/r_k)$ on the closed interval. Retained complex-conjugate pairs and repeated roots contribute with their full multiplicities. An interval can have several local maxima of unequal heights; an interior stationary point need not be the global maximizer.



The input domain includes degrees up to 64, clusters of complex-conjugate roots close to the real axis, and critical points close to endpoints or to one another. There is no minimum critical-point separation assumption. Return each maximum and overshoot with error at most $10^{-9}\max(1,|y_{\text{ref}}|)$. The numerical representation, search strategy, and treatment of all competing extrema are part of the implementation.



The exact interval maximum is



$$

M_j = \max\left(0, \max_{x \in \text{interior critical points}} \pi(x)\right),

$$



where the $0$ accounts for the endpoint values $\pi(r_j) = \pi(r_{j+1}) = 0$. The spline overshoot is $C_j(\widehat{x}_j) - M_j$ and may be negative when the Hermite estimate is below the true maximum.



Return one row per examined interval, in the same order as the spline table, with columns



$$

[r_j,\ r_{j+1},\ M_j,\ C_j(\widehat{x}_j) - M_j].

$$



Invalid input (empty or non-one-dimensional array, non-finite entries, or a pipeline stage raising ValueError) must raise ValueError.

Returns
-------
A real NumPy array of shape (n_examined, 4) containing each examined interval, the exact maximum of the residual polynomial on it, and the Hermite-spline overshoot.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evaluate_exact_interval_maxima(
    harmonic_ritz_roots: np.ndarray,
) -> np.ndarray:
    """
    Compute exact residual-polynomial maxima on the examined intervals.

    Parameters
    ----------
    harmonic_ritz_roots : np.ndarray
        One-dimensional array of nonzero harmonic Ritz roots with
        complex roots occurring in conjugate pairs.

    Returns
    -------
    np.ndarray
        Real array of shape (n_examined, 4) with columns
        [left_root, right_root, exact_maximum, spline_overshoot].

    Raises
    ------
    ValueError
        If the root array is empty or not one-dimensional, contains
        non-finite entries, or an upstream stage rejects the input.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_evaluate_exact_interval_maxima(
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

    balanced_roots = _oracle_apply_balance_method2(
        roots
    )

    real_root_derivatives = _oracle_evaluate_real_root_derivatives(
        roots
    )

    spline_candidates = _oracle_construct_hermite_spline_candidates(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               '\n'
               'harmonic_ritz_roots = np.array([\n'
               '    -5.0,\n'
               '    -1.3,\n'
               '    0.9,\n'
               '    2.0 + 1.5j,\n'
               '    2.0 - 1.5j\n'
               '], dtype=complex)\n',
      'call': 'evaluate_exact_interval_maxima(\n    harmonic_ritz_roots\n)',
      'gold_call': '_oracle_evaluate_exact_interval_maxima(\n    harmonic_ritz_roots\n)'},
     {'setup': 'import numpy as np\n'
               '\n'
               'harmonic_ritz_roots = np.array([\n'
               '    -6.0,\n'
               '    -2.0,\n'
               '    0.8,\n'
               '    3.5\n'
               '], dtype=complex)\n',
      'call': 'evaluate_exact_interval_maxima(\n    harmonic_ritz_roots\n)',
      'gold_call': '_oracle_evaluate_exact_interval_maxima(\n    harmonic_ritz_roots\n)'},
     {'setup': 'import numpy as np\n'
               '\n'
               'harmonic_ritz_roots = np.array([\n'
               '    1.0,\n'
               '    -1.1111111111111112,\n'
               '    -20.0 + 20.0j,\n'
               '    -20.0 - 20.0j\n'
               '], dtype=complex)\n',
      'call': 'evaluate_exact_interval_maxima(\n    harmonic_ritz_roots\n)',
      'gold_call': '_oracle_evaluate_exact_interval_maxima(\n    harmonic_ritz_roots\n)'},
     {'setup': 'import numpy as np\n'
               '\n'
               'harmonic_ritz_roots = np.array([\n'
               '    0.5,\n'
               '    0.5,\n'
               '    -3.0,\n'
               '    -6.0\n'
               '], dtype=complex)\n',
      'call': 'evaluate_exact_interval_maxima(\n    harmonic_ritz_roots\n)',
      'gold_call': '_oracle_evaluate_exact_interval_maxima(\n    harmonic_ritz_roots\n)'},
     {'setup': 'import numpy as np\n'
               '\n'
               'harmonic_ritz_roots = np.array([\n'
               '    -3.0,\n'
               '    -1.0,\n'
               '    1.0678380296485983,\n'
               '    1.6094271405406886,\n'
               '    2.6866715986911203,\n'
               '    4.287768888388513,\n'
               '    6.395177052675791,\n'
               '    8.985806886701518,\n'
               '    12.03127490787994,\n'
               '    15.498214331265899,\n'
               '    19.348640643033054,\n'
               '    23.540367766756177,\n'
               '    28.027470262892436,\n'
               '    32.76078649750764,\n'
               '    37.68845726742523,\n'
               '    42.75649398050857,\n'
               '    47.90937016597429,\n'
               '    53.09062983402571,\n'
               '    58.243506019491434,\n'
               '    63.31154273257478,\n'
               '    68.23921350249236,\n'
               '    72.97252973710758,\n'
               '    77.45963223324384,\n'
               '    81.65135935696695,\n'
               '    85.5017856687341,\n'
               '    88.96872509212005,\n'
               '    92.01419311329849,\n'
               '    94.60482294732421,\n'
               '    96.71223111161149,\n'
               '    98.31332840130888,\n'
               '    99.39057285945933,\n'
               '    99.9321619703514\n'
               '], dtype=complex)\n',
      'call': 'evaluate_exact_interval_maxima(\n    harmonic_ritz_roots\n)',
      'gold_call': '_oracle_evaluate_exact_interval_maxima(\n    harmonic_ritz_roots\n)'},
     {'setup': 'import numpy as np\n'
               'harmonic_ritz_roots = np.array([1/1024, 1., 1j, -1j], dtype=complex)\n',
      'call': 'evaluate_exact_interval_maxima(harmonic_ritz_roots)',
      'gold_call': '_oracle_evaluate_exact_interval_maxima(harmonic_ritz_roots)'},
     {'setup': 'import numpy as np\n'
               'harmonic_ritz_roots = np.array([1/1024, 1., .25j, -.25j], dtype=complex)\n',
      'call': 'evaluate_exact_interval_maxima(harmonic_ritz_roots)',
      'gold_call': '_oracle_evaluate_exact_interval_maxima(harmonic_ritz_roots)'},
     {'setup': 'import numpy as np\n'
               'centers = np.cos((2*np.arange(30)+1)*np.pi/60)\n'
               'centers = (centers-centers[::-1])/2\n'
               'harmonic_ritz_roots = np.r_[1/1024, 1., centers+1e-3j, centers-1e-3j]\n',
      'call': 'evaluate_exact_interval_maxima(harmonic_ritz_roots)',
      'gold_call': '_oracle_evaluate_exact_interval_maxima(harmonic_ritz_roots)'},
     {'setup': 'import numpy as np\n'
               'centers = 0.9*np.cos((2*np.arange(30)+1)*np.pi/60)+0.05\n'
               'harmonic_ritz_roots = np.r_[1/1024, 1., centers+1e-5j, centers-1e-5j]\n',
      'call': 'evaluate_exact_interval_maxima(harmonic_ritz_roots)',
      'gold_call': '_oracle_evaluate_exact_interval_maxima(harmonic_ritz_roots)'},
     {'setup': 'import numpy as np\n'
               'harmonic_ritz_roots = np.array([1/1024, -2., 1., -.4+1e-5j, -.4-1e-5j, .3+2e-5j, '
               '.3-2e-5j])\n',
      'call': 'evaluate_exact_interval_maxima(harmonic_ritz_roots)',
      'gold_call': '_oracle_evaluate_exact_interval_maxima(harmonic_ritz_roots)'},
     {'setup': 'import numpy as np\n'
               '\n'
               '\n'
               'def _run_invalid(fn, *args):\n'
               '    try:\n'
               '        fn(*args)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '\n'
               'import numpy as np\n'
               '\n'
               'harmonic_ritz_roots = np.array([\n'
               '    -5.0,\n'
               '    -1.3,\n'
               '    2.0 + 1.5j\n'
               '], dtype=complex)\n',
      'call': '_run_invalid(\n    evaluate_exact_interval_maxima, harmonic_ritz_roots\n)',
      'gold_call': '_run_invalid(\n    _oracle_evaluate_exact_interval_maxima, harmonic_ritz_roots\n)'}]
