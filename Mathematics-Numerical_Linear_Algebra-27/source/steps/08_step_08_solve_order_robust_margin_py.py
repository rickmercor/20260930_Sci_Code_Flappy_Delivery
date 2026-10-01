"""
Run the full exact-region, mixed-precision batch calculation and return the largest guaranteed robustness margin.

The orchestrator connects the rounded-factor partition to the stateful batch solver, labels all safe strata, and compares the relative margins of every safe component for every supplied configuration. Endpoint inclusion must be retained when deciding attainment. Compute the supremal margin from the supplied inputs; do not substitute a stored expected result.



For each configuration theta, G(lambda;theta) is the minimum matching fixed/adaptive true-residual ratio over every ordering and batch position. Use one common theta on the entire uncertainty interval; the result is the largest supremal margin of the safe components.

Returns
-------
float: unrounded supremal relative uncertainty half-width in parts per million; the prompt displays three decimals.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_order_robust_margin(domain: tuple = ((287, 256), (289, 256)),
                             configurations: tuple | None = None, threshold: float = 2.0) -> float:
    """Compute the complete benchmark margin in parts per million.

    Parameters
    ----------
    domain : tuple
        Closed rational-pair subinterval of [287/256,289/256], including a
        singleton if requested. Use the factor_rounding_regions descriptor.
    configurations : tuple or None
        Nonempty tuple of distinct (m2,m3,c) triples with (m2,m3) among
        (3,4),(4,3),(6,2) and c among 5,11,17,23,31. None means all 15 triples,
        ordered by that nesting order and increasing c.
    threshold : float
        Finite positive required fixed/adaptive true-residual ratio.
        The prompt uses 2.0. Other values are test instances of the same
        robustness calculation, not changes to the solver algorithm.

        Construct the ORIGINAL n=48 matrix using zero-based indices modulo
        48: A[i,i]=1+(1+i%3)/4096, A[i,i+1]=-7/16, A[i,i-1]=-3/16,
        A[i,i+8]=-1/4, A[i,i-8]=-1/8; unspecified entries are zero.
        The three rhs rows have entries:
        b0[i]=1+(-1)**i/4+(i%7)/32,
        b1[i]=(-1)**i*(1+(i%5)/16),
        b2[i]=(((3*i)%11)-5)/8+(i%2)/32.
        Use four-row block factors, three nested FGMRES levels, two
        Richardson corrections, and exactly two outer cycles of length two.
        For every factor state and triple evaluate all ordered batches using
        batch_residuals. The gain is the minimum matching fixed/adaptive
        ratio over all permutations and positions. Resolve every real
        multiplier through the exact factor partition, not a sampled grid.

    Returns
    -------
    margin_ppm : float
        Native Python float within 1e-6 absolute error of the supremal
        1e6*d, for some common configuration and interval
        [alpha*(1-d), alpha*(1+d)] wholly contained in the safe domain.
        Internal values are not rounded to three decimals. A safe singleton
        has margin 0.0. An empty safe set has no finite supremal margin and
        raises ValueError rather than returning an invented zero.

    Raises
    ------
    ValueError
        If domain is not a valid closed rational subinterval, configurations
        are empty, duplicated or outside the allowed set, threshold is not
        finite and positive, any child calculation has a documented
        breakdown/nonfinite result, any adaptive residual used as a ratio
        denominator is zero, or the safe set is empty.
    """
    return margin_ppm  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _f3r_benchmark_data():
    n = 48
    i = np.arange(n)
    A = np.zeros((n,n), dtype=np.float64)
    A[i,i] = 1 + (1+i%3)/4096
    for offset,value in ((1,-7/16),(-1,-3/16),(8,-1/4),(-8,-1/8)):
        A[i,(i+offset)%n] = value
    B = np.stack((1+(-1.0)**i/4+(i%7)/32,
                  (-1.0)**i*(1+(i%5)/16),
                  (((3*i)%11)-5)/8+(i%2)/32))
    return A,B

def _oracle_solve_order_robust_margin(domain: tuple = ((287, 256), (289, 256)),
                                     configurations: tuple | None = None, threshold: float = 2.0) -> float:
    if not isinstance(domain, (tuple,list)) or len(domain) != 2:
        raise ValueError("domain must contain two rational endpoints")
    a,b = map(_f3r_fraction,domain)
    if not _Fraction(287,256) <= a <= b <= _Fraction(289,256):
        raise ValueError("domain is outside the benchmark interval")
    allowed = tuple((u,v,c) for u,v in ((3,4),(4,3),(6,2)) for c in (5,11,17,23,31))
    if configurations is None:
        configs = allowed
    else:
        if not isinstance(configurations,(tuple,list)) or not configurations:
            raise ValueError("configurations must be nonempty")
        configs = []
        for row in configurations:
            if not isinstance(row,(tuple,list)) or len(row) != 3:
                raise ValueError("Invalid configuration")
            row = tuple(_f3r_integer(v,'configuration entry') for v in row)
            if row not in allowed or row in configs:
                raise ValueError("Invalid or duplicate configuration")
            configs.append(row)
        configs = tuple(configs)
    try:
        threshold = float(threshold)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError("threshold must be positive and finite") from exc
    if threshold <= 0 or not np.isfinite(threshold):
        raise ValueError("threshold must be positive and finite")
    A,B = _f3r_benchmark_data()
    boundaries,strata,lower,upper = _oracle_factor_rounding_regions(np.diag(A),domain)
    gains = np.empty((len(lower), len(configs)), dtype=np.float64)
    # Batch independent factor states. Each row keeps its own adaptive history.
    # Reuse only fixed-weight controls with the same nesting, matrix and factors.
    # Bound temporary memory while batching independent factor states.
    for start in range(0, len(lower), 64):
        stop = min(start + 64, len(lower))
        fixed_controls = {}
        for j, theta in enumerate(configs):
            nesting = theta[:2]
            fixed, adaptive, weights, counters = _f3r_gold_batch_residuals_many(
                A, B, lower[start:stop], upper[start:stop], theta, 2, 2,
                fixed_controls.get(nesting))
            if nesting not in fixed_controls:
                fixed_controls[nesting] = fixed[:, 0, :].copy()
            if np.any(adaptive <= 0):
                raise ValueError("Adaptive residual denominator must be positive")
            ratios = np.divide(fixed, adaptive, dtype=np.float64)
            gains[start:stop, j] = np.min(ratios, axis=(1, 2))
    components = _oracle_safe_component_margins(boundaries,strata,gains,threshold)
    if len(components) == 0:
        raise ValueError("No safe multiplier exists")
    low,high = float(components[:,5].max()),float(components[:,6].max())
    return float(low + (high-low)/2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and declared-invalid test specifications."""
    return [
        # normal: continuous safe subinterval
        {'setup': '\n'
                  'import math\n'
                  'def _case_scalar(actual):\n'
                  '    return int(type(actual) is float and math.isfinite(actual) and abs(actual-(44.5335114673792)) '
                  '<= 1e-6)\n',
         'call': '_case_scalar(solve_order_robust_margin(((11227, 10000), (11228, 10000)),((4, 3, 5),),2.0))',
         'gold_call': '_case_scalar(_oracle_solve_order_robust_margin(((11227, 10000), (11228, 10000)),((4, 3, '
                      '5),),2.0))'},
        # boundary: safe singleton margin
        {'setup': '\n'
                  'import math\n'
                  'def _case_scalar(actual):\n'
                  '    return int(type(actual) is float and math.isfinite(actual) and abs(actual-(0.0)) <= 1e-6)\n',
         'call': '_case_scalar(solve_order_robust_margin(((9, 8), (9, 8)),((4, 3, 5),),2.0))',
         'gold_call': '_case_scalar(_oracle_solve_order_robust_margin(((9, 8), (9, 8)),((4, 3, 5),),2.0))'},
        # edge: lower safe boundary lies inside the domain
        {'setup': '\n'
                  'import math\n'
                  'def _case_scalar(actual):\n'
                  '    return int(type(actual) is float and math.isfinite(actual) and '
                  'abs(actual-(20.900842462214406)) <= 1e-6)\n',
         'call': '_case_scalar(solve_order_robust_margin(((11225, 10000), (11226, 10000)),((4, 3, 5),),2.0))',
         'gold_call': '_case_scalar(_oracle_solve_order_robust_margin(((11225, 10000), (11226, 10000)),((4, 3, '
                      '5),),2.0))'},
        # invalid: no configurations
        {'setup': '\n'
                  '\n'
                  'def _case_raise(fn):\n'
                  '    try:\n'
                  '        fn()\n'
                  '        return 0\n'
                  '    except ValueError:\n'
                  '        return 1\n'
                  '    except Exception:\n'
                  '        return 2',
         'call': '_case_raise(lambda: solve_order_robust_margin(configurations=()))',
         'gold_call': '_case_raise(lambda: _oracle_solve_order_robust_margin(configurations=()))'},
        # invalid: empty safe set
        {'setup': '\n'
                  '\n'
                  'def _case_raise(fn):\n'
                  '    try:\n'
                  '        fn()\n'
                  '        return 0\n'
                  '    except ValueError:\n'
                  '        return 1\n'
                  '    except Exception:\n'
                  '        return 2',
         'call': '_case_raise(lambda: solve_order_robust_margin(((9,8),(9,8)),((4,3,5),),1e6))',
         'gold_call': '_case_raise(lambda: _oracle_solve_order_robust_margin(((9,8),(9,8)),((4,3,5),),1e6))'},
    ]
