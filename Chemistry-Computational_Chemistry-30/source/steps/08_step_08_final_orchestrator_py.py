"""
Combine source retrieval, exact tail constraint, physical bounds and computed gap to report the globally maximal dimensionless α. Include the admissible family, α=1 gap bound, mode-validity argument and conditional interpretation.

The result is a deterministic idealized benchmark, not a raw-data fit. Check SI time units and eV thermal units, conservation, nonnegative rates, observable stable relaxation and the limits of transferring solution-phase computed energies to a PMMA lifetime.

Returns
-------
float, globally maximal dimensionless alpha at the retrieved ROKS T2-T1 gap
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from typing import Optional

def solve_triplet_bound(source: Optional[np.ndarray] = None,
                        kappa: float = 1.0) -> float:
    """Return the maximal common escape multiplier at the computed gap.

    If source is None, retrieve the nominal Note S2, Fig. S26 and Table
    S9 inputs from Dou et al., DOI 10.1038/s41377-025-02063-x.
    Otherwise source uses the eight-value source-unit layout documented
    by prepare_inputs. kappa is the known positive T1 radiative rate.
    Chain the previous computational steps: unit conversion, exact
    lifetime elimination, global bounds, admissible boundary, generator
    and dominant-mode calculation. Use public step functions here;
    reference/oracle implementations chain only oracle functions.
    The two-temperature step is a separate proposed experiment, not
    extra input required for this one-temperature bound.

    Returns
    -------
    float
        Maximal alpha at the supplied or retrieved ROKS gap, attained at
        zero T1 nonradiative loss. This is a bound, not an identified gap.

    Raises
    ------
    ValueError
        For nonreal/nonfinite inputs, wrong source shape, nonpositive
        source lifetimes/rates/thermal energy or kappa, inverted triplet
        energy order, decay<=kappa, nonfinite converted values, a target
        outside the positive fast-block elimination domain (a<=decay,
        u+c<=decay, B<=0 or H<=0), nonfinite bounds/matrix/spectral
        results, a nonphysical generator, a nonstable/nonsimple/nonreal
        dominant pole or singular eigenvector basis. Also raise if the
        constructed boundary fails admissible_point's rounding guard,
        its dominant decay differs from the target by more than 1e-6
        relative, or its T1 tail amplitude is nonpositive. The matrix
        and spectral numerical conventions are those of slow_mode.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve_triplet_bound(source=None,kappa=1.0):
    import numpy as np
    if source is None:
        source = np.array([6.0,6.0e7,7.5e5,30.0,25.6,120.0,2.953,2.912])
    tau_s,i,u,c,theta,decay,gap,k = _oracle_prepare_inputs(source,kappa)
    a,B,H = _oracle_tail_coefficients(tau_s,i,u,c,decay)
    zmax,minimum_gap,maximum_alpha = _oracle_kinetic_bounds(H,decay,k,theta,gap)
    point = _oracle_admissible_point(H,decay,k,theta,gap,maximum_alpha)
    if point[3] != 1.0:
        raise ValueError("constructed boundary is not physically admissible")
    matrix = _oracle_population_generator(tau_s,i,u,c,point[0],point[1])
    rate,amplitude = _oracle_slow_mode(matrix)
    if abs(rate-decay) > 1e-6*decay or amplitude <= 0:
        raise ValueError("target is not an observable dominant tail")
    return float(maximum_alpha)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Valid cases compare outputs; invalid cases explicitly require ValueError."""
    return [
        {
            "setup": "",
            "call": "solve_triplet_bound(None,1.0)",
            "gold_call": "_oracle_solve_triplet_bound(None,1.0)"
        },
        {
            "setup": "import numpy as np\ns=np.array([6.,6e7,7.5e5,30.,25.6,120.,2.912,2.912])",
            "call": "solve_triplet_bound(s,1.0)",
            "gold_call": "_oracle_solve_triplet_bound(s,1.0)"
        },
        {
            "setup": "",
            "call": "solve_triplet_bound(None,8.0)",
            "gold_call": "_oracle_solve_triplet_bound(None,8.0)"
        },
        {
            "setup": "def _expect_value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return 1\n    raise AssertionError(\"Expected ValueError for invalid input\")",
            "call": "_expect_value_error(solve_triplet_bound, None,9.0)",
            "gold_call": "_expect_value_error(_oracle_solve_triplet_bound, None,9.0)"
        }
    ]
