"""
Given the stage model, its projection interval $dt$ and the requested number $m$ of reduced age classes, return its reduction to $m$ equal-width age classes spanning the same ages as the age-classified model defined in step 2. Let $n$ be the number of classes in that age-classified model. The reduced projection interval is the width of one reduced age class, and the demographic timing is that specified in step 3. Use the diagnostic age grid fixed by the returned resolution convention below for quantities associated with the matrix being reduced.

Construct the consistent reduced matrix with step 6 and, for comparison, the flow-matching collapse of the same matrix with step 4. Certify the result before reporting it: the consistent matrix must pass all four requirements of step 7, its survival probabilities must equal those of the flow-matching collapse, and the Perron root of the matrix being reduced, raised to the number of sub-classes per class, must equal the growth rate of the age-classified model. Raise RuntimeError when the largest of these discrepancies exceeds the certificate tolerance, since the reduced model is then not the consistent one.

Return the fertility of the oldest reduced class as the headline result together with the elasticity of the reduced growth rate to it from step 5, and alongside them the expansion, the whole reduced model, its reproductive values scaled so that the first equals one, and the demographic parameters of the consistent reduction, of the flow-matching collapse and of the age-classified model from step 8, the generation times in the units of dt.

This stage answers the question the chain exists for. Given a stage-classified model whose last stage is a plus-group, a projection interval dt and a number m of reduced age classes, it expresses the model as an age-classified Leslie model, reduces that model to an m-class Leslie model over the correspondingly longer interval in a way that agrees with it in growth rate, stable age structure, reproductive values and elasticities, and reports the demographic parameters that show what the reduction preserves and what it does not.

1. Step 1 returns the Perron root, stable distribution and reproductive values of an irreducible matrix, choosing the root by largest real part so that imprimitive matrices are handled.

2. Step 2 unrolls the plus-group of the stage model into age classes and truncates the expansion at the shortest length whose growth rate agrees with the stage model's to within the stated relative tolerance.

3. Step 3 returns the projection matrix on a finer age grid with the same demographic timing as the original model.

4. Step 4 collapses a projection matrix over blocks of whole classes and several intervals by matching summed interstage flows, and reports the effectiveness of that fit.

5. Step 5 returns the elasticities of the Perron root to every entry of a projection matrix.

6. Step 6 returns the reduced matrix consistent in growth rate, grouped stable distribution, reproductive values and elasticities, together with its reproductive values and effectiveness.

7. Step 7 measures how far a reduced matrix departs from each of the four consistency requirements.

8. Step 8 computes the growth rate, net reproductive rate, generation time and Demetrius entropy of a Leslie matrix.

The elasticity of the reduced growth rate to the oldest reduced fertility, taken from step 5, is the share of the growth rate that the graded entry carries, and it equals the summed elasticity of the corresponding block of the resolved model's projection over the reduced interval.

Returns
-------
dict holding the float oldest_fertility, the fertility of the oldest reduced class, and the float oldest_fertility_elasticity, the elasticity of the reduced growth rate to that fertility; the integer age_classes, the length n of the age expansion, and the floats stage_growth_rate and expansion_error, the growth rate of the stage model and the relative error of the expansion's growth rate; the np.ndarray reduced_fertility of shape (m,) and reduced_survival of shape (m - 1,); the floats reduced_growth_rate and reduced_interval; the np.ndarray reduced_reproductive_values of shape (m,) with first entry one; the float growth_rate of the age-classified model; the integer resolution, the number of equal-width sub-classes per original age class in the diagnostic grid, defined as one when $m$ divides $n$ and $m$ otherwise; the integer peripheral_count, the number of eigenvalues on the spectral circle of the matrix on that grid; the np.ndarray standard_fertility of shape (m,), the fertilities of the flow-matching collapse; the floats balanced_effectiveness and standard_effectiveness, the effectiveness defined in step 6 and step 4 respectively; the floats net_reproductive_rate, generation_time and entropy of the consistent reduction, the same three prefixed original_ for the age-classified model and prefixed standard_ for the flow-matching collapse; the floats standard_reproductive_value_residual and standard_elasticity_residual; and the float certificate_residual.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reduce_stage_model(
    stage_matrix: np.ndarray,
    reduced_classes: int,
    interval: float = 1.0,
    tolerance: float = 1e-3,
    certificate_tol: float = 1e-9,
) -> dict:
    """Reduce a stage model with a plus-group to fewer, wider age classes consistently.

    Parameters
    ----------
    stage_matrix : np.ndarray
        Stage-classified matrix with Leslie pattern plus a positive southeast entry, shape (s, s).
        Its subdiagonal survival probabilities lie in (0, 1], its last stage is fertile, and its
        southeast stasis probability lies strictly between zero and one.
    reduced_classes : int
        Number of reduced age classes, from one to the length of the age expansion.
    interval : float
        Projection interval of the stage model, above zero.
    tolerance : float
        Relative tolerance on the growth rate fixing the length of the age expansion.
    certificate_tol : float
        Largest discrepancy accepted in the consistency certificate.

    Returns
    -------
    dict
        Under the keys oldest_fertility, oldest_fertility_elasticity, age_classes,
        stage_growth_rate, expansion_error, reduced_fertility, reduced_survival,
        reduced_growth_rate, reduced_interval, reduced_reproductive_values, growth_rate,
        resolution, peripheral_count, standard_fertility, balanced_effectiveness,
        standard_effectiveness, net_reproductive_rate, generation_time, entropy,
        original_net_reproductive_rate, original_generation_time, original_entropy,
        standard_net_reproductive_rate, standard_generation_time, standard_entropy,
        standard_reproductive_value_residual, standard_elasticity_residual and
        certificate_residual.
        resolution is one when reduced_classes divides age_classes, and reduced_classes
        otherwise; it fixes the number of equal-width sub-classes per original age class
        in the diagnostic grid.

    Raises
    ------
    ValueError
        When the stage matrix or tolerance fails the checks of step 2, when reduced_classes is
        not an integer from one to the length of the expansion, or when interval or
        certificate_tol is not finite and above zero. A count given as a bool, or as a float
        even when its value is integral, is not accepted as an integer, and a bool or a
        non-numeric value is not accepted as a real number.
    RuntimeError
        When the consistency certificate exceeds certificate_tol.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numbers

import numpy as np


def _check_count(value, name, lower):
    """Validate an integer count not below a lower bound and return it as an int."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(name + " must be an integer")
    if int(value) < lower:
        raise ValueError(name + " must be at least " + str(lower))
    return int(value)


def _check_positive(value, name):
    """Validate a finite real number above zero and return it as a float."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(name + " must be a real number")
    x = float(value)
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError(name + " must be finite and above zero")
    return x


def _oracle_reduce_stage_model(
    stage_matrix: np.ndarray,
    reduced_classes: int,
    interval: float = 1.0,
    tolerance: float = 1e-3,
    certificate_tol: float = 1e-9,
) -> dict:
    """Reference implementation."""
    expansion = _oracle_age_expand_stage_model(stage_matrix, tolerance)  # noqa: F821
    fertility = expansion["fertility"]
    survival = expansion["survival"]
    original = _oracle_disaggregate_leslie(fertility, survival, 1)  # noqa: F821
    n = original.shape[0]
    m = _check_count(reduced_classes, "reduced_classes", 1)
    if m > n:
        raise ValueError("reduced_classes must not exceed the number of age classes")
    dt = _check_positive(interval, "interval")
    tol = _check_positive(certificate_tol, "certificate_tol")

    if n % m == 0:
        resolution = 1
        fine = original
        steps = n // m
    else:
        resolution = m
        fine = _oracle_disaggregate_leslie(fertility, survival, m)  # noqa: F821
        steps = n

    original_triplet = _oracle_perron_triplet(original)  # noqa: F821
    fine_triplet = _oracle_perron_triplet(fine)  # noqa: F821
    consistent = _oracle_elasticity_consistent_aggregate(fine, m, steps)  # noqa: F821
    standard = _oracle_interstage_flow_aggregate(fine, m, steps)  # noqa: F821
    b = consistent["reduced_matrix"]
    bs = standard["reduced_matrix"]

    own = _oracle_consistency_residuals(fine, m, steps, b)  # noqa: F821
    other = _oracle_consistency_residuals(fine, m, steps, bs)  # noqa: F821
    survival_gap = max([0.0] + [abs(b[i + 1, i] - bs[i + 1, i]) for i in range(m - 1)])
    rate_gap = abs(fine_triplet["growth_rate"] ** resolution - original_triplet["growth_rate"]) \
        / original_triplet["growth_rate"]
    certificate = max(max(own.values()), survival_gap, rate_gap)
    if certificate > tol:
        raise RuntimeError("the reduced model fails the consistency certificate by %.3e" % certificate)

    elasticities = _oracle_elasticity_matrix(b)  # noqa: F821
    reduced_interval = dt * n / m
    mine = _oracle_demographic_parameters(b, reduced_interval)  # noqa: F821
    theirs = _oracle_demographic_parameters(bs, reduced_interval)  # noqa: F821
    base = _oracle_demographic_parameters(original, dt)  # noqa: F821

    values = consistent["reduced_reproductive_values"]
    return {
        "oldest_fertility": float(b[0, m - 1]),
        "oldest_fertility_elasticity": float(elasticities[0, m - 1]),
        "age_classes": n,
        "stage_growth_rate": expansion["stage_growth_rate"],
        "expansion_error": expansion["relative_error"],
        "reduced_fertility": b[0, :].copy(),
        "reduced_survival": np.array([b[i + 1, i] for i in range(m - 1)]),
        "reduced_growth_rate": mine["growth_rate"],
        "reduced_interval": reduced_interval,
        "reduced_reproductive_values": values / values[0],
        "growth_rate": original_triplet["growth_rate"],
        "resolution": resolution,
        "peripheral_count": fine_triplet["peripheral_count"],
        "standard_fertility": bs[0, :].copy(),
        "balanced_effectiveness": consistent["effectiveness"],
        "standard_effectiveness": standard["effectiveness"],
        "net_reproductive_rate": mine["net_reproductive_rate"],
        "generation_time": mine["generation_time"],
        "entropy": mine["entropy"],
        "original_net_reproductive_rate": base["net_reproductive_rate"],
        "original_generation_time": base["generation_time"],
        "original_entropy": base["entropy"],
        "standard_net_reproductive_rate": theirs["net_reproductive_rate"],
        "standard_generation_time": theirs["generation_time"],
        "standard_entropy": theirs["entropy"],
        "standard_reproductive_value_residual": other["reproductive_value_residual"],
        "standard_elasticity_residual": other["elasticity_residual"],
        "certificate_residual": float(certificate),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return self-contained, single-invocation numeric test cases."""
    setup = "import numpy as np\n\ndef _flatten(value):\n    if isinstance(value, dict):\n        parts = [_flatten(value[key]) for key in sorted(value)]\n        return np.concatenate(parts) if parts else np.empty(0, dtype=float)\n    if isinstance(value, (list, tuple)):\n        parts = [_flatten(item) for item in value]\n        return np.concatenate(parts) if parts else np.empty(0, dtype=float)\n    return np.asarray(value, dtype=float).ravel()\n\ndef project(value):\n    raw = _flatten(value)\n    missing = np.isnan(raw)\n    return np.concatenate((np.where(missing, 0.0, raw), missing.astype(float)))\n\ndef expect_value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n\ndef expect_runtime_error(fn, *args):\n    try:\n        fn(*args)\n    except RuntimeError:\n        return np.array([1.0])\n    return np.array([0.0])\n\nSTAGE = np.array([[0.0, 0.3, 1.0, 2.0],\n                  [0.6, 0.0, 0.0, 0.0],\n                  [0.0, 0.7, 0.0, 0.0],\n                  [0.0, 0.0, 0.85, 0.55]])\n"
    return [
        {
            "setup": setup + "",
            "call": "project(reduce_stage_model(STAGE, 3))",
            "gold_call": "project(_oracle_reduce_stage_model(STAGE, 3))",
        },
        {
            "setup": setup + "OTHER = np.array([[0.2, 1.1, 1.6], [0.5, 0.0, 0.0], [0.0, 0.8, 0.3]])\n",
            "call": "project(reduce_stage_model(OTHER, 2))",
            "gold_call": "project(_oracle_reduce_stage_model(OTHER, 2))",
        },
        {
            "setup": setup + "",
            "call": "project(reduce_stage_model(STAGE, 3, 1.0, 1e-2))",
            "gold_call": "project(_oracle_reduce_stage_model(STAGE, 3, 1.0, 1e-2))",
        },
        {
            "setup": setup + "",
            "call": "expect_runtime_error(reduce_stage_model, STAGE, 3, 1.0, 1e-3, 1e-300)",
            "gold_call": "expect_runtime_error(_oracle_reduce_stage_model, STAGE, 3, 1.0, 1e-3, 1e-300)",
        },
    ]
