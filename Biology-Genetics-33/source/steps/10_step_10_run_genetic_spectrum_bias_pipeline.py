"""
Chain the sub-problem functions 01-09 end to end on the full-sibling testbed and return the value the moment-matching estimator assigns to the requested quantity when the two variance components do not share an eigenbasis.

The whole measurement is a chain of exact algebraic steps followed by one root find: tabulate the design and covariance functionals, evaluate the limiting trace moments they imply, and invert the moment map under a working model that assumes a shared eigenbasis. The returned quantity is what an analyst would report from an arbitrarily large study of this design.

Returns
-------
float: the requested reported value of the moment-matching fit, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def run_genetic_spectrum_bias_pipeline(rotation_angle: float = 0.6283185307179586,
                                       tau_genetic: float = 1.4,
                                       rho_genetic: float = 0.35,
                                       tau_residual: float = 0.6,
                                       rho_residual: float = 0.8,
                                       n_traits: int = 240,
                                       n_families: int = 300,
                                       size_cycle: tuple = (1, 2, 3),
                                       quantity: str = "rho_genetic") -> float:
    """Run the full joint moment-matching fit on the full-sibling testbed.

    Family ``i`` contains ``size_cycle[i % len(size_cycle)]`` individuals, an
    unknown population mean is carried as a fixed effect, and the two variance
    components are the step spectra of the family and individual levels with the
    family-level eigenbasis rotated away from the coordinate basis. All three
    parameters are fitted together from the first three trace moments of the
    between-family sum-of-squares matrix.

    Parameters
    ----------
    rotation_angle : float
        Angle in radians between the two eigenbases; finite.
    tau_genetic : float
        True nonzero eigenvalue of the family-level component; greater than
        zero.
    rho_genetic : float
        True fraction of trait directions carrying it, in ``(0, 1]`` and
        strictly below ``rho_residual``.
    tau_residual : float
        True nonzero eigenvalue of the individual-level component; greater than
        zero.
    rho_residual : float
        Known fraction of trait directions carrying it, in ``(0, 1]``.
    n_traits : int
        Number of measured traits; an integer of at least two.
    n_families : int
        Number of families in the design; a positive integer.
    size_cycle : tuple
        Repeating pattern of family sizes; a non-empty sequence of integers of
        at least one.
    quantity : str
        Which reported value to return: ``"rho_genetic"`` for the fitted
        fraction of trait directions carrying additive genetic variance,
        ``"tau_genetic"`` for the fitted family-level eigenvalue,
        ``"tau_residual"`` for the fitted individual-level eigenvalue, or
        ``"moment_1"``, ``"moment_2"`` and ``"moment_3"`` for the three limiting
        between-family trace moments.

    Returns
    -------
    reported_value : float
        The requested quantity, dimensionless for the fractions and in trait
        variance units otherwise, as a native Python float.

    Raises
    ------
    ValueError
        If ``rotation_angle`` is not finite, if either ``tau`` is not a finite
        number greater than zero, if either ``rho`` is not a finite number in
        ``(0, 1]``, if ``rho_genetic`` is not strictly below ``rho_residual``,
        if ``n_traits`` is not an integer value of at least two, if
        ``n_families`` is not an integer value of at least one, if
        ``size_cycle`` is not a non-empty sequence of integer values of at least
        one, or if ``quantity`` is not one of the six listed names.
    """
    return reported_value  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_genetic_spectrum_bias_pipeline(rotation_angle: float = 0.6283185307179586,
                                               tau_genetic: float = 1.4,
                                               rho_genetic: float = 0.35,
                                               tau_residual: float = 0.6,
                                               rho_residual: float = 0.8,
                                               n_traits: int = 240,
                                               n_families: int = 300,
                                               size_cycle: tuple = (1, 2, 3),
                                               quantity: str = "rho_genetic") -> float:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(value))

    # -- Validate the orchestrator inputs.
    if not _is_number(rotation_angle):
        raise ValueError("rotation_angle must be a finite number")
    for name, value in (("tau_genetic", tau_genetic), ("tau_residual", tau_residual)):
        if not _is_number(value) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite number greater than zero")
    for name, value in (("rho_genetic", rho_genetic), ("rho_residual", rho_residual)):
        if not _is_number(value) or not (0.0 < float(value) <= 1.0):
            raise ValueError(f"{name} must be a finite number in the interval (0, 1]")
    if not float(rho_genetic) < float(rho_residual):
        raise ValueError("rho_genetic must be strictly below rho_residual")
    if not _is_number(n_traits) or float(n_traits) != float(int(n_traits)) or int(n_traits) < 2:
        raise ValueError("n_traits must be an integer value of at least two")
    if not _is_number(n_families) or float(n_families) != float(int(n_families)) or int(n_families) < 1:
        raise ValueError("n_families must be an integer value of at least one")
    cycle = list(size_cycle) if not isinstance(size_cycle, (str, bytes)) else []
    if len(cycle) < 1 or not all(_is_number(v) and float(v) == float(int(v)) and int(v) >= 1
                                 for v in cycle):
        raise ValueError("size_cycle must be a non-empty sequence of integers of at least one")
    if quantity not in ("rho_genetic", "tau_genetic", "tau_residual",
                        "moment_1", "moment_2", "moment_3"):
        raise ValueError("quantity must be one of 'rho_genetic', 'tau_genetic', "
                         "'tau_residual', 'moment_1', 'moment_2' or 'moment_3'")

    n_traits = int(n_traits)
    n_families = int(n_families)
    cycle = [int(v) for v in cycle]
    family_sizes = np.array([cycle[i % len(cycle)] for i in range(n_families)], dtype=np.int64)

    # -- Sub-problem 03: the two level increments of the nested design, with the
    #    unknown population mean removed from the coarser one. The joint fit uses
    #    the between-family increment alone.
    projectors = _oracle_build_nested_design_projectors(family_sizes, True)

    # -- Sub-problem 04: the design functionals, tabulated to word length three
    #    because three moments are matched.
    design_between = _oracle_evaluate_design_functionals(projectors[0], family_sizes,
                                                         n_traits, 3)

    # -- Sub-problems 05 and 06: the true covariance components and the traces of
    #    their ordered products, which is where the eigenbasis mismatch enters.
    components = _oracle_build_variance_components(n_traits, tau_genetic, rho_genetic,
                                                   tau_residual, rho_residual, rotation_angle)
    spectral = _oracle_evaluate_spectral_functionals(components, 3)

    # -- Sub-problems 01 and 02: the combinatorial skeleton of each moment.
    block_tables = [_oracle_compute_kreweras_block_table(
        _oracle_enumerate_noncrossing_pairings(order)) for order in (1, 2, 3)]

    # -- Sub-problem 07: the limiting trace moments the design actually produces.
    moments = np.array([_oracle_evaluate_moment_expansion(design_between, spectral, table)
                        for table in block_tables], dtype=float)

    # -- Sub-problems 08 and 09: the joint fit, whose inner loop rebuilds the
    #    working model's functionals for each candidate parameter triple.
    estimates = _oracle_solve_joint_moment_equations(design_between, moments, rho_residual)

    reported = {"tau_genetic": float(estimates[0]),
                "rho_genetic": float(estimates[1]),
                "tau_residual": float(estimates[2]),
                "moment_1": float(moments[0]),
                "moment_2": float(moments[1]),
                "moment_3": float(moments[2])}
    return float(reported[quantity])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Integration: the reported configuration (normal scenario) ---
        {
            "setup": """import numpy as np
""",
            "call": "round(run_genetic_spectrum_bias_pipeline(), 9)",
            "gold_call": "round(_oracle_run_genetic_spectrum_bias_pipeline(), 9)",
        },
        # --- Integration: the fitted family-level eigenvalue of the same run ---
        {
            "setup": """import numpy as np
quantity = 'tau_genetic'
""",
            "call": "round(run_genetic_spectrum_bias_pipeline(0.6283185307179586, 1.4, 0.35, 0.6, 0.8, 240, 300, (1, 2, 3), quantity), 9)",
            "gold_call": "round(_oracle_run_genetic_spectrum_bias_pipeline(0.6283185307179586, 1.4, 0.35, 0.6, 0.8, 240, 300, (1, 2, 3), quantity), 9)",
        },
        # --- Integration: the fitted individual-level eigenvalue, which the joint
        #     fit biases even though its own spectrum is not rotated ---
        {
            "setup": """import numpy as np
quantity = 'tau_residual'
""",
            "call": "round(run_genetic_spectrum_bias_pipeline(0.6283185307179586, 1.4, 0.35, 0.6, 0.8, 240, 300, (1, 2, 3), quantity), 9)",
            "gold_call": "round(_oracle_run_genetic_spectrum_bias_pipeline(0.6283185307179586, 1.4, 0.35, 0.6, 0.8, 240, 300, (1, 2, 3), quantity), 9)",
        },
        # --- Integration: the third between-family trace moment, which pins the
        #     order-three design and covariance functionals together ---
        {
            "setup": """import numpy as np
quantity = 'moment_3'
""",
            "call": "round(run_genetic_spectrum_bias_pipeline(0.6283185307179586, 1.4, 0.35, 0.6, 0.8, 240, 300, (1, 2, 3), quantity), 9)",
            "gold_call": "round(_oracle_run_genetic_spectrum_bias_pipeline(0.6283185307179586, 1.4, 0.35, 0.6, 0.8, 240, 300, (1, 2, 3), quantity), 9)",
        },
        # --- Integration (boundary): a vanishing angle, where the working model is
        #     correct and the fit must return the true occupied fraction ---
        {
            "setup": """import numpy as np
""",
            "call": "round(run_genetic_spectrum_bias_pipeline(0.0, 1.4, 0.35, 0.6, 0.8, 240, 300, (1, 2, 3), 'rho_genetic'), 9)",
            "gold_call": "round(_oracle_run_genetic_spectrum_bias_pipeline(0.0, 1.4, 0.35, 0.6, 0.8, 240, 300, (1, 2, 3), 'rho_genetic'), 9)",
        },
        # --- Integration: a quarter turn, the most adversarial alignment of the
        #     two eigenbases ---
        {
            "setup": """import numpy as np
angle = np.pi / 2.0
""",
            "call": "round(run_genetic_spectrum_bias_pipeline(angle, 1.4, 0.35, 0.6, 0.8, 240, 300, (1, 2, 3), 'rho_genetic'), 9)",
            "gold_call": "round(_oracle_run_genetic_spectrum_bias_pipeline(angle, 1.4, 0.35, 0.6, 0.8, 240, 300, (1, 2, 3), 'rho_genetic'), 9)",
        },
        # --- Integration (edge): a different trait count, family-size pattern and
        #     spectrum, which exercises every step away from the testbed ---
        {
            "setup": """import numpy as np
""",
            "call": "round(run_genetic_spectrum_bias_pipeline(0.9, 2.2, 0.2, 0.45, 0.55, 150, 80, (2, 3, 4, 1), 'rho_genetic'), 9)",
            "gold_call": "round(_oracle_run_genetic_spectrum_bias_pipeline(0.9, 2.2, 0.2, 0.45, 0.55, 150, 80, (2, 3, 4, 1), 'rho_genetic'), 9)",
        },
        # --- Invalid: a genetic fraction at or above the residual fraction, for
        #     which the working model's mixed moments are not the ones used ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_genetic_spectrum_bias_pipeline(0.6283185307179586, 1.4, 0.9, 0.6, 0.8, 240, 300, (1, 2, 3), 'rho_genetic')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_genetic_spectrum_bias_pipeline(0.6283185307179586, 1.4, 0.9, 0.6, 0.8, 240, 300, (1, 2, 3), 'rho_genetic')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: an unrecognised reported quantity ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_genetic_spectrum_bias_pipeline(0.6283185307179586, 1.4, 0.35, 0.6, 0.8, 240, 300, (1, 2, 3), 'heritability')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_genetic_spectrum_bias_pipeline(0.6283185307179586, 1.4, 0.35, 0.6, 0.8, 240, 300, (1, 2, 3), 'heritability')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
