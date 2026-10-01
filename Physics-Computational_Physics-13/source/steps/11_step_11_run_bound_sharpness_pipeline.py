"""
Chain the sub-problem functions 01-10 end to end on the multibody testbed and return the bound sharpness of the hybrid preconditioner.

This step runs the whole measurement end to end. It (i) assembles the mechanical block of the saddle-point system with sub-problem 01 and the constraint Jacobian with sub-problem 02, (ii) builds the multigrid hierarchy whose per-level ingredients are the aggregation of sub-problem 03 and the smoothed prolongator of sub-problem 04, (iii) materialises the V-cycle as an explicit approximate inverse with sub-problem 05, (iv) measures the spectral-equivalence constant of that approximate inverse against the mechanical block with sub-problem 06, (v) recovers the kinematic topology from the Jacobian, cuts its cycles and clusters the constraints with sub-problem 07, (vi) assembles the block-diagonal-plus-low-rank regularised Schur approximation with sub-problem 08, (vii) forms the block-triangular preconditioned operator and measures the worst deviation of its nonzero spectrum from unity with sub-problem 09, and (viii) divides that deviation by the theoretical bound implied by the spectral-equivalence constant with sub-problem 10.

The returned scalar answers a question the theory cannot answer by itself: whether the eigenvalue bound derived for the idealised preconditioner, which depends on the multigrid approximation quality and on nothing else, remains predictive once the constraint block has been sparsified into small clusters, patched with a low-rank correction across the kinematic loops and shifted for redundancy. A ratio of order one would confirm the idealisation. A ratio far above one localises the spectral spread in the constraint treatment rather than in the multigrid, and identifies which component would have to be improved first.

Returns
-------
float: the bound sharpness of the hybrid preconditioner for this configuration, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def run_bound_sharpness_pipeline(n_bodies: int = 24, n_chords: int = 4,
                                 h: float = 1.0e-3, beta: float = 0.25,
                                 k_scale: float = 1.0e8, theta: float = 0.25,
                                 omega: float = 0.6666666666666666,
                                 block_size: int = 6, n_min: int = 24,
                                 max_levels: int = 4, max_cluster: int = 6,
                                 eps_rel: float = 1.0e-8) -> float:
    """Run the full bound-sharpness measurement on the multibody testbed.

    Parameters
    ----------
    n_bodies : int
        Number of rigid bodies in the mechanism (n_bodies >= 4).
    n_chords : int
        Number of long-range chords added to the closed chain (n_chords >= 0).
    h : float
        Integration time step in seconds (h > 0).
    beta : float
        Newmark parameter of the implicit integrator (0 < beta <= 1).
    k_scale : float
        Reference joint stiffness in N/m (k_scale > 0).
    theta : float
        Strength-of-connection threshold, 0 < theta <= 1.
    omega : float
        Damping factor of the prolongator smoothing sweep, 0 < omega <= 1.
    block_size : int
        Size of the smoother blocks on the finest level (block_size >= 1).
    n_min : int
        Level order at or below which the operator is inverted exactly
        (n_min >= 1).
    max_levels : int
        Maximum number of coarsening steps before an exact solve
        (max_levels >= 0).
    max_cluster : int
        Maximum number of constraints per cluster (max_cluster >= 1).
    eps_rel : float
        Tikhonov shift relative to the matrix one-norm, eps_rel >= 0.

    Returns
    -------
    sharpness : float
        Observed spectral deviation of the preconditioned saddle-point
        operator divided by its theoretical bound, as a native Python float.
    """
    return sharpness  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_bound_sharpness_pipeline(n_bodies: int = 24, n_chords: int = 4,
                                         h: float = 1.0e-3, beta: float = 0.25,
                                         k_scale: float = 1.0e8, theta: float = 0.25,
                                         omega: float = 0.6666666666666666,
                                         block_size: int = 6, n_min: int = 24,
                                         max_levels: int = 4, max_cluster: int = 6,
                                         eps_rel: float = 1.0e-8) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import glob
    import importlib.util
    import os
    import sys

    import numpy as np

    # -- Resolve the oracle functions of sub-problems 01-10. Preference order:
    #    (1) already present in the executing namespace (shared-namespace
    #    harness), (2) loaded from a sibling sub-problem file matched by name
    #    pattern (standalone execution; file prefixes may vary).
    #    The gold path never falls back to a public (submitted) implementation:
    #    binding one would put the same defect on both sides of the differential
    #    comparison and make it vacuous, so an unresolvable step raises instead.
    def _resolve_step(oracle_name, pattern):
        namespace = globals()
        candidate = namespace.get(oracle_name)
        if callable(candidate):
            return candidate
        search_dirs = []
        roots = []
        if "__file__" in namespace:
            roots.append(os.path.dirname(os.path.abspath(namespace["__file__"])))
        roots.append(os.getcwd())
        if sys.argv and sys.argv[0]:
            roots.append(os.path.dirname(os.path.abspath(sys.argv[0])))
        # Each root, its sub_problems/ child and its parent (and that parent's
        # sub_problems/) are searched, so the gold resolves whether the harness
        # runs from the task root, from sub_problems/, or from a copy of this
        # file placed one level away from its siblings.
        for root in roots:
            parent = os.path.dirname(root)
            search_dirs += [root, os.path.join(root, "sub_problems"),
                            parent, os.path.join(parent, "sub_problems")]
        seen = set()
        search_dirs = [d for d in search_dirs if not (d in seen or seen.add(d))]
        for directory in search_dirs:
            for path in sorted(glob.glob(os.path.join(directory, pattern))):
                spec = importlib.util.spec_from_file_location(
                    os.path.basename(path)[:-3], path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, oracle_name):
                    return getattr(module, oracle_name)
        raise RuntimeError(f"cannot resolve required step function {oracle_name}")

    build_stiffness = _resolve_step(
        "_oracle_assemble_effective_stiffness", "*assemble_effective_stiffness*.py")
    build_jacobian = _resolve_step(
        "_oracle_assemble_constraint_jacobian", "*assemble_constraint_jacobian*.py")
    build_aggregates = _resolve_step(
        "_oracle_build_strength_aggregates", "*build_strength_aggregates*.py")
    build_prolongator = _resolve_step(
        "_oracle_build_smoothed_prolongator", "*build_smoothed_prolongator*.py")
    build_vcycle = _resolve_step(
        "_oracle_assemble_vcycle_operator", "*assemble_vcycle_operator*.py")
    spectral_equivalence = _resolve_step(
        "_oracle_compute_spectral_equivalence", "*compute_spectral_equivalence*.py")
    partition_graph = _resolve_step(
        "_oracle_partition_constraint_graph", "*partition_constraint_graph*.py")
    build_schur = _resolve_step(
        "_oracle_assemble_schur_preconditioner", "*assemble_schur_preconditioner*.py")
    spectral_deviation = _resolve_step(
        "_oracle_compute_spectral_deviation", "*compute_spectral_deviation*.py")
    bound_sharpness = _resolve_step(
        "_oracle_compute_bound_sharpness", "*compute_bound_sharpness*.py")

    # -- Validate the orchestrator inputs.
    for name, value, floor in (("n_bodies", n_bodies, 4), ("n_chords", n_chords, 0),
                               ("block_size", block_size, 1), ("n_min", n_min, 1),
                               ("max_levels", max_levels, 0),
                               ("max_cluster", max_cluster, 1)):
        if not (isinstance(value, (int, np.integer)) and not isinstance(value, bool)
                and int(value) >= floor):
            raise ValueError(f"{name} must be an integer >= {floor}")
    for name, value in (("h", h), ("k_scale", k_scale)):
        if not (isinstance(value, (int, float)) and np.isfinite(value)
                and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    for name, value in (("beta", beta), ("theta", theta), ("omega", omega)):
        if not (isinstance(value, (int, float)) and np.isfinite(value)
                and 0.0 < float(value) <= 1.0):
            raise ValueError(f"{name} must be a finite number in the interval (0, 1]")
    if not (isinstance(eps_rel, (int, float)) and np.isfinite(eps_rel)
            and float(eps_rel) >= 0.0):
        raise ValueError("eps_rel must be a finite number >= 0")

    # -- Sub-problems 01-02: the two blocks of the saddle-point system.
    stiffness = build_stiffness(int(n_bodies), int(n_chords), float(h),
                                float(beta), float(k_scale))
    jacobian = build_jacobian(int(n_bodies), int(n_chords))

    # -- Sub-problems 03-04: the finest-level coarsening, then 05: the V-cycle.
    aggregates = build_aggregates(stiffness, float(theta))
    prolongator = build_prolongator(stiffness, aggregates, float(omega))
    vcycle_operator = build_vcycle(stiffness, aggregates, prolongator,
                                   float(theta), float(omega),
                                   int(block_size), int(n_min), int(max_levels))

    # -- Sub-problem 06: quality of that approximate inverse.
    gamma = spectral_equivalence(stiffness, vcycle_operator)

    # -- Sub-problems 07-08: the sparsified, loop-corrected, regularised Schur block.
    partition = partition_graph(jacobian, 6, int(max_cluster))
    schur_prec = build_schur(jacobian, vcycle_operator, partition, float(eps_rel))

    # -- Sub-problem 09: spectrum of the preconditioned saddle-point operator.
    deviation = spectral_deviation(stiffness, jacobian, vcycle_operator, schur_prec)

    # -- Sub-problem 10: observed spread measured against the theoretical bound.
    return float(bound_sharpness(gamma, deviation))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Integration: small mechanism, whole pipeline (normal scenario) ---
        {
            "setup": """import numpy as np
n_bodies = 10
""",
            "call": "run_bound_sharpness_pipeline(n_bodies)",
            "gold_call": "_oracle_run_bound_sharpness_pipeline(n_bodies)",
        },
        # --- Integration: final-answer configuration ---
        {
            "setup": """import numpy as np
n_bodies = 24
n_chords = 4
""",
            "call": "run_bound_sharpness_pipeline(n_bodies, n_chords)",
            "gold_call": "_oracle_run_bound_sharpness_pipeline(n_bodies, n_chords)",
        },
        # --- Integration (boundary): open cluster cap of one and no chords ---
        {
            "setup": """import numpy as np
n_bodies = 8
n_chords = 0
max_cluster = 1
""",
            "call": "run_bound_sharpness_pipeline(n_bodies, n_chords, max_cluster=max_cluster)",
            "gold_call": "_oracle_run_bound_sharpness_pipeline(n_bodies, n_chords, max_cluster=max_cluster)",
        },
        # --- Integration (edge): tiny step, scalar smoother, heavier regularisation ---
        {
            "setup": """import numpy as np
n_bodies = 12
n_chords = 3
""",
            "call": "run_bound_sharpness_pipeline(n_bodies, n_chords, 1.0e-4, 0.3, 5.0e7, 0.25, 2.0 / 3.0, 1, 12, 3, 4, 1.0e-6)",
            "gold_call": "_oracle_run_bound_sharpness_pipeline(n_bodies, n_chords, 1.0e-4, 0.3, 5.0e7, 0.25, 2.0 / 3.0, 1, 12, 3, 4, 1.0e-6)",
        },
        # --- Invalid: mechanism too small to close a chain ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_bound_sharpness_pipeline(3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_bound_sharpness_pipeline(3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: negative Tikhonov shift ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_bound_sharpness_pipeline(8, eps_rel=-1.0e-8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_bound_sharpness_pipeline(8, eps_rel=-1.0e-8)
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
