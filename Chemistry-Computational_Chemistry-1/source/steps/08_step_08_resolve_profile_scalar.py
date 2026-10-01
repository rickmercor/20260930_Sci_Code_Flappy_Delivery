"""
Apply the complete analysis and return the selected scalar.

x_lo, x_hi, y_lo, y_hi are finite increasing bounds of the sampled rectangle.



n_grid is an integer >= 2 giving the number of cell-centred nodes per dimension.



n_bins is an integer >= 2 giving the number of uniform bins.



cv_lo, cv_hi are finite increasing bounds of the binned coordinate range.



c and k are the finite amplitude and angular frequency of the reduced coordinate.



dt, gamma, mass and kbt are finite strictly positive scales of the stochastic update.



scale, kappa and xc define the surface amplitude, confinement stiffness and centre.



Apply the complete source-defined analysis matching the supplied inputs and return one finite dimensionless scalar.

Returns
-------
float: the reported scalar, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Apply the complete analysis and return the selected scalar."""

import numpy as np
from math import erf


def resolve_profile_scalar(x_lo: float, x_hi: float, y_lo: float, y_hi: float, n_grid: int, n_bins: int, cv_lo: float, cv_hi: float, c: float, k: float, dt: float, gamma: float, mass: float, kbt: float, scale: float = 0.05, kappa: float = 2.0, xc: tuple = (-0.5, 0.75)) -> float:
    """Apply the complete analysis and return the selected scalar.

    Parameters
    ----------
    x_lo, x_hi
        Increasing bounds of the first coordinate.
    y_lo, y_hi
        Increasing bounds of the second coordinate.
    n_grid
        Number of cell-centred nodes per dimension (``n_grid >= 2``).
    n_bins
        Number of uniform bins (``n_bins >= 2``).
    cv_lo, cv_hi
        Increasing bounds of the binned coordinate range.
    c, k
        Amplitude and angular frequency of the reduced coordinate.
    dt, gamma
        Strictly positive step size and relaxation rate.
    mass, kbt
        Strictly positive inertia and thermal energy scales.
    scale, kappa, xc
        Surface amplitude, confinement stiffness and centre.

    Returns
    -------
    float
        The reported scalar, as a native Python float.

    Raises
    ------
    ValueError
        If ``n_grid`` or ``n_bins`` is below 2, or if any collective-variable
        bin receives no grid point, which leaves its conditional average
        undefined. A ``ValueError`` raised by an earlier step for its own
        invalid inputs propagates unchanged.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_resolve_profile_scalar(
    x_lo,
    x_hi,
    y_lo,
    y_hi,
    n_grid,
    n_bins,
    cv_lo,
    cv_hi,
    c,
    k,
    dt,
    gamma,
    mass,
    kbt,
    scale=0.05,
    kappa=2.0,
    xc=(-0.5, 0.75),
):
    """Reference implementation for resolve_profile_scalar."""
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import glob
    import importlib.util
    import os
    import sys

    import numpy as np

    # -- Resolve the oracle functions of sub-problems 01-07. Preference order:
    #    (1) already present in the executing namespace (shared-namespace
    #    harness), (2) loaded from a sibling sub-problem file matched by name.
    #    There is deliberately no fallback to the public names: in a shared
    #    namespace those are the candidate's implementations, and falling back
    #    to them would let the gold side of the comparison execute candidate
    #    code. If no oracle can be found the orchestrator fails loudly.
    def _resolve_step(oracle_name, pattern):
        namespace = globals()
        candidate = namespace.get(oracle_name)
        if callable(candidate):
            return candidate
        search_dirs = []
        if "__file__" in namespace:
            search_dirs.append(os.path.dirname(os.path.abspath(namespace["__file__"])))
        cwd = os.getcwd()
        search_dirs += [cwd, os.path.join(cwd, "sub_problems")]
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
        raise RuntimeError("cannot resolve required step function " + oracle_name)

    evaluate_reference_surface = _resolve_step(
        "_oracle_evaluate_reference_surface", "*evaluate_reference_surface*.py")
    evaluate_reduced_surface = _resolve_step(
        "_oracle_evaluate_reduced_surface", "*evaluate_reduced_surface*.py")
    compute_propagated_means = _resolve_step(
        "_oracle_compute_propagated_means", "*compute_propagated_means*.py")
    compute_separation_fractions = _resolve_step(
        "_oracle_compute_separation_fractions", "*compute_separation_fractions*.py")
    derive_coordinate_geometry = _resolve_step(
        "_oracle_derive_coordinate_geometry", "*derive_coordinate_geometry*.py")
    assemble_gradient_terms = _resolve_step(
        "_oracle_assemble_gradient_terms", "*assemble_gradient_terms*.py")
    aggregate_binned_averages = _resolve_step(
        "_oracle_aggregate_binned_averages", "*aggregate_binned_averages*.py")

    n = int(n_grid)
    if n < 2 or int(n_bins) < 2: raise ValueError("n_grid and n_bins must be >= 2")
    gx = x_lo + (np.arange(n) + 0.5)*(x_hi - x_lo)/n
    gy = y_lo + (np.arange(n) + 0.5)*(y_hi - y_lo)/n
    XX, YY = np.meshgrid(gx, gy, indexing="ij")
    P = np.stack([XX.ravel(), YY.ravel()], axis=1)
    tg = evaluate_reference_surface(P, scale, kappa, xc)
    dr = evaluate_reduced_surface(P, scale, kappa, xc)
    lw = -(tg[:, 0] - tg[:, 0].min())
    p0 = np.zeros_like(P)
    mt = compute_propagated_means(p0, -tg[:, 1:3], dt, gamma)   # force = -gradient
    md = compute_propagated_means(p0, -dr[:, 1:3], dt, gamma)
    beta = compute_separation_fractions(mt, md, dt, gamma, mass, kbt)
    geom = derive_coordinate_geometry(P, c, k)
    D    = assemble_gradient_terms(tg[:, 1:3], geom)
    edges = np.linspace(cv_lo, cv_hi, int(n_bins) + 1)
    mf = aggregate_binned_averages(geom[:, 0], D,    lw, edges)
    mb = aggregate_binned_averages(geom[:, 0], beta, lw, edges)
    if np.any(np.isnan(mf)) or np.any(np.isnan(mb)): raise ValueError("an empty CV bin left a conditional average undefined")
    centers = 0.5*(edges[:-1] + edges[1:]); h = centers[1] - centers[0]
    F = np.concatenate([[0.0], np.cumsum(0.5*h*(mf[:-1] + mf[1:]))])
    return float(F[int(np.argmax(mb))] - F[int(np.argmin(mb))])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for resolve_profile_scalar."""
    return [
            {
                    "setup": "import numpy as np\n",
                    "call": "resolve_profile_scalar(-1.8, 1.3, -0.4, 2.1, 220, 14, -1.5, 1.15, 0.30, 1.8, 0.03, 1.5, 1.0, 1.0)",
                    "gold_call": "_oracle_resolve_profile_scalar(-1.8, 1.3, -0.4, 2.1, 220, 14, -1.5, 1.15, 0.30, 1.8, 0.03, 1.5, 1.0, 1.0)",
                    "tol": 1e-09
            },
            {
                    "setup": "import numpy as np\n",
                    "call": "resolve_profile_scalar(-1.8, 1.3, -0.4, 2.1, 180, 12, -1.5, 1.1, 0.35, 1.5, 0.02, 1.0, 1.0, 1.0)",
                    "gold_call": "_oracle_resolve_profile_scalar(-1.8, 1.3, -0.4, 2.1, 180, 12, -1.5, 1.1, 0.35, 1.5, 0.02, 1.0, 1.0, 1.0)",
                    "tol": 1e-09
            },
            {
                    "setup": "import numpy as np\n",
                    "call": "resolve_profile_scalar(-1.8, 1.3, -0.4, 2.1, 150, 10, -1.4, 1.0, 0.20, 2.0, 0.05, 2.0, 1.0, 1.0)",
                    "gold_call": "_oracle_resolve_profile_scalar(-1.8, 1.3, -0.4, 2.1, 150, 10, -1.4, 1.0, 0.20, 2.0, 0.05, 2.0, 1.0, 1.0)",
                    "tol": 1e-09
            },
            {
                    "setup": "import numpy as np\n\n",
                    "call": "resolve_profile_scalar(-1.8, 1.3, -0.4, 2.1, 220, 14, -1.5, 1.1, 0.35, 1.5, 0.02, 1.0, 1.0, 1.0)",
                    "gold_call": "_oracle_resolve_profile_scalar(-1.8, 1.3, -0.4, 2.1, 220, 14, -1.5, 1.1, 0.35, 1.5, 0.02, 1.0, 1.0, 1.0)",
                    "tol": 1e-09
            },
            {
                    "setup": "import numpy as np\n\n",
                    "call": "resolve_profile_scalar(-1.8, 1.3, -0.4, 2.1, 260, 16, -1.6, 1.2, 0.50, 1.5, 0.02, 1.0, 1.0, 1.0)",
                    "gold_call": "_oracle_resolve_profile_scalar(-1.8, 1.3, -0.4, 2.1, 260, 16, -1.6, 1.2, 0.50, 1.5, 0.02, 1.0, 1.0, 1.0)",
                    "tol": 1e-09
            },
            {
                    "setup": "import numpy as np\n\n",
                    "call": "resolve_profile_scalar(-1.8, 1.3, -0.4, 2.1, 240, 12, -1.4, 1.0, 0.35, 1.5, 0.10, 0.5, 2.0, 0.5, 0.05, 3.0, (-0.4, 0.8))",
                    "gold_call": "_oracle_resolve_profile_scalar(-1.8, 1.3, -0.4, 2.1, 240, 12, -1.4, 1.0, 0.35, 1.5, 0.10, 0.5, 2.0, 0.5, 0.05, 3.0, (-0.4, 0.8))",
                    "tol": 1e-09
            },
            {
                    "setup": "import numpy as np\n\n",
                    "call": "resolve_profile_scalar(-1.8, 1.3, -0.4, 2.1, 200, 16, -1.6, 1.2, 0.35, 1.5, 0.02, 1.0, 4.0, 0.25, 0.05, 2.0, (-0.5, 0.75))",
                    "gold_call": "_oracle_resolve_profile_scalar(-1.8, 1.3, -0.4, 2.1, 200, 16, -1.6, 1.2, 0.35, 1.5, 0.02, 1.0, 4.0, 0.25, 0.05, 2.0, (-0.5, 0.75))",
                    "tol": 1e-09
            },
            {
                    "setup": "import numpy as np\ndef run_model():\n    try:\n        resolve_profile_scalar(-1.8, 1.3, -0.4, 2.1, 5, 60, -1.6, 1.2, 0.35, 1.5, 0.02, 1.0, 1.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_resolve_profile_scalar(-1.8, 1.3, -0.4, 2.1, 5, 60, -1.6, 1.2, 0.35, 1.5, 0.02, 1.0, 1.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
                    "call": "run_model()",
                    "gold_call": "run_gold()",
                    "tol": 0.0
            }
    ]
