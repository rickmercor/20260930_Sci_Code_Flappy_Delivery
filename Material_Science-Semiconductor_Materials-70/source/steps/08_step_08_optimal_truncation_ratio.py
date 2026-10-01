"""
Everything assembled so far exists to answer one question: for a fixed computational grid, how many Fourier modes should the represented field actually keep. The habitual answer is all of them, M = N, and it is a habit rather than a result. The grid and the truncation do different jobs. More grid points improve the trapezoidal estimate of each Fourier coefficient and nothing else, so their benefit saturates. More modes let the represented field resolve the material interface more sharply, which helps, but they also admit the coefficients whose trapezoidal estimates are worst, because the quadrature error of a coefficient grows with its frequency index, and near a discontinuity those same high modes are the ones that ring. Two effects of opposite sign in the same parameter produce an interior optimum, and where it falls is a property of the microstructure and the contrast rather than of the algorithm.

This stage runs the whole chain once for each candidate truncation on a stated list, forms the energy-norm error of each converged field against the true microstructure, and reports the truncation that minimises it as a fraction of the grid size. The list is stated rather than searched so that the answer is a finite deterministic computation and not the outcome of an optimiser.

Two contrasts deserve to be reported alongside it, because each is a place where a plausible shortcut gives a different answer.

The first is the macroscopic conductivity. Over the sweep run here its relative error against the exact value falls monotonically as the truncation rises, so judged on the effective property alone the best truncation on this list is the largest one and the habit appears vindicated. The fall of the effective conductivity itself is a theorem rather than an observation, since the converged field minimises a fixed convex energy over spaces that grow with the truncation; that its error falls with it needs the sampled problem to overestimate the exact value as well, which holds here by measurement. It would be wrong to explain it by saying the mean mode cannot feel the high frequencies: the mean current is a convolution of the conductivity spectrum with the field spectrum, so changing the retained content does change it, and the package's own numbers show it changing. What can be said is that the quantity is a single spatial average, so the oscillations that dominate the local error enter it only through that average and are not weighted by their amplitude wherever they occur. Anyone who measures the discretisation by the effective property alone will therefore not see the optimum the local field has, and may conclude that the two parameters need not be separated at all.

The second is the generalised Green operator built on difference quotients. Changing the derivative rule changes which direction each mode of the field is projected onto, and it is a reasonable expectation that a rule with a built-in length scale would damp the ringing. Running the same sweep with it settles that by measurement rather than expectation. What that sweep reports is not the energy-norm error, and the distinction is not cosmetic. The identity that turns the energy-norm error into an energy needs the computed field to be a genuine periodic gradient plus the applied mean. The spectral rule guarantees that; a difference rule does not, because it projects each mode onto the conjugated difference symbol rather than onto the wavevector, and the resulting field carries a component of order one per cent outside the gradient subspace. The cross term in the expansion therefore no longer collapses onto the same constant, so the quantity reported for the difference rule is the excess of its apparent conductivity over the exact one and nothing more. It is a legitimate diagnostic, it is the quantity the two families are compared on, and it coincides with the energy-norm error only in the spectral case. It carries no variational certificate of non-negativity, and no conclusion about the true local-field error of the difference rule follows from it.

Returns
-------
dict, the optimal truncation ratio, with the errors, conductivities, reconstruction peaks and sweep diagnostics that establish it.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def optimal_truncation_ratio(
    donor_density_matrix: float,
    mobility_matrix: float,
    donor_density_inclusion: float,
    mobility_inclusion: float,
    period: float,
    mean_field,
    n_grid: int,
    mode_step: int,
    refinement: int,
    spacing_ratio: float,
    tolerance: float,
    max_iterations: int,
) -> dict:
    """Sweep the candidate truncations and report the one that minimises the energy-norm error.

    Parameters
    ----------
    donor_density_matrix : float
        Ionised donor density of the surrounding phase in reciprocal cubic metre.
    mobility_matrix : float
        Electron mobility of the surrounding phase in square metre per volt second.
    donor_density_inclusion : float
        Ionised donor density of the embedded phase in reciprocal cubic metre.
    mobility_inclusion : float
        Electron mobility of the embedded phase in square metre per volt second.
    period : float
        Cell edge in metre, above zero.
    mean_field : sequence
        Two components of the applied mean electric field in volt per metre.
    n_grid : int
        The number N of grid intervals, even and above zero.
    mode_step : int
        Spacing of the candidate truncations, even, above zero and dividing N.
    refinement : int
        Whole refinement factor for the reconstruction, at least two.
    spacing_ratio : float
        Difference spacing of the generalised operator as a multiple of the grid spacing.
    tolerance : float
        Residual at which each iteration stops, above zero.
    max_iterations : int
        Largest number of sweeps permitted per candidate, above zero.

    Returns
    -------
    dict
        Under the keys optimal_ratio, optimal_modes, optimal_error, error_at_full, error_ratio, macroscopic_at_optimum, macroscopic_at_full, exact_macroscopic, macro_error_at_optimum, macro_error_at_full, c_matrix, c_inclusion, contrast, sampled_area_fraction, peak_ratio_at_optimum, peak_ratio_at_full, iterations_at_optimum, generalised_optimal_ratio, generalised_optimal_modes, generalised_optimal_excess, candidates and errors. candidates and errors are sequences of the same length, holding the candidate truncations in increasing order and the energy-norm error of each. Every other entry is a scalar.

    Raises
    ------
    ValueError
        When any physical argument fails to be finite and above zero, when N or the candidate spacing fails to be a positive even integer, when the spacing does not divide N, when the refinement factor is below two, or when any single solve fails to converge within max_iterations sweeps.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _even_positive(value, label):
    """Return an argument as an int once it is known to be a positive even integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("%s must be an integer" % label)
    out = int(value)
    if out <= 0 or out % 2 != 0:
        raise ValueError("%s must be a positive even integer" % label)
    return out


def _positive_float(value, label):
    """Return an argument as a float once it is known to be finite and above zero."""
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError("%s must be finite and above zero" % label)
    return out


def _oracle_optimal_truncation_ratio(
    donor_density_matrix: float,
    mobility_matrix: float,
    donor_density_inclusion: float,
    mobility_inclusion: float,
    period: float,
    mean_field,
    n_grid: int,
    mode_step: int,
    refinement: int,
    spacing_ratio: float,
    tolerance: float,
    max_iterations: int,
) -> dict:
    """Reference implementation: reruns the whole chain once per candidate truncation."""
    big_n = _even_positive(n_grid, "n_grid")
    step = _even_positive(mode_step, "mode_step")
    if big_n % step != 0:
        raise ValueError("mode_step must divide n_grid")
    if isinstance(refinement, bool) or not isinstance(refinement, (int, np.integer)):
        raise ValueError("refinement must be an integer")
    factor = int(refinement)
    if factor < 2:
        raise ValueError("refinement must be at least two")
    length = _positive_float(period, "period")
    ratio_h = _positive_float(spacing_ratio, "spacing_ratio")
    tol = _positive_float(tolerance, "tolerance")

    phases = _oracle_resolve_phase_conductivities(                      # noqa: F821
        donor_density_matrix, mobility_matrix,
        donor_density_inclusion, mobility_inclusion, 0.25,
    )
    c1 = phases["c_matrix"]
    c2 = phases["c_inclusion"]
    c0 = 0.5 * (c1 + c2)

    board = _oracle_checkerboard_conductivity(big_n, c1, c2, length)    # noqa: F821
    conductivity = board["conductivity"]
    spacing = length / (big_n + 1)

    candidates = tuple(range(step, big_n + 1, step))
    errors = []
    macroscopic = []
    peaks = []
    iterations = []
    generalised_excess = []
    for modes in candidates:
        lattice = _oracle_spectral_lattice(big_n, modes, length, c0)    # noqa: F821
        solved = _oracle_moulinec_suquet_field(                        # noqa: F821
            conductivity, lattice["symbol"], lattice["green"], lattice["retained"],
            c0, mean_field, length, tol, max_iterations,
        )
        rebuilt = _oracle_spectral_reconstruction(                     # noqa: F821
            solved["field_transform"], big_n, modes, factor,
        )
        scored = _oracle_energy_norm_error(                            # noqa: F821
            rebuilt["fine_field"], modes, c1, c2, mean_field,
        )
        errors.append(scored["energy_norm_error"])
        macroscopic.append(solved["macroscopic_conductivity"])
        peaks.append(rebuilt["peak_ratio"])
        iterations.append(solved["iterations"])

        modified = _oracle_generalised_green_operator(                  # noqa: F821
            lattice["wavevector"], ratio_h * spacing, c0, lattice["retained"],
        )
        solved_g = _oracle_moulinec_suquet_field(                       # noqa: F821
            conductivity, modified["symbol"], modified["green"], lattice["retained"],
            c0, mean_field, length, tol, max_iterations,
        )
        rebuilt_g = _oracle_spectral_reconstruction(                    # noqa: F821
            solved_g["field_transform"], big_n, modes, factor,
        )
        scored_g = _oracle_energy_norm_error(                           # noqa: F821
            rebuilt_g["fine_field"], modes, c1, c2, mean_field,
        )
        # not an energy-norm error: the difference rule breaks the compatibility the
        # identity needs, so this is an apparent-conductivity excess
        generalised_excess.append(scored_g["energy_norm_error"])

    best = int(np.argmin(errors))
    best_g = int(np.argmin(generalised_excess))
    exact = _oracle_energy_norm_error(                                  # noqa: F821
        np.ones((2, 4 * step + 1, 4 * step + 1)), step, c1, c2, (1.0, 0.0),
    )["exact_macroscopic"]
    full = len(candidates) - 1
    return {
        "optimal_ratio": candidates[best] / float(big_n),
        "optimal_modes": int(candidates[best]),
        "optimal_error": errors[best],
        "error_at_full": errors[full],
        "error_ratio": errors[full] / errors[best],
        "macroscopic_at_optimum": macroscopic[best],
        "macroscopic_at_full": macroscopic[full],
        "exact_macroscopic": exact,
        "macro_error_at_optimum": abs(macroscopic[best] - exact) / exact,
        "macro_error_at_full": abs(macroscopic[full] - exact) / exact,
        "c_matrix": c1,
        "c_inclusion": c2,
        "contrast": phases["contrast"],
        "sampled_area_fraction": board["sampled_area_fraction"],
        "peak_ratio_at_optimum": peaks[best],
        "peak_ratio_at_full": peaks[full],
        "iterations_at_optimum": int(iterations[best]),
        "generalised_optimal_ratio": candidates[best_g] / float(big_n),
        "generalised_optimal_modes": int(candidates[best_g]),
        "generalised_optimal_excess": generalised_excess[best_g],
        "candidates": candidates,
        "errors": tuple(errors),
    }

# =============================================================================
# TEST CASES
# =============================================================================

FLAT = """
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        out = []
        for v in x:
            out.extend(flat(v))
        return tuple(out)
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    if isinstance(x, bool):
        return (int(x),)
    return (x,)
"""

BASE = "1.0e22, 0.1200, 5.0e24, 0.0240, 2.0e-6, (1.0e5, 0.0)"


def test_cases():
    return [
        {
            # the whole chain at two reduced grids: the minimiser, the errors either
            # side of it, and the contrast between the field optimum and the fact that
            # the macroscopic error is smallest at the largest truncation
            "setup": """
def digest(out):
    return (round(out["optimal_ratio"], 12), out["optimal_modes"],
            round(out["optimal_error"], 12), round(out["error_at_full"], 12),
            int(out["optimal_error"] < out["error_at_full"]),
            int(out["macro_error_at_full"] < out["macro_error_at_optimum"]),
            round(out["macroscopic_at_optimum"], 8), round(out["exact_macroscopic"], 9),
            round(out["contrast"], 10), round(out["sampled_area_fraction"], 12),
            round(out["generalised_optimal_excess"], 12), out["generalised_optimal_modes"],
            round(out["peak_ratio_at_full"], 10), round(out["peak_ratio_at_optimum"], 10),
            round(out["macroscopic_at_full"], 8), round(out["macro_error_at_full"], 12),
            round(out["macro_error_at_optimum"], 12), round(out["error_ratio"], 10),
            out["iterations_at_optimum"], round(out["c_matrix"], 8),
            round(out["c_inclusion"], 6), tuple(out["candidates"]))
""" + FLAT,
            "call": "flat((digest(optimal_truncation_ratio(" + BASE + ", 16, 4, 4, 0.5, 1.0e-9, 20000)), digest(optimal_truncation_ratio(" + BASE + ", 24, 4, 4, 0.5, 1.0e-9, 20000))))",
            "gold_call": "flat((digest(_oracle_optimal_truncation_ratio(" + BASE + ", 16, 4, 4, 0.5, 1.0e-9, 20000)), digest(_oracle_optimal_truncation_ratio(" + BASE + ", 24, 4, 4, 0.5, 1.0e-9, 20000))))",
        },
        {
            # the reported quantities must not move with the cell size, the strength of
            # the loading, or a common rescaling of both conductivities, since the
            # answer is dimensionless and the problem is linear
            "setup": """
def digest(out):
    return (round(out["optimal_ratio"], 12), round(out["optimal_error"], 10),
            round(out["error_ratio"], 8), round(out["generalised_optimal_ratio"], 12),
            round(out["generalised_optimal_excess"], 10), round(out["peak_ratio_at_full"], 10))
def compare(fn):
    a = digest(fn(1.0e22, 0.1200, 5.0e24, 0.0240, 2.0e-6, (1.0e5, 0.0), 16, 4, 4, 0.5, 1.0e-9, 20000))
    b = digest(fn(1.0e22, 0.1200, 5.0e24, 0.0240, 5.0e-4, (1.0e5, 0.0), 16, 4, 4, 0.5, 1.0e-9, 20000))
    c = digest(fn(1.0e22, 0.1200, 5.0e24, 0.0240, 2.0e-6, (0.0, 3.0e3), 16, 4, 4, 0.5, 1.0e-9, 20000))
    d = digest(fn(1.0e22, 0.6000, 5.0e24, 0.1200, 2.0e-6, (1.0e5, 0.0), 16, 4, 4, 0.5, 1.0e-9, 20000))
    return (a, tuple(round(x - y, 9) for x, y in zip(a, b)),
            tuple(round(x - y, 9) for x, y in zip(a, c)),
            tuple(round(x - y, 9) for x, y in zip(a, d)))
""" + FLAT,
            "call": "flat(compare(optimal_truncation_ratio))",
            "gold_call": "flat(compare(_oracle_optimal_truncation_ratio))",
        },
        {
            # the shape of the curve is recorded rather than asserted, because on denser
            # candidate lists it acquires a mod-four alternation and is monotone on
            # neither side; what is asserted is that refining the reconstruction changes
            # nothing, the integration being already exact
            "setup": """
def digest(out):
    e = list(out["errors"])
    k = e.index(min(e))
    falling = int(all(e[i] > e[i + 1] for i in range(k)))
    rising = int(all(e[i] < e[i + 1] for i in range(k, len(e) - 1)))
    return (k, len(e), falling, rising, int(0 < k < len(e) - 1),
            round(out["optimal_ratio"], 12), round(out["optimal_error"], 12))
def compare(fn):
    a = digest(fn(1.0e22, 0.1200, 5.0e24, 0.0240, 2.0e-6, (1.0e5, 0.0), 16, 2, 4, 0.5, 1.0e-9, 20000))
    b = digest(fn(1.0e22, 0.1200, 5.0e24, 0.0240, 2.0e-6, (1.0e5, 0.0), 16, 2, 7, 0.5, 1.0e-9, 20000))
    return (a, tuple(round(x - y, 10) for x, y in zip(a, b)))
""" + FLAT,
            "call": "flat(compare(optimal_truncation_ratio))",
            "gold_call": "flat(compare(_oracle_optimal_truncation_ratio))",
        },
        {
            "setup": """
def verdict(fn, **kw):
    a = dict(donor_density_matrix=1.0e22, mobility_matrix=0.1200,
             donor_density_inclusion=5.0e24, mobility_inclusion=0.0240,
             period=2.0e-6, mean_field=(1.0e5, 0.0), n_grid=8, mode_step=2,
             refinement=4, spacing_ratio=0.5, tolerance=1.0e-9, max_iterations=20000)
    a.update(kw)
    try:
        fn(**a)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(optimal_truncation_ratio), verdict(optimal_truncation_ratio, n_grid=7), verdict(optimal_truncation_ratio, mode_step=3), verdict(optimal_truncation_ratio, mode_step=6), verdict(optimal_truncation_ratio, refinement=1), verdict(optimal_truncation_ratio, period=0.0), verdict(optimal_truncation_ratio, spacing_ratio=-1.0), verdict(optimal_truncation_ratio, tolerance=0.0), verdict(optimal_truncation_ratio, max_iterations=2), verdict(optimal_truncation_ratio, mobility_matrix=0.0), verdict(optimal_truncation_ratio, mean_field=(0.0, 0.0))))",
            "gold_call": "flat((verdict(_oracle_optimal_truncation_ratio), verdict(_oracle_optimal_truncation_ratio, n_grid=7), verdict(_oracle_optimal_truncation_ratio, mode_step=3), verdict(_oracle_optimal_truncation_ratio, mode_step=6), verdict(_oracle_optimal_truncation_ratio, refinement=1), verdict(_oracle_optimal_truncation_ratio, period=0.0), verdict(_oracle_optimal_truncation_ratio, spacing_ratio=-1.0), verdict(_oracle_optimal_truncation_ratio, tolerance=0.0), verdict(_oracle_optimal_truncation_ratio, max_iterations=2), verdict(_oracle_optimal_truncation_ratio, mobility_matrix=0.0), verdict(_oracle_optimal_truncation_ratio, mean_field=(0.0, 0.0))))",
        },
    ]
