"""
Compose the law-screened target and robust full-Maxwell benchmark.

Return the best law-screened common-passband guarantee ratio.



``energies`` is a finite strictly increasing one-dimensional supplied-sample

grid with at least five nodes. Interpolate target tensors and reconstruct

the signed semi-infinite Maxwell response. The closed `$enz_window$` selects

supplied samples; it does not create clipped boundary samples or evaluate a

continuous interpolant there. Expand each finite comparison boundary by

exactly one outward IEEE-754 ``nextafter`` step so a decimal endpoint that

differs from its intended grid node by one representable value remains

included. Then extract the retained spectrum's signed peak,

connected half-maximum support, and positive-to-negative lobe spacing.

Build target ENZ coordinates, evaluate the universal strength and

lobe-spacing laws on `$angle_grid$`, and apply the nominal agreement and

interval-wide gyrotropy-validity screens. Intersect every target Maxwell

support in each gate/alignment rectangle, multiply the common width by

the minimum peak, and select one fixed nominal setting.



Independently reconstruct the benchmark response from exactly its supplied

energy-plus-six tensor rows, with no densification or direct analytic-curve

optimization. At each angle, PCHIP the resulting contrast values and extract

Step 3 metrics when both signed lobes exist. For each fixed benchmark nominal angle whose complete closed

alignment window contains valid metrics, intersect all peak-signed

half-maximum supports, multiply the common width by the minimum peak, and

maximize this product with exact ties going to the smaller nominal angle.

Divide the selected target product by that robust 0.16-T benchmark

product. ``drifts[1]`` is an exact integer multiple of the uniform angle

spacing. Compose all seven prior public functions; private reference-implementation

names are not public. Absolute accuracy: 2e-6.

Returns
-------
One finite target-to-benchmark robust common-passband guarantee ratio
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def robust_surface_advantage(
    mu_nodes: np.ndarray,
    energies: np.ndarray,
    tensors: np.ndarray,
    mu_grid: np.ndarray,
    angle_grid: np.ndarray,
    benchmark: np.ndarray,
    enz_window: np.ndarray,
    drifts: np.ndarray,
    validity_limit: float,
    agreement_limits: np.ndarray,
) -> float:
    """Return the best law-screened common-passband guarantee ratio.

    ``mu_nodes``, ``tensors``, and ``mu_grid`` retain their Step 1 input
    roles, ``angle_grid`` retains its Step 2 role, and ``benchmark`` is the
    supplied increasing energy-plus-six-tensor-column table with shape (K,7).
    The target tensor energy-axis length must equal ``len(energies)``.

    ``energies`` is a finite strictly increasing one-dimensional supplied-sample
    grid with at least five nodes. Interpolate target tensors and reconstruct
    the signed semi-infinite Maxwell response. The closed ``enz_window`` selects
    supplied samples; it does not create clipped boundary samples or evaluate a
    continuous interpolant there. Expand each finite comparison boundary by
    exactly one outward IEEE-754 ``nextafter`` step so a decimal endpoint that
    differs from its intended grid node by one representable value remains
    included. Then extract the retained spectrum's signed peak,
    connected half-maximum support, and positive-to-negative lobe spacing.
    Extract target metrics spectrum by spectrum. A rejected spectrum omits only
    target gate/alignment windows containing that spectrum; it must not abort
    the remaining target calculation, and unaffected windows remain eligible.
    Build target ENZ coordinates, evaluate the universal strength and
    lobe-spacing laws on ``angle_grid``, and apply the nominal agreement and
    interval-wide gyrotropy-validity screens. Intersect every target Maxwell
    support in each gate/alignment rectangle, multiply the common width by
    the minimum peak, and select one fixed nominal setting.

    Independently reconstruct the benchmark response from exactly its supplied
    energy-plus-six tensor rows, with no densification or direct analytic-curve
    optimization. At each angle, PCHIP the resulting contrast values and extract
    Step 3 metrics when both signed lobes exist. For each fixed benchmark nominal angle whose complete closed
    alignment window contains valid metrics, intersect all peak-signed
    half-maximum supports, multiply the common width by the minimum peak, and
    maximize this product with exact ties going to the smaller nominal angle.
    Divide the selected target product by that robust 0.16-T benchmark
    product. ``drifts[0]`` is a closed physical chemical-potential half-width:
    include exactly the supplied grid nodes within it. ``drifts[1]`` is an exact
    integer multiple of the uniform angle spacing. Compose all seven prior public
    functions; private reference-implementation names are not public. Raise ``ValueError``
    for malformed, nonfinite, unordered, incompatible, or inherited-contract
    inputs, including tensor/energy-axis mismatch, invalid windows or drifts,
    or the absence of an admissible target or benchmark passband. Absolute
    accuracy: 2e-6.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Compose the law-screened target and robust full-Maxwell benchmark."""

import numpy as np

def _benchmark_common_guarantee(benchmark, angles, angle_drift):
    table = np.asarray(benchmark, float)
    theta = np.asarray(angles, float)
    if (table.ndim != 2 or table.shape[1] != 7 or len(table) < 5
            or not np.all(np.isfinite(table)) or not np.all(np.diff(table[:, 0]) > 0)):
        raise ValueError("benchmark must be an increasing finite (E,7) table")
    spacing = np.diff(theta)
    if (len(spacing) == 0
            or not np.allclose(spacing, spacing[0], rtol=0, atol=1e-12)):
        raise ValueError("angle grid must be uniform")
    radius = int(round(float(angle_drift) / spacing[0]))
    if (radius < 0
            or abs(radius * spacing[0] - angle_drift) > 1e-10):
        raise ValueError("angle drift must be a nonnegative integer grid multiple")
    responses = _oracle_voigt_surface_responses(table[:, 1:], theta)
    metrics = {}
    for j in range(len(theta)):
        try:
            metrics[j] = _oracle_full_maxwell_metrics(
                table[:, 0], responses[j:j+1]
            )[0]
        except ValueError:
            continue

    best = None
    for j in range(radius, len(theta) - radius):
        indices = range(j-radius, j+radius+1)
        if any(index not in metrics for index in indices):
            continue
        block = np.asarray([metrics[index] for index in indices])
        dmin = float(np.min(block[:, 0]))
        common_left = float(np.max(block[:, 2]))
        common_right = float(np.min(block[:, 3]))
        if common_right <= common_left:
            continue
        dloc = int(np.flatnonzero(block[:, 0] == dmin)[0])
        lloc = int(np.flatnonzero(block[:, 2] == common_left)[0])
        rloc = int(np.flatnonzero(block[:, 3] == common_right)[0])
        width = common_right - common_left
        row = np.asarray((
            theta[j], dmin, common_left, common_right, width, dmin*width,
            theta[j-radius+dloc], theta[j-radius+lloc], theta[j-radius+rloc],
        ))
        if best is None or row[5] > best[5]:
            best = row
    if best is None:
        raise ValueError("benchmark has no valid common alignment passband")
    return best

def _oracle_robust_surface_advantage(
    mu_nodes: np.ndarray,
    energies: np.ndarray,
    tensors: np.ndarray,
    mu_grid: np.ndarray,
    angle_grid: np.ndarray,
    benchmark: np.ndarray,
    enz_window: np.ndarray,
    drifts: np.ndarray,
    validity_limit: float,
    agreement_limits: np.ndarray,
) -> float:
    energy = np.asarray(energies, float)
    window = np.asarray(enz_window, float)
    if (energy.ndim != 1 or len(energy) < 5 or not np.all(np.isfinite(energy))
            or np.any(np.diff(energy) <= 0)):
        raise ValueError("energies must be a finite increasing one-dimensional grid")
    if (window.shape != (2,) or not np.all(np.isfinite(window))
            or window[0] >= window[1]):
        raise ValueError("enz_window must contain finite increasing bounds")
    lower = np.nextafter(window[0], -np.inf)
    upper = np.nextafter(window[1], np.inf)
    selected = (energy >= lower) & (energy <= upper)
    if np.count_nonzero(selected) < 5:
        raise ValueError("enz_window must retain at least five energy nodes")

    fields = _oracle_material_fields(mu_nodes, tensors, mu_grid)
    if fields.ndim != 3 or fields.shape[1] != len(energy):
        raise ValueError("tensor energy axis must match energies")
    responses = _oracle_voigt_surface_responses(fields, angle_grid)
    metric_shape = responses.shape[:-1]
    full = np.full(metric_shape + (7,), np.nan)
    spectrum_valid = np.ones(metric_shape, dtype=bool)
    for index in np.ndindex(metric_shape):
        try:
            full[index] = _oracle_full_maxwell_metrics(
                energy[selected], responses[index][selected][None, :]
            )[0]
        except ValueError:
            spectrum_valid[index] = False
    energy_column = np.broadcast_to(
        energy[None, selected, None], (len(mu_grid), selected.sum(), 1)
    )
    target_spectra = np.concatenate((energy_column, fields[:, selected, :]), axis=2)
    coordinates = _oracle_enz_design_coordinates(list(target_spectra))
    target_laws = _oracle_universal_law_metrics(coordinates, angle_grid)
    drift_values = np.asarray(drifts, float)
    design_mu = np.asarray(mu_grid, float)
    if (drift_values.shape != (2,) or not np.all(np.isfinite(drift_values))
            or np.any(drift_values < 0) or design_mu.ndim != 1
            or len(design_mu) < 2 or not np.all(np.isfinite(design_mu))
            or np.any(np.diff(design_mu) <= 0)):
        raise ValueError("drifts and chemical-potential grid must be finite and valid")
    mu_spacing = np.diff(design_mu)
    if not np.allclose(mu_spacing, mu_spacing[0], rtol=0, atol=1e-12):
        raise ValueError("chemical-potential grid must be uniform")
    mu_ratio = drift_values[0] / mu_spacing[0]
    nearest_mu_radius = int(round(mu_ratio))
    if abs(mu_ratio - nearest_mu_radius) <= 1e-10:
        mu_radius = nearest_mu_radius
    else:
        mu_radius = int(np.floor(np.nextafter(mu_ratio, np.inf)))
    effective_drifts = (mu_radius * mu_spacing[0], drift_values[1])
    guarantees = _oracle_window_guarantees(
        mu_grid,
        angle_grid,
        full,
        target_laws,
        coordinates[:, 5],
        effective_drifts,
        validity_limit,
        agreement_limits,
        spectrum_valid,
    )
    optimum = _oracle_select_operating_point(guarantees)
    benchmark_result = _benchmark_common_guarantee(
        benchmark, angle_grid, drifts[1]
    )
    return float(optimum[12] / benchmark_result[5])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common_setup = """import copy as _input_copy

def _isolated_call(fn, *args, **kwargs):
    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))
    return fn(*copied_args, **copied_kwargs)

import numpy as np

def _value_error_code(fn, *args):
    try:
        _isolated_call(fn, *args)
    except ValueError:
        return 1.0
    except Exception:
        return -1.0
    return 0.0
mu = np.array([-0.333, -0.3, -0.267, -0.233, -0.2, -0.167, -0.133])
e = np.linspace(0.06, 0.24, 19)
m = mu[:, None]
w = e[None, :]
e0 = 0.17 + 0.08 * m
loss = 8 + 0 * m
g = 1 + 0 * m
t = np.stack([80 * (w - e0), loss + 0 * w, 80 * (w - e0), loss + 0 * w, (0.04 + 12 * (m + 0.22) ** 2) * g + 0 * w, g + 0 * w], axis=-1)
be = np.array([0.13, 0.14, 0.15, 0.16, 0.17])
benchmark = np.column_stack((be, 160 * (be - 0.153), 0.8 + 0 * be, 160 * (be - 0.153), 0.8 + 0 * be, 0.015 + 0 * be, -0.12 + 0 * be))
"""
    return [
        {
            "setup": common_setup + """mg = np.linspace(-0.31, -0.155, 41)
ag = np.linspace(5.0, 65.0, 17)
""",
            "call": (
                "float(_isolated_call(robust_surface_advantage, mu, e, t, mg, ag, benchmark, (0.06, 0.24), "
                "(0.019375, 7.5), 0.2, (0.95, 0.95)))"
            ),
            "gold_call": (
                "(lambda v: v if np.isclose(v, 2.9309277140800547, rtol=0, atol=2e-06) else (_ for _ in "
                "()).throw(AssertionError('independent Step 8 pin "
                "drifted')))(float(_isolated_call(_oracle_robust_surface_advantage, mu, e, t, mg, ag, benchmark, "
                "(0.06, 0.24), (0.019375, 7.5), 0.2, (0.95, 0.95))))"
            ),
            "tol": 2e-06,
        },
        {
            "setup": common_setup + """mu = mu + 0.004
t = t.copy()
t[..., 0] *= 1.15
t[..., 1] *= 1.05
t[..., 2] *= 0.9
t[..., 3] *= 0.97
t[..., 5] *= 1.2
mg = np.linspace(-0.31, -0.155, 49)
ag = np.linspace(6.0, 66.0, 21)
""",
            "call": (
                "float(_isolated_call(robust_surface_advantage, mu, e, t, mg, ag, benchmark, (0.07, 0.23), "
                "(0.012916666666666667, 6.0), 0.2, (0.27, 0.25)))"
            ),
            "gold_call": (
                "(lambda v: v if np.isclose(v, 3.047185772968821, rtol=0, atol=2e-06) else (_ for _ in "
                "()).throw(AssertionError('independent Step 8 pin "
                "drifted')))(float(_isolated_call(_oracle_robust_surface_advantage, mu, e, t, mg, ag, benchmark, "
                "(0.07, 0.23), (0.012916666666666667, 6.0), 0.2, (0.27, 0.25))))"
            ),
            "tol": 2e-06,
        },
        {
            "setup": common_setup + """e = e * 1.01
benchmark = benchmark.copy()
benchmark[:, 5:7] *= 1.2
mg = np.linspace(-0.31, -0.155, 57)
ag = np.linspace(5.0, 65.0, 25)
""",
            "call": (
                "float(_isolated_call(robust_surface_advantage, mu, e, t, mg, ag, benchmark, (0.09, 0.23), "
                "(0.02214285714285714, 5.0), 0.055, (0.95, 0.95)))"
            ),
            "gold_call": (
                "(lambda v: v if np.isclose(v, 1.627965363054241, rtol=0, atol=2e-06) else (_ for _ in "
                "()).throw(AssertionError('independent Step 8 pin "
                "drifted')))(float(_isolated_call(_oracle_robust_surface_advantage, mu, e, t, mg, ag, benchmark, "
                "(0.09, 0.23), (0.02214285714285714, 5.0), 0.055, (0.95, 0.95))))"
            ),
            "tol": 2e-06,
        },
        {
            "setup": common_setup + """t = t.copy()
t[0, :, 4] = 4.0
mg = np.linspace(mu[0], mu[-1], 41)
ag = np.linspace(5.0, 65.0, 17)
""",
            "call": (
                "float(_isolated_call(robust_surface_advantage, mu, e, t, mg, ag, benchmark, (0.06, 0.24), (0.01,"
                " 7.5), 0.2, (0.95, 0.95)))"
            ),
            "gold_call": (
                "(lambda v: v if np.isclose(v, 3.139040113207104, rtol=0, atol=2e-06) else (_ for _ in "
                "()).throw(AssertionError('independent Step 8 pin "
                "drifted')))(float(_isolated_call(_oracle_robust_surface_advantage, mu, e, t, mg, ag, benchmark, "
                "(0.06, 0.24), (0.01, 7.5), 0.2, (0.95, 0.95))))"
            ),
            "tol": 2e-06,
        },
        {
            "setup": common_setup + """mg = np.linspace(-0.31, -0.155, 41)
ag = np.linspace(5.0, 65.0, 17)
bad_e = e[:-1]
""",
            "call": (
                "_value_error_code(robust_surface_advantage, mu, bad_e, t, mg, ag, benchmark, (0.06, 0.23), "
                "(0.019375, 7.5), 0.2, (0.95, 0.95))"
            ),
            "gold_call": (
                "_value_error_code(_oracle_robust_surface_advantage, mu, bad_e, t, mg, ag, benchmark, (0.06, "
                "0.23), (0.019375, 7.5), 0.2, (0.95, 0.95))"
            ),
            "tol": 0,
        },
        {
            "setup": common_setup + """mg = np.linspace(-0.31, -0.155, 41)
ag = np.linspace(5.0, 65.0, 17)
bad_t = t[:, :-1, :]
""",
            "call": (
                "_value_error_code(robust_surface_advantage, mu, e, bad_t, mg, ag, benchmark, (0.06, 0.23), "
                "(0.019375, 7.5), 0.2, (0.95, 0.95))"
            ),
            "gold_call": (
                "_value_error_code(_oracle_robust_surface_advantage, mu, e, bad_t, mg, ag, benchmark, (0.06, "
                "0.23), (0.019375, 7.5), 0.2, (0.95, 0.95))"
            ),
            "tol": 0,
        },
    ]
