"""
Run the leading-order chain at every quasi-momentum in the list, in this order. Step 1 resolves the contrast bookkeeping. Step 2 partitions the cell and returns its edge lengths; it is run once, as is step 3, which forms the element stiffness of one background cube. Then, at each quasi-momentum in turn, step 4 assembles the quasi-periodic exterior stiffness, step 5 condenses it onto the resonator surface and tests the resulting map against all six rigid motions, and step 6 solves the generalised problem against the inertia of the rigid resonator, supplied in the same basis, to give the six branch frequencies. Step 7 supplies the dilute-limit reference.

Then, at the quasi-momentum attaining the leading-order upper edge and there only, run the finite-contrast chain: step 8 assembles the full two-phase pencil of the cell, and step 9 returns its eight lowest eigenfrequencies, each converged to a relative residual below one part in 1e8. The six lowest of those are the finite-contrast counterparts of the six leading-order branches; which of the eight is the first ordinary branch of the cell is for the solver to identify, and it is reported as the ordinary branch. The graded correction is the fourth lowest finite-contrast frequency minus the leading-order upper edge, in hertz.

Report the leading-order quantities alongside: both edges and the width of the leading-order gap, the branch maxima summed over the six branches, the position of the leading-order lower edge within the dilute-limit interval for a ball of the same volume, and the shear wavelength of the background at the leading-order upper edge divided by the lattice constant.

Returns
-------
dict, holding correction_hertz, the graded finite-contrast correction; finite_contrast_edge_hertz and full_pencil_hertz, the corrected edge and the eight lowest finite-contrast frequencies at the attaining quasi-momentum; ordinary_branch_hertz, the first of them that is an ordinary branch of the cell; bandgap_edge_hertz and bandgap_edge_angular for the leading-order upper edge; gap_upper_hertz, gap_lower_hertz, gap_width_hertz and bandwidth_sum_hertz; argmax_index and argmax_branch for the entry attaining the upper edge, lower_argmax_index and lower_argmax_branch for the lower edge; band_tops, frequencies, ball_hertz_min, ball_hertz_max, position_in_interval, tau and wavelength_ratio.

Assemble the whole calculation and report the finite-contrast correction to the upper edge of the first subwavelength bandgap: the difference, in hertz, between the edge the full two-phase problem gives at the stated contrast and the edge the leading-order reduction gives.

The reduction carries six branches, one for each rigid motion of the resonator, and is exact only in the limit of vanishing contrast. At the stated finite contrast the Bloch frequencies of the two-phase cell differ from the leading-order ones. This stage measures that difference at one quasi-momentum, the one at which the leading-order upper edge is attained.

Sorted ascending at each quasi-momentum, the first three leading-order branches are the ones built on the translations and the last three the ones built on the rotations, at every quasi-momentum in the prescribed list. The two families behave differently as the quasi-momentum goes to zero, and which of them descends there, which does not, and why, is part of the task rather than something this stage settles. The leading-order lower edge of the gap is the largest value the first family attains and its upper edge the smallest value the second attains, both over the prescribed list; the leading-order edge this stage corrects is the upper one.

Returns
-------
dict, holding correction_hertz, the graded finite-contrast correction; finite_contrast_edge_hertz and full_pencil_hertz, the corrected edge and the eight lowest finite-contrast frequencies at the attaining quasi-momentum; ordinary_branch_hertz, the first of them that is an ordinary branch of the cell; bandgap_edge_hertz and bandgap_edge_angular for the leading-order upper edge; gap_upper_hertz, gap_lower_hertz, gap_width_hertz and bandwidth_sum_hertz; argmax_index and argmax_branch for the entry attaining the upper edge, lower_argmax_index and lower_argmax_branch for the lower edge; band_tops, frequencies, ball_hertz_min, ball_hertz_max, position_in_interval, tau and wavelength_ratio.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def report_finite_contrast_correction(
    lattice_constant: float,
    n_side: int,
    spans: tuple,
    lam: float,
    mu: float,
    rho: float,
    delta: float,
    eps: float,
    alphas: tuple,
    rel_tol: float,
) -> dict:
    """Scan the prescribed quasi-momenta for the leading-order gap, then correct its upper edge at finite contrast with the full two-phase pencil, and report the correction.

    Parameters
    ----------
    lattice_constant : float
        Edge of the cubic unit cell in metre.
    n_side : int
        Elements along each edge of the cell.
    spans : tuple
        Elements spanned by the resonator along each axis.
    lam : float
        First Lame parameter of the background in pascal.
    mu : float
        Shear modulus of the background in pascal.
    rho : float
        Background density in kilogram per cubic metre.
    delta : float
        Reciprocal stiffness contrast.
    eps : float
        Reciprocal density contrast.
    alphas : tuple
        Quasi-momenta to scan.
    rel_tol : float
        Relative residual at which each leading-order solve stops. The two-phase
        pencil is solved to the residual bound the prompt states, 1e-8, which
        this argument does not alter.

    Returns
    -------
    dict
        Under the keys correction_hertz, finite_contrast_edge_hertz, full_pencil_hertz,
        ordinary_branch_hertz, bandgap_edge_hertz, bandgap_edge_angular, gap_upper_hertz,
        gap_lower_hertz, gap_width_hertz, bandwidth_sum_hertz, argmax_index, argmax_branch,
        lower_argmax_index, lower_argmax_branch, band_tops, frequencies, ball_hertz_min,
        ball_hertz_max, position_in_interval, tau and wavelength_ratio.
        correction_hertz is the graded quantity: the fourth lowest finite-contrast frequency at
        the quasi-momentum attaining the leading-order upper edge, minus that leading-order
        edge, in hertz. finite_contrast_edge_hertz is that fourth lowest frequency and
        full_pencil_hertz the float array of the eight lowest, ascending; ordinary_branch_hertz
        is the first of them that is an ordinary branch of the cell. bandgap_edge_hertz is the leading-order upper edge and equals
        gap_upper_hertz; argmax_index and argmax_branch locate it, the branch index lying in
        three to five. gap_lower_hertz is the leading-order lower edge, located by
        lower_argmax_index and lower_argmax_branch with that branch index in zero to two, and
        gap_width_hertz is their difference. band_tops holds the largest value each of the six
        leading-order branches attains and frequencies is the (n_alpha, 6) table, sorted
        ascending at each quasi-momentum. position_in_interval places the leading-order LOWER
        edge within the dilute interval, and wavelength_ratio is the background shear
        wavelength at the leading-order upper edge divided by the lattice constant.

    Raises
    ------
    ValueError
        When alphas is empty, when any argument fails the checks the earlier stages impose, or when the scan produces no finite frequency.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_report_finite_contrast_correction(
    lattice_constant: float,
    n_side: int,
    spans: tuple,
    lam: float,
    mu: float,
    rho: float,
    delta: float,
    eps: float,
    alphas: tuple,
    rel_tol: float,
) -> dict:
    """Reference implementation chaining every earlier stage."""
    grid = tuple(alphas)
    if len(grid) == 0:
        raise ValueError("alphas must hold at least one quasi-momentum")

    scaling = _oracle_resolve_contrast_scaling(lam, mu, rho, delta, eps)  # noqa: F821
    part = _oracle_partition_unit_cell(lattice_constant, n_side, spans)   # noqa: F821
    elem = _oracle_hex_element_stiffness(lam, mu, part["h"])              # noqa: F821

    span = np.asarray(spans, dtype=np.float64)
    lo_corner = ((np.asarray([n_side] * 3, dtype=np.float64) - span) // 2)
    centre = (lo_corner + span / 2.0) * part["h"]
    mass = (rho / eps) * part["volume_D"]
    sd = np.asarray(part["sides"], dtype=np.float64)
    moments = np.array([mass * (sd[1] ** 2 + sd[2] ** 2) / 12.0,
                        mass * (sd[0] ** 2 + sd[2] ** 2) / 12.0,
                        mass * (sd[0] ** 2 + sd[1] ** 2) / 12.0])
    inertia = np.diag(np.concatenate([np.full(3, mass), moments]))

    rows = []
    for alpha in grid:
        blocks = _oracle_assemble_bloch_exterior(                          # noqa: F821
            part["elements"], part["surface_nodes"], part["free_nodes"],
            elem["element_stiffness"], elem["corner_signs"], n_side, alpha,
        )
        cap = _oracle_capacity_schur_matrix(                               # noqa: F821
            blocks["K_ff"], blocks["K_fc"], blocks["K_cc"], blocks["surface_dofs"],
            n_side, part["h"], centre, rel_tol,
        )
        freq = _oracle_subwavelength_frequencies(                          # noqa: F821
            cap["capacity"], part["volume_D"], inertia, rho, eps,
        )
        rows.append(freq["hertz"])

    table = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(table)):
        raise ValueError("the scan produced a frequency that is not finite")

    lower = table[:, :3]
    upper = table[:, 3:]
    lower_flat = int(np.argmax(lower))
    lower_idx, lower_branch = divmod(lower_flat, 3)
    gap_lower = float(lower[lower_idx, lower_branch])
    upper_flat = int(np.argmin(upper))
    upper_idx, upper_branch = divmod(upper_flat, 3)
    gap_upper = float(upper[upper_idx, upper_branch])
    if not gap_upper > gap_lower:
        raise ValueError("the sampled branches overlap: no first gap")
    edge = gap_upper

    ball = _oracle_dilute_ball_reference(lam, mu, rho, eps, part["volume_D"])  # noqa: F821
    lo, hi = ball["hertz_min"], ball["hertz_max"]

    pencil = _oracle_assemble_full_bloch_pencil(                          # noqa: F821
        n_side, spans, elem["element_stiffness"], elem["corner_signs"], part["h"],
        rho, delta, eps, grid[upper_idx],
    )
    spec = _oracle_full_pencil_spectrum(pencil["K_full"], pencil["M_full"], 8, 1.0e-8)  # noqa: F821
    full = np.asarray(spec["hertz"], dtype=np.float64)
    corrected = float(full[3])

    tops = table.max(axis=0)
    return {
        "correction_hertz": corrected - edge,
        "finite_contrast_edge_hertz": corrected,
        "full_pencil_hertz": full,
        "ordinary_branch_hertz": float(full[6]),
        "bandwidth_sum_hertz": float(tops.sum()),
        "gap_upper_hertz": gap_upper,
        "gap_lower_hertz": gap_lower,
        "gap_width_hertz": gap_upper - gap_lower,
        "bandgap_edge_hertz": edge,
        "bandgap_edge_angular": edge * 2.0 * np.pi,
        "argmax_index": int(upper_idx),
        "argmax_branch": int(upper_branch) + 3,
        "lower_argmax_index": int(lower_idx),
        "lower_argmax_branch": int(lower_branch),
        "band_tops": tops,
        "frequencies": table,
        "ball_hertz_min": float(lo),
        "ball_hertz_max": float(hi),
        "position_in_interval": float((gap_lower - lo) / (hi - lo)),
        "tau": float(scaling["tau"]),
        "wavelength_ratio": float(scaling["c_s"] / edge / float(lattice_constant)),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
PI = np.pi
SMALL = ((PI, 0.0, 0.0), (0.0, 0.0, PI), (PI, PI, PI))
def digest(out):
    return (round(float(out["correction_hertz"]), 5), round(float(out["finite_contrast_edge_hertz"]), 5),
            tuple(round(float(v), 5) for v in out["full_pencil_hertz"]), round(float(out["ordinary_branch_hertz"]), 5),
            round(float(out["bandwidth_sum_hertz"]), 6),
            round(float(out["bandgap_edge_hertz"]), 6), int(out["argmax_index"]), int(out["argmax_branch"]),
            tuple(round(float(v), 6) for v in out["band_tops"]), tuple(int(n) for n in out["frequencies"].shape),
            # the whole table, not only its column maxima: replacing every row by the branch
            # maxima leaves band_tops and the edge unchanged
            round(float(out["frequencies"].sum()), 5),
            round(float(out["frequencies"].min()), 6),
            round(float((out["frequencies"] ** 2).sum()), 3),
            round(float(out["tau"]), 10), round(float(out["position_in_interval"]), 6))
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        return tuple(v for e in x for v in flat(e))
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    return (int(x),) if isinstance(x, bool) else (x,)
""",
            "call": "flat(digest(report_finite_contrast_correction(0.02, 10, (4, 3, 2), 1.5e6, 5.0e5, 1200.0, 1.0e-2, 5.0e-3, SMALL, 1e-12)))",
            "gold_call": "flat(digest(_oracle_report_finite_contrast_correction(0.02, 10, (4, 3, 2), 1.5e6, 5.0e5, 1200.0, 1.0e-2, 5.0e-3, SMALL, 1e-12)))",
        },
        {
            "setup": """import numpy as np
PI = np.pi
ONE = ((0.0, 0.0, PI),)
def digest(out):
    return (round(float(out["correction_hertz"]), 5), tuple(round(float(v), 5) for v in out["full_pencil_hertz"]),
            round(float(out["bandwidth_sum_hertz"]), 6),
            round(float(out["bandgap_edge_hertz"]), 6), round(float(out["bandgap_edge_angular"]), 4),
            int(out["argmax_index"]), int(out["argmax_branch"]),
            round(float(out["frequencies"][0, 0]), 6), round(float(out["wavelength_ratio"]), 6),
            round(float(out["ball_hertz_min"]), 6), round(float(out["ball_hertz_max"]), 6))
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        return tuple(v for e in x for v in flat(e))
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    return (int(x),) if isinstance(x, bool) else (x,)
""",
            "call": "flat(digest(report_finite_contrast_correction(0.02, 8, (4, 3, 2), 1.5e6, 5.0e5, 1200.0, 1.0e-2, 5.0e-3, ONE, 1e-12)))",
            "gold_call": "flat(digest(_oracle_report_finite_contrast_correction(0.02, 8, (4, 3, 2), 1.5e6, 5.0e5, 1200.0, 1.0e-2, 5.0e-3, ONE, 1e-12)))",
        },
        {
            "setup": """import numpy as np
PI = np.pi
OK = ((PI, 0.0, 0.0),)
def verdict(fn, al=OK, n=8, spans=(4, 3, 2), eps=5e-3, L=0.02):
    try:
        fn(L, n, spans, 1.5e6, 5.0e5, 1200.0, 1.0e-2, eps, al, 1e-12)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        return tuple(v for e in x for v in flat(e))
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    return (int(x),) if isinstance(x, bool) else (x,)
def verdicts(fn):
    return flat((verdict(fn, al=()), verdict(fn, al=((0.0, 0.0, 0.0),)), verdict(fn, n=3), verdict(fn, spans=(4, 3)), verdict(fn, eps=1.0), verdict(fn, L=0.0), verdict(fn)))
""",
            'call': 'verdicts(report_finite_contrast_correction)',
            'gold_call': 'verdicts(_oracle_report_finite_contrast_correction)',
        },
    ]
