"""
Recover one intentionally omitted EFT coefficient from finite-spectrum consistency.

The input EFT array must contain exactly one NaN marking the omitted coefficient.
Treat that coefficient as an unknown x, construct the transformed coefficients
Q(s)=A'(s)/A(s), and impose singularity of the largest source-prescribed leading
Hankel probe. The real algebraic candidates are then filtered by the full source
spectral reconstruction: an admissible completion must produce a nonempty
rank-deficient finite spectrum whose generalized spectral values are real within
imag_tol and whose physical locations are distinct.

Parameters
----------
b_incomplete : np.ndarray
    One-dimensional real EFT coefficient array containing exactly one NaN. The
    constant coefficient b_0 must be present and nonzero.
rank_rtol : float
    Positive finite relative singular-value threshold used in the rank test.
imag_tol : float
    Nonnegative finite tolerance for generalized-eigenvalue imaginary parts.

Returns
-------
missing_value : float
    The unique admissible real value of the omitted EFT coefficient.

Raises
------
ValueError
    If the input is malformed, the omitted coefficient is b_0, the determinant
    condition is degenerate, or the stated admissibility conditions do not leave
    exactly one real completion.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def recover_missing_eft_coefficient(
    b_incomplete: "np.ndarray", rank_rtol: float, imag_tol: float
) -> float:
    """Recover the unique missing EFT coefficient consistent with a real finite spectrum."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import numpy as np
from scipy.optimize import root_scalar


def _oracle_recover_missing_eft_coefficient(
    b_incomplete: "np.ndarray", rank_rtol: float, imag_tol: float
) -> float:
    """Recover the unique missing EFT coefficient consistent with a real finite spectrum."""
    raw = np.asarray(b_incomplete)
    if np.iscomplexobj(raw) and np.any(np.imag(raw[np.isfinite(raw)]) != 0.0):
        raise ValueError("b_incomplete must be real-valued")
    try:
        coeff = np.asarray(raw, dtype=float)
        rank_tol = float(rank_rtol)
        eig_tol = float(imag_tol)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("inputs must be real numeric values") from exc

    if coeff.ndim != 1 or coeff.size < 4:
        raise ValueError("b_incomplete must be a one-dimensional array of length at least four")
    missing = np.flatnonzero(np.isnan(coeff))
    if missing.size != 1:
        raise ValueError("b_incomplete must contain exactly one NaN marking the omitted coefficient")
    if np.any(np.isinf(coeff)):
        raise ValueError("b_incomplete must not contain infinite values")
    missing_index = int(missing[0])
    if missing_index == 0 or not np.isfinite(coeff[0]) or coeff[0] == 0.0:
        raise ValueError("b[0] must be known, finite, and nonzero")
    if not np.isfinite(rank_tol) or rank_tol <= 0.0:
        raise ValueError("rank_rtol must be strictly positive and finite")
    if not np.isfinite(eig_tol) or eig_tol < 0.0:
        raise ValueError("imag_tol must be nonnegative and finite")

    # Polynomial helpers use ascending powers of the unknown coefficient x.
    def poly_add(a, b):
        n = max(len(a), len(b))
        out = np.zeros(n, dtype=float)
        out[: len(a)] += a
        out[: len(b)] += b
        while out.size > 1 and abs(out[-1]) <= 1.0e-14 * max(1.0, float(np.max(np.abs(out)))):
            out = out[:-1]
        return out

    def poly_sub(a, b):
        return poly_add(a, -np.asarray(b, dtype=float))

    def poly_mul(a, b):
        out = np.polynomial.polynomial.polymul(a, b)
        while out.size > 1 and abs(out[-1]) <= 1.0e-14 * max(1.0, float(np.max(np.abs(out)))):
            out = out[:-1]
        return out

    b_poly = []
    for j, value in enumerate(coeff):
        if j == missing_index:
            b_poly.append(np.array([0.0, 1.0], dtype=float))
        else:
            b_poly.append(np.array([float(value)], dtype=float))

    # Construct c_k(x) directly from A Q = A'. Division is only by known b_0.
    c_poly = []
    for n in range(coeff.size - 1):
        rhs = (n + 1) * b_poly[n + 1]
        for k in range(n):
            rhs = poly_sub(rhs, poly_mul(c_poly[k], b_poly[n - k]))
        c_poly.append(rhs / coeff[0])

    # Configuration depends on the available transformed sequence length, not on x.
    dummy_c = np.zeros(coeff.size - 1, dtype=float)
    r, probe_size = _oracle_choose_source_configuration(dummy_c)
    required = r + 2 * (probe_size - 1)
    if required >= len(c_poly):
        raise ValueError("insufficient transformed data for the maximal probe")

    # Determinant polynomial of C_r^(probe_size)(x), via the Leibniz formula.
    matrix = [
        [c_poly[r + i + j] for j in range(probe_size)]
        for i in range(probe_size)
    ]
    determinant = np.array([0.0], dtype=float)
    for perm in itertools.permutations(range(probe_size)):
        inversions = sum(
            perm[i] > perm[j]
            for i in range(probe_size)
            for j in range(i + 1, probe_size)
        )
        term = np.array([1.0], dtype=float)
        for i, j in enumerate(perm):
            term = poly_mul(term, matrix[i][j])
        determinant = poly_add(determinant, -term if inversions % 2 else term)

    scale = max(1.0, float(np.max(np.abs(determinant))))
    while determinant.size > 1 and abs(determinant[-1]) <= 1.0e-12 * scale:
        determinant = determinant[:-1]
    if determinant.size <= 1 or np.all(np.abs(determinant) <= 1.0e-14 * scale):
        raise ValueError("the finite-spectrum determinant condition does not isolate the missing coefficient")

    roots = np.polynomial.polynomial.polyroots(determinant)
    root_imag_tol = max(1.0e-8, 100.0 * eig_tol)
    real_seeds = sorted(
        float(z.real)
        for z in roots
        if np.isfinite(z.real) and np.isfinite(z.imag) and abs(z.imag) <= root_imag_tol
    )
    if not real_seeds:
        raise ValueError("the determinant condition has no real candidate completion")

    def determinant_at(x):
        completed = coeff.copy()
        completed[missing_index] = float(x)
        c = _oracle_compute_log_derivative_coefficients(completed)
        rr, size = _oracle_choose_source_configuration(c)
        probe = _oracle_build_hankel_probe(c, rr, size)
        return float(np.linalg.det(probe))

    refined = []
    for seed in real_seeds:
        delta = 1.0e-6 * max(1.0, abs(seed))
        try:
            sol = root_scalar(
                determinant_at,
                x0=seed,
                x1=seed + delta,
                method="secant",
                xtol=1.0e-14,
                rtol=1.0e-14,
                maxiter=100,
            )
            candidate = float(sol.root) if sol.converged else seed
        except (ValueError, RuntimeError, OverflowError, ZeroDivisionError):
            candidate = seed
        if not any(abs(candidate - old) <= 1.0e-8 * max(1.0, abs(candidate), abs(old)) for old in refined):
            refined.append(candidate)

    admissible = []
    for candidate in refined:
        completed = coeff.copy()
        completed[missing_index] = candidate
        try:
            c = _oracle_compute_log_derivative_coefficients(completed)
            rr, size = _oracle_choose_source_configuration(c)
            probe = _oracle_build_hankel_probe(c, rr, size)
            d = _oracle_infer_finite_spectrum_size(probe, rank_tol)
            if d < 1 or d >= size:
                continue
            current, shifted = _oracle_build_shifted_hankel_pencil(c, rr, d)
            spectrum = _oracle_recover_spectral_locations(current, shifted, eig_tol)
        except (ValueError, np.linalg.LinAlgError):
            continue

        locations = spectrum[:, 1]
        if locations.size > 1:
            gaps = np.diff(locations)
            loc_scale = max(1.0, float(np.max(np.abs(locations))))
            if np.any(np.abs(gaps) <= 1.0e-8 * loc_scale):
                continue
        admissible.append(candidate)

    # Merge any numerically duplicated admissible roots.
    unique = []
    for candidate in sorted(admissible):
        if not unique or abs(candidate - unique[-1]) > 1.0e-8 * max(1.0, abs(candidate), abs(unique[-1])):
            unique.append(candidate)
    if len(unique) != 1:
        raise ValueError("the stated finite-spectrum conditions do not select a unique real completion")
    return float(unique[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, scaled, and small-normalization cases."""
    return [
        {
            "setup": "import numpy as np\nb=np.array([1.255458,0.076295438118,-0.035757230423273274,-0.075823351552458039254394,-0.083749978089300131915797567674,-0.078338731191377881605205437737333754,np.nan,-0.057419982857692864178695909526227899110314926714,-0.047357424351643017636639948800385481733980392431793594,-0.038625674826665615587987604470468827548796529186628134990074,-0.031299339343808186683923366066895834773625422979666139953711748154],dtype=float)",
            "call": "round(recover_missing_eft_coefficient(b,1e-10,1e-10),12)",
            "gold_call": "round(_oracle_recover_missing_eft_coefficient(b,1e-10,1e-10),12)",
        },
        {
            "setup": "import numpy as np\nb=2*np.array([1.255458,0.076295438118,-0.035757230423273274,-0.075823351552458039254394,-0.083749978089300131915797567674,-0.078338731191377881605205437737333754,-0.068275581072225571993077829698251907397434,-0.057419982857692864178695909526227899110314926714,-0.047357424351643017636639948800385481733980392431793594,-0.038625674826665615587987604470468827548796529186628134990074,-0.031299339343808186683923366066895834773625422979666139953711748154],dtype=float); b[6]=np.nan",
            "call": "round(recover_missing_eft_coefficient(b,1e-10,1e-10),12)",
            "gold_call": "round(_oracle_recover_missing_eft_coefficient(b,1e-10,1e-10),12)",
        },
        {
            "setup": "import numpy as np\nb=0.5*np.array([1.255458,0.076295438118,-0.035757230423273274,-0.075823351552458039254394,-0.083749978089300131915797567674,-0.078338731191377881605205437737333754,-0.068275581072225571993077829698251907397434,-0.057419982857692864178695909526227899110314926714,-0.047357424351643017636639948800385481733980392431793594,-0.038625674826665615587987604470468827548796529186628134990074,-0.031299339343808186683923366066895834773625422979666139953711748154],dtype=float); b[6]=np.nan",
            "call": "round(recover_missing_eft_coefficient(b,1e-10,1e-10),12)",
            "gold_call": "round(_oracle_recover_missing_eft_coefficient(b,1e-10,1e-10),12)",
        },
    ]
