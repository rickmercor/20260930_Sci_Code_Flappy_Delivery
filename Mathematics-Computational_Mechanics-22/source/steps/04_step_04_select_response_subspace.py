"""
Certify the spectrum a retained subspace produces and what it leaves behind.

Deflating part of the contact energy has a predictable effect on the symmetric

pencil, and that prediction is what makes a selection worth doing: every

direction the retained subspace keeps collapses onto a single repeated

eigenvalue while every direction it does not keep survives, displaced by the

response it still carries. The subspace may arrive either as its orthogonal

projector or as a rectangular spanning factor whose columns need not be

orthogonal or independent. In the latter representation only the numerical

column space matters. Which directions survive is decided by that space, not by

the ordering of the supplied levels; an admissible space need only reduce the

response and need not be the dominant one. Directions lying in the numerical

null space of the response carry nothing and must not be counted among the

survivors. Take the measured spectrum from a congruence that keeps the problem

symmetric; forming the nonsymmetric product of one operator with the inverse of

the other discards that structure. Report the measured spectrum, the

theoretical prediction, the sharpest constant bounding the relative inverse

error left on the omitted part, the conditioning that results, the worst

interaction left outside the retained space, and the conditioning ceiling that

worst interaction guarantees on its own. The source also separates this ideal

metric theorem from a fixed approximate-core regime. When selection and the

Woodbury state use an SPD surrogate $\widetilde M$ while the unchanged

reference operator still contains $M$, the exact surrogate spectrum remains

available, but the true pair is governed by the spectral-equivalence and

omitted-contact bound of Appendix C.

Returns
-------
tuple containing two finite nondecreasing float arrays of shape (n,), four finite floats: measured spectrum, predicted spectrum, certificate in [0, 1), the preconditioned condition number, at least one, the worst omitted interaction, nonnegative, and the posterior condition bound, at least one; then a finite float array of shape (5,) holding c1, c2, delta, the robust Appendix-C ceiling, and the measured true-pair condition number
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def certify_preconditioned_spectrum(
    condensed_core: np.ndarray,
    interaction_rows: np.ndarray,
    response_levels: np.ndarray,
    retained_projector: np.ndarray | None,
    retained_factor: np.ndarray | None = None,
    reference_core: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, np.ndarray]:
    r"""Return measured and predicted spectra, two certificates, and two ratios.

    Write $H=M+U^\top U$ for the fully contacted symmetric operator and
    $P=M+U^\top\Pi U$ for the operator retaining only the selected part of the
    contact energy. The measured spectrum is the ascending spectrum of the
    pencil $(H,P)$. The predicted spectrum is what the deflation theorem gives
    for that pencil, ascending and of length n, from the response that $\Pi$
    actually leaves outside the retained space. The certificate is the sharpest
    constant $c$ for which the inverse action of $P$ reproduces the inverse
    action of $H$ on every right-hand side to within $c$, the discrepancy
    measured in the core energy norm and the right-hand side in its dual norm.
    The worst omitted interaction is the largest value that the squared
    Euclidean norm of the contact action $\Pi$ discards attains relative to the
    core energy of the displacement producing it, and the posterior bound is
    the ceiling on the measured conditioning that this single number
    guarantees for any admissible projector. The final value is the ratio of
    the largest measured eigenvalue to the smallest.

    Here condensed_core is the fixed SPD core $\widetilde M$ used for response
    selection and inverse application. If reference_core is supplied, it is
    the SPD core $M$ in the unchanged reference operator. Let $c_1,c_2$ be the
    extremal generalized eigenvalues of $(M,\widetilde M)$, let
    $E=U^\top(I-\Pi)U$, and let $\delta$ be the largest generalized eigenvalue
    of $(E,\widetilde M)$. The final array is

    ``[c1, c2, delta, max(c2 + delta, 1) / min(c1, 1), kappa_true]``,

    where kappa_true is the measured condition number of
    $(M+U^\top U,\widetilde M+U^\top\Pi U)$. With no reference core, take
    $M=\widetilde M$; the robust ceiling then reduces to the exact posterior
    ceiling.

    Supply exactly one retained-space representation. If retained_factor is
    None, retained_projector must be a finite symmetric idempotent matrix of
    shape (m, m). Otherwise retained_projector must be None and retained_factor
    must be a finite matrix of shape (m, k), including k=0 and k greater than
    m. Its orthogonal projector is formed from the left singular vectors whose
    singular values exceed max(m, k, 1) times machine epsilon times the largest
    singular value; an all-zero factor represents the empty space. Raises
    ValueError unless condensed_core is finite symmetric positive definite;
    reference_core, when supplied, is finite symmetric positive definite with
    shape (n, n);
    interaction_rows has shape (m, n), including (0, n) and m greater than n;
    response_levels is a finite nonnegative nonincreasing vector of shape (m,)
    reproducing the response eigenvalues to rtol 1e-9 and atol 1e-11; the
    represented subspace leaves the response invariant to rtol 1e-8 and atol
    1e-10; P is positive definite; the measured and predicted spectra agree to
    rtol 1e-8 and atol 1e-10; and the measured conditioning does not exceed the
    posterior bound by more than 1e-8; and the measured true-pair conditioning
    does not exceed its Appendix-C ceiling by more than 1e-8.

    Parameters
    ----------
    condensed_core : np.ndarray
        Symmetric positive-definite core of shape (n, n).
    interaction_rows : np.ndarray
        Ordered interaction factor of shape (m, n).
    response_levels : np.ndarray
        Nonincreasing response levels of shape (m,).
    retained_projector : np.ndarray or None
        Response-invariant orthogonal projector of shape (m, m), or None when
        retained_factor supplies the representation.
    retained_factor : np.ndarray or None
        Optional rectangular spanning factor of shape (m, k).
    reference_core : np.ndarray or None
        Optional unchanged SPD reference core M of shape (n, n).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float, float, float, float, np.ndarray]
        Measured ascending spectrum of shape (n,), predicted ascending spectrum
        of shape (n,), the omitted-response certificate, the condition number
        of the measured spectrum, the worst omitted interaction, and the
        posterior condition bound, followed by the five-entry approximate-core
        certificate described above.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_certify_preconditioned_spectrum(
    condensed_core: np.ndarray,
    interaction_rows: np.ndarray,
    response_levels: np.ndarray,
    retained_projector: np.ndarray | None,
    retained_factor: np.ndarray | None = None,
    reference_core: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, np.ndarray]:
    """Reference theorem-level spectral certification."""
    core = np.asarray(condensed_core, dtype=float)
    rows = np.asarray(interaction_rows, dtype=float)
    levels = np.asarray(response_levels, dtype=float)
    reference = (
        core if reference_core is None else np.asarray(reference_core, dtype=float)
    )
    supplied_projector = retained_projector
    supplied_factor = retained_factor
    if core.ndim != 2 or core.shape[0] != core.shape[1] or core.shape[0] == 0:
        raise ValueError("condensed_core must be a nonempty square matrix")
    n_dof = core.shape[0]
    if rows.ndim != 2 or rows.shape[1] != n_dof:
        raise ValueError("interaction_rows must have shape (m, n)")
    n_rows = rows.shape[0]
    if levels.shape != (n_rows,):
        raise ValueError("response_levels must have shape (m,)")
    if reference.shape != (n_dof, n_dof):
        raise ValueError("reference_core must have shape (n, n)")
    if (supplied_projector is None) == (supplied_factor is None):
        raise ValueError("supply exactly one retained-space representation")
    if not all(np.all(np.isfinite(x)) for x in (core, rows, levels, reference)):
        raise ValueError("all inputs must be finite")
    if not np.allclose(core, core.T, rtol=0.0, atol=1e-12):
        raise ValueError("condensed_core must be symmetric")
    try:
        core_factor = np.linalg.cholesky(core)
    except np.linalg.LinAlgError as exc:
        raise ValueError("condensed_core must be positive definite") from exc
    if not np.allclose(reference, reference.T, rtol=0.0, atol=1e-12):
        raise ValueError("reference_core must be symmetric")
    try:
        np.linalg.cholesky(reference)
    except np.linalg.LinAlgError as exc:
        raise ValueError("reference_core must be positive definite") from exc
    if np.any(levels < -1e-12) or np.any(np.diff(levels) > 1e-12):
        raise ValueError("response_levels must be nonnegative and nonincreasing")
    if supplied_factor is None:
        projector = np.asarray(supplied_projector, dtype=float)
        if projector.shape != (n_rows, n_rows):
            raise ValueError("retained_projector must have shape (m, m)")
        if not np.all(np.isfinite(projector)):
            raise ValueError("all inputs must be finite")
        if not np.allclose(projector, projector.T, rtol=1e-10, atol=1e-12):
            raise ValueError("retained_projector must be symmetric")
        if not np.allclose(projector @ projector, projector, rtol=1e-9, atol=1e-11):
            raise ValueError("retained_projector must be idempotent")
    else:
        spanning = np.asarray(supplied_factor, dtype=float)
        if spanning.ndim != 2 or spanning.shape[0] != n_rows:
            raise ValueError("retained_factor must have shape (m, k)")
        if not np.all(np.isfinite(spanning)):
            raise ValueError("all inputs must be finite")
        if n_rows == 0 or spanning.shape[1] == 0:
            retained = np.zeros((n_rows, 0), dtype=float)
        else:
            left, singular_values, _ = np.linalg.svd(spanning, full_matrices=False)
            largest = float(singular_values[0]) if singular_values.size else 0.0
            rank_tolerance = (
                max(spanning.shape[0], spanning.shape[1], 1)
                * np.finfo(float).eps
                * largest
            )
            retained = left[:, singular_values > rank_tolerance]
        projector = retained @ retained.T

    response = rows @ np.linalg.solve(core, rows.T)
    response = 0.5 * (response + response.T)
    measured_levels = np.linalg.eigvalsh(response)[::-1]
    measured_levels = np.maximum(measured_levels, 0.0)
    if not np.allclose(levels, measured_levels, rtol=1e-9, atol=1e-11):
        raise ValueError("response_levels do not match the metric response")
    if n_rows > 0 and not np.allclose(
        response @ projector, projector @ response, rtol=1e-8, atol=1e-10
    ):
        raise ValueError("retained_projector is not response-invariant")

    # The surviving response is the one carried by the orthogonal complement of
    # the retained space, which is fixed by the projector rather than by the
    # position of a level in the sorted list.
    if n_rows > 0:
        weights, directions = np.linalg.eigh(projector)
        complement = directions[:, weights < 0.5]
        if complement.shape[1] > 0:
            omitted_all = np.linalg.eigvalsh(complement.T @ response @ complement)
            omitted_all = np.maximum(omitted_all[::-1], 0.0)
        else:
            omitted_all = np.zeros(0, dtype=float)
    else:
        omitted_all = np.zeros(0, dtype=float)

    preconditioner = core + rows.T @ projector @ rows
    surrogate_reference = core + rows.T @ rows
    preconditioner = 0.5 * (preconditioner + preconditioner.T)
    surrogate_reference = 0.5 * (surrogate_reference + surrogate_reference.T)
    try:
        factor = np.linalg.cholesky(preconditioner)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "the reduced preconditioner must be positive definite"
        ) from exc
    left = np.linalg.solve(factor, surrogate_reference)
    whitened = np.linalg.solve(factor, left.T).T
    whitened = 0.5 * (whitened + whitened.T)
    measured_spectrum = np.linalg.eigvalsh(whitened)

    scale = max(1.0, float(levels[0])) if n_rows else 1.0
    positive_tolerance = max(n_dof, n_rows, 1) * np.finfo(float).eps * scale
    omitted = omitted_all[omitted_all > positive_tolerance]
    if omitted.size > n_dof:
        raise ValueError("more omitted levels than displacement dimensions")
    predicted_spectrum = np.concatenate(
        [np.ones(n_dof - omitted.size, dtype=float), 1.0 + omitted]
    )
    predicted_spectrum.sort()
    if not np.allclose(measured_spectrum, predicted_spectrum, rtol=1e-8, atol=1e-10):
        raise ValueError("measured generalized spectrum violates the prediction")
    omitted_level = float(omitted[0]) if omitted.size else 0.0
    certificate = omitted_level / (1.0 + omitted_level)
    posterior_bound = 1.0 + omitted_level
    smallest = float(measured_spectrum[0])
    if smallest <= 0.0:
        raise ValueError("the preconditioned pencil must be positive definite")
    condition_number = float(measured_spectrum[-1] / smallest)
    if condition_number > posterior_bound + 1e-8:
        raise ValueError("the measured conditioning exceeds the posterior bound")

    # Appendix C: certify the unchanged reference operator when response
    # selection and Woodbury application use a fixed approximate SPD core.
    scaled_reference = np.linalg.solve(core_factor, reference)
    scaled_reference = np.linalg.solve(core_factor, scaled_reference.T).T
    scaled_reference = 0.5 * (scaled_reference + scaled_reference.T)
    equivalence = np.linalg.eigvalsh(scaled_reference)
    c1 = float(equivalence[0])
    c2 = float(equivalence[-1])

    complement_projector = np.eye(n_rows, dtype=float) - projector
    omitted_matrix = rows.T @ (complement_projector @ rows)
    omitted_matrix = 0.5 * (omitted_matrix + omitted_matrix.T)
    scaled_omitted = np.linalg.solve(core_factor, omitted_matrix)
    scaled_omitted = np.linalg.solve(core_factor, scaled_omitted.T).T
    scaled_omitted = 0.5 * (scaled_omitted + scaled_omitted.T)
    delta = max(0.0, float(np.linalg.eigvalsh(scaled_omitted)[-1]))
    robust_bound = float(max(c2 + delta, 1.0) / min(c1, 1.0))

    true_reference = reference + rows.T @ rows
    true_reference = 0.5 * (true_reference + true_reference.T)
    robust_left = np.linalg.solve(factor, true_reference)
    robust_whitened = np.linalg.solve(factor, robust_left.T).T
    robust_whitened = 0.5 * (robust_whitened + robust_whitened.T)
    robust_spectrum = np.linalg.eigvalsh(robust_whitened)
    if robust_spectrum[0] <= 0.0:
        raise ValueError("the true preconditioned pencil must be positive definite")
    robust_condition = float(robust_spectrum[-1] / robust_spectrum[0])
    if robust_condition > robust_bound + 1e-8:
        raise ValueError("the true conditioning exceeds the Appendix-C bound")
    robust_certificate = np.array(
        [c1, c2, delta, robust_bound, robust_condition], dtype=float
    )
    return (
        measured_spectrum,
        predicted_spectrum,
        float(certificate),
        condition_number,
        float(omitted_level),
        float(posterior_bound),
        robust_certificate,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return projector and spanning-factor paths plus invalid inputs."""
    return [
        {
            "setup": """import numpy as np
M = np.array([[40.5,-1.5,.2,0,.25],[-1.5,12.666666666666666,-.633333333333333,.433333333333333,-.25],[.2,-.633333333333333,3.166666666666667,-.866666666666667,.1],[0,.433333333333333,-.866666666666667,1.866666666666667,-.1],[.25,-.25,.1,-.1,.925]])
U = np.array([[8,0,0,0,0],[0,4,0,0,0],[0,0,2,.2,0],[0,0,.1,1.5,1.2]], dtype=float)
G = U @ np.linalg.solve(M, U.T)
w, q = np.linalg.eigh((G + G.T) / 2)
order = np.argsort(w)[::-1]
levels = np.maximum(w[order], 0.0)
Pi = q[:, order[:2]] @ q[:, order[:2]].T
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in certify_preconditioned_spectrum(M, U, levels, Pi)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_certify_preconditioned_spectrum(M, U, levels, Pi)])",
        },
        {
            "setup": """import numpy as np
M = np.array([[40.5,-1.5,.2,0,.25],[-1.5,12.666666666666666,-.633333333333333,.433333333333333,-.25],[.2,-.633333333333333,3.166666666666667,-.866666666666667,.1],[0,.433333333333333,-.866666666666667,1.866666666666667,-.1],[.25,-.25,.1,-.1,.925]])
U = np.array([[8,0,0,0,0],[0,4,0,0,0],[0,0,2,.2,0],[0,0,.1,1.5,1.2]], dtype=float)
G = U @ np.linalg.solve(M, U.T)
w, q = np.linalg.eigh((G + G.T) / 2)
order = np.argsort(w)[::-1]
levels = np.maximum(w[order], 0.0)
Pi = q[:, order[2:]] @ q[:, order[2:]].T
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in certify_preconditioned_spectrum(M, U, levels, Pi)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_certify_preconditioned_spectrum(M, U, levels, Pi)])",
        },
        {
            "setup": """import numpy as np
M = np.array([[40.5,-1.5,.2,0,.25],[-1.5,12.666666666666666,-.633333333333333,.433333333333333,-.25],[.2,-.633333333333333,3.166666666666667,-.866666666666667,.1],[0,.433333333333333,-.866666666666667,1.866666666666667,-.1],[.25,-.25,.1,-.1,.925]])
U = np.array([[8,0,0,0,0],[0,4,0,0,0],[0,0,2,.2,0],[0,0,.1,1.5,1.2]], dtype=float)
G = U @ np.linalg.solve(M, U.T)
w, q = np.linalg.eigh((G + G.T) / 2)
order = np.argsort(w)[::-1]
levels = np.maximum(w[order], 0.0)
keep = order[[0, 2]]
Pi = q[:, keep] @ q[:, keep].T
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in certify_preconditioned_spectrum(M, U, levels, Pi)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_certify_preconditioned_spectrum(M, U, levels, Pi)])",
        },
        {
            "setup": """import numpy as np
M = np.array([[40.5,-1.5,.2,0,.25],[-1.5,12.666666666666666,-.633333333333333,.433333333333333,-.25],[.2,-.633333333333333,3.166666666666667,-.866666666666667,.1],[0,.433333333333333,-.866666666666667,1.866666666666667,-.1],[.25,-.25,.1,-.1,.925]])
U = np.array([[8,0,0,0,0],[0,4,0,0,0],[0,0,2,.2,0],[0,0,.1,1.5,1.2]], dtype=float)
G = U @ np.linalg.solve(M, U.T)
w, q = np.linalg.eigh((G + G.T) / 2)
order = np.argsort(w)[::-1]
levels = np.maximum(w[order], 0.0)
keep = order[[1]]
Pi = q[:, keep] @ q[:, keep].T
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in certify_preconditioned_spectrum(M, U, levels, Pi)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_certify_preconditioned_spectrum(M, U, levels, Pi)])",
        },
        {
            "setup": """import numpy as np
M = np.diag([3.0, 1.0])
U = np.zeros((0, 2))
levels = np.zeros(0)
Pi = np.zeros((0, 0))
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in certify_preconditioned_spectrum(M, U, levels, Pi)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_certify_preconditioned_spectrum(M, U, levels, Pi)])",
        },
        {
            "setup": """import numpy as np
M = np.diag([2.0, 3.0, 4.0])
U = np.diag([3.0, 2.0, 1.0])
G = U @ np.linalg.solve(M, U.T)
w = np.linalg.eigvalsh(G)
levels = w[::-1]
Pi = np.eye(3)
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in certify_preconditioned_spectrum(M, U, levels, Pi)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_certify_preconditioned_spectrum(M, U, levels, Pi)])",
        },
        {
            "setup": """import numpy as np
M = np.array([[2.0, .2], [.2, 1.0]])
U = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, .5]])
G = U @ np.linalg.solve(M, U.T)
w, q = np.linalg.eigh((G + G.T) / 2)
order = np.argsort(w)[::-1]
levels = np.maximum(w[order], 0.0)
Pi = q[:, order[:1]] @ q[:, order[:1]].T
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in certify_preconditioned_spectrum(M, U, levels, Pi)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_certify_preconditioned_spectrum(M, U, levels, Pi)])",
        },
        {
            "setup": """import numpy as np
M = np.array([[5.0, .4], [.4, 2.0]])
U = np.array([[3.0, 0.0], [0.0, 1.0], [-1.0, .5], [.25, 2.0]])
G = U @ np.linalg.solve(M, U.T)
w, q = np.linalg.eigh((G + G.T) / 2)
order = np.argsort(w)[::-1]
levels = np.maximum(w[order], 0.0)
Pi = q[:, order[:1]] @ q[:, order[:1]].T
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in certify_preconditioned_spectrum(M, U, levels, Pi)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_certify_preconditioned_spectrum(M, U, levels, Pi)])",
        },
        {
            "setup": """import numpy as np
M = np.eye(2)
U = np.diag([2.0, 1.0])
levels = np.array([4.0, 1.0])
Pi = np.array([[0.0, 0.0], [0.0, 1.0]])
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in certify_preconditioned_spectrum(M, U, levels, Pi)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_certify_preconditioned_spectrum(M, U, levels, Pi)])",
        },
        {
            "setup": """import numpy as np
M = np.eye(2)
U = np.diag([2.0, 1.0])
levels = np.array([1.0, 4.0])
Pi = np.array([[1.0, 0.0], [0.0, 0.0]])
def run_model():
    try:
        certify_preconditioned_spectrum(M, U, levels, Pi)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_certify_preconditioned_spectrum(M, U, levels, Pi)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
M = np.eye(2)
U = np.diag([2.0, 1.0])
levels = np.array([4.0, 1.0])
Pi = np.array([[1.0, 0.4], [0.4, 0.0]])
def run_model():
    try:
        certify_preconditioned_spectrum(M, U, levels, Pi)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_certify_preconditioned_spectrum(M, U, levels, Pi)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
M = np.eye(2)
U = np.diag([2.0, 1.0])
levels = np.array([4.0, 1.0])
Pi = np.array([[0.5, 0.5], [0.5, 0.5]])
def run_model():
    try:
        certify_preconditioned_spectrum(M, U, levels, Pi)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_certify_preconditioned_spectrum(M, U, levels, Pi)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
M = np.eye(3)
U = np.diag([2.0, 2.0, 1.0])
levels = np.array([4.0, 4.0, 1.0])
v = np.array([1.0, 1.0, 0.0])
Q = np.column_stack([v, -3.0 * v])
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in certify_preconditioned_spectrum(M, U, levels, None, Q)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_certify_preconditioned_spectrum(M, U, levels, None, Q)])",
        },
        {
            "setup": """import numpy as np
M = np.diag([2.0, 3.0, 5.0, 7.0])
U = np.diag([4.0, 3.0, 2.0, 1.0])
G = U @ np.linalg.solve(M, U.T)
levels = np.linalg.eigvalsh(G)[::-1]
Q = np.array([[0.0, 0.0, 0.0], [2.0, -1.0, 5.0], [0.0, 0.0, 0.0], [1.0, 3.0, -2.0]])
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in certify_preconditioned_spectrum(M, U, levels, None, Q)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_certify_preconditioned_spectrum(M, U, levels, None, Q)])",
        },
        {
            "setup": """import numpy as np
M = np.diag([3.0, 1.0])
U = np.zeros((2, 2))
levels = np.zeros(2)
Q = np.zeros((2, 0))
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in certify_preconditioned_spectrum(M, U, levels, None, Q)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_certify_preconditioned_spectrum(M, U, levels, None, Q)])",
        },
        {
            "setup": """import numpy as np
M = np.eye(2)
U = np.diag([2.0, 1.0])
levels = np.array([4.0, 1.0])
Q = np.array([[1.0, 2.0, -1.0, .5], [0.0, 0.0, 0.0, 0.0]])
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in certify_preconditioned_spectrum(M, U, levels, None, Q)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_certify_preconditioned_spectrum(M, U, levels, None, Q)])",
        },
        {
            "setup": """import numpy as np
M = np.eye(2)
U = np.diag([2.0, 1.0])
levels = np.array([4.0, 1.0])
Pi = np.eye(2)
Q = np.eye(2)[:, :1]
def run_model():
    try:
        certify_preconditioned_spectrum(M, U, levels, Pi, Q)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_certify_preconditioned_spectrum(M, U, levels, Pi, Q)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
M = np.eye(2)
U = np.diag([2.0, 1.0])
levels = np.array([4.0, 1.0])
Q = np.array([[1.0], [1.0]])
def run_model():
    try:
        certify_preconditioned_spectrum(M, U, levels, None, Q)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_certify_preconditioned_spectrum(M, U, levels, None, Q)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
Mtilde = np.array([[2.0, .3], [.3, 1.4]])
M = np.array([[2.8, .1], [.1, 1.8]])
U = np.array([[1.5, -.2], [.3, 1.1], [-.7, .4]])
G = U @ np.linalg.solve(Mtilde, U.T)
w, q = np.linalg.eigh((G + G.T) / 2)
order = np.argsort(w)[::-1]
levels = np.maximum(w[order], 0.0)
Q = q[:, order[:1]] @ np.array([[2.0, -1.0]])
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in certify_preconditioned_spectrum(Mtilde, U, levels, None, Q, M)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_certify_preconditioned_spectrum(Mtilde, U, levels, None, Q, M)])",
        },
        {
            "setup": """import numpy as np
K0 = np.diag([3.0, 1.5, .8])
Vp = np.array([[1.0, -.4, .2], [.2, .7, -1.1]])
Gp = Vp @ np.linalg.solve(K0, Vp.T)
wp, qp = np.linalg.eigh((Gp + Gp.T) / 2)
lead = qp[:, np.argmax(wp):np.argmax(wp)+1]
Rp = lead.T @ Vp
Mtilde = K0 + Rp.T @ Rp
M = K0 + Vp.T @ Vp
U = np.array([[1.2, 0.0, -.3], [0.0, .8, .5], [.4, -.2, .9]])
G = U @ np.linalg.solve(Mtilde, U.T)
w, q = np.linalg.eigh((G + G.T) / 2)
order = np.argsort(w)[::-1]
levels = np.maximum(w[order], 0.0)
Pi = q[:, order[:2]] @ q[:, order[:2]].T
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in certify_preconditioned_spectrum(Mtilde, U, levels, Pi, None, M)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_certify_preconditioned_spectrum(Mtilde, U, levels, Pi, None, M)])",
        },
        {
            "setup": """import numpy as np
Mtilde = np.eye(2)
M = np.diag([1.0, -1.0])
U = np.eye(2)
levels = np.ones(2)
Pi = np.eye(2)
def run_model():
    try:
        certify_preconditioned_spectrum(Mtilde, U, levels, Pi, None, M)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_certify_preconditioned_spectrum(Mtilde, U, levels, Pi, None, M)
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
