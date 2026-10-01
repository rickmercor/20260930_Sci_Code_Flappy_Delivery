"""
Given the two matrices of the pencil and a starting guess for the frequency, run two-sided shifted inverse iteration until the correction falls below the requested tolerance or the iteration budget is exhausted, and return the frequency, the right and left eigenvectors in the normalisation described above, the size of the last correction and the number of iterations taken. The pencil is written with the eigenvalue equal to the product of the imaginary unit and the frequency, so the frequency is minus the imaginary unit times the eigenvalue. The iteration starts from the deterministic vector whose entries are the reciprocals of one plus the node index, on both sides.

The pencil of step 3 has a singular second operator, so it cannot be reduced to a standard eigenvalue problem, and a dense generalised eigensolver applied to it returns a spectrum in which the physical modes are a small minority. Most of what comes back are artefacts of the discretisation: the compactified problem has an irregular singular point at infinity, where one of the two solutions of the underlying equation varies like exp(-2 lambda / sigma) and is therefore invisible to any polynomial basis, and the branch cut of the asymptotically flat background seeds further spurious eigenvalues along the imaginary axis. Those artefacts move when the resolution changes while genuine modes do not, which is the standard way of telling them apart, but it means that reading a mode off a dense solve requires filtering rather than sorting.

A targeted solver avoids the problem, seeded by the sweep of step 4. Shifted inverse iteration applied to the shifted pencil converges to the eigenvector nearest the shift, and because the shifted matrix is close to singular at a genuine eigenvalue a single solve already produces an accurate vector. Iterating on both sides at once, the right vector from the shifted matrix and the left vector from its conjugate transpose, gives the ingredients for a two-sided Rayleigh quotient update of the eigenvalue: projecting the residual of the shifted pencil on the left vector and dividing by the same projection of the second operator applied to the right vector yields a correction that is accurate to third order, so a handful of iterations exhausts double precision from a shift good to one or two digits. The left vector is needed in any case for the condition number, so computing it costs nothing extra.

Two practical points. The shifted matrix becomes exactly singular once the eigenvalue is converged to working precision, and the factorisation then fails outright; the iteration must therefore keep its last successful pair of vectors rather than overwrite them with the output of a failed solve. And the eigenvectors are determined only up to a complex scale, so a definite convention is required before they can be compared or reported: unit Euclidean length, with the phase fixed by making the entry of largest modulus real and positive. The condition number itself is invariant under that scale, but nothing else is.

Returns
-------
dict holding the float frequency_real and the float frequency_imag, the real and imaginary parts of the converged frequency; the complex arrays right_vector and left_vector in unit Euclidean norm with the entry of largest modulus made real and positive; the float correction, the modulus of the last accepted update to the frequency; and the int iterations, the number of accepted updates.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def regular_mode(
    operator_a: np.ndarray,
    operator_b: np.ndarray,
    frequency_guess: complex,
    tolerance: float,
    max_iterations: int,
) -> dict:
    """Locate one regular mode of the pencil by two-sided shifted inverse iteration.

    Parameters
    ----------
    operator_a : np.ndarray
        Second-order member of the pencil.
    operator_b : np.ndarray
        First-order member of the pencil.
    frequency_guess : complex
        Starting guess for the frequency, as a complex number. A guess on the imaginary axis at
        height y is the complex number with zero real part and imaginary part y, not the real
        number y.
    tolerance : float
        Relative tolerance on the frequency correction.
    max_iterations : int
        Iteration budget.

    Returns
    -------
    dict
        Under the keys frequency_real, frequency_imag, right_vector, left_vector, correction and
        iterations.

    Raises
    ------
    ValueError
        When the two matrices are not square, finite and of the same shape with side at least five,
        when the guess is not finite, when the tolerance is not finite and strictly positive, or when
        the iteration budget is not a positive integer.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import warnings

import numpy as np
import scipy.linalg as sla


def _validate_pencil(operator_a, operator_b, frequency_guess, tolerance, max_iterations):
    a = np.asarray(operator_a, dtype=complex)
    b = np.asarray(operator_b, dtype=complex)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 5:
        raise ValueError("operator_a must be a square array of side at least five")
    if b.shape != a.shape:
        raise ValueError("operator_b must have the same shape as operator_a")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("the pencil matrices must be finite")
    guess = complex(frequency_guess)
    if not np.isfinite(guess.real) or not np.isfinite(guess.imag):
        raise ValueError("frequency_guess must be finite")
    tol = float(tolerance)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tolerance must be finite and strictly positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, (int, np.integer)):
        raise ValueError("max_iterations must be an integer")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be at least one")
    return a, b, guess, tol, int(max_iterations)


def _fix_phase(vector):
    """Unit Euclidean norm with the entry of largest modulus made real and positive."""
    scaled = vector / np.linalg.norm(vector)
    lead = scaled[int(np.argmax(np.abs(scaled)))]
    return scaled * (np.conj(lead) / abs(lead))


def _oracle_regular_mode(
    operator_a: np.ndarray,
    operator_b: np.ndarray,
    frequency_guess: complex,
    tolerance: float,
    max_iterations: int,
) -> dict:
    """Reference implementation."""
    a, b, guess, tol, budget = _validate_pencil(
        operator_a, operator_b, frequency_guess, tolerance, max_iterations
    )
    size = a.shape[0]
    start = (1.0 / (1.0 + np.arange(size))).astype(complex)
    right = start / np.linalg.norm(start)
    left = right.copy()
    eigenvalue = 1j * guess
    correction = float("inf")
    taken = 0
    for _ in range(budget):
        shifted = a - eigenvalue * b
        try:
            with warnings.catch_warnings():
                # once the eigenvalue is converged the shifted pencil is singular to working
                # precision, which is the stopping signal rather than a fault
                warnings.simplefilter("ignore", sla.LinAlgWarning)
                factors = sla.lu_factor(shifted)
                trial_right = sla.lu_solve(factors, right)
                trial_left = sla.lu_solve(factors, left, trans=2)
        except Exception:
            break
        if not (np.all(np.isfinite(trial_right)) and np.all(np.isfinite(trial_left))):
            break
        trial_right = trial_right / np.linalg.norm(trial_right)
        trial_left = trial_left / np.linalg.norm(trial_left)
        pairing = np.conj(trial_left) @ (b @ trial_right)
        if pairing == 0.0 or not np.isfinite(pairing):
            break
        step = (np.conj(trial_left) @ (shifted @ trial_right)) / pairing
        if not np.isfinite(step):
            break
        right, left = trial_right, trial_left
        eigenvalue = eigenvalue + step
        correction = float(abs(step))
        taken += 1
        if correction < tol * max(abs(eigenvalue), 1.0):
            break
    frequency = -1j * eigenvalue
    return {
        "frequency_real": float(frequency.real),
        "frequency_imag": float(frequency.imag),
        "right_vector": _fix_phase(right),
        "left_vector": _fix_phase(left),
        "correction": correction,
        "iterations": taken,
    }

# =============================================================================
# TEST CASES
# =============================================================================

FLAT = """
def flat(x):
    if isinstance(x, dict):
        return flat([x[k] for k in sorted(x)])
    if isinstance(x, (tuple, list)):
        out = []
        for v in x:
            out.extend(flat(v))
        return tuple(out)
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    if isinstance(x, complex):
        return (round(x.real, 9), round(x.imag, 9))
    if isinstance(x, bool):
        return (int(x),)
    return (x,)
"""

SETUP = """
import numpy as np
def tangherlini_pencil(n, d, ell, s):
    # Chebyshev-Lobatto collocation of the total transmission pencil, written out in closed form
    # so that no other step's function is needed to build this case's input
    k = np.arange(n + 1)
    x = np.cos(np.pi * k / n)
    c = np.where((k == 0) | (k == n), 2.0, 1.0) * (-1.0) ** k
    diff = x[:, None] - x[None, :]
    base = np.outer(c, 1.0 / c) / (diff + np.eye(n + 1))
    base = base - np.diag(base.sum(axis=1))
    nodes = (1.0 - x) / 2.0
    dmat = -2.0 * base
    m = d - 3
    p = nodes ** 2 - nodes ** (m + 2)
    dp = 2.0 * nodes - (m + 2) * nodes ** (m + 1)
    q = ell * (ell + d - 3) + (d - 2) * (d - 4) / 4.0 + (1.0 - s ** 2) * (d - 2) ** 2 * nodes ** m / 4.0
    a = p[:, None] * (dmat @ dmat) + dp[:, None] * dmat - np.diag(q)
    return {"operator_a": a, "operator_b": 2.0 * dmat, "nodes": nodes}
P14 = tangherlini_pencil(60, 14, 2, 2.0)
P4 = tangherlini_pencil(8, 4, 2, 2.0)
def digest(out):
    return (round(out["frequency_real"], 10), round(out["frequency_imag"], 10),
            np.round(np.abs(out["right_vector"]), 9), np.round(np.abs(out["left_vector"]), 9))
"""


def test_cases():
    return [
        {
            # the graded sector: the single mode on the positive imaginary frequency axis
            "setup": SETUP + FLAT,
            "call": "flat(digest(regular_mode(P14['operator_a'], P14['operator_b'], 1.1j, 1e-13, 200)))",
            "gold_call": "flat(digest(_oracle_regular_mode(P14['operator_a'], P14['operator_b'], 1.1j, 1e-13, 200)))",
        },
        {
            # the four-dimensional gravitational vector sector, whose only total transmission mode sits
            # at exactly four times the imaginary unit with the polynomial eigenfunction 1 + 3 sigma / 4;
            # the solver must reach both from a two-digit shift
            "setup": SETUP + """
def exact_case(fn):
    out = fn(P4["operator_a"], P4["operator_b"], 3.9j, 1e-13, 200)
    frequency = complex(out["frequency_real"], out["frequency_imag"])
    target = 1.0 + 0.75 * P4["nodes"]
    target = target / np.linalg.norm(target)
    aligned = np.max(np.abs(np.abs(out["right_vector"]) - target))
    return (int(abs(frequency - 4.0j) < 1e-5), int(aligned < 1e-5), int(out["iterations"] >= 1))
""" + FLAT,
            "call": "flat(exact_case(regular_mode))",
            "gold_call": "flat(exact_case(_oracle_regular_mode))",
        },
        {
            # the converged mode is purely imaginary, annihilates the shifted pencil, is returned in the
            # stated normalisation, and does not move when the resolution is raised
            "setup": SETUP + """
def consistency(fn):
    out = fn(P14["operator_a"], P14["operator_b"], 1.1j, 1e-13, 200)
    frequency = complex(out["frequency_real"], out["frequency_imag"])
    A, B = P14["operator_a"], P14["operator_b"]
    x, z = out["right_vector"], out["left_vector"]
    shifted = A - (1j * frequency) * B
    finer = tangherlini_pencil(90, 14, 2, 2.0)
    other = fn(finer["operator_a"], finer["operator_b"], 1.1j, 1e-13, 200)
    lead = x[int(np.argmax(np.abs(x)))]
    return (int(abs(out["frequency_real"]) < 1e-12),
            int(np.max(np.abs(shifted @ x)) < 1e-9),
            int(np.max(np.abs(np.conj(z) @ shifted)) < 1e-9),
            int(abs(np.linalg.norm(x) - 1.0) < 1e-12 and abs(lead.imag) < 1e-14 and lead.real > 0.0),
            int(abs(out["frequency_imag"] - other["frequency_imag"]) < 1e-11))
""" + FLAT,
            "call": "flat(consistency(regular_mode))",
            "gold_call": "flat(consistency(_oracle_regular_mode))",
        },
        {
            # edge: an iteration budget of one, which must still return a normalised pair of vectors
            # and a finite frequency rather than failing
            "setup": SETUP + FLAT,
            "call": "flat(digest(regular_mode(P14['operator_a'], P14['operator_b'], 1.1j, 1e-13, 1)))",
            "gold_call": "flat(digest(_oracle_regular_mode(P14['operator_a'], P14['operator_b'], 1.1j, 1e-13, 1)))",
        },
        {
            "setup": SETUP + """
def verdict(fn, **kw):
    args = dict(operator_a=P14["operator_a"], operator_b=P14["operator_b"],
                frequency_guess=1.1j, tolerance=1e-13, max_iterations=200)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(regular_mode, operator_a=np.ones((3, 3))), "
                    "verdict(regular_mode, operator_b=np.ones((4, 5))), "
                    "verdict(regular_mode, frequency_guess=complex('nan')), "
                    "verdict(regular_mode, tolerance=0.0), "
                    "verdict(regular_mode, max_iterations=0)))",
            "gold_call": "flat((verdict(_oracle_regular_mode, operator_a=np.ones((3, 3))), "
                         "verdict(_oracle_regular_mode, operator_b=np.ones((4, 5))), "
                         "verdict(_oracle_regular_mode, frequency_guess=complex('nan')), "
                         "verdict(_oracle_regular_mode, tolerance=0.0), "
                         "verdict(_oracle_regular_mode, max_iterations=0)))",
        },
    ]
