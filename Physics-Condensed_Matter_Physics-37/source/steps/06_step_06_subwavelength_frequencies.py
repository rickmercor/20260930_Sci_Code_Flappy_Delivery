"""
Return the ordinary frequencies in hertz as well as the angular frequencies, since the frequencies a band diagram is read in are the ordinary ones, and sort ascending so that entry zero is the lowest of the six branches at this quasi-momentum.

Returns
-------
dict, holding the angular and hertz frequency sextuples, the resonator mass and the cross_check_defect.

Convert the capacity matrix into resonant frequencies. Treating the transmission problem as a perturbation of the static limit, projecting the variational form onto the six rigid motions the reduction retains and keeping the leading term gives a generalised eigenvalue problem rather than a simple ratio,

$$Q v = omega^2 M v,$$

with Q the capacity matrix and M the inertia the rigid body presents to those six motions. The six columns of the trial basis do not share one unit: the three constant fields are dimensionless and the three rotations carry metre. The eigenvalues of Q on their own are therefore not a spectrum and must not be divided by anything to make one; the frequencies come from the pencil.

What M is, is part of the physics and not a formality, and it is supplied to this stage rather than formed here: it is the inertia the uniform rigid resonator presents to the six motions, expressed in the same basis, the same column order and about the same reference point as the columns of Q. A translation answers to the mass and a rotation to a moment of inertia, so its entries are not interchangeable, and dividing a rotational entry of Q by the mass gives a frequency wrong by many orders.

The bookkeeping in the contrast is the other trap and deserves stating plainly. The stiffness contrast delta does not appear. It cancels: the perturbation parameter that multiplies the surface term is delta, but the interior inertia carries tau squared, which is delta over eps, and the ratio leaves eps alone. Carrying delta into the formula in place of eps inflates every frequency by exactly the factor tau. Equivalently the combination rho times eps to the minus one is the resonator density rho_in, so the mass is rho_in times volume_D and every moment follows from it. Both forms must agree; computing the frequencies with the background density and an explicit contrast factor and again with the resonator density directly, then comparing, is the cheapest available check that the contrasts have been handled correctly.

Returns
-------
dict, holding the angular and hertz frequency sextuples, the resonator mass and the cross_check_defect.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def subwavelength_frequencies(
    capacity: np.ndarray,
    volume_D: float,
    inertia: np.ndarray,
    rho: float,
    eps: float,
) -> dict:
    """Turn the six by six capacity matrix into the six subwavelength resonant frequencies.

    Parameters
    ----------
    capacity : np.ndarray
        The (6, 6) Hermitian positive definite capacity matrix from the condensation.
    volume_D : float
        Resonator volume in cubic metre, above zero.
    inertia : np.ndarray
        The (6, 6) inertia matrix of the uniform rigid resonator, expressed in the same six-motion
        basis as capacity: the same column order, the same reference point for the rotations. Its
        entries carry the units the basis requires.
    rho : float
        Background density in kilogram per cubic metre, above zero.
    eps : float
        Reciprocal density contrast, above zero and below one.

    Returns
    -------
    dict
        Under the keys angular, hertz, mass and cross_check_defect.
        The generalised problem is capacity v = omega squared times inertia v. Solve that
        generalised problem; the eigenvalues of capacity on their own are not the physical
        spectrum, because the translation columns are dimensionless and the rotation
        columns carry metre.
        mass is the resonator mass rho divided by eps, times volume_D.
        angular and hertz hold the six frequencies ascending, in radian per second and
        in hertz. cross_check_defect is the largest absolute difference in hertz
        between the frequencies formed with the background density and an explicit
        contrast factor and those formed with the resonator density directly.

    Raises
    ------
    ValueError
        When capacity is not a (6, 6) Hermitian positive definite array, when volume_D or rho is not finite and above zero, when inertia is not a (6, 6) finite Hermitian positive definite array, or when eps falls outside the open interval from zero to one.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _pencil(Qh, Mmat):
    """Eigenvalues of the pencil Qh v = w Mmat v through the Cholesky factor of Mmat, ascending."""
    L = np.linalg.cholesky(Mmat)
    Li = np.linalg.inv(L)
    return np.sort(np.linalg.eigvalsh(Li @ Qh @ Li.conj().T))


def _oracle_subwavelength_frequencies(
    capacity: np.ndarray,
    volume_D: float,
    inertia: np.ndarray,
    rho: float,
    eps: float,
) -> dict:
    """Reference implementation."""
    Q = np.asarray(capacity)
    if Q.ndim != 2 or Q.shape != (6, 6):
        raise ValueError("capacity must be a six by six array")
    if not np.all(np.isfinite(Q)):
        raise ValueError("capacity must be finite")
    dgq = np.sqrt(np.abs(np.diag(Q)))
    dgq = np.where(dgq > 0.0, dgq, 1.0)
    if float((np.abs(Q - Q.conj().T) / np.outer(dgq, dgq)).max()) > 1.0e-6:
        raise ValueError("capacity must be Hermitian to working accuracy")
    Mi = np.asarray(inertia)
    if Mi.ndim != 2 or Mi.shape != (6, 6) or not np.all(np.isfinite(Mi)):
        raise ValueError("inertia must be a finite six by six array")
    if np.abs(Mi - Mi.conj().T).max() > 1.0e-9 * max(float(np.abs(Mi).max()), 1e-300):
        raise ValueError("inertia must be Hermitian")
    Mh = 0.5 * (Mi + Mi.conj().T)
    if not np.all(np.linalg.eigvalsh(Mh) > 0.0):
        raise ValueError("inertia must be positive definite")
    vol = float(volume_D)
    if not np.isfinite(vol) or vol <= 0.0:
        raise ValueError("volume_D must be finite and above zero")
    rho = float(rho)
    if not np.isfinite(rho) or rho <= 0.0:
        raise ValueError("rho must be finite and above zero")
    eps = float(eps)
    if not np.isfinite(eps) or not 0.0 < eps < 1.0:
        raise ValueError("eps must lie strictly between zero and one")

    mass = (rho / eps) * vol
    Qh = 0.5 * (Q + Q.conj().T)
    w = _pencil(Qh, Mh)
    if not np.all(w > 0.0):
        raise ValueError("the reduced pencil must be positive definite")
    wb = _pencil(Qh, Mh * eps) * eps
    first = np.sqrt(wb)
    second = np.sqrt(w)
    defect = float(np.abs(first - second).max() / (2.0 * np.pi))
    return {
        "angular": second,
        "hertz": second / (2.0 * np.pi),
        "mass": mass,
        "cross_check_defect": defect,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": r"""import copy as _input_copy
def _invoke_with_fresh_inputs(fn, *args, **kwargs):
    # deep-copied arguments keep candidate, reference and repeated calls independent
    args, kwargs = _input_copy.deepcopy((args, kwargs))
    return fn(*args, **kwargs)

import numpy as np
def digest(out):
    return (tuple(round(float(v), 8) for v in out["hertz"]),
            tuple(round(float(v), 6) for v in out["angular"]),
            round(float(out["mass"]), 12),
            # a flag, not the value: the two algebraic forms agree to the roundoff floor
            int(float(out["cross_check_defect"]) <= 1e-8 * float(max(out["hertz"]))))
# the condensed capacity at (0, 0, pi) on the graded mesh, translations then rotations
BETA = np.array([
    [3.95315689e+04, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 4.11755178e+04, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 9.92448840e+04, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.41605900e+00, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.88937400e+00, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 2.17464400e+00]])
# the inertia of the graded box about its centroid, in the same basis as BETA
M6 = np.diag([0.09, 0.09, 0.09, 6.09375e-07, 9.375e-07, 1.171875e-06])
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        return tuple(v for e in x for v in flat(e))
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    return (int(x),) if isinstance(x, bool) else (x,)
""",
            'call': 'flat(digest(_invoke_with_fresh_inputs(subwavelength_frequencies, BETA, 3.75e-07, M6, 1200.0, 0.005)))',
            'gold_call': 'flat(digest(_invoke_with_fresh_inputs(_oracle_subwavelength_frequencies, BETA, 3.75e-07, M6, 1200.0, 0.005)))',
        },
        {
            "setup": r"""import copy as _input_copy
def _invoke_with_fresh_inputs(fn, *args, **kwargs):
    # deep-copied arguments keep candidate, reference and repeated calls independent
    args, kwargs = _input_copy.deepcopy((args, kwargs))
    return fn(*args, **kwargs)

import numpy as np
def digest(out):
    return (tuple(round(float(v), 8) for v in out["hertz"]), round(float(out["mass"]), 12))
# halving eps doubles the resonator density, so the inertia supplied doubles with it, which must
# lower every frequency by exactly the square root of two; the off-diagonal entries of the
# capacity couple translation to rotation, and the inertia here carries a coupling block too
# the coupling entries sit well inside the Cauchy-Schwarz bound set by the two diagonal
# scales, so the matrix stays positive definite while translation and rotation genuinely mix
A = np.array([
    [9.0, 0.0, 0.0, 0.0, 1.039230484541e-03, 0.0],
    [0.0, 1.0, 0.0, 3.535533905933e-04, 0.0, 0.0],
    [0.0, 0.0, 4.0, 0.0, 0.0, 6.0e-04],
    [0.0, 3.535533905933e-04, 0.0, 2.0e-06, 0.0, 0.0],
    [1.039230484541e-03, 0.0, 0.0, 0.0, 3.0e-06, 0.0],
    [0.0, 0.0, 6.0e-04, 0.0, 0.0, 4.0e-06]])
M1 = np.array([[0.2, 0.0, 0.0, 0.0, 2.0e-04, 0.0], [0.0, 0.2, 0.0, -2.0e-04, 0.0, 0.0], [0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
               [0.0, -2.0e-04, 0.0, 4.0e-07, 0.0, 0.0], [2.0e-04, 0.0, 0.0, 0.0, 6.0e-07, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 8.0e-07]])
M2 = 2.0 * M1
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        return tuple(v for e in x for v in flat(e))
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    return (int(x),) if isinstance(x, bool) else (x,)
""",
            'call': 'flat((digest(_invoke_with_fresh_inputs(subwavelength_frequencies, A, 2e-06, M1, 1000.0, 0.01)), digest(_invoke_with_fresh_inputs(subwavelength_frequencies, A, 2e-06, M2, 1000.0, 0.005))))',
            'gold_call': 'flat((digest(_invoke_with_fresh_inputs(_oracle_subwavelength_frequencies, A, 2e-06, M1, 1000.0, 0.01)), digest(_invoke_with_fresh_inputs(_oracle_subwavelength_frequencies, A, 2e-06, M2, 1000.0, 0.005))))',
        },
        {
            "setup": r"""import copy as _input_copy
def _invoke_with_fresh_inputs(fn, *args, **kwargs):
    # deep-copied arguments keep candidate, reference and repeated calls independent
    args, kwargs = _input_copy.deepcopy((args, kwargs))
    return fn(*args, **kwargs)

import numpy as np
GOOD = np.diag([3.0, 4.0, 5.0, 1.0e-06, 2.0e-06, 3.0e-06])
GM = np.diag([0.1, 0.1, 0.1, 2.0e-07, 3.0e-07, 4.0e-07])
SKEWM = GM.copy(); SKEWM[0, 4] = 1.0e-03
def verdict(fn, w=GOOD, v=1e-6, mi=GM, r=1200.0, e=5e-3):
    try:
        _invoke_with_fresh_inputs(fn, w, v, mi, r, e)
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
    return flat((verdict(fn, w=np.eye(4)), verdict(fn, w=-GOOD), verdict(fn, w=GOOD * np.nan), verdict(fn, v=0.0), verdict(fn, mi=np.eye(4)), verdict(fn, mi=-GM), verdict(fn, mi=SKEWM), verdict(fn, r=-1.0), verdict(fn, e=1.0), verdict(fn, e=0.0), verdict(fn)))
""",
            'call': 'verdicts(subwavelength_frequencies)',
            'gold_call': 'verdicts(_oracle_subwavelength_frequencies)',
        },
    ]
