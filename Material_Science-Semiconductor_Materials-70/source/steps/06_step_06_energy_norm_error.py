"""
The quality of a computed field is measured against the exact field in the energy norm the problem itself supplies,

$$err = integral of (Eth - Enum) . c . (Eth - Enum) / integral of Eth . c . Eth,$$

with Eth the exact field of the real microstructure and Enum the computed one. Written that way the measure appears to need the exact field, which for this microstructure is an involved complex-variable construction. It does not. Expand the numerator,

$$integral of (Eth - Enum) . c . (Eth - Enum) = integral of Eth . c . Eth - 2 * integral of Eth . c . Enum + integral of Enum . c . Enum,$$

and treat the two ingredients separately. The exact current Jth = c * Eth is divergence free and periodic. The computed field is, by construction, a periodic gradient added to the prescribed mean: every retained non-zero mode of it lies along the direction the Green operator projects onto, and its mean mode is the loading. The Hill-Mandel lemma applies to that pairing and gives

$$average of Jth . Enum = average of Jth . average of Enum = cth * |Ebar|^2,$$

where cth is the exact effective conductivity, because the average of the computed field is the applied mean field exactly. The same lemma applied to the exact field against itself gives the first term as cth * |Ebar|^2 as well. Two of the three terms therefore collapse onto the same constant and

$$err = average of c * |Enum|^2 / (cth * |Ebar|^2) - 1.$$

Nothing about the exact field survives except the single number cth. Two things follow. The measure is computable from the numerical field alone once cth is known, and it is non-negative for every admissible field, because the exact solution is the one that minimises the energy over all periodic gradients with the prescribed mean; it vanishes only when the computed field is the exact one. A negative value is therefore not a small numerical accident but a sign that something in the chain is wrong.

The conductivity appearing in that average is the conductivity of the real material. Its interface sits exactly at half the cell edge and its area fraction is exactly one quarter, whatever the grid did to it. Using the sampled conductivity instead turns the average into the discrete energy of the fixed point, which equals the computed macroscopic conductivity identically, and the measure then collapses to a comparison of two effective conductivities that carries none of the information about the field. That substitution is the single most consequential error available here and it is silent: it produces a smooth, plausible curve with a different shape.

The average can be formed exactly rather than by quadrature. The computed field retains the modes with index of magnitude at most M/2 in each direction, so the square of its magnitude reaches index M in each direction, and its coefficients W are recovered without error by transforming it on any grid long enough to represent every index from minus M to M separately. An odd grid of 2M + 1 points already does that; an even grid needs 2M + 2, because an even-length transform reaches one index further in the negative direction than the positive. The conductivity is piecewise constant, c1 plus (c2 - c1) times the indicator of the quarter square, and the integral of a single exponential over that square factorises into

$$g(p) = integral from 0 to 1/2 of exp(2 * pi * i * p * u) du,$$

which is 1/2 at p = 0, zero at every other even p, and i / (pi * p) at odd p. The average is then c1 times W at the mean index plus (c2 - c1) times the sum of W times g(p) times g(q) over all indices, with no quadrature error at any stage and therefore no refinement study to perform.

The one number that must come from outside is cth. For this microstructure, a square of side half the cell edge repeated on a square lattice, it is known in closed form, and it is not any mean-field estimate: it is the exact effective conductivity of a boundary value problem solved by complex-variable methods, and it can be recognised by two properties. Exchanging the two phase conductivities multiplies it by the same construction applied to the exchanged pair to give exactly c1 times c2, and in the limit of a perfectly conducting square it approaches a finite multiple of c1 rather than diverging, because a square at quarter area fraction does not percolate.

Returns
-------
dict, the exact cell energy of the field against the real microstructure, the closed-form effective conductivity, and the energy-norm error that follows.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def energy_norm_error(
    fine_field,
    n_modes: int,
    c_matrix: float,
    c_inclusion: float,
    mean_field,
) -> dict:
    """Integrate the computed field against the true microstructure and turn the result into the energy-norm error.

    Parameters
    ----------
    fine_field : array
        Reconstructed field of shape (2, nf, nf) in volt per metre.
    n_modes : int
        Truncation order M, even and above zero.
    c_matrix : float
        Conductivity of the surrounding phase in siemens per metre, above zero.
    c_inclusion : float
        Conductivity of the embedded phase in siemens per metre, above zero.
    mean_field : sequence
        Two components of the applied mean electric field in volt per metre.

    Returns
    -------
    dict
        Under the keys cell_energy, mean_square_field, inclusion_integral, exact_macroscopic, apparent_conductivity and energy_norm_error.

    Raises
    ------
    ValueError
        When M fails to be a positive even integer, when the field is not a real array of shape (2, nf, nf) whose length cannot represent every index from minus M to M, when either conductivity fails to be finite and above zero, or when the mean field is not two finite components of which at least one is non-zero.
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


def _half_cell_weights(n_fine):
    """Return the exact cell-averaged integral of each exponential over the lower half period."""
    index = np.fft.fftfreq(n_fine, d=1.0 / n_fine).astype(int)
    weights = np.zeros(n_fine, complex)
    weights[index == 0] = 0.5
    odd = (index % 2 != 0)
    weights[odd] = 1j / (np.pi * index[odd])
    return weights


def _oracle_energy_norm_error(
    fine_field,
    n_modes: int,
    c_matrix: float,
    c_inclusion: float,
    mean_field,
) -> dict:
    """Reference implementation."""
    small_m = _even_positive(n_modes, "n_modes")
    c1 = _positive_float(c_matrix, "c_matrix")
    c2 = _positive_float(c_inclusion, "c_inclusion")

    field = np.asarray(fine_field, dtype=float)
    if field.ndim != 3 or field.shape[0] != 2 or field.shape[1] != field.shape[2]:
        raise ValueError("fine_field must have shape (2, nf, nf)")
    n_fine = field.shape[1]
    # the squared magnitude reaches index M in each direction, so every index from -M to M
    # must be separately representable by a transform of this length
    reach = np.fft.fftfreq(n_fine, d=1.0 / n_fine).astype(int)
    if int(reach.min()) > -small_m or int(reach.max()) < small_m:
        raise ValueError("fine_field must carry enough points to represent every index from "
                         "minus n_modes to n_modes")
    if not np.all(np.isfinite(field)):
        raise ValueError("fine_field must be finite")

    mean = np.asarray(mean_field, dtype=float).ravel()
    if mean.size != 2 or not np.all(np.isfinite(mean)):
        raise ValueError("mean_field must be two finite components")
    mean_norm_sq = float(mean @ mean)
    if mean_norm_sq <= 0.0:
        raise ValueError("mean_field must not vanish")

    square = field[0] ** 2 + field[1] ** 2
    # exact because the square of the magnitude is a trigonometric polynomial of order 2M
    coefficients = np.fft.fft2(square) / float(n_fine) ** 2
    weights = _half_cell_weights(n_fine)
    inclusion = float(np.real(np.sum(coefficients * weights[:, None] * weights[None, :])))
    mean_square = float(np.real(coefficients[0, 0]))

    energy = c1 * mean_square + (c2 - c1) * inclusion
    exact = c1 * math.sqrt((c1 + 3.0 * c2) / (3.0 * c1 + c2))
    apparent = energy / mean_norm_sq
    return {
        "cell_energy": energy,
        "mean_square_field": mean_square,
        "inclusion_integral": inclusion,
        "exact_macroscopic": exact,
        "apparent_conductivity": apparent,
        "energy_norm_error": apparent / exact - 1.0,
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


def test_cases():
    return [
        {
            # a uniform field: the average must be the exact rule of mixtures at area
            # fraction one quarter, which fixes the indicator weights independently of
            # any spectral content
            "setup": """
import math
import numpy as np
def uniform(nf, a, b):
    f = np.zeros((2, nf, nf))
    f[0] = a
    f[1] = b
    return f
def digest(fn, nf, M, a, b, c1, c2):
    out = fn(uniform(nf, a, b), M, c1, c2, (a, b))
    want = (0.75 * c1 + 0.25 * c2) * (a * a + b * b)
    # a uniform field carries the rule-of-mixtures energy, so the error it must report follows
    # from that and the closed-form effective conductivity alone, written out here rather than
    # read back from the call under test
    cth = c1 * math.sqrt((c1 + 3.0 * c2) / (3.0 * c1 + c2))
    want_err = (0.75 * c1 + 0.25 * c2) / cth - 1.0
    return (round(out["cell_energy"] - want, 9), round(out["mean_square_field"] - (a * a + b * b), 12),
            round(out["inclusion_integral"] - 0.25 * (a * a + b * b), 12),
            round(out["apparent_conductivity"] - (0.75 * c1 + 0.25 * c2), 9),
            round(out["energy_norm_error"] - want_err, 12), round(out["energy_norm_error"], 10))
""" + FLAT,
            "call": "flat((digest(energy_norm_error, 33, 8, 1.0, 0.0, 1.0, 100.0), digest(energy_norm_error, 41, 10, 3.0, 4.0, 2.0, 50.0), digest(energy_norm_error, 65, 16, 1.0e5, 0.0, 192.26119608, 19226.119608)))",
            "gold_call": "flat((digest(_oracle_energy_norm_error, 33, 8, 1.0, 0.0, 1.0, 100.0), digest(_oracle_energy_norm_error, 41, 10, 3.0, 4.0, 2.0, 50.0), digest(_oracle_energy_norm_error, 65, 16, 1.0e5, 0.0, 192.26119608, 19226.119608)))",
        },
        {
            # single planted modes, whose integral over the quarter square is known in
            # closed form, so the parity structure of the weights is tested directly
            "setup": """
import numpy as np
def planted(nf, p, q, amp):
    x = np.arange(nf) / nf
    f = np.zeros((2, nf, nf))
    f[0] = np.sqrt(np.maximum(amp * (1.0 + np.cos(2.0 * np.pi * (p * x[:, None] + q * x[None, :]))), 0.0))
    return f
def digest(fn, nf, M, p, q):
    out = fn(planted(nf, p, q, 1.0), M, 1.0, 5.0, (1.0, 0.0))
    return (round(out["mean_square_field"], 11), round(out["inclusion_integral"], 11),
            round(out["cell_energy"], 11), round(out["energy_norm_error"], 11),
            round(out["apparent_conductivity"], 11))
""" + FLAT,
            "call": "flat((digest(energy_norm_error, 49, 4, 1, 0), digest(energy_norm_error, 49, 4, 2, 0), digest(energy_norm_error, 49, 4, 1, 1), digest(energy_norm_error, 49, 4, 3, 2), digest(energy_norm_error, 49, 4, 0, 0)))",
            "gold_call": "flat((digest(_oracle_energy_norm_error, 49, 4, 1, 0), digest(_oracle_energy_norm_error, 49, 4, 2, 0), digest(_oracle_energy_norm_error, 49, 4, 1, 1), digest(_oracle_energy_norm_error, 49, 4, 3, 2), digest(_oracle_energy_norm_error, 49, 4, 0, 0)))",
        },
        {
            # the closed-form effective conductivity: its duality under exchange of the
            # phases, its perfect-conductor limit, its behaviour at unit contrast, and
            # the fact that it is scaled rather than shifted by a common factor
            "setup": """
import numpy as np
def cth(fn, c1, c2):
    f = np.zeros((2, 9, 9))
    f[0] = 1.0
    return fn(f, 2, c1, c2, (1.0, 0.0))["exact_macroscopic"]
def digest(fn):
    a = cth(fn, 1.0, 100.0)
    b = cth(fn, 100.0, 1.0)
    return (round(a, 10), round(a * b - 100.0, 9), round(cth(fn, 1.0, 1.0) - 1.0, 12),
            round(cth(fn, 1.0, 1.0e12) - np.sqrt(3.0), 5), round(cth(fn, 7.0, 700.0) / a, 10))
""" + FLAT,
            "call": "flat(digest(energy_norm_error))",
            "gold_call": "flat(digest(_oracle_energy_norm_error))",
        },
        {
            # the reported error, checked against a field whose integrals are elementary
            "setup": """
import math
import numpy as np
def cosine(nf, p, a, b):
    x = np.arange(nf) / nf
    f = np.zeros((2, nf, nf))
    f[0] = a + b * np.cos(2.0 * np.pi * p * x[:, None])
    return f
def expected(a, b, c1, c2):
    # cell mean a, mean square a^2 + b^2 / 2; over the quarter square a whole cosine integrates
    # to zero and its square to a quarter, giving a^2 / 4 + b^2 / 8
    mean_square = a * a + 0.5 * b * b
    inclusion = 0.25 * a * a + 0.125 * b * b
    energy = c1 * mean_square + (c2 - c1) * inclusion
    cth = c1 * math.sqrt((c1 + 3.0 * c2) / (3.0 * c1 + c2))
    return mean_square, inclusion, energy, energy / (a * a) / cth - 1.0
def digest(fn, nf, M, p, a, b, c1, c2):
    out = fn(cosine(nf, p, a, b), M, c1, c2, (a, 0.0))
    ms, inc, en, err = expected(a, b, c1, c2)
    return (round(out["mean_square_field"] - ms, 11), round(out["inclusion_integral"] - inc, 11),
            round(out["cell_energy"] - en, 8), round(out["energy_norm_error"] - err, 11),
            round(out["energy_norm_error"], 10))
def homogeneous(fn, nf, M):
    # with a single conductivity the exact field is uniform, so a uniform field is the exact
    # one and the reported error must vanish identically
    f = np.zeros((2, nf, nf))
    f[0] = 2.5
    out = fn(f, M, 7.0, 7.0, (2.5, 0.0))
    return (round(out["energy_norm_error"], 14), round(out["exact_macroscopic"] - 7.0, 12))
""" + FLAT,
            "call": "flat((digest(energy_norm_error, 41, 6, 3, 2.0, 0.5, 1.0, 100.0), digest(energy_norm_error, 41, 6, 1, 1.0, 1.0, 2.0, 7.0), digest(energy_norm_error, 61, 8, 4, 3.0, 1.5, 192.26119608, 19226.119608), homogeneous(energy_norm_error, 41, 6)))",
            "gold_call": "flat((digest(_oracle_energy_norm_error, 41, 6, 3, 2.0, 0.5, 1.0, 100.0), digest(_oracle_energy_norm_error, 41, 6, 1, 1.0, 1.0, 2.0, 7.0), digest(_oracle_energy_norm_error, 61, 8, 4, 3.0, 1.5, 192.26119608, 19226.119608), homogeneous(_oracle_energy_norm_error, 41, 6)))",
        },
        {
            "setup": """
import numpy as np
def verdict(fn, nf=41, M=8, c1=1.0, c2=100.0, E=(1.0, 0.0), shape=None):
    f = np.zeros((2, nf, nf)) if shape is None else np.zeros(shape)
    f[0] = 1.0 if shape is None else 1.0
    try:
        fn(f, M, c1, c2, E)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(energy_norm_error), verdict(energy_norm_error, M=7), verdict(energy_norm_error, M=0), verdict(energy_norm_error, nf=20), verdict(energy_norm_error, nf=16, M=8), verdict(energy_norm_error, nf=17, M=8), verdict(energy_norm_error, nf=18, M=8), verdict(energy_norm_error, c1=0.0), verdict(energy_norm_error, c2=float('inf')), verdict(energy_norm_error, E=(0.0, 0.0)), verdict(energy_norm_error, shape=(2, 41, 40))))",
            "gold_call": "flat((verdict(_oracle_energy_norm_error), verdict(_oracle_energy_norm_error, M=7), verdict(_oracle_energy_norm_error, M=0), verdict(_oracle_energy_norm_error, nf=20), verdict(_oracle_energy_norm_error, nf=16, M=8), verdict(_oracle_energy_norm_error, nf=17, M=8), verdict(_oracle_energy_norm_error, nf=18, M=8), verdict(_oracle_energy_norm_error, c1=0.0), verdict(_oracle_energy_norm_error, c2=float('inf')), verdict(_oracle_energy_norm_error, E=(0.0, 0.0)), verdict(_oracle_energy_norm_error, shape=(2, 41, 40))))",
        },
    ]
