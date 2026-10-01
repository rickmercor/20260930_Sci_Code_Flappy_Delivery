"""
```
Apply one rational propagation increment to the centered Fourier envelope: d[0]*U plus the sum of d[j+1] times the auxiliary field for pole b[j]. Every auxiliary solve uses the same original U. Return the updated field and one factor list per pole in the current pole order. Reuse a cache only for unchanged ordered poles, grid, and material; the right-hand side may change. Pass None after changing the operator inputs or permuting coefficients.
```

The partial-fraction expansion evaluates R(X)U, where R(X)=d[0]*I+sum_j d[j+1]*(I+b[j]*X)^(-1) and X=M_tilde-I-D. Poles and their associated residues must remain paired. Simultaneously permuting b and d[1:] leaves the rational operator unchanged, while d[0] stays fixed. Cached factorizations belong to particular resolvents, so a changed pole order requires rebuilding the cache or consistently reordering its entries. The Gold implementation calls the preceding _oracle_wjsolver; the candidate calls the public wjsolver. Mathematically equivalent rational evaluation remains valid.

Returns
-------
return np.zeros_like(U, dtype=complex), []
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def apply_pade(kx, ky, mx, B, C, k0, U, b, d, lu_cache=None):
    """Apply one partial-fraction Pade step to the centered field U.

    U2 = d[0]*U + sum_j d[j+1]*(I + b[j]*X)^(-1) @ U, with
    X = M_tilde - I - diag((kx**2 + ky[column]**2) / k0**2)
    for each column. Every auxiliary solve uses the original U.
    U is square; kx and ky match its dimensions; k0 is positive.
    mx is an integer shift; B and C are real. The coefficient arrays
    b and d have shapes (P,) and (P+1,), with matched pole-residue pairs.
    Return (U2, lu_cache), with one factor list per ordered pole.
    Reuse a cache only for unchanged b, grid, and material; U may change.
    Pass None after changing operator inputs or permuting coefficients.
    """
    return np.zeros_like(U, dtype=complex), []

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_apply_pade(kx, ky, mx, B, C, k0, U, b, d, lu_cache=None):
    """Apply R(X)U with X=M_tilde-I-D and matched pole/residue pairs.

    R(X)=d[0]*I+sum_j d[j+1]*(I+b[j]*X)^(-1).
    Here M_tilde is the positive index-squared modulation matrix and
    D=diag((kx**2+ky[column]**2)/k0**2) for each ky column.
    Cache entries belong to the same ordered b, grid, and material values;
    callers must pass None if any of those values change.
    """
    spectrum = np.asarray(U, dtype=complex)
    poles = np.asarray(b, dtype=complex)
    residues = np.asarray(d, dtype=complex)

    if not np.isfinite(k0) or k0 <= 0:
        raise ValueError("k0 must be finite and positive.")
    if spectrum.ndim != 2 or spectrum.shape[0] != spectrum.shape[1]:
        raise ValueError("U must be a square 2D array.")

    size = spectrum.shape[0]
    if np.asarray(kx).shape != (size,) or np.asarray(ky).shape != (size,):
        raise ValueError("kx and ky must match the field dimensions.")
    if poles.ndim != 1 or residues.shape != (poles.size + 1,):
        raise ValueError("b and d must have shapes (P,) and (P+1,).")
    if not all(
        np.all(np.isfinite(values))
        for values in (spectrum, poles, residues)
    ):
        raise ValueError("The field and coefficients must be finite.")

    if lu_cache is None:
        lu_cache = [None] * poles.size
    if len(lu_cache) != poles.size:
        raise ValueError("lu_cache must contain one entry per pole.")

    updated = residues[0] * spectrum
    for index, pole in enumerate(poles):
        auxiliary, lu_cache[index] = _oracle_wjsolver(
            kx, ky, mx, B, C, pole, k0, spectrum, lu_cache[index]
        )
        updated += residues[index + 1] * auxiliary

    return updated, lu_cache

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np

ny = 8
kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=0.5))
ky = kx.copy()
rng = np.random.default_rng(2)
U = rng.random((ny, ny)) + 1j * rng.random((ny, ny))
k0 = 20.0
b, d = PadeCoeff(P=1, t=0.1)


def run_model():
    result, _ = apply_pade(kx, ky, 1, 0.1, 1.4, k0, U, b, d)
    return result


def run_gold():
    result, _ = _oracle_apply_pade(kx, ky, 1, 0.1, 1.4, k0, U, b, d)
    return result
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """
import numpy as np

ny = 8
kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=0.5))
ky = kx.copy()
U = np.eye(ny, dtype=complex)
k0 = 20.0
b, d = PadeCoeff(P=3, t=0.2)


def run_model():
    result, _ = apply_pade(kx, ky, ny // 2, 0.1, 1.4, k0, U, b, d)
    return result


def run_gold():
    result, _ = _oracle_apply_pade(
        kx, ky, ny // 2, 0.1, 1.4, k0, U, b, d
    )
    return result
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """
import numpy as np

ny = 8
kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=0.5))
ky = kx.copy()
rng = np.random.default_rng(3)
U = rng.random((ny, ny)) + 1j * rng.random((ny, ny))
k0 = 20.0
b, d = PadeCoeff(P=2, t=0.05)


def run_model():
    first, cache = apply_pade(kx, ky, ny, 0.1, 1.4, k0, U, b, d)
    second, _ = apply_pade(
        kx, ky, ny, 0.1, 1.4, k0, U, b, d, lu_cache=cache
    )
    return first, second


def run_gold():
    first, cache = _oracle_apply_pade(kx, ky, ny, 0.1, 1.4, k0, U, b, d)
    second, _ = _oracle_apply_pade(
        kx, ky, ny, 0.1, 1.4, k0, U, b, d, lu_cache=cache
    )
    return first, second
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """
import numpy as np

ny = 4
kx = np.array([-1.0, -0.5, 0.0, 0.5])
ky = np.array([-1.2, -0.6, 0.0, 0.6])
k0 = 2.0
mx = 1
B = 0.07
C = 1.1
first_field = np.eye(ny, dtype=complex)
second_field = (1.0 + 0.25j) * np.roll(first_field, 1, axis=0)
b, d = _oracle_PadeCoeff(P=4, t=0.2)
order = np.array([2, 0, 3, 1])
permuted_residues = np.concatenate((d[:1], d[1:][order]))
expected_first = np.array([
    [
        0.9988869681419341 - 0.04530257339048966j,
        3.1356592770703795e-05 + 0.00785630177509492j,
        -5.310942467462443e-05 - 2.1101685280562046e-05j,
        3.1356592770759306e-05 + 0.00785630177509492j,
    ],
    [
        0.000317275442821896 + 0.0092806876270724j,
        0.9999260594875123 + 0.005625074294982445j,
        -0.00012120542873698265 + 0.007092626542655976j,
        -5.7868157947504884e-05 - 4.453334220560351e-05j,
    ],
    [
        -8.459050845988081e-05 - 0.00017581747677295964j,
        -6.337019286906803e-05 + 0.007382566519237479j,
        0.9997498121643477 + 0.01999174147991046j,
        -6.33701928691513e-05 + 0.007382566519237476j,
    ],
    [
        0.0003172754428218405 + 0.0092806876270724j,
        -5.7868157947504884e-05 - 4.4533342205601775e-05j,
        -0.0001212054287368855 + 0.007092626542655978j,
        0.9999260594875122 + 0.005625074294982441j,
    ],
], dtype=complex)
expected_second = np.array([
    [
        -0.0020028964639462465 + 0.00936000648777787j,
        -4.671827628729974e-05 - 5.983269531311536e-05j,
        -0.0019153948379431185 + 0.007500336704605523j,
        1.0032268245856117 + 0.23644190077074767j,
    ],
    [
        1.0053345144831594 + 0.22723030112814568j,
        -0.0019090118226784275 + 0.007366723971020216j,
        -4.781794812119822e-05 - 3.4271220733558816e-05j,
        -0.0019327188510029902 + 0.00786414092328759j,
    ],
    [
        -0.00196584971357764 + 0.008566233092428314j,
        0.9969694974697768 + 0.2616027321262189j,
        -0.0018943620644009188 + 0.007062325185471747j,
        -4.671827628724423e-05 - 5.983269531309454e-05j,
    ],
    [
        -4.0900841029034574e-05 - 0.00019061757045585326j,
        -0.0019090118226785108 + 0.007366723971020199j,
        0.9962901615095301 + 0.26418274316373824j,
        -0.0019327188510029347 + 0.007864140923287605j,
    ],
], dtype=complex)


def run_model():
    first, cache = apply_pade(kx, ky, mx, B, C, k0, first_field, b, d)
    second, _ = apply_pade(
        kx, ky, mx, B, C, k0, second_field, b, d, lu_cache=cache
    )
    permuted, _ = apply_pade(
        kx, ky, mx, B, C, k0, first_field, b[order], permuted_residues
    )
    return first, second, permuted
""",
            "call": "run_model()",
            "gold_call": "(expected_first, expected_second, expected_first)",
        },
    ]
