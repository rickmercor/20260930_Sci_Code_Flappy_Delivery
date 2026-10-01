"""
Before any pulse is fired, the padded cell can be asked directly what it rings at. Seek displacements that oscillate at a single angular frequency and the momentum balance turns into a generalised eigenvalue problem: the discrete stiffness operator on one side, the voxel voxel_density on the other, and the squared angular frequencies as the eigenvalues. Both operators are symmetric, so those eigenvalues come out real and non-negative. The answers are the baseline of this task, the frequencies against which the pulsed reading will later be compared.

The stiffness operator is built from the link moduli of the previous stage. Its action is applied in the transformed domain: multiply the transform of the displacement by the symbol of the one-voxel forward difference to get the extension of every link, weight each link by its own modulus to get the force it carries, and multiply by the conjugate symbol to gather those forces back onto the voxels. Pairing a difference with its own adjoint in this way is what makes the operator symmetric and positive semi-definite. An odd voxel count keeps the integer frequency set symmetric about zero so that no frequency lands on Nyquist.

Padding suited to a time march will not serve here. Stiffness that vanished on the padding would leave the pencil singular over every padding voxel, so the padding used for this stage is very compliant instead and carries no mass at all. Being massless is what makes it removable: padding voxels contribute no inertia, so they can be condensed out of the stiffness operator exactly, leaving a reduced pencil over the specimen voxels alone whose mass matrix is positive definite.

One mode of each family is then discarded. A constant displacement is annihilated by the forward difference, so it is an exact null vector of the stiffness operator over the whole cell, and the reduced operator inherits it exactly; whatever small frequency the solver returns for that mode is numerical dust. What remains, in ascending order, are the natural frequencies of the axial family and of the transverse family. One transverse family is enough, since both transverse components ride on the same modulus row.

Returns
-------
dict, the natural frequencies in hertz of the longitudinal and transverse families, ascending, null mode dropped.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_padded_eigenproblem(
    face_modulus: np.ndarray,
    voxel_density: np.ndarray,
    voxel_edge: float,
    n_specimen: int,
    n_modes: int,
) -> dict:
    """Read the natural frequencies of the embedded specimen off the padded harmonic pencil.

    Parameters
    ----------
    face_modulus : np.ndarray
        Modulus of every link in pascal, one row per component, shape (3, n_total).
    voxel_density : np.ndarray
        Voxel voxel_density, shape (n_total,), above zero across the specimen and exactly zero across the padding.
    voxel_edge : float
        Edge of one voxel in metre.
    n_specimen : int
        How many voxels at the head of the cell belong to the specimen.
    n_modes : int
        How many frequencies to list for each family.

    Returns
    -------
    dict
        Under the keys longitudinal and transverse, float64 arrays of shape (n_modes,) in hertz.

    Raises
    ------
    ValueError
        When face_modulus is not shaped (3, n_total), when voxel_density is not shaped (n_total,), when n_total comes out even, when voxel_edge fails to sit above zero, when n_specimen is not an integer between one and n_total minus one, when the voxel_density fails to stay above zero across the specimen or to vanish across the padding, when a link beyond the specimen fails to stay above zero, or when n_modes is not an integer between one and n_specimen minus one.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import eigh


def _link_difference_symbol(n_total, voxel_edge):
    """Transform multiplier that shifts a field by one voxel and subtracts, over an odd cell."""
    if n_total % 2 == 0:
        raise ValueError("an even voxel count is outside this configuration")
    modes = np.fft.fftfreq(n_total, d=1.0 / n_total)
    return (np.exp(2j * np.pi * modes / n_total) - 1.0) / voxel_edge


def _stiffness_action(field, face_row, symbol):
    """Extend every link, weight it by its own modulus, and gather the forces back onto the voxels."""
    extension = np.fft.ifft(symbol * np.fft.fft(field, axis=-1), axis=-1).real
    force = np.fft.fft(face_row * extension, axis=-1)
    return np.fft.ifft(np.conj(symbol) * force, axis=-1).real


def _dense_operator(face_row, symbol):
    """Stiffness operator of one family, read off its action on the whole identity at once."""
    images = _stiffness_action(np.eye(face_row.size), face_row, symbol)
    return 0.5 * (images + images.T)


def _family_spectrum(face_row, mass, symbol, n_specimen, n_modes):
    """Natural frequencies of one family once the massless padding has been condensed away."""
    operator = _dense_operator(face_row, symbol)
    kept = operator[:n_specimen, :n_specimen]
    coupling = operator[:n_specimen, n_specimen:]
    padding = operator[n_specimen:, n_specimen:]
    reduced = kept - coupling @ np.linalg.solve(padding, coupling.T)
    reduced = 0.5 * (reduced + reduced.T)
    squared = eigh(reduced, np.diag(mass[:n_specimen]), eigvals_only=True)
    return np.sqrt(np.maximum(squared, 0.0))[1:n_modes + 1] / (2.0 * np.pi)


def _oracle_solve_padded_eigenproblem(
    face_modulus: np.ndarray,
    voxel_density: np.ndarray,
    voxel_edge: float,
    n_specimen: int,
    n_modes: int,
) -> dict:
    """Reference implementation."""
    links = np.asarray(face_modulus, dtype=float)
    mass = np.asarray(voxel_density, dtype=float)
    if links.ndim != 2 or links.shape[0] != 3:
        raise ValueError("face_modulus wants three rows, one per displacement component")
    n_total = links.shape[1]
    if mass.shape != (n_total,):
        raise ValueError("density wants exactly one entry per voxel")
    if n_total % 2 == 0:
        raise ValueError("an even voxel count is outside this configuration")
    edge = float(voxel_edge)
    if edge <= 0.0:
        raise ValueError("voxel_size wants a value above zero")
    if not isinstance(n_specimen, (int, np.integer)) or not 1 <= int(n_specimen) <= n_total - 1:
        raise ValueError("n_specimen wants an integer between one and n_total minus one")
    kept = int(n_specimen)
    if mass[:kept].min() <= 0.0 or np.any(mass[kept:] != 0.0):
        raise ValueError("mass belongs on the specimen voxels and nowhere else")
    if links[:, kept:].min() <= 0.0:
        raise ValueError("every link beyond the specimen wants a modulus above zero")
    if not isinstance(n_modes, (int, np.integer)) or not 1 <= int(n_modes) <= kept - 1:
        raise ValueError("n_modes wants an integer between one and n_specimen minus one")
    wanted = int(n_modes)

    symbol = _link_difference_symbol(n_total, edge)
    return {
        "longitudinal": _family_spectrum(links[0], mass, symbol, kept, wanted),
        "transverse": _family_spectrum(links[1], mass, symbol, kept, wanted),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
COUNT, BODY = 33, 28
LINKS = np.zeros((3, COUNT))
LINKS[:, :BODY] = 1.0e11
MASS = np.full(COUNT, 4506.3)
def verdict(fn):
    try:
        fn(LINKS, MASS, 2.5e-5, BODY, 3)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": "verdict(solve_padded_eigenproblem)",
            "gold_call": "verdict(_oracle_solve_padded_eigenproblem)",
        },
        {
            "setup": """import numpy as np
COUNT, BODY, EDGE = 141, 129, 5.0e-3 / 129
LINKS = np.zeros((3, COUNT))
LINKS[0, :BODY] = 1.77e11
LINKS[1, :BODY] = 4.1e10
LINKS[2, :BODY] = 4.1e10
LINKS[:, BODY:] = 1e-7 * LINKS[:, :BODY].mean(axis=1, keepdims=True)
MASS = np.zeros(COUNT)
MASS[:BODY] = 4506.3
def digest(out):
    # a uniform bar puts the discrete modes at c sin(n pi h / (2 l)) / (pi h), padding aside
    span = BODY * EDGE
    rungs = np.arange(1, 6)
    axial = np.sqrt(1.77e11 / 4506.3) / (np.pi * EDGE) * np.sin(rungs * np.pi * EDGE / (2 * span))
    shear = np.sqrt(4.1e10 / 4506.3) / (np.pi * EDGE) * np.sin(rungs * np.pi * EDGE / (2 * span))
    return (out["longitudinal"].shape,
            round(float(np.abs(out["longitudinal"] / axial - 1.0).max()), 7),
            round(float(np.abs(out["transverse"] / shear - 1.0).max()), 7),
            tuple(np.round(out["longitudinal"] / 1e6, 6)), tuple(np.round(out["transverse"] / 1e6, 6)))
""",
            "call": "digest(solve_padded_eigenproblem(LINKS, MASS, EDGE, BODY, 5))",
            "gold_call": "digest(_oracle_solve_padded_eigenproblem(LINKS, MASS, EDGE, BODY, 5))",
        },
        {
            "setup": """import numpy as np
COUNT, BODY, EDGE = 141, 129, 5.0e-3 / 129
draw = np.random.default_rng(21)
LINKS = np.zeros((3, COUNT))
LINKS[0, :BODY] = np.repeat(draw.uniform(1.3e11, 2.7e11, 43), 3)
LINKS[1, :BODY] = np.repeat(draw.uniform(3.6e10, 5.5e10, 43), 3)
LINKS[2] = LINKS[1]
LINKS[:, BODY:] = 1e-7 * LINKS[:, :BODY].mean(axis=1, keepdims=True)
MASS = np.zeros(COUNT)
MASS[:BODY] = 4506.3
def digest(out):
    axial, shear = out["longitudinal"], out["transverse"]
    return (int(np.all(np.diff(axial) > 0)), int(np.all(np.diff(shear) > 0)),
            int(axial[0] > shear[0]), tuple(np.round(axial / 1e6, 6)), tuple(np.round(shear / 1e6, 6)))
""",
            "call": "digest(solve_padded_eigenproblem(LINKS, MASS, EDGE, BODY, 6))",
            "gold_call": "digest(_oracle_solve_padded_eigenproblem(LINKS, MASS, EDGE, BODY, 6))",
        },
    ]
