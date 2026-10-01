"""
Run the complete neutral periodic charge model and return the longitudinal shift of its highest optical frequency.

Use the same conserved charges and coordinate derivatives in the electrostatic Hessian and periodic Born tensors. Add the supplied short-range Hessian, enforce the specified acoustic constraints and evaluate the optical frequency shift. The linear equilibrium counterterm contributes no second derivative.

Returns
-------
return shift
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_phonon_shift(positions, box, species, pair_weights, cutoff, bias, coefficients, softness, alpha, real_extent, reciprocal_extent, short_hessian, masses, direction):
    """Compute the neutral periodic model's longitudinal optical shift.

    Parameters
    ----------
    positions : (N, 3) real array, N >= 2
        Cartesian coordinates in a fixed orthorhombic cell.
    box : (3,) positive real array
        Cell lengths.
    species : (N,) integer array
        Type indices in [0, S).
    pair_weights : (S, S) real array
        Directed weights indexed by central and neighbor types.
    cutoff : positive float
        Radius of the complete periodic moment sum.
    bias : (S,) real array
        Type-dependent raw-charge intercepts.
    coefficients : (S, 5) real array
        Coefficients of the five invariant features.
    softness : (S,) positive real array
        Fixed type weights for charge redistribution.
    alpha : positive float
        Ewald splitting parameter.
    real_extent : nonnegative integer
        Real-image indices range from -real_extent to +real_extent.
    reciprocal_extent : (3,) nonnegative integer array
        Reciprocal-index bounds along the three cell axes.
    short_hessian : (3*N, 3*N) real array
        Cartesian short-range Hessian. A linear counterterm cancels the
        electrostatic force at the reference geometry and has zero Hessian.
    masses : (N,) positive real array
        Atomic masses.
    direction : (3,) nonzero real array
        Longitudinal propagation direction.

    Call the public functions from steps 1-7 in this order:

    1. moment_jets(positions, box, species, pair_weights, cutoff)
       returns (m, g, h). Include all images with 0 < |d| < cutoff
       and weight c*(1-|d|/cutoff)^4, including nonzero self images.
       Pack moments as M0, M1, and row-major M2.

    2. invariant_jets(m, g, h)
       returns (f, df, ddf). Feature order:
       M0, M0^2, M1.M1, M2:M2, M1.M2.M1.

    3. redistributed_charge_jets(
           f, df, ddf, species, bias, coefficients, softness, 0.0
       )
       returns (q, jq, hq). The total charge is exactly zero.
       The redistribution weight of atom i is
       softness[species[i]] / sum_j softness[species[j]].

    4. ewald_kernel_jets(
           positions, box, alpha, real_extent, reciprocal_extent
       )
       returns (k, kg, kh). Include both signs of reciprocal modes,
       the Ewald self term and the background term.

    5. electrostatic_jet(q, jq, hq, k, kg, kh)
       returns (energy, force, he), retaining all charge and kernel
       derivatives of E = 0.5*q.T@k@q.

    6. periodic_born_tensors(positions, box, q, jq)
       returns z0 with atom, polarization and displacement indices.
       Use the phase-compensated periodic dipole convention of step 6.

    7. longitudinal_shift(
           he + short_hessian, z0, masses, box, direction
       )
       returns the final scalar. Apply the Cartesian acoustic
       corrections and use the mass-weighted optical subspace.
       Isotropic electronic screening cancels from the correction.

    Earlier public step functions are available in the runtime namespace.
    All model parameters are supplied as arguments.

    Returns
    -------
    shift : float
        sqrt(lambda_longitudinal) - sqrt(lambda_analytic), using the
        largest eigenvalue of each optical matrix. Reduced units;
        no 2*pi frequency conversion.

    Raises
    ------
    ValueError
        If required array shapes are invalid, N < 2, an input is
        nonfinite, box lengths, cutoff, alpha, softness or masses are
        nonpositive, species indices are nonintegral or out of range,
        image extents are negative or nonintegral, distinct atoms have
        periodic separation below 1e-10, direction has zero norm,
        short_hessian has the wrong shape or nonfinite entries, an
        intermediate jet is invalid under the called step's contract,
        or either largest optical eigenvalue is nonpositive.

    Runtime
    -------
    Import required libraries inside the submitted function and inside any
    helper that uses them. Do not assume that np, itertools or erfc exists
    in the function's global namespace. Imports in test setup are separate.
    """
    return shift

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve_phonon_shift(positions, box, species, pair_weights, cutoff, bias, coefficients, softness, alpha, real_extent, reciprocal_extent, short_hessian, masses, direction):
    """Reference implementation of the complete neutral pipeline."""
    import numpy as np
    r = np.asarray(positions, dtype=float)
    short = np.asarray(short_hessian, dtype=float)
    if r.ndim != 2 or r.shape[1:] != (3,) or len(r) < 2:
        raise ValueError('positions must have shape (N, 3), N >= 2')
    size = 3 * len(r)
    if short.shape != (size, size) or not np.all(np.isfinite(short)):
        raise ValueError('short_hessian must be finite with shape (3*N, 3*N)')
    moments, moment_first, moment_second = _oracle_moment_jets(positions, box, species, pair_weights, cutoff)
    features, feature_first, feature_second = _oracle_invariant_jets(moments, moment_first, moment_second)
    charges, charge_first, charge_second = _oracle_redistributed_charge_jets(features, feature_first, feature_second, species, bias, coefficients, softness, 0.0)
    kernel, kernel_first, kernel_second = _oracle_ewald_kernel_jets(positions, box, alpha, real_extent, reciprocal_extent)
    _, _, electrostatic_hessian = _oracle_electrostatic_jet(charges, charge_first, charge_second, kernel, kernel_first, kernel_second)
    born_scaled = _oracle_periodic_born_tensors(positions, box, charges, charge_first)
    shift = _oracle_longitudinal_shift(electrostatic_hessian + short, born_scaled, masses, box, direction)
    return float(shift)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base = """import numpy as np

laplacian = np.array([
    [ 2.7, -1.1, -0.7, -0.9],
    [-1.1,  3.0, -1.3, -0.6],
    [-0.7, -1.3,  3.2, -1.2],
    [-0.9, -0.6, -1.2,  2.7],
])

data = {
    "positions": np.array([
        [0.35, 0.55, 0.80],
        [2.05, 1.65, 2.70],
        [3.55, 3.25, 1.60],
        [1.25, 4.20, 4.75],
    ]),
    "box": np.array([4.7, 5.1, 6.2]),
    "species": np.array([0, 1, 0, 1]),
    "pair_weights": np.array([
        [0.9, 1.2],
        [0.7, 1.05],
    ]),
    "cutoff": 5.4,
    "bias": np.array([0.65, -0.58]),
    "coefficients": np.array([
        [0.120, -0.025, 0.015, -0.004, 0.003],
        [-0.085, 0.020, -0.010, 0.005, -0.002],
    ]),
    "softness": np.array([1.0, 1.7]),
    "alpha": 0.72,
    "real_extent": 2,
    "reciprocal_extent": np.array([4, 4, 5]),
    "short_hessian": np.kron(laplacian, np.eye(3)),
    "masses": np.array([1.0, 1.8, 1.3, 2.1]),
    "direction": np.array([2.0, -1.0, 3.0]),
}
"""
    copy_gold = '\nimport copy\ndata_g = copy.deepcopy(data)\n'
    invalid = base + (
        '\ndata["short_hessian"] = np.zeros((1, 1))\n\n'
        'def _raise_code(fn, d):\n'
        '    try:\n'
        '        fn(**d)\n'
        '    except ValueError:\n'
        '        return 1\n'
        '    except Exception:\n'
        '        raise\n'
        '    raise AssertionError("Expected ValueError")\n'
    )
    return [
        {
            'setup': base + copy_gold,
            'call': 'solve_phonon_shift(**data)',
            'gold_call': '_oracle_solve_phonon_shift(**data_g)',
        },
        {
            'setup': base + '\ndata["coefficients"] = np.zeros((2, 5))\n' + copy_gold,
            'call': 'solve_phonon_shift(**data)',
            'gold_call': '_oracle_solve_phonon_shift(**data_g)',
        },
        {
            'setup': (base + '\ndata["coefficients"] = np.zeros((2, 5))\n'
                           + 'data["bias"] = np.zeros(2)\n' + copy_gold),
            'call': 'solve_phonon_shift(**data)',
            'gold_call': '_oracle_solve_phonon_shift(**data_g)',
        },
        {
            'setup': invalid + copy_gold,
            'call': '_raise_code(solve_phonon_shift, data)',
            'gold_call': '_raise_code(_oracle_solve_phonon_shift, data_g)',
        },
    ]
