"""
Orchestrate the fixed two-layer calculation using the seven preceding public functions and the conventions in the main problem.



Only fill_first varies; all second-layer tangents are zero. Return one float64 array ordered as



[J, rho_2, D, T_plus1, R_total, T_total,

 updates_layer1, updates_layer2, d_flux_total].



Use layer 1’s two-update residual diagnostics for J and rho_2. The remaining entries come from the converged two-layer optical calculation.

This step assembles the previously implemented numerical components into the fixed physical calculation defined by the main problem.



Preserve the physical layer order, use only propagating exterior orders for power totals, and retain all seven harmonics internally. The primary outputs come from layer 1 after two rational updates; the optical quantities come from the converged cascade.

Returns
-------
Return one float64 NumPy array of shape (9,), not a tuple or scalar:  [J, rho_2, D, T_plus1, R_total, T_total, updates1, updates2, d_flux_total]  The entries are in exactly that order. The update counts are integer-valued but stored as float64. Only J belongs inside the model-facing <final_answer> tags. The function returns the complete nine-entry numerical array, not tagged text.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_sensitivity(fill_first: float = 0.43) -> np.ndarray:
    """Compute the fixed-update residual and optical sensitivities.

    Parameters
    ----------
    fill_first
        Fill fraction of the first periodic layer.

    Returns
    -------
    np.ndarray
        Float64 array ordered as
        [J, rho_2, D, T_plus1, R_total, T_total,
        updates_layer1, updates_layer2, d_flux_total].
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve_sensitivity(fill_first: float = 0.43) -> np.ndarray:
    global np, solve, expm_frechet
    import numpy as np
    from scipy.linalg import expm_frechet, solve

    harmonics = np.arange(-3, 4)
    kappa = 0.23 + 0.76 * harmonics
    vacuum_root = np.sqrt(
        (kappa ** 2 - 1).astype(np.complex128)
    )
    layers = [
        (fill_first, 0.0, 4.0, 3.1),
        (0.57, 0.19, 2.89, 2.4),
    ]

    scattering_pairs = []
    update_counts = []
    primary_rho = None
    primary_sensitivity = None

    for layer_index, (
        fill,
        offset,
        epsilon_high,
        depth,
    ) in enumerate(layers):
        coefficients, coefficient_tangent = (
            _oracle_reciprocal_fourier(
                harmonics,
                fill,
                offset,
                epsilon_high,
            )
        )

        if layer_index != 0:
            coefficient_tangent = np.zeros_like(
                coefficient_tangent
            )

        operators = _oracle_tm_operators(
            coefficients,
            coefficient_tangent,
            kappa,
        )

        (
            root,
            root_tangent,
            updates,
            _,
            rho_two,
            rho_tangent_two,
        ) = _oracle_rotated_root(
            operators[4],
            operators[5],
        )

        if layer_index == 0:
            primary_rho = rho_two
            primary_sensitivity = rho_tangent_two

        modal = _oracle_transformed_modal(
            operators[2],
            operators[3],
            root,
            root_tangent,
        )
        propagation = _oracle_layer_propagation(
            root,
            root_tangent,
            depth,
        )
        scattering_pairs.append(
            _oracle_layer_scattering(
                *modal,
                *propagation,
                vacuum_root,
            )
        )
        update_counts.append(updates)

    scattering, tangent = _oracle_redheffer_compose(
        *scattering_pairs[0],
        *scattering_pairs[1],
    )

    transmission = scattering[1, 0, :, 3]
    transmission_tangent = tangent[1, 0, :, 3]
    reflection = scattering[0, 0, :, 3]
    reflection_tangent = tangent[0, 0, :, 3]

    propagating = np.abs(kappa) < 1
    weights = np.zeros_like(kappa)
    weights[propagating] = (
        np.sqrt(1 - kappa[propagating] ** 2)
        / np.sqrt(1 - kappa[3] ** 2)
    )

    transmitted = weights * np.abs(transmission) ** 2
    reflected = weights * np.abs(reflection) ** 2

    transmitted_tangent = (
        2
        * weights
        * np.real(
            np.conj(transmission) * transmission_tangent
        )
    )
    reflected_tangent = (
        2
        * weights
        * np.real(
            np.conj(reflection) * reflection_tangent
        )
    )

    return np.array(
        [
            primary_sensitivity,
            primary_rho,
            transmitted_tangent[4],
            transmitted[4],
            reflected.sum(),
            transmitted.sum(),
            *update_counts,
            (
                transmitted_tangent
                + reflected_tangent
            ).sum(),
        ],
        dtype=np.float64,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '', 'call': 'solve_sensitivity(0.39)', 'gold_call': '_oracle_solve_sensitivity(0.39)'},
     {'setup': '', 'call': 'solve_sensitivity(0.43)', 'gold_call': '_oracle_solve_sensitivity(0.43)'},
     {'setup': '', 'call': 'solve_sensitivity(0.47)', 'gold_call': '_oracle_solve_sensitivity(0.47)'},
     {'setup': 'import numpy as np\nfill_boundary = float(np.nextafter(1.0, 0.0))',
      'call': 'solve_sensitivity(fill_boundary)',
      'gold_call': '_oracle_solve_sensitivity(fill_boundary)'}]
