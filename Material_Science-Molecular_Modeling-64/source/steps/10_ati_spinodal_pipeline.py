"""
Compose ATI inference and select a conditional two-row stability bound.

The orchestrator evaluates the active-set certificate J for complete, single-deletion, and conditional double-deletion data. The extended numerical return retains sensitivity diagnostics for code-level verification, while the reasoning response reports only the decision-bearing certificate.

Returns
-------
return np.asarray([pair_lower, complete_lower, first_lower, first_row, second_row, valid_second_count, pair_root, pair_profile_half_width, pair_coefficient_count, robust_alpha, robust_beta, pair_local_standard_error, first_minus_pair, complete_minus_pair, block_count, runner_up_row, runner_up_bound, runner_up_feasible, local_delta_bound], dtype=float)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def ati_spinodal_pipeline(gap_blocks, lambdas, compositions,
                          solution_volumes_a3, temperature_k,
                          mass_reference_u, mass_solute_u,
                          volume_reference_a3, pressure_target_gpa,
                          bulk_modulus_gpa, bulk_derivative,
                          volume_zero_a3, calibration_compositions,
                          calibration_residuals_ev, sigma_grid_ev,
                          length_grid, noise_sigma_ev,
                          monitor_index=3, z_score=1.645,
                          delta_aicc=4.0, coexistence_sigma_radius=1.0):
    """Compose ATI inference and a sequential synchronized-block stress test.

    Refit all single synchronized-row deletions and select the smallest bound,
    then condition on that deletion and refit every remaining second deletion.
    Exclude a second candidate if any stage raises ValueError, including loss
    of positive-definite block covariance. Select the smallest surviving pair
    bound; ties use the smaller original one-based row number at each round.
    Return float64(19): [pair_bound,complete_bound,first_deletion_bound,
    first_row,second_row,valid_second_count,pair_root,pair_profile_half_width,
    pair_coefficient_count,robust_alpha,robust_beta,local_delta_standard_error,
    first_minus_pair,complete_minus_pair,original_block_count]. Every fold
    independently re-estimates all covariances and coefficients. The last four
    entries are [unconditional_runner_up_row,unconditional_runner_up_bound,
    runner_up_conditional_feasible,local_delta_lower_bound].
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_ati_spinodal_pipeline(gap_blocks, lambdas, compositions,
                                  solution_volumes_a3, temperature_k,
                                  mass_reference_u, mass_solute_u,
                                  volume_reference_a3, pressure_target_gpa,
                                  bulk_modulus_gpa, bulk_derivative,
                                  volume_zero_a3, calibration_compositions,
                                  calibration_residuals_ev, sigma_grid_ev,
                                  length_grid, noise_sigma_ev,
                                  monitor_index=3, z_score=1.645,
                                  delta_aicc=4.0, coexistence_sigma_radius=1.0):
    """Compose all stages and stress-test the bound by synchronized block deletion."""
    import numpy as np
    blocks = np.asarray(gap_blocks, dtype=float)
    x = np.asarray(compositions, dtype=float)
    volumes = np.asarray(solution_volumes_a3, dtype=float)
    monitor = int(monitor_index)
    if blocks.ndim != 3 or blocks.shape[0] != x.size or blocks.shape[2] != np.asarray(lambdas).size:
        raise ValueError("gap_blocks must have shape (C,B,L)")
    if volumes.shape != x.shape or not (0 <= monitor < x.size):
        raise ValueError("volume vector and monitor_index must match compositions")
    if blocks.shape[1] < np.asarray(lambdas).size + 2:
        raise ValueError("need at least L+2 synchronized blocks for leave-one-block-out stability")

    def run_once(active_blocks):
        certificates = []
        for index in range(x.size):
            statistics = _oracle_gap_block_statistics(active_blocks[index])
            ti = _oracle_quadratic_gls_ti(lambdas, statistics)
            mass = _oracle_alchemical_mass_correction(
                x[index], temperature_k, mass_reference_u, mass_solute_u
            )
            pv = _oracle_birch_murnaghan_pv(
                volumes[index], volume_reference_a3, pressure_target_gpa,
                bulk_modulus_gpa, bulk_derivative, volume_zero_a3
            )
            certificates.append(
                np.array([ti[0] + mass[0] + pv[0], ti[1], ti[0], mass[0], pv[0]], dtype=float)
            )
        mixing = _oracle_mixing_curve_covariance(
            x, np.asarray(certificates), temperature_k
        )
        augmented = _oracle_augment_finite_size_covariance(
            mixing, calibration_compositions, calibration_residuals_ev,
            sigma_grid_ev, length_grid, noise_sigma_ev
        )
        rk = _oracle_fit_redlich_kister(augmented, temperature_k)
        screened = _oracle_common_tangent_screen(
            rk, temperature_k, sigma_radius=coexistence_sigma_radius
        )
        spinodal = _oracle_spinodal_lower_confidence(
            screened, temperature_k, z_score=z_score, delta_aicc=delta_aicc
        )
        chosen_row = screened[np.flatnonzero(screened[:, 0] == spinodal[3])[0]]
        return spinodal, chosen_row

    full_spinodal, _ = run_once(blocks)
    deletion_candidates = []
    for omitted in range(blocks.shape[1]):
        fold_spinodal, fold_row = run_once(np.delete(blocks, omitted, axis=1))
        deletion_candidates.append((float(fold_spinodal[0]), omitted,
                                    fold_spinodal, fold_row))
    first_lower, first_omitted, first_spinodal, _ = min(
        deletion_candidates, key=lambda item: (item[0], item[1])
    )
    _, runner_up_omitted, runner_up_spinodal, _ = sorted(
        deletion_candidates, key=lambda item: (item[0], item[1])
    )[1]
    second_candidates = []
    runner_up_conditional_feasible = 0.0
    for second_omitted in range(blocks.shape[1]):
        if second_omitted == first_omitted:
            continue
        try:
            pair_spinodal, pair_row = run_once(
                np.delete(blocks, [first_omitted, second_omitted], axis=1)
            )
        except ValueError:
            continue
        if second_omitted == runner_up_omitted:
            runner_up_conditional_feasible = 1.0
        second_candidates.append((float(pair_spinodal[0]), second_omitted,
                                  pair_spinodal, pair_row))
    if not second_candidates:
        raise ValueError("no covariance-feasible conditional second deletion")
    robust_lower, second_omitted, chosen_spinodal, chosen_row = min(
        second_candidates, key=lambda item: (item[0], item[1])
    )
    return np.array([
        robust_lower, full_spinodal[0], first_lower, float(first_omitted + 1),
        float(second_omitted + 1), float(len(second_candidates)),
        chosen_spinodal[1], chosen_spinodal[2], chosen_spinodal[3],
        chosen_row[31], chosen_row[32], chosen_spinodal[6],
        first_lower - robust_lower, full_spinodal[0] - robust_lower,
        float(blocks.shape[1]), float(runner_up_omitted + 1),
        float(runner_up_spinodal[0]), runner_up_conditional_feasible,
        float(chosen_spinodal[1] - z_score * chosen_spinodal[6])
    ], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup="import numpy as np\nv=lambda s:np.fromstring(s,sep=',')\nlam=v('0,.25,.5,.75,1')\nH=1-2*((np.array([0,21,12,25,3,22,15,26])[:,None]>>np.arange(4,-1,-1))&1)\nchol=np.zeros((5,5));chol[np.tril_indices(5)]=v('1,.35,.93675,.15,.25,.95656,.08,.12,.30,.94234,.05,.10,.16,.28,.94048')\nresid=H@chol.T\na=v('-.046395623,-.119592843,-.204877463,-.299644351,-.408949543,-.530757393,-.617605610')\nb=-.063-.003*np.arange(7);b[-1]=-.08\nc=.0415+.0015*np.arange(7);c[-1]=.05\nmeans=a[:,None]+b[:,None]*lam+c[:,None]*lam**2\nscales=.0015*(1+.08*np.arange(7))\nblocks=means[:,None,:]+scales[:,None,None]*resid[None,:,:]\nx=v('.15,.30,.45,.60,.75,.90,1')\nvolumes=v('39,36.70,34.35,32.40,30.45,28.75,27.80')\nfs_x=v('.12,.28,.46,.64,.82,1')\nfs_r=v('.00185,.00062,-.00048,-.00131,-.00037,.00091')\nfs_sigma=v('.0008,.0012,.0016,.0020,.0025,.0032')\nfs_length=v('.06,.12,.20,.32,.50,.80')"
    rows=[{'call': 'ati_spinodal_pipeline(blocks*1.0,lam,x,volumes,520.0,22.98976928,6.94,41.552,0.0,5.143,3.943,41.552,fs_x,fs_r*1.0,fs_sigma,fs_length,0.00035,monitor_index=3,z_score=1.645,delta_aicc=4.0,coexistence_sigma_radius=1.0)', 'gold_call': '_oracle_ati_spinodal_pipeline(blocks*1.0,lam,x,volumes,520.0,22.98976928,6.94,41.552,0.0,5.143,3.943,41.552,fs_x,fs_r*1.0,fs_sigma,fs_length,0.00035,monitor_index=3,z_score=1.645,delta_aicc=4.0,coexistence_sigma_radius=1.0)'}, {'call': 'ati_spinodal_pipeline(blocks*1.0,lam,x,volumes+np.array([0.1,0.0,-0.1,0.0,0.1,0.0,-0.1]),500.0,22.98976928,6.94,41.552,0.0,5.143,3.943,41.552,fs_x,fs_r*0.8,fs_sigma,fs_length,0.00035,monitor_index=2,z_score=1.28,delta_aicc=2.0,coexistence_sigma_radius=0.75)', 'gold_call': '_oracle_ati_spinodal_pipeline(blocks*1.0,lam,x,volumes+np.array([0.1,0.0,-0.1,0.0,0.1,0.0,-0.1]),500.0,22.98976928,6.94,41.552,0.0,5.143,3.943,41.552,fs_x,fs_r*0.8,fs_sigma,fs_length,0.00035,monitor_index=2,z_score=1.28,delta_aicc=2.0,coexistence_sigma_radius=0.75)'}, {'call': 'ati_spinodal_pipeline(blocks*1.0,lam,x,volumes+np.array([-0.1,0.1,0.0,-0.1,0.0,0.1,0.0]),540.0,22.98976928,6.94,41.552,0.0,5.143,3.943,41.552,fs_x,fs_r*1.2,fs_sigma,fs_length,0.00035,monitor_index=4,z_score=1.96,delta_aicc=6.0,coexistence_sigma_radius=1.25)', 'gold_call': '_oracle_ati_spinodal_pipeline(blocks*1.0,lam,x,volumes+np.array([-0.1,0.1,0.0,-0.1,0.0,0.1,0.0]),540.0,22.98976928,6.94,41.552,0.0,5.143,3.943,41.552,fs_x,fs_r*1.2,fs_sigma,fs_length,0.00035,monitor_index=4,z_score=1.96,delta_aicc=6.0,coexistence_sigma_radius=1.25)'}, {'call': 'ati_spinodal_pipeline(blocks*1.0,lam,x,volumes+np.array([0.0,-0.08,0.05,0.02,-0.04,0.06,0.0]),490.0,22.98976928,6.94,41.552,0.0,5.143,3.943,41.552,fs_x,fs_r*0.6,fs_sigma,fs_length,0.00035,monitor_index=1,z_score=1.1,delta_aicc=3.0,coexistence_sigma_radius=0.5)', 'gold_call': '_oracle_ati_spinodal_pipeline(blocks*1.0,lam,x,volumes+np.array([0.0,-0.08,0.05,0.02,-0.04,0.06,0.0]),490.0,22.98976928,6.94,41.552,0.0,5.143,3.943,41.552,fs_x,fs_r*0.6,fs_sigma,fs_length,0.00035,monitor_index=1,z_score=1.1,delta_aicc=3.0,coexistence_sigma_radius=0.5)'}, {'call': 'ati_spinodal_pipeline(blocks*1.0,lam,x,volumes+np.array([0.05,0.02,-0.06,0.08,0.0,-0.05,0.03]),500.0,22.98976928,6.94,41.552,0.0,5.143,3.943,41.552,fs_x,fs_r*1.4,fs_sigma,fs_length,0.00035,monitor_index=5,z_score=1.5,delta_aicc=5.0,coexistence_sigma_radius=1.1)', 'gold_call': '_oracle_ati_spinodal_pipeline(blocks*1.0,lam,x,volumes+np.array([0.05,0.02,-0.06,0.08,0.0,-0.05,0.03]),500.0,22.98976928,6.94,41.552,0.0,5.143,3.943,41.552,fs_x,fs_r*1.4,fs_sigma,fs_length,0.00035,monitor_index=5,z_score=1.5,delta_aicc=5.0,coexistence_sigma_radius=1.1)'}, {'call': 'ati_spinodal_pipeline(blocks*1.0,lam,x,volumes+np.array([-0.04,0.07,0.02,-0.03,0.05,0.0,-0.02]),505.0,22.98976928,6.94,41.552,0.0,5.143,3.943,41.552,fs_x,fs_r*0.9,fs_sigma,fs_length,0.00035,monitor_index=0,z_score=0.8,delta_aicc=1.0,coexistence_sigma_radius=0.9)', 'gold_call': '_oracle_ati_spinodal_pipeline(blocks*1.0,lam,x,volumes+np.array([-0.04,0.07,0.02,-0.03,0.05,0.0,-0.02]),505.0,22.98976928,6.94,41.552,0.0,5.143,3.943,41.552,fs_x,fs_r*0.9,fs_sigma,fs_length,0.00035,monitor_index=0,z_score=0.8,delta_aicc=1.0,coexistence_sigma_radius=0.9)'}, {'call': 'ati_spinodal_pipeline(blocks*1.0,lam,x,volumes+np.array([0.03,-0.02,0.06,0.0,-0.07,0.04,0.01]),515.0,22.98976928,6.94,41.552,0.0,5.143,3.943,41.552,fs_x,fs_r*1.1,fs_sigma,fs_length,0.00035,monitor_index=2,z_score=2.0,delta_aicc=8.0,coexistence_sigma_radius=0.65)', 'gold_call': '_oracle_ati_spinodal_pipeline(blocks*1.0,lam,x,volumes+np.array([0.03,-0.02,0.06,0.0,-0.07,0.04,0.01]),515.0,22.98976928,6.94,41.552,0.0,5.143,3.943,41.552,fs_x,fs_r*1.1,fs_sigma,fs_length,0.00035,monitor_index=2,z_score=2.0,delta_aicc=8.0,coexistence_sigma_radius=0.65)'}, {'call': 'ati_spinodal_pipeline(blocks*1.0,lam,x,volumes+np.array([-0.02,0.04,-0.05,0.07,0.01,-0.03,0.02]),525.0,22.98976928,6.94,41.552,0.0,5.143,3.943,41.552,fs_x,fs_r*1.3,fs_sigma,fs_length,0.00035,monitor_index=6,z_score=1.72,delta_aicc=4.5,coexistence_sigma_radius=0.85)', 'gold_call': '_oracle_ati_spinodal_pipeline(blocks*1.0,lam,x,volumes+np.array([-0.02,0.04,-0.05,0.07,0.01,-0.03,0.02]),525.0,22.98976928,6.94,41.552,0.0,5.143,3.943,41.552,fs_x,fs_r*1.3,fs_sigma,fs_length,0.00035,monitor_index=6,z_score=1.72,delta_aicc=4.5,coexistence_sigma_radius=0.85)'}]
    for row in rows: row['setup']=setup
    return rows
