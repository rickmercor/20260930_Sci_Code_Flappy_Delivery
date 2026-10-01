"""
Assemble the fixed-hyperparameter block-sparse mixed-model trace.

The rotated model is y_rot = X_rot beta + g_rot + epsilon. Sparse cis effects

beta are governed hierarchically by block indicators zeta_b and

annotation-informed SNP indicators gamma_j, while g_rot represents the dense polygenic

background in the eigenbasis of the genetic relationship matrix. Each sweep

first evaluates TSS-based SNP priors, then visits every LD block using a

collapsed block gate and, when active, a sequential within-block SNP scan. The

dense Gaussian effect is sampled after the sparse pass, followed by the ordered

Metropolis update of the annotation coefficients. The supplied uniform and

normal tables replace stochastic draws and make the trace deterministic. The

variance ratios and block prior remain fixed, post-sweep beta vectors are

retained, and their componentwise mean defines the sparse-cis predictor

X_test beta_mean.

Inputs

------

X_rot, y_rot, eigenvalues : rotated training data and GRM spectrum

positions_bp, tss_bp, block_indices : genomic annotations and LD partition

X_test, n_sweeps : held-out genotypes and trace length

u_block, u_snp, z_beta, z_g : fixed block, SNP, slab, and dense variates

mh_proposal_normals, mh_uniforms : fixed annotation-update variates

sigma2, eta_beta, eta_g, pi_block : fixed model hyperparameters

alpha_init, kappa_init : initial annotation coefficients

alpha_prior_mean, kappa_prior_mean : annotation prior centers

alpha_prior_var, kappa_prior_var : annotation prior variances

alpha_step, kappa_step : annotation random-walk scales

Returns

-------

first_prediction : sparse-cis predicted expression for the first held-out row

The final step chains the reduced fixed-hyperparameter trace across sweeps and returns the requested held-out scalar. The supplied random-variate tables make the computation deterministic

Returns
-------
float, the first held-out sparse-cis expression prediction as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def run_full_pipeline(
    X_rot: np.ndarray,
    y_rot: np.ndarray,
    eigenvalues: np.ndarray,
    positions_bp: np.ndarray,
    tss_bp: float,
    block_indices: list,
    X_test: np.ndarray,
    n_sweeps: int,
    u_block: np.ndarray,
    u_snp: np.ndarray,
    z_beta: np.ndarray,
    z_g: np.ndarray,
    mh_proposal_normals: np.ndarray,
    mh_uniforms: np.ndarray,
    sigma2: float,
    eta_beta: float,
    eta_g: float,
    pi_block: float,
    alpha_init: float,
    kappa_init: float,
    alpha_prior_mean: float,
    kappa_prior_mean: float,
    alpha_prior_var: float,
    kappa_prior_var: float,
    alpha_step: float,
    kappa_step: float,
) -> float:
    """Run the fixed-hyperparameter trace and return the first test prediction.

    Parameters
    ----------
    X_rot, y_rot, eigenvalues : np.ndarray
        Rotated training quantities and GRM eigenvalues.
    positions_bp : np.ndarray
        SNP genomic positions in base pairs.
    tss_bp : float
        Transcription start site position in base pairs.
    block_indices : list
        Ordered disjoint arrays that partition the SNP indices.
    X_test : np.ndarray
        Held-out genotype matrix.
    n_sweeps : int
        Positive number of deterministic sweeps.
    u_block, u_snp, z_beta, z_g : np.ndarray
        Supplied block, SNP, slab, and dense-effect variates by sweep.
    mh_proposal_normals, mh_uniforms : np.ndarray
        Supplied annotation proposal and acceptance variates by sweep.
    sigma2, eta_beta, eta_g : float
        Fixed positive residual, slab-ratio, and polygenic-ratio parameters.
    pi_block : float
        Fixed block inclusion prior probability.
    alpha_init, kappa_init : float
        Initial annotation coefficients.
    alpha_prior_mean, kappa_prior_mean : float
        Annotation prior means.
    alpha_prior_var, kappa_prior_var : float
        Positive annotation prior variances.
    alpha_step, kappa_step : float
        Positive annotation proposal scales.

    Raises
    ------
    ValueError
        If the sweep count or random-variate table dimensions are invalid.

    Returns
    -------
    first_prediction : float
        Sparse-cis predicted expression for the first held-out individual.
    """
    return first_prediction  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_run_full_pipeline(
    X_rot: np.ndarray,
    y_rot: np.ndarray,
    eigenvalues: np.ndarray,
    positions_bp: np.ndarray,
    tss_bp: float,
    block_indices: list,
    X_test: np.ndarray,
    n_sweeps: int,
    u_block: np.ndarray,
    u_snp: np.ndarray,
    z_beta: np.ndarray,
    z_g: np.ndarray,
    mh_proposal_normals: np.ndarray,
    mh_uniforms: np.ndarray,
    sigma2: float,
    eta_beta: float,
    eta_g: float,
    pi_block: float,
    alpha_init: float,
    kappa_init: float,
    alpha_prior_mean: float,
    kappa_prior_mean: float,
    alpha_prior_var: float,
    kappa_prior_var: float,
    alpha_step: float,
    kappa_step: float,
) -> float:
    """Reference implementation chaining every earlier step."""
    X = np.asarray(X_rot, dtype=np.float64)
    expression = np.asarray(y_rot, dtype=np.float64)
    eigvals = np.asarray(eigenvalues, dtype=np.float64)
    positions = np.asarray(positions_bp, dtype=np.float64)
    test_genotypes = np.asarray(X_test, dtype=np.float64)
    block_uniforms = np.asarray(u_block, dtype=np.float64)
    snp_uniforms = np.asarray(u_snp, dtype=np.float64)
    slab_normals = np.asarray(z_beta, dtype=np.float64)
    dense_normals = np.asarray(z_g, dtype=np.float64)
    annotation_normals = np.asarray(mh_proposal_normals, dtype=np.float64)
    annotation_uniforms = np.asarray(mh_uniforms, dtype=np.float64)
    if isinstance(n_sweeps, bool) or not isinstance(n_sweeps, (int, np.integer)):
        raise ValueError("n_sweeps must be a positive integer")
    sweeps = int(n_sweeps)
    if sweeps <= 0:
        raise ValueError("n_sweeps must be a positive integer")
    if X.ndim != 2:
        raise ValueError("X_rot must be two dimensional")
    n, p = X.shape
    n_blocks = len(block_indices)
    expected_shapes = [
        (block_uniforms, (sweeps, n_blocks), "u_block"),
        (snp_uniforms, (sweeps, p), "u_snp"),
        (slab_normals, (sweeps, p), "z_beta"),
        (dense_normals, (sweeps, n), "z_g"),
    ]
    for values, expected_shape, name in expected_shapes:
        if values.shape != expected_shape or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite with shape {expected_shape}")
    if annotation_normals.ndim != 3 or annotation_normals.shape[0] != sweeps or annotation_normals.shape[2] != 2:
        raise ValueError("mh_proposal_normals must have shape (n_sweeps, q, 2)")
    if annotation_uniforms.shape != annotation_normals.shape[:2]:
        raise ValueError("mh_uniforms must have shape (n_sweeps, q)")
    if not np.all(np.isfinite(annotation_normals)) or not np.all(np.isfinite(annotation_uniforms)):
        raise ValueError("annotation variates must be finite")
    if expression.shape != (n,) or eigvals.shape != (n,) or positions.shape != (p,):
        raise ValueError("training inputs have incompatible dimensions")
    if test_genotypes.ndim != 2 or test_genotypes.shape[1] != p or test_genotypes.shape[0] == 0:
        raise ValueError("X_test must be nonempty with p columns")

    beta = np.zeros(p, dtype=np.float64)
    gamma = np.zeros(p, dtype=bool)
    zeta = np.zeros(n_blocks, dtype=bool)
    g_rot = np.zeros(n, dtype=np.float64)
    alpha = float(alpha_init)
    kappa = float(kappa_init)
    beta_samples = []

    for sweep in range(sweeps):
        annotation = _oracle_compute_annotation_probabilities(  # noqa: F821
            positions, tss_bp, alpha, kappa
        )
        distances_mb = annotation[0]
        inclusion_probabilities = annotation[1]
        residual = expression - X @ beta - g_rot

        for block_number, block in enumerate(block_indices):
            indices = np.asarray(block, dtype=np.int64)
            residual_without_block = residual + X[:, indices] @ beta[indices]
            block_update = _oracle_compute_collapsed_block_probability(  # noqa: F821
                X,
                indices,
                residual_without_block,
                sigma2,
                eta_beta,
                inclusion_probabilities,
                pi_block,
            )
            zeta[block_number] = (
                block_uniforms[sweep, block_number]
                < block_update[1]
            )
            if zeta[block_number]:
                scan = _oracle_scan_active_block(  # noqa: F821
                    X,
                    indices,
                    residual,
                    beta,
                    gamma,
                    sigma2,
                    eta_beta,
                    inclusion_probabilities,
                    snp_uniforms[sweep, indices],
                    slab_normals[sweep, indices],
                )
                residual = scan[:n].copy()
                beta = scan[n : n + p].copy()
                gamma = scan[n + p : n + 2 * p].astype(bool)
            else:
                residual = residual_without_block
                beta[indices] = 0.0
                gamma[indices] = False

        dense_update = _oracle_sample_polygenic_background(  # noqa: F821
            expression,
            X,
            beta,
            eigvals,
            sigma2,
            eta_g,
            dense_normals[sweep],
        )
        g_rot = dense_update[0]
        annotation_update = _oracle_update_annotation_prior(  # noqa: F821
            alpha,
            kappa,
            distances_mb,
            gamma,
            zeta,
            block_indices,
            annotation_normals[sweep],
            annotation_uniforms[sweep],
            alpha_prior_mean,
            kappa_prior_mean,
            alpha_prior_var,
            kappa_prior_var,
            alpha_step,
            kappa_step,
        )
        alpha = float(annotation_update[0])
        kappa = float(annotation_update[1])
        beta_samples.append(beta.copy())

    summary = _oracle_summarize_sparse_prediction(  # noqa: F821
        np.asarray(beta_samples), test_genotypes
    )
    return float(summary[-1])

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


_MAIN_SETUP = """import numpy as np
X_rot = np.array([[1.3416407864998738, 1.3416407864998738, 1.3416407864998738, 1.3416407864998738, 1.3416407864998738, 1.3416407864998738], [1.0488088481701516, -1.0488088481701516, 1.0488088481701516, -1.0488088481701516, 1.0488088481701516, -1.0488088481701516], [1.02469507659596, 1.02469507659596, -1.02469507659596, -1.02469507659596, 0.0, 0.0], [0.6123724356957945, -0.6123724356957945, -0.6123724356957945, 0.6123724356957945, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
y_rot = np.array([1.3, 0.4, -0.9, 1.0, -0.5])
eigenvalues = np.array([1.8, 1.1, 0.7, 0.25, 0.0])
positions = np.array([999000, 1000000, 1050000, 1200000, 1700000, 2200000])
blocks = [np.array([0, 1, 2]), np.array([3, 4, 5])]
X_test = np.array([[0.6, 0.55, -0.2, 0.9, -0.4, 0.82], [-0.3, -0.28, 1.1, -0.5, 0.7, -0.47]])
u_block = np.array([[0.15, 0.72], [0.58, 0.22], [0.3088, 0.63]])
u_snp = np.array([[0.12, 0.62, 0.18, 0.77, 0.18, 0.54], [0.43, 0.08, 0.71, 0.29, 0.66, 0.15], [0.21, 0.47, 0.05, 0.81, 0.34, 0.58]])
z_beta = np.array([[0.35, -0.8, 1.1, -0.25, 0.6, -1.2], [-0.4, 0.95, -0.55, 0.7, -1.05, 0.2], [1.25, -0.3, 0.45, -0.9, 0.15, 0.85]])
z_g = np.array([[0.2, -0.7, 1.1, 0.3, -0.4], [-0.6, 0.5, -0.2, 1.2, 0.1], [0.8, -0.1, 0.4, -0.5, 1.0]])
mh_normals = np.array([[[0.4, -0.6], [-1.1, 0.3], [0.7, -0.2], [0.2, 0.9]], [[-0.5, 0.8], [0.9, -1.0], [-0.3, 0.4], [1.2, -0.7]], [[0.6, 0.5], [-0.8, -0.4], [0.1, 1.1], [-1.0, 0.2]]])
mh_uniforms = np.array([[0.99, 0.95, 0.4, 0.97], [0.95, 0.2, 0.99, 0.7], [0.85, 0.95, 0.8, 0.99]])
args = (X_rot, y_rot, eigenvalues, positions, 1000000.0, blocks, X_test, 3, u_block, u_snp, z_beta, z_g, mh_normals, mh_uniforms, 0.55, 0.85, 0.65, 0.38, -1.55, -0.9, -2.9444389791664403, 0.0, 4.0, 4.0, 0.3, 0.5)
"""


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": _MAIN_SETUP,
            "call": "float(run_full_pipeline(*args))",
            "gold_call": "float(_oracle_run_full_pipeline(*args))",
        },
        {
            "setup": """import numpy as np
X = np.zeros((1, 1))
y = np.array([2.0])
eigenvalues = np.array([0.0])
positions = np.array([10.0])
blocks = [np.array([0])]
X_test = np.array([[3.0]])
u_block = np.array([[0.9]])
u_snp = np.array([[0.5]])
z_beta = np.array([[0.0]])
z_g = np.array([[7.0]])
mh_normals = np.array([[[0.0, 0.0]]])
mh_uniforms = np.array([[0.5]])
args = (X, y, eigenvalues, positions, 10.0, blocks, X_test, 1, u_block, u_snp, z_beta, z_g, mh_normals, mh_uniforms, 1.0, 1.0, 1.0, 0.2, -2.0, 0.0, -2.0, 0.0, 4.0, 4.0, 0.3, 0.5)
""",
            "call": "float(run_full_pipeline(*args))",
            "gold_call": "float(_oracle_run_full_pipeline(*args))",
        },
        {
            "setup": """import numpy as np
X = np.zeros((1, 1))
y = np.array([1.0])
eigenvalues = np.array([0.0])
positions = np.array([10.0])
blocks = [np.array([0])]
X_test = np.array([[1.0]])
u_block = np.array([[0.5]])
u_snp = np.array([[0.5]])
z_beta = np.array([[0.0]])
z_g = np.array([[0.0]])
mh_normals = np.array([[[0.0, 0.0]]])
mh_uniforms = np.array([[0.5]])
args = (X, y, eigenvalues, positions, 10.0, blocks, X_test, 2, u_block, u_snp, z_beta, z_g, mh_normals, mh_uniforms, 1.0, 1.0, 1.0, 0.2, -2.0, 0.0, -2.0, 0.0, 4.0, 4.0, 0.3, 0.5)
def run_model():
    try:
        run_full_pipeline(*args)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_run_full_pipeline(*args)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
