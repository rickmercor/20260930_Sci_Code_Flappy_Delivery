"""
Construct the transcription-start-site-informed SNP inclusion prior.

Regulatory variants nearer a gene's transcription start site can receive larger

prior support for a nonzero cis effect. For SNP j, the annotation is the absolute

distance a_j = |position_j - tss| / 10^6 in megabases, the prior log-odds are

ell_j = alpha + kappa * a_j, and the inclusion probability is

pi_j = 1 / (1 + exp(-ell_j)). Alpha controls the baseline log-odds at the TSS,

while kappa controls how those odds vary with distance. The base-pair to

megabase conversion is part of the model convention, and the logistic transform

is evaluated stably so extreme finite coefficients approach zero or one without

overflow.

Inputs

------

positions_bp : one-dimensional SNP positions in base pairs

tss_bp : transcription start site position in base pairs

alpha : annotation-prior intercept

kappa : annotation-prior distance coefficient

Returns

-------

distances_mb : absolute SNP-to-TSS distances in megabases

inclusion_probabilities : annotation-informed probabilities pi_j

Genomic distance to the transcription start site provides a compact biological annotation for cis-regulatory prior support. This step converts the supplied genomic positions into the annotation values and prior probabilities used by the later latent-state updates.

Returns
-------
float64 array of shape (2, p): row 0 contains distances_mb and row 1 contains inclusion_probabilities
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_annotation_probabilities(
    positions_bp: np.ndarray,
    tss_bp: float,
    alpha: float,
    kappa: float,
) -> np.ndarray:
    """Compute TSS distances and annotation-informed inclusion probabilities.

    Parameters
    ----------
    positions_bp : np.ndarray
        One-dimensional SNP positions in base pairs.
    tss_bp : float
        Transcription start site position in base pairs.
    alpha : float
        Logistic-prior intercept.
    kappa : float
        Logistic-prior TSS-distance coefficient.

    Raises
    ------
    ValueError
        If positions are empty or not a finite one-dimensional array, or if a
        scalar parameter is not finite.

    Returns
    -------
    result : np.ndarray
        Float64 array of shape (2, p). Row 0 is `distances_mb`; row 1 is
        `inclusion_probabilities`.
    """
    return result  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_compute_annotation_probabilities(
    positions_bp: np.ndarray,
    tss_bp: float,
    alpha: float,
    kappa: float,
) -> np.ndarray:
    """Reference implementation."""
    positions = np.asarray(positions_bp, dtype=np.float64)
    if positions.ndim != 1 or positions.size == 0 or not np.all(np.isfinite(positions)):
        raise ValueError("positions_bp must be a nonempty finite one-dimensional array")
    try:
        tss = float(tss_bp)
        intercept = float(alpha)
        coefficient = float(kappa)
    except (TypeError, ValueError) as exc:
        raise ValueError("tss_bp, alpha, and kappa must be finite scalars") from exc
    if not np.all(np.isfinite([tss, intercept, coefficient])):
        raise ValueError("tss_bp, alpha, and kappa must be finite scalars")

    distances_mb = np.abs(positions - tss) / 1_000_000.0
    linear_predictor = intercept + coefficient * distances_mb
    inclusion_probabilities = np.exp(-np.logaddexp(0.0, -linear_predictor))
    return np.vstack((distances_mb, inclusion_probabilities))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
positions = np.array([999000.0, 1000000.0, 1050000.0, 2200000.0])
""",
            "call": "compute_annotation_probabilities(positions, 1000000.0, -1.55, -0.9)",
            "gold_call": "_oracle_compute_annotation_probabilities(positions, 1000000.0, -1.55, -0.9)",
        },
        {
            "setup": """import numpy as np
positions = np.array([42.0])
""",
            "call": "compute_annotation_probabilities(positions, 42.0, 1000.0, -1000.0)",
            "gold_call": "_oracle_compute_annotation_probabilities(positions, 42.0, 1000.0, -1000.0)",
        },
        {
            "setup": """import numpy as np
positions = np.array([[1.0, 2.0]])
def run_model():
    try:
        compute_annotation_probabilities(positions, 1.0, 0.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_compute_annotation_probabilities(positions, 1.0, 0.0, 0.0)
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
