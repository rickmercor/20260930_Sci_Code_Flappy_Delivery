"""
The deterministic imputation pipeline encodes phased genotypes, constructs

overlapping token windows, derives normalized physical-coordinate bias, combines

rotary attention with the directional genomic bias, and applies the local CNN

bottleneck. A frozen masked-language head decodes every masked occurrence,

overlap averaging consolidates repeated loci, and mean phase-aware negative

log-likelihood supplies the final scalar.

Inputs

------

beta: relative genomic positional bias coefficient

overlap: number of shared SNP tokens between consecutive six-SNP windows

Returns

-------

masked_nll: native float reported inside final_answer tags

Returns
-------
float, the phase-aware masked mean negative log-likelihood as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_full_pipeline(beta: float = -1.4, overlap: int = 3) -> float:
    """Run the frozen GenoBERT-style toy imputation pipeline.

    Parameters
    ----------
    beta : float
        Finite coefficient multiplying relative genomic positional bias.
    overlap : int
        Integer overlap in ``[0, 6)`` for six-SNP windows.

    Raises
    ------
    ValueError
        If beta is not finite or overlap is not an integer in ``[0, 6)``.

    Returns
    -------
    masked_nll : float
        Mean phase-aware negative log-likelihood over the fixed masked sites.
    """
    return masked_nll  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_run_full_pipeline(beta: float = -1.4, overlap: int = 3) -> float:
    """Reference implementation chaining every earlier step."""
    if not np.isfinite(beta):
        raise ValueError("beta must be finite")
    if (
        not isinstance(overlap, (int, np.integer))
        or isinstance(overlap, (bool, np.bool_))
        or overlap < 0
        or overlap >= 6
    ):
        raise ValueError("overlap must be an integer in [0, 6)")

    alleles = np.array(
        [
            [[0, 0], [0, 1], [0, 0], [1, 1], [1, 0], [0, 1], [1, 1], [0, 0], [1, 0]],
            [[0, 1], [0, 1], [1, 0], [1, 1], [0, 0], [1, 0], [1, 1], [0, 1], [0, 0]],
            [[1, 1], [1, 0], [1, 0], [0, 1], [0, 0], [1, 1], [0, 1], [0, 0], [1, 1]],
            [[0, 0], [0, 0], [0, 1], [1, 0], [1, 1], [0, 1], [0, 0], [1, 1], [1, 0]],
        ],
        dtype=int,
    )
    masked_sites = np.array(
        [[0, 1], [0, 6], [1, 2], [1, 7], [2, 3], [2, 8], [3, 4], [3, 6]],
        dtype=int,
    )
    genomic_positions = np.array(
        [100.0, 107.0, 125.0, 126.0, 170.0, 205.0, 206.0, 290.0, 450.0]
    )

    encoded = _oracle_encode_phased_genotypes(alleles, masked_sites)  # noqa: F821
    segments = _oracle_segment_genotype_windows(  # noqa: F821
        encoded, genomic_positions, window_size=6, overlap=overlap
    )
    bias_pack = _oracle_build_genomic_bias(segments)  # noqa: F821
    projected = _oracle_project_rotary_qkv(  # noqa: F821
        segments, embedding_dim=4, rope_base=10000.0
    )
    attention_output = _oracle_apply_genomic_attention(  # noqa: F821
        projected, bias_pack, beta=beta
    )
    hidden = _oracle_normalize_attention_residual(  # noqa: F821
        projected, attention_output, encoder_depth=1
    )
    encoded_hidden = _oracle_apply_cnn_bottleneck(  # noqa: F821
        hidden, encoder_depth=1, bottleneck_factor=2.0, kernel_size=3
    )
    pooled = _oracle_merge_masked_probabilities(  # noqa: F821
        encoded_hidden, segments, masked_sites
    )
    return _oracle_score_masked_likelihood(pooled)  # noqa: F821

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "beta = -1.4\noverlap = 3\n",
            "call": "float(run_full_pipeline(beta=beta, overlap=overlap))",
            "gold_call": "float(_oracle_run_full_pipeline(beta=beta, overlap=overlap))",
        },
        {
            "setup": "beta = 0.0\noverlap = 0\n",
            "call": "float(run_full_pipeline(beta=beta, overlap=overlap))",
            "gold_call": "float(_oracle_run_full_pipeline(beta=beta, overlap=overlap))",
        },
        {
            "setup": """def run_model():
    try:
        run_full_pipeline(overlap=6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_run_full_pipeline(overlap=6)
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
