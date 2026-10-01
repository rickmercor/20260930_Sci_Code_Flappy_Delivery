"""
Return the action of the leave-one-chromosome-out phenotypic covariance on a set

of vectors, without forming that covariance.

Correcting a trait for polygenic background before testing a region requires the

phenotypic covariance carried by every chromosome except the one under test, so

that the variants being tested do not also appear among the variants used to

correct for. Dropping the chromosome under test changes nothing else about the

fitted model: the genetic part is the sum of X_t X_t^T over the retained

chromosomes, divided by the same scaling factor m the pooled relatedness matrix

was built with and weighted by sigma_g^2, and the residual part stays sigma_e^2

times the identity. Only the action of the covariance on a vector is ever

needed, so the blocks are applied one after another and accumulated, and the

covariance itself is never assembled.

Returns
-------
np.ndarray with the shape of the input vectors, float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def loco_covariance_product(
    blocks: list,
    focal_index: int,
    sigma_g_squared: float,
    sigma_e_squared: float,
    vectors: "np.ndarray",
) -> "np.ndarray":
    """Apply the leave-one-chromosome-out covariance to one or more vectors.

    Args:
        blocks: list of arrays, one per chromosome, each of shape (n, p_t)
            holding the centred and frequency-scaled genotypes of that
            chromosome's retained variants.
        focal_index: index into blocks of the chromosome under test, whose
            variants are excluded from the covariance.
        sigma_g_squared: the genetic variance component.
        sigma_e_squared: the residual variance component.
        vectors: array of shape (n,) or (n, k) to apply the covariance to.

    Returns:
        np.ndarray with the same shape as vectors, holding the product.

    Raises:
        ValueError: if blocks is empty, if focal_index is out of range, if the
            blocks disagree on the number of individuals, or if excluding the
            focal chromosome leaves no variants to build the covariance from.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_loco_covariance_product(
    blocks: list,
    focal_index: int,
    sigma_g_squared: float,
    sigma_e_squared: float,
    vectors: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation."""
    if len(blocks) == 0:
        raise ValueError("at least one genotype block is required")
    focal_index = int(focal_index)
    if not 0 <= focal_index < len(blocks):
        raise ValueError("focal_index is out of range")
    matrices = [np.asarray(block, dtype=float) for block in blocks]
    n_individuals = matrices[0].shape[0]
    for block in matrices:
        if block.ndim != 2 or block.shape[0] != n_individuals:
            raise ValueError("blocks disagree on the number of individuals")
    retained = sum(block.shape[1] for index, block in enumerate(matrices)
                   if index != focal_index)
    if retained == 0:
        raise ValueError("no variants remain once the focal chromosome is dropped")
    pooled = sum(float(np.trace(block @ block.T)) for block in matrices)
    if pooled <= 0.0:
        raise ValueError("the genotype blocks carry no variance")
    scale = pooled / n_individuals
    target = np.asarray(vectors, dtype=float)
    accumulated = np.zeros_like(target)
    for index, block in enumerate(matrices):
        if index == focal_index:
            continue
        accumulated = accumulated + block @ (block.T @ target)
    return float(sigma_e_squared) * target + (
        float(sigma_g_squared) / scale) * accumulated

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
blocks = [
    [[0.0198, 0.7766], [-0.5912, -1.7604], [-0.1022, -1.5674], [0.3768, 1.1646], [0.2968, 1.3866]],
    [[0.1546, -0.1674, 0.2012], [1.3636, -0.2144, 0.2802], [-0.8694, -0.1494, 0.2012], [0.1126, 0.0736, 0.1532], [-0.7614, 0.4576, -0.8358]],
    [[-0.4056, 0.934], [-1.2246, -0.421], [2.3594, -1.369], [-2.0786, -0.803], [1.3494, 1.659]],
]
blocks = [np.array(b, dtype=float) for b in blocks]
trait = np.array([10.4, 12.1, 9.7, 11.5, 13.2])

def rnd(a):
    return [round(float(v), 10) for v in list(a)]
""",
            "call": "rnd(loco_covariance_product(blocks, 0, 0.93, 0.07, np.array([1.0, -2.0, 0.5, 0.25, -1.5])))",
            "gold_call": "rnd(_oracle_loco_covariance_product(blocks, 0, 0.93, 0.07, np.array([1.0, -2.0, 0.5, 0.25, -1.5])))",
        },
        {
            "setup": """import numpy as np
blocks = [
    [[0.0198, 0.7766], [-0.5912, -1.7604], [-0.1022, -1.5674], [0.3768, 1.1646], [0.2968, 1.3866]],
    [[0.1546, -0.1674, 0.2012], [1.3636, -0.2144, 0.2802], [-0.8694, -0.1494, 0.2012], [0.1126, 0.0736, 0.1532], [-0.7614, 0.4576, -0.8358]],
    [[-0.4056, 0.934], [-1.2246, -0.421], [2.3594, -1.369], [-2.0786, -0.803], [1.3494, 1.659]],
]
blocks = [np.array(b, dtype=float) for b in blocks]
trait = np.array([10.4, 12.1, 9.7, 11.5, 13.2])

def rnd2(a):
    return [[round(float(v), 10) for v in row] for row in a]

import numpy as np
mat = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5], [0.5, -1.0]])
""",
            "call": "rnd2(loco_covariance_product(blocks, 1, 0.5, 0.5, mat))",
            "gold_call": "rnd2(_oracle_loco_covariance_product(blocks, 1, 0.5, 0.5, mat))",
        },
        {
            "setup": """import numpy as np
blocks = [
    [[0.0198, 0.7766], [-0.5912, -1.7604], [-0.1022, -1.5674], [0.3768, 1.1646], [0.2968, 1.3866]],
    [[0.1546, -0.1674, 0.2012], [1.3636, -0.2144, 0.2802], [-0.8694, -0.1494, 0.2012], [0.1126, 0.0736, 0.1532], [-0.7614, 0.4576, -0.8358]],
    [[-0.4056, 0.934], [-1.2246, -0.421], [2.3594, -1.369], [-2.0786, -0.803], [1.3494, 1.659]],
]
blocks = [np.array(b, dtype=float) for b in blocks]
trait = np.array([10.4, 12.1, 9.7, 11.5, 13.2])

def rnd(a):
    return [round(float(v), 10) for v in list(a)]
""",
            "call": "rnd(loco_covariance_product(blocks, 2, 0.93, 0.07, np.array([1.0, -2.0, 0.5, 0.25, -1.5])))",
            "gold_call": "rnd(_oracle_loco_covariance_product(blocks, 2, 0.93, 0.07, np.array([1.0, -2.0, 0.5, 0.25, -1.5])))",
        },
        {
            "setup": """import numpy as np
blocks = [
    [[0.0198, 0.7766], [-0.5912, -1.7604], [-0.1022, -1.5674], [0.3768, 1.1646], [0.2968, 1.3866]],
    [[0.1546, -0.1674, 0.2012], [1.3636, -0.2144, 0.2802], [-0.8694, -0.1494, 0.2012], [0.1126, 0.0736, 0.1532], [-0.7614, 0.4576, -0.8358]],
    [[-0.4056, 0.934], [-1.2246, -0.421], [2.3594, -1.369], [-2.0786, -0.803], [1.3494, 1.659]],
]
blocks = [np.array(b, dtype=float) for b in blocks]
trait = np.array([10.4, 12.1, 9.7, 11.5, 13.2])

def run_model():
    try:
        loco_covariance_product(blocks, 3, 0.93, 0.07, np.array([1.0, -2.0, 0.5, 0.25, -1.5]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_oracle():
    try:
        _oracle_loco_covariance_product(blocks, 3, 0.93, 0.07, np.array([1.0, -2.0, 0.5, 0.25, -1.5]))
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
