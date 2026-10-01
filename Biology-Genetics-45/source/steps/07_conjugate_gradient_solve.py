"""
Solve the leave-one-chromosome-out system against the standardised phenotype and

return the solution together with the number of iterations spent on it.

The prediction that corrects a trait for polygenic background is proportional to

the inverse of the phenotypic covariance applied to the phenotype. At the sample

sizes these methods target that covariance is neither inverted nor assembled,

and the system is not carried to convergence. Starting from a zero solution, the

vector returned after k iterations is the one that makes the covariance-weighted

distance to the exact solution as small as it can be over the subspace spanned

by the right-hand side and its first k - 1 images under the covariance. Iterations

stop early once the Euclidean norm of the residual has fallen to the stated

fraction of the norm of the right-hand side, and otherwise continue to the

stated budget. The budget is what binds in practice, so the vector returned is a

partial solve, and the number of iterations behind it is part of what the

method reports.

Returns
-------
tuple (np.ndarray of shape (n,) float64, int number of iterations performed)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def conjugate_gradient_solve(
    blocks: list,
    focal_index: int,
    sigma_g_squared: float,
    sigma_e_squared: float,
    phenotype: "np.ndarray",
    tolerance: float = 1e-5,
    max_iterations: int = 6,
) -> tuple:
    """Solve the leave-one-chromosome-out system against the phenotype.

    Args:
        blocks: list of arrays, one per chromosome, each of shape (n, p_t)
            holding that chromosome's centred, frequency-scaled genotypes.
        focal_index: index into blocks of the chromosome under test, excluded
            from the covariance.
        sigma_g_squared: the genetic variance component.
        sigma_e_squared: the residual variance component.
        phenotype: array of shape (n,) of raw phenotype values, standardised
            here to zero mean and y^T y = n before it becomes the right-hand
            side of the system.
        tolerance: iterations stop once the Euclidean norm of the residual has
            fallen to this fraction of the norm of the right-hand side.
        max_iterations: the largest number of iterations that may be taken, which is
            the dimension of the subspace the returned vector is optimal over
            when the tolerance is not reached first.

    Returns:
        tuple (solution, iterations) where solution is an (n,) float array and
        iterations is the native int count of iterations actually taken.

    Raises:
        ValueError: if the phenotype length does not match the blocks, if the
            phenotype has zero variance, if tolerance is not positive, or if
            max_iterations is below 1.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_conjugate_gradient_solve(
    blocks: list,
    focal_index: int,
    sigma_g_squared: float,
    sigma_e_squared: float,
    phenotype: "np.ndarray",
    tolerance: float = 1e-5,
    max_iterations: int = 6,
) -> tuple:
    """Reference implementation."""
    matrices = [np.asarray(block, dtype=float) for block in blocks]
    if len(matrices) == 0:
        raise ValueError("at least one genotype block is required")
    focal_index = int(focal_index)
    if not 0 <= focal_index < len(matrices):
        raise ValueError("focal_index is out of range")
    n_individuals = matrices[0].shape[0]
    trait = np.asarray(phenotype, dtype=float).ravel()
    if trait.size != n_individuals:
        raise ValueError("phenotype length does not match the genotype blocks")
    if float(tolerance) <= 0.0:
        raise ValueError("tolerance must be strictly positive")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be at least 1")
    centred = trait - trait.mean()
    scale = float(centred @ centred)
    if scale <= 0.0:
        raise ValueError("phenotype has zero variance")
    right_hand_side = centred * np.sqrt(n_individuals / scale)

    retained = sum(block.shape[1] for index, block in enumerate(matrices)
                   if index != focal_index)
    if retained == 0:
        raise ValueError("no variants remain once the focal chromosome is dropped")
    pooled = sum(float(np.trace(block @ block.T)) for block in matrices)
    if pooled <= 0.0:
        raise ValueError("the genotype blocks carry no variance")
    scale = pooled / n_individuals
    weight = float(sigma_g_squared) / scale

    def _apply(vector):
        accumulated = np.zeros_like(vector)
        for index, block in enumerate(matrices):
            if index == focal_index:
                continue
            accumulated = accumulated + block @ (block.T @ vector)
        return float(sigma_e_squared) * vector + weight * accumulated

    solution = np.zeros(n_individuals, dtype=float)
    residual = right_hand_side.copy()
    direction = right_hand_side.copy()
    delta_new = float(residual @ residual)
    threshold = float(tolerance) * float(np.linalg.norm(right_hand_side))
    iterations = 0
    for _ in range(int(max_iterations)):
        product = _apply(direction)
        step = delta_new / float(direction @ product)
        solution = solution + step * direction
        residual = residual - step * product
        iterations += 1
        delta_old, delta_new = delta_new, float(residual @ residual)
        if np.sqrt(delta_new) <= threshold:
            break
        direction = residual + (delta_new / delta_old) * direction
    return solution, int(iterations)

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

def pack(r):
    return np.concatenate([
        np.round(np.asarray(r[0], dtype=float).ravel(), 10),
        np.asarray([r[1]], dtype=float),
    ])
""",
            "call": "pack(conjugate_gradient_solve(blocks, 0, 0.93, 0.07, trait, 1e-5, 3))",
            "gold_call": "pack(_oracle_conjugate_gradient_solve(blocks, 0, 0.93, 0.07, trait, 1e-5, 3))",
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

def pack(r):
    return np.concatenate([
        np.round(np.asarray(r[0], dtype=float).ravel(), 10),
        np.asarray([r[1]], dtype=float),
    ])
""",
            "call": "pack(conjugate_gradient_solve(blocks, 0, 0.93, 0.07, trait, 0.05, 20))",
            "gold_call": "pack(_oracle_conjugate_gradient_solve(blocks, 0, 0.93, 0.07, trait, 0.05, 20))",
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

def pack(r):
    return np.concatenate([
        np.round(np.asarray(r[0], dtype=float).ravel(), 10),
        np.asarray([r[1]], dtype=float),
    ])
""",
            "call": "pack(conjugate_gradient_solve(blocks, 2, 0.5, 0.5, trait, 1e-9, 6))",
            "gold_call": "pack(_oracle_conjugate_gradient_solve(blocks, 2, 0.5, 0.5, trait, 1e-9, 6))",
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
        conjugate_gradient_solve(blocks, 0, 0.93, 0.07, trait, 1e-5, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_oracle():
    try:
        _oracle_conjugate_gradient_solve(blocks, 0, 0.93, 0.07, trait, 1e-5, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
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
        conjugate_gradient_solve(blocks, 0, 0.93, 0.07, trait, 0.0, 6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_oracle():
    try:
        _oracle_conjugate_gradient_solve(blocks, 0, 0.93, 0.07, trait, 0.0, 6)
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
