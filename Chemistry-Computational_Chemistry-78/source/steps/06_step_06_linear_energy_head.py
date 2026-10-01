"""
Evaluate the paper's linear reaction-energy head.

Map the supplied graph embedding to one scalar using the frozen
linear weights. Dropout is not applied. Do not run a new DFT job.

Returns
-------
return energy
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def linear_energy_head(
    graph_embedding: np.ndarray,
    linear_weights: dict,
) -> float:
    """
    Evaluate the linear energy head.

    Parameters
    ----------
    graph_embedding : np.ndarray
        Pooled embedding with shape (8,).
    linear_weights : dict
        Frozen tensors W with shape (1, 8) and b with shape (1,).

    Returns
    -------
    float
        Predicted reaction energy in eV.

    Raises
    ------
    ValueError
        If the embedding length or linear weights are incompatible, or
        if values are non-finite.
    """
    return energy

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_linear_energy_head(
    graph_embedding: np.ndarray,
    linear_weights: dict,
) -> float:
    hidden = 8
    z = np.asarray(graph_embedding, dtype=float).reshape(-1)
    if z.shape[0] != hidden:
        raise ValueError("graph_embedding must have length 8")
    if not np.all(np.isfinite(z)):
        raise ValueError("graph_embedding contains non-finite values")
    w = np.asarray(linear_weights["W"], dtype=float)
    b = np.asarray(linear_weights["b"], dtype=float).reshape(-1)
    if w.shape != (1, hidden) or b.shape != (1,):
        raise ValueError("linear head has incompatible weights")
    return float((z @ w.T + b).reshape(-1)[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np

graph_embedding = np.array([0.2, -0.1, 0.4, 0.0, 0.3, -0.2, 0.1, 0.5])
linear_weights = {
    "W": np.array([[0.1, -0.2, 0.3, 0.0, -0.1, 0.2, 0.4, -0.3]]),
    "b": np.array([0.05]),
}

def run_model():
    return linear_energy_head(
        np.array(graph_embedding, copy=True),
        {"W": np.array(linear_weights["W"], copy=True), "b": np.array(linear_weights["b"], copy=True)},
    )

def run_gold():
    return _oracle_linear_energy_head(
        np.array(graph_embedding, copy=True),
        {"W": np.array(linear_weights["W"], copy=True), "b": np.array(linear_weights["b"], copy=True)},
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

graph_embedding = np.ones(7)
linear_weights = {
    "W": np.ones((1, 8)),
    "b": np.zeros(1),
}

def run_model():
    try:
        linear_energy_head(
            np.array(graph_embedding, copy=True),
            {"W": np.array(linear_weights["W"], copy=True), "b": np.array(linear_weights["b"], copy=True)},
        )
        return 0
    except ValueError:
        return 1

def run_gold():
    try:
        _oracle_linear_energy_head(
            np.array(graph_embedding, copy=True),
            {"W": np.array(linear_weights["W"], copy=True), "b": np.array(linear_weights["b"], copy=True)},
        )
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

graph_embedding = np.linspace(-0.4, 0.3, 8)
linear_weights = {
    "W": np.linspace(0.2, -0.2, 8).reshape(1, 8),
    "b": np.array([-0.15]),
}

def run_model():
    return linear_energy_head(
        np.array(graph_embedding, copy=True),
        {"W": np.array(linear_weights["W"], copy=True), "b": np.array(linear_weights["b"], copy=True)},
    )

def run_gold():
    return _oracle_linear_energy_head(
        np.array(graph_embedding, copy=True),
        {"W": np.array(linear_weights["W"], copy=True), "b": np.array(linear_weights["b"], copy=True)},
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

graph_embedding = np.linspace(-0.4, 0.3, 8)
linear_weights = {
    "W": np.linspace(0.2, -0.2, 8).reshape(1, 8),
    "b": np.array([0.25]),
}

def run_model():
    return linear_energy_head(
        np.array(graph_embedding, copy=True),
        {"W": np.array(linear_weights["W"], copy=True), "b": np.array(linear_weights["b"], copy=True)},
    )

def run_gold():
    return _oracle_linear_energy_head(
        np.array(graph_embedding, copy=True),
        {"W": np.array(linear_weights["W"], copy=True), "b": np.array(linear_weights["b"], copy=True)},
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

graph_embedding = np.ones(8)
linear_weights = {
    "W": np.ones((8, 1)),
    "b": np.zeros(1),
}

def run_model():
    try:
        linear_energy_head(
            np.array(graph_embedding, copy=True),
            {"W": np.array(linear_weights["W"], copy=True), "b": np.array(linear_weights["b"], copy=True)},
        )
        return 0
    except ValueError:
        return 1

def run_gold():
    try:
        _oracle_linear_energy_head(
            np.array(graph_embedding, copy=True),
            {"W": np.array(linear_weights["W"], copy=True), "b": np.array(linear_weights["b"], copy=True)},
        )
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
