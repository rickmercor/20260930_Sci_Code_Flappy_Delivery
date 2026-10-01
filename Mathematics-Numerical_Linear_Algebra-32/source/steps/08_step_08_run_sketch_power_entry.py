"""
Orchestrator. Call steps 1-7 in order: construct A, draw the three test matrices, form (Y, W, Z), amplify the rangefinder for q steps, take thin factors, recover B from the sketched corange, and return the requested entry of the rank-r reconstruction. This is the scalar required by the prompt.

The end-to-end method is a one-pass rangefinder amplified by a wider sketch, followed by a sketched corange solve and rank truncation. On the prompt instance (12-by-10, seeds 7 and 11, widths 4/7/8, q=1, rank 3, entry (2, 0)) this recovers 0.18187566628973476.

Returns
-------
native Python float: the requested reconstruction entry. For the prompt instance this is 0.18187566628973476
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def run_sketch_power_entry(
    m: int,
    n: int,
    s: np.ndarray,
    data_seed: int,
    sketch_seed: int,
    s_width: int,
    d: int,
    l: int,
    q: int,
    rank: int,
    row: int,
    col: int,
) -> float:
    """End-to-end one-pass amplified reconstruction; return one entry."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import importlib.util
import sys
from pathlib import Path

import numpy as np

def _step_roots():
    roots = []
    try:
        roots.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    roots.append(Path.cwd())
    for entry in list(sys.path):
        if entry:
            roots.append(Path(entry))
    out = []
    seen = set()
    for root in roots:
        try:
            key = str(root.resolve())
        except OSError:
            continue
        if key in seen or not root.is_dir():
            continue
        seen.add(key)
        out.append(root)
    return out


def _load_step(key):
    """Load a sibling step module whose filename ends with ``{key}.py``."""
    for root in _step_roots():
        matches = sorted(root.glob(f"*{key}.py"))
        if not matches:
            continue
        path = matches[0]
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    raise ImportError(f"could not load sub-problem {key}")


def _bind_oracle(public_name):
    g = globals()
    oracle_name = "_oracle_" + public_name
    if oracle_name in g and callable(g[oracle_name]):
        return g[oracle_name]
    if public_name in g and callable(g[public_name]) and public_name != "run_sketch_power_entry":
        fn = g[public_name]
        try:
            mod = _load_step(public_name)
            return getattr(mod, oracle_name, fn)
        except ImportError:
            return fn
    mod = _load_step(public_name)
    return getattr(mod, oracle_name)


construct_flat_spectrum_matrix = _bind_oracle("construct_flat_spectrum_matrix")
draw_gaussian_test_matrices = _bind_oracle("draw_gaussian_test_matrices")
form_one_pass_sketches = _bind_oracle("form_one_pass_sketches")
amplify_rangefinder = _bind_oracle("amplify_rangefinder")
rangefinder_thin_factors = _bind_oracle("rangefinder_thin_factors")
sketched_coefficient_matrix = _bind_oracle("sketched_coefficient_matrix")
reconstruction_entry = _bind_oracle("reconstruction_entry")


def _unpack_three(pack):
    pack = np.asarray(pack, dtype=float).reshape(-1)
    if pack.size < 6:
        raise ValueError("pack is too short")
    mA, nA, mB, nB, mC, nC = [int(round(float(x))) for x in pack[:6]]
    if min(mA, nA, mB, nB, mC, nC) < 1:
        raise ValueError("packed shapes must be positive")
    need = 6 + mA * nA + mB * nB + mC * nC
    if pack.size != need:
        raise ValueError("pack length does not match header")
    i = 6
    A = pack[i : i + mA * nA].reshape(mA, nA)
    i += mA * nA
    B = pack[i : i + mB * nB].reshape(mB, nB)
    i += mB * nB
    C = pack[i:].reshape(mC, nC)
    return A, B, C


def _oracle_run_sketch_power_entry(
    m, n, s, data_seed, sketch_seed, s_width, d, l, q, rank, row, col
):
    if not isinstance(q, (int, np.integer)):
        raise ValueError("q must be an integer")
    q = int(q)
    if q < 1:
        raise ValueError("require q >= 1")
    A = construct_flat_spectrum_matrix(m, n, s, data_seed)
    test_pack = draw_gaussian_test_matrices(m, n, s_width, d, l, sketch_seed)
    _, Psi, _ = _unpack_three(test_pack)
    sketch_pack = form_one_pass_sketches(A, test_pack)
    Y, W, Z = _unpack_three(sketch_pack)
    Yhat = amplify_rangefinder(Y, Z, q)
    stacked = rangefinder_thin_factors(Yhat)
    Q = stacked[: A.shape[0]]
    B = sketched_coefficient_matrix(Q, Psi, W)
    return reconstruction_entry(Q, B, rank, row, col)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
m, n, s_width, d, l, q, rank, row, col = 12, 10, 4, 7, 8, 1, 3, 2, 0
s = np.array([4.0, 3.2, 2.6, 2.1, 1.7, 1.4, 1.15, 0.95, 0.8, 0.7])
data_seed, sketch_seed = 7, 11
""",
            "call": "run_sketch_power_entry(m, n, s, data_seed, sketch_seed, s_width, d, l, q, rank, row, col)",
            "gold_call": "_oracle_run_sketch_power_entry(m, n, s, data_seed, sketch_seed, s_width, d, l, q, rank, row, col)",
        },
        {
            "setup": """import numpy as np
m, n, s_width, d, l, q, rank, row, col = 6, 4, 2, 4, 3, 1, 2, 1, 0
s = np.array([5.0, 2.0, 0.8, 0.3])
data_seed, sketch_seed = 1, 4
""",
            "call": "run_sketch_power_entry(m, n, s, data_seed, sketch_seed, s_width, d, l, q, rank, row, col)",
            "gold_call": "_oracle_run_sketch_power_entry(m, n, s, data_seed, sketch_seed, s_width, d, l, q, rank, row, col)",
        },
        {
            "setup": """import numpy as np
m, n, s_width, d, l, q, rank, row, col = 4, 3, 1, 2, 2, 1, 1, 0, 0
s = np.array([3.0, 1.5, 0.4])
data_seed, sketch_seed = 0, 2
""",
            "call": "run_sketch_power_entry(m, n, s, data_seed, sketch_seed, s_width, d, l, q, rank, row, col)",
            "gold_call": "_oracle_run_sketch_power_entry(m, n, s, data_seed, sketch_seed, s_width, d, l, q, rank, row, col)",
        },
        {
            "setup": """import numpy as np
s = np.array([2.0, 1.0])
def run_model():
    try:
        run_sketch_power_entry(3, 2, s, 0, 1, 1, 2, 2, 0, 1, 0, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_sketch_power_entry(3, 2, s, 0, 1, 1, 2, 2, 0, 1, 0, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
s = np.array([2.0, 0.0])
def run_model():
    try:
        run_sketch_power_entry(2, 2, s, 0, 1, 1, 2, 2, 1, 1, 0, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_sketch_power_entry(2, 2, s, 0, 1, 1, 2, 2, 1, 1, 0, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
