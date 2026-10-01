"""
Compute the flooding acceleration with the source averaging order.

Ragged trajectory ensembles require an explicit distinction between averaging
over simulations first and averaging over each trajectory first.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ensemble_time_log_acceleration(
    restored_bias: 'np.ndarray',
    lengths: 'np.ndarray',
    temperature: float,
    gamma: float,
) -> 'np.ndarray':
    """Return log acceleration factors for ragged trajectory sets.

    At each stored frame, average the exponential only over trajectories still
    present, then average those frame means uniformly over time. Use
    ``R=0.008314462618`` kJ mol^-1 K^-1, matching the pinned implementation,
    and a stable log-domain evaluation. Padding in ``restored_bias`` is ignored
    through ``lengths``.

    Raises
    ------
    ValueError
        If the arrays have incompatible shapes or unsupported dtypes, if any
        length lies outside the available frame range, if ``temperature`` is
        nonfinite or not strictly positive, if ``gamma`` is nonfinite or lies
        outside the closed interval [0, 1], if an active bias sample is
        nonfinite, or if the returned log acceleration is nonfinite.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _eatr_logmeanexp(values: 'np.ndarray') -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.mean(np.exp(values - maximum))))


def _oracle_ensemble_time_log_acceleration(
    restored_bias: 'np.ndarray',
    lengths: 'np.ndarray',
    temperature: float,
    gamma: float,
) -> 'np.ndarray':
    try:
        bias = np.asarray(restored_bias)
        lens = np.asarray(lengths)
        temp = float(temperature)
        quality = float(gamma)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("acceleration inputs must be numeric") from exc
    if bias.ndim != 3 or lens.shape != bias.shape[:2] or bias.shape[0] == 0:
        raise ValueError("incompatible bias and length shapes")
    if (
        not np.issubdtype(bias.dtype, np.number) or not np.isrealobj(bias)
        or not np.issubdtype(lens.dtype, np.integer)
    ):
        raise ValueError("bias must be real numeric and lengths integral")
    bias = bias.astype(float, copy=False)
    if (
        np.any(lens < 1) or np.any(lens > bias.shape[2])
        or not np.isfinite(temp) or temp <= 0.0
        or not np.isfinite(quality) or not 0.0 <= quality <= 1.0
    ):
        raise ValueError("invalid lengths, temperature, or gamma")
    beta = 1.0 / (0.008314462618 * temp)
    output = np.empty(bias.shape[0], dtype=float)
    for set_index in range(bias.shape[0]):
        frame_logs = []
        for frame in range(int(np.max(lens[set_index]))):
            running = lens[set_index] > frame
            active = bias[set_index, running, frame]
            if np.any(~np.isfinite(active)):
                raise ValueError("active restored bias samples must be finite")
            frame_logs.append(_eatr_logmeanexp(beta * quality * active))
        output[set_index] = _eatr_logmeanexp(np.asarray(frame_logs))
    if np.any(~np.isfinite(output)):
        raise ValueError("log acceleration is not finite")
    return output

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nv=np.array([[[1.,2.,0.],[2.,3.,4.]],[[0.,1.,2.],[1.,2.,0.]],[[2.,2.,2.],[3.,3.,3.]]]); l=np.array([[2,3],[3,2],[3,3]])", "call": "ensemble_time_log_acceleration(v,l,300.,.6)", "gold_call": "_oracle_ensemble_time_log_acceleration(v,l,300.,.6)", "tol": 1e-12},
        {"setup": "import numpy as np\nv=np.arange(18,dtype=float).reshape(3,2,3); l=np.full((3,2),3)", "call": "ensemble_time_log_acceleration(v,l,300.,0.)", "gold_call": "_oracle_ensemble_time_log_acceleration(v,l,300.,0.)", "tol": 1e-12},
        {"setup": "import numpy as np\nv=np.full((3,2,2),1e307); l=np.full((3,2),2)", "call": "ensemble_time_log_acceleration(v,l,1e307,1.)", "gold_call": "_oracle_ensemble_time_log_acceleration(v,l,1e307,1.)", "tol": 1e-12},
        {"setup": "import numpy as np\ndef check(fn):\n out=[]\n for v,l,t,g in ((np.zeros((3,2,2)),np.full((3,2),3),300.,.5),(np.full((3,2,2),np.nan),np.full((3,2),2),300.,.5),(np.zeros((3,2,2)),np.full((3,2),2),300.,1.1)):\n  try: fn(v,l,t,g)\n  except ValueError: out.append(1)\n  except Exception: out.append(2)\n  else: out.append(0)\n return np.array(out)", "call": "check(ensemble_time_log_acceleration)", "gold_call": "check(_oracle_ensemble_time_log_acceleration)", "tol": 0.0},
    ]
