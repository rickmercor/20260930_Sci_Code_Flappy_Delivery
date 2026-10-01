"""
Validate a ragged OPES archive and restore the reported bias offset.

OPES reports a shifted flooding bias, so the set barrier must be restored
before rate accelerations are evaluated.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def restore_opes_bias(
    raw_bias: 'np.ndarray',
    lengths: 'np.ndarray',
    events: 'np.ndarray',
    barriers: 'np.ndarray',
    dt: float,
    temperature: float,
) -> 'np.ndarray':
    """Return a finite bias tensor with the OPES BARRIER offset restored.

    Parameters
    ----------
    raw_bias : np.ndarray
        Real numeric array with shape (n_set, n_traj, n_frame), where
        n_set >= 3, n_traj >= 1, and n_frame >= 2. Samples at frame indices
        smaller than the corresponding value in ``lengths`` are active and
        must be finite. Padding samples at frame indices greater than or equal
        to ``lengths`` may be nonfinite.
    lengths : np.ndarray
        Integer array with shape (n_set, n_traj). Every entry must satisfy
        2 <= lengths[i, j] <= n_frame.
    events : np.ndarray
        Real numeric binary array with shape (n_set, n_traj). Every entry
        must be finite and equal to either 0 or 1.
    barriers : np.ndarray
        Real numeric array with shape (n_set,). All values must be finite
        and distinct.
    dt : float
        Finite positive time step.
    temperature : float
        Finite positive temperature.

    Returns
    -------
    np.ndarray
        Float array with the same shape as ``raw_bias``. The barrier for each
        set is added to every active bias sample. Padding samples are replaced
        by zero. All returned values are finite.

    Raises
    ------
    ValueError
        If the archive arrays are not real numeric data with the required
        shapes, if n_set < 3, n_traj < 1, or n_frame < 2, if ``lengths`` is
        not an integer array or contains values outside [2, n_frame], if
        ``events`` contains nonfinite or non-binary values, if ``barriers``
        contains nonfinite or duplicate values, if ``dt`` or ``temperature``
        is not finite and positive, if an active bias sample is nonfinite,
        if the arrays cannot be represented as float64 where required, or if
        restoring the barrier produces a nonfinite float64 result.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_restore_opes_bias(
    raw_bias: 'np.ndarray',
    lengths: 'np.ndarray',
    events: 'np.ndarray',
    barriers: 'np.ndarray',
    dt: float,
    temperature: float,
) -> 'np.ndarray':
    try:
        raw = np.asarray(raw_bias)
        lens = np.asarray(lengths)
        ev = np.asarray(events)
        barr = np.asarray(barriers)
        step = float(dt)
        temp = float(temperature)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("archive inputs must be numeric") from exc
    if (
        raw.ndim != 3 or raw.shape[0] < 3 or raw.shape[1] == 0 or raw.shape[2] < 2
        or lens.shape != raw.shape[:2] or ev.shape != raw.shape[:2]
        or barr.shape != (raw.shape[0],)
    ):
        raise ValueError("incompatible archive shapes")
    if (
        not np.issubdtype(raw.dtype, np.number)
        or not np.issubdtype(lens.dtype, np.integer)
        or not np.issubdtype(ev.dtype, np.number)
        or not np.issubdtype(barr.dtype, np.number)
        or not np.isrealobj(raw) or not np.isrealobj(ev) or not np.isrealobj(barr)
    ):
        raise ValueError("archive arrays must be real numeric data")
    try:
        raw = raw.astype(float, copy=False)
        evf = ev.astype(float, copy=False)
        barr = barr.astype(float, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("archive arrays must fit float64") from exc
    if (
        np.any(lens < 2) or np.any(lens > raw.shape[2])
        or np.any(~np.isfinite(evf)) or np.any((evf != 0.0) & (evf != 1.0))
        or np.any(~np.isfinite(barr)) or len(np.unique(barr)) != len(barr)
        or not np.isfinite(step) or step <= 0.0
        or not np.isfinite(temp) or temp <= 0.0
    ):
        raise ValueError("invalid lengths, events, barriers, dt, or temperature")
    active = np.arange(raw.shape[2])[None, None, :] < lens[:, :, None]
    if np.any(~np.isfinite(raw[active])):
        raise ValueError("active bias samples must be finite")
    restored = np.zeros(raw.shape, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = raw + barr[:, None, None]
    restored[active] = shifted[active]
    if np.any(~np.isfinite(restored)):
        raise ValueError("restored bias exceeded the float64 range")
    return restored

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common = "import numpy as np\nr=np.array([[[-2.,-1.,np.nan],[-1.5,-1.,-.5]],[[-3.,-2.,np.nan],[-2.5,-2.,-1.]] ,[[-4.,-3.,np.nan],[-3.5,-3.,-2.]]])\nl=np.array([[2,3],[2,3],[2,3]])\ne=np.array([[1,0],[1,1],[0,1]])\nb=np.array([3.,5.,7.])"
    return [
        {"setup": common, "call": "restore_opes_bias(r,l,e,b,.1,300.)", "gold_call": "_oracle_restore_opes_bias(r,l,e,b,.1,300.)", "tol": 1e-12},
        {"setup": common, "call": "restore_opes_bias(r,l,e,b,1e-6,1e6)", "gold_call": "_oracle_restore_opes_bias(r,l,e,b,1e-6,1e6)", "tol": 1e-12},
        {"setup": "import numpy as np\nr=np.full((3,1,2),-8e307); l=np.full((3,1),2); e=np.ones((3,1)); b=np.array([8e307,8e307-1e292,8e307-2e292])", "call": "restore_opes_bias(r,l,e,b,1.,300.)", "gold_call": "_oracle_restore_opes_bias(r,l,e,b,1.,300.)", "tol": 1e-12},
        {"setup": "import numpy as np\ndef check(fn):\n out=[]\n base=(np.zeros((3,2,3)),np.full((3,2),2),np.ones((3,2)),np.arange(3.))\n for r,l,e,b in ((base[0],np.zeros((3,2),int),base[2],base[3]),(base[0],base[1],np.full((3,2),2),base[3]),(np.full((3,2,3),np.nan),base[1],base[2],base[3])):\n  try: fn(r,l,e,b,.1,300.)\n  except ValueError: out.append(1)\n  except Exception: out.append(2)\n  else: out.append(0)\n return np.array(out)", "call": "check(restore_opes_bias)", "gold_call": "check(_oracle_restore_opes_bias)", "tol": 0.0},
    ]
