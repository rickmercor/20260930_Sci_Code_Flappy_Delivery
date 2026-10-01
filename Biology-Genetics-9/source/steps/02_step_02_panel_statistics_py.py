"""
Panel and switching evidence

Panel labels identify the reference panel that supplies each composite state allele at the current marker. From the state and adjacent marker posterior evidence, obtain the panel copying probabilities and aligned dimensionless switching increments.

Inputs gamma and same are aligned, finite, nonnegative arrays with shape (M,H), where H is at least 2. Every row of gamma must sum to one. Every row of same must sum to at most one. panels is an aligned integer array with values from 0 through npanels minus 1, and npanels is an integer greater than or equal to 1.

Return the panel probability array with shape (M,npanels) and the switching increment array with shape (M,). The switching increment at marker 0 is zero.

Raise ValueError if any input is outside the stated domain.

Returns
-------
Tuple: panel probabilities (M,J), switching increments (M,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def panel_statistics(gamma: np.ndarray, same: np.ndarray, panels: np.ndarray, npanels: \
    int) -> tuple[np.ndarray, np.ndarray]:
    (
        'Tuple: panel probabilities (M,J), switching increments (M,'
        '). Raises ValueError for inputs outside the stated domain.'
    )
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np
from scipy.special import logsumexp

def _array(value, ndim=None):
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('numeric input required') from exc
    if not np.isfinite(a).all() or a.size == 0:
        raise ValueError('finite nonempty input required')
    if ndim is not None and a.ndim != ndim:
        raise ValueError('wrong array dimension')
    return a

def _require(condition):
    if not condition:
        raise ValueError('input outside the declared domain')

def _oracle_panel_statistics(gamma: np.ndarray, same: np.ndarray, panels: np.ndarray, \
    npanels: int) -> tuple[np.ndarray, np.ndarray]:
    import math
    import numpy as np
    from scipy.special import logsumexp
    gamma = _array(gamma, 2); same = _array(same, 2)
    j = _array(panels, 2)
    npanels = float(_array(npanels, 0))
    if not (gamma.shape == same.shape == j.shape and gamma.shape[1] >= 2):
        raise ValueError('input outside the declared domain')
    if not (int(npanels) == npanels and npanels >= 1):
        raise ValueError('input outside the declared domain')
    if not (np.all(j == np.floor(j)) and np.all((j >= 0) & (j < npanels))):
        raise ValueError('input outside the declared domain')
    if not (np.all(gamma >= 0) and np.allclose(gamma.sum(1), 1, atol=1e-9)):
        raise ValueError('input outside the declared domain')
    if not (np.all(same >= 0) and np.all(same.sum(1) <= 1+1e-9)):
        raise ValueError('input outside the declared domain')
    npanels = int(npanels)

    panels = np.asarray(panels,int)
    gamma, same = np.asarray(gamma,float), np.asarray(same,float)
    p = np.zeros((len(gamma),npanels))
    for m in range(len(gamma)):
        for h in range(gamma.shape[1]):
            p[m,panels[m,h]] += gamma[m,h]
    tau = gamma.shape[1]/(gamma.shape[1]-1.0)*(1.0-same.sum(axis=1))
    tau[0] = 0.
    return p, tau

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def _pack(value):
    values = value if isinstance(value, tuple) else (value,)
    pieces = []
    for v in values:
        a = np.asarray(v, dtype=float)
        pieces.append(np.r_[float(a.ndim), a.shape, a.ravel()])
    return np.concatenate(pieces)

def _error_code(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0

def test_cases():
    return [{'setup': (
        'import numpy as np\nga=np.array([[.2,.3,.5],[.5,.1,.4]]); s'
        'a=np.array([[.2,.3,.5],[.1,.01,.2]]); j=np.array([[0,1,2],[2,0,1]])'
    ), 'call': '_pack(panel_statistics(ga,sa,j,3))', 'gold_call': \
        '_pack(_oracle_panel_statistics(ga,sa,j,3))'}, {'setup': (
        'import numpy as np\nga=np.array([[.5,.5]]); sa=ga.copy(); j'
        '=np.array([[0,0]])'
    ), 'call': '_pack(panel_statistics(ga,sa,j,1))', 'gold_call': \
        '_pack(_oracle_panel_statistics(ga,sa,j,1))'}, {'setup': (
        'import numpy as np\nga=np.array([[.1,.2,.7],[.3,.2,.5],[.6,'
        '.1,.3]]); sa=np.array([[.1,.2,.7],[0.,0.,0.],[.1,.05,.03]]'
        '); j=np.array([[2,1,0],[0,1,2],[1,1,1]])'
    ), 'call': '_pack(panel_statistics(ga,sa,j,4))', 'gold_call': \
        '_pack(_oracle_panel_statistics(ga,sa,j,4))'}, {'setup': (
        'import numpy as np\nga=np.array([[.1,.9]]); sa=ga.copy(); j'
        '=np.array([[0,2]])'
    ), 'call': '_error_code(panel_statistics,ga,sa,j,2)', 'gold_call': \
        '_error_code(_oracle_panel_statistics,ga,sa,j,2)'}]
