"""
Replay aligned symmetric-proposal searches and return the earliest minimum-energy state visited by every run.

Replay the matching treatment's random-walk acceptance rule on a finite state space with symmetric proposals. energies has finite shape (K,N), temperatures finite positive shape (K,), and move_maps integer shape (M,N); every map is an involution on 0 through N-1. moves is an integer array of shape (R,H) indexing those maps; uniforms has the same shape with values in [0,1). K,N,M,R are positive and H may be zero. start is an integer state index. Each run starts at start; each column is one attempted move. Compare an uphill acceptance probability to the aligned uniform with a strict inequality. The recorded run representative is the earliest visited state attaining that run's minimum energy, including the initial state; rejected proposals are not visited. Return integer state indices with shape (K,R). Invalid arrays or indices raise ValueError. Do not mutate inputs.

Returns
-------
run_minima : np.ndarray — integer array of shape (K, R) containing state indices.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def replay_state_search(
    energies: np.ndarray,
    temperatures: np.ndarray,
    move_maps: np.ndarray,
    moves: np.ndarray,
    uniforms: np.ndarray,
    start: int,
) -> np.ndarray:
    """Return each setup and run's earliest visited minimum-energy state.
 
    Parameters
    ----------
    energies
        Finite setup-by-state energy array of shape (K, N).
    temperatures
        Positive setup temperatures of shape (K,).
    move_maps, moves, uniforms, start
        Valid involutive maps, aligned replay inputs, and start-state index.
 
    Returns
    -------
    np.ndarray
        Integer state-index array of shape (K, R).
    """
    return run_minima

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_replay_state_search(
    energies: np.ndarray,
    temperatures: np.ndarray,
    move_maps: np.ndarray,
    moves: np.ndarray,
    uniforms: np.ndarray,
    start: int,
) -> np.ndarray:
    import numpy as np
    e=np.asarray(energies,dtype=float); temp=np.asarray(temperatures,dtype=float)
    raw_maps=np.asarray(move_maps); raw_moves=np.asarray(moves)
    u=np.asarray(uniforms,dtype=float)
    if e.ndim!=2 or not e.size or temp.shape!=(e.shape[0],) or raw_maps.ndim!=2 or raw_maps.shape[0]<1 or raw_maps.shape[1]!=e.shape[1] or raw_moves.ndim!=2 or not raw_moves.shape[0] or u.shape!=raw_moves.shape:
        raise ValueError('invalid search array shapes')
    if not np.issubdtype(raw_maps.dtype,np.integer) or not np.issubdtype(raw_moves.dtype,np.integer):
        raise ValueError('integer maps and move indices required')
    maps=raw_maps.astype(int); m=raw_moves.astype(int)
    if not all(np.all(np.isfinite(a)) for a in [e,temp,u]) or np.any(temp<=0) or np.any((u<0)|(u>=1)):
        raise ValueError('invalid temperature or uniforms')
    if np.any((maps<0)|(maps>=e.shape[1])) or np.any((m<0)|(m>=len(maps))) or not isinstance(start,(int,np.integer)) or not 0<=start<e.shape[1]:
        raise ValueError('invalid state or move index')
    if any(not np.array_equal(row[row],np.arange(e.shape[1])) for row in maps):
        raise ValueError('move maps must be involutions')
    out=np.empty((len(e),len(m)),dtype=int)
    for k in range(len(e)):
        for r in range(len(m)):
            cur=best=int(start)
            for t in range(m.shape[1]):
                prop=int(maps[m[r,t],cur]); delta=e[k,prop]-e[k,cur]
                if delta<=0 or u[r,t]<np.exp(-delta/temp[k]): cur=prop
                if e[k,cur]<e[k,best]: best=cur
            out[k,r]=best
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'e=np.array([[3.,1.,-2.,0.],[0.,-1.,2.,4.]]);t=np.array([1.,2.]);maps=np.array([[1,0,3,2],[0,2,1,3]]);m=np.array([[0,1,0,1],[1,0,1,0]]);u=np.array([[.2,.7,.1,.9],[.4,.3,.5,.1]])\n',
      'call': 'replay_state_search(e,t,maps,m,u,0)',
      'gold_call': '_oracle_replay_state_search(e,t,maps,m,u,0)'},
     {'setup': 'import numpy as np\n'
               'e=np.zeros((1,2));t=np.ones(1);maps=np.array([[1,0]]);m=np.zeros((2,0),dtype=int);u=np.empty((2,0))\n',
      'call': 'replay_state_search(e,t,maps,m,u,1)',
      'gold_call': '_oracle_replay_state_search(e,t,maps,m,u,1)'},
     {'setup': 'import numpy as np\n'
               'e=np.zeros((1,2));t=np.ones(1);maps=np.array([[1,0]]);m=np.zeros((2,3),dtype=int);u=np.zeros((2,3))\n',
      'call': 'replay_state_search(e,t,maps,m,u,1)',
      'gold_call': '_oracle_replay_state_search(e,t,maps,m,u,1)'},
     {'setup': 'import numpy as np\n'
               'e=np.array([[0.,1.,-1.]]);t=np.ones(1);maps=np.array([[1,0,2],[0,2,1]]);m=np.array([[0,1],[0,1]]);u=np.array([[np.exp(-1.),0.],[0.,0.]])\n',
      'call': 'replay_state_search(e,t,maps,m,u,0)',
      'gold_call': '_oracle_replay_state_search(e,t,maps,m,u,0)'},
     {'setup': 'import numpy as np\n'
               'e=np.zeros((1,3));t=np.ones(1);maps=np.array([[1,2,0]]);m=np.zeros((1,1),dtype=int);u=np.zeros((1,1))\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        replay_state_search(e,t,maps,m,u,0)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_replay_state_search(e,t,maps,m,u,0)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'}]
