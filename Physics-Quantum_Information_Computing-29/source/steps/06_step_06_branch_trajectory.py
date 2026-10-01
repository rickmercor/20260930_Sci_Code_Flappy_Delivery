"""
Return the retained population of a finite predictive-reliability branching decoder. Initialize fault-to-detector messages to H times the prior vector, run masked_run(H,prior,syndrome,messages,all_unmasked,initial_iters) and select the initial node with path_statistics(). Each round expands retained paths in their current order, assigning their selected node zero then one. Each child copies its parent's final messages, extends its own fixed mask and runs masked_run() for inner_iters. Its next node and score come from path_statistics() on that call's accumulator and actual iterations. Converged children stay eligible. Rank children with rank_paths() and keep width entries. Track distinct successful full corrections. Return (scores,paths,initial_node,leading_iterations,leading_sum,result_count), with scores descending and paths shaped (population,rounds,2). H and prior satisfy masked_run's domain; H must have at least rounds+2 ones per row and n>rounds. rounds is a nonboolean integer in [1,6], width in [1,8] and both iteration budgets are nonboolean integer-valued scalars in [1,32]. These bounds allow at most 63 conditional calls, so a 64-distinct-solution collection limit never terminates this benchmark before its round limit. Invalid domains or upstream magnitude failures raise ValueError. Do not mutate inputs.

The selected conditional histories are neither an exhaustive search nor a minimum-weight certificate. Warm messages affect the future trajectory, while reliability sums reset at each child call. Pruning by the current-run normalized predictive score couples inference, early stopping and the retained search population.

Returns
-------
tuple, numeric retained-population diagnostics
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def branch_trajectory(H, prior, syndrome, initial_iters=12, inner_iters=8, rounds=6, width=4):
    """Run the bounded predictive branching trajectory.

    Parameters
    ----------
    H : array_like
        Binary detector matrix with row degree at least rounds+2.
    prior : array_like
        Real log odds within the conditional-run domain.
    syndrome : array_like
        Binary original detector outcomes.
    initial_iters : int
        Root iteration budget in [1,32].
    inner_iters : int
        Child iteration budget in [1,32].
    rounds : int
        Nonboolean number of branching rounds in [1,6].
    width : int
        Nonboolean retained width in [1,8].

    Returns
    -------
    tuple
        Scores, path pairs, initial node, leading run count, leading absolute
        reliability sum and distinct correction count.

    Raises
    ------
    ValueError
        If any stated domain or upstream calculation bound fails.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_branch_trajectory(H, prior, syndrome, initial_iters=12, inner_iters=8, rounds=6, width=4):
    for x,hi in ((rounds,6),(width,8)):
        if isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) or not 1<=x<=hi:raise ValueError('search budget')
    for x in (initial_iters,inner_iters):
        a=np.asarray(x)
        if a.ndim!=0 or a.dtype.kind not in 'iuf' or not np.isfinite(a) or a!=np.floor(a) or not 1<=a<=32:raise ValueError('iteration budget')
    h=np.asarray(H);l=np.asarray(prior)
    if h.ndim!=2 or h.dtype.kind not in 'biufc' or np.any(h.imag!=0) or not np.all(np.isfinite(h)) or np.any((h!=0)&(h!=1)):raise ValueError('matrix')
    h=h.real.astype(int)
    if h.shape[1]<=rounds or np.any(h.sum(axis=1)<rounds+2):raise ValueError('degree guard')
    if l.shape!=(h.shape[1],) or l.dtype.kind not in 'biufc' or np.any(l.imag!=0) or not np.all(np.isfinite(l)) or np.any(np.abs(l)>16):raise ValueError('prior')
    l=l.real.astype(float);fixed=np.full(len(l),-1,dtype=int)
    root=_oracle_masked_run(h,l,syndrome,h*l,fixed,initial_iters)
    j,_,_=_oracle_path_statistics(root[1],fixed,root[2])
    initial_node=j
    beam=[(root,fixed,(),j,0.,0.)]
    results={tuple(root[3])} if root[4] else set()
    for depth in range(rounds):
        children=[]
        for state,mask,path,nxt,score,A in beam:
            for value in (0,1):
                f=mask.copy();f[nxt]=value
                out=_oracle_masked_run(h,l,syndrome,state[0],f,inner_iters)
                node,sc,total=_oracle_path_statistics(out[1],f,out[2])
                if out[4]:results.add(tuple(out[3]))
                children.append((out,f,path+((nxt,value),),node,sc,total))
        indices=_oracle_rank_paths(np.array([x[4] for x in children]),np.array([x[2] for x in children],dtype=int),width)
        beam=[children[int(i)] for i in indices]
    return np.array([x[4] for x in beam]),np.array([x[2] for x in beam],dtype=int),int(initial_node),int(beam[0][0][2]),float(beam[0][5]),int(len(results))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\nH=np.array([[1,1,1,1,0],[0,1,1,1,1]]);L=np.array([1.,-.5,2.,.3,.75]);s=np.array([1,0])',
            'call': 'branch_trajectory(H.copy(),L.copy(),s.copy(),3,4,2,2)',
            'gold_call': '_oracle_branch_trajectory(H.copy(),L.copy(),s.copy(),3,4,2,2)',
        },
        {
            'setup': 'import numpy as np\nH=np.ones((1,3),int);L=np.array([.25,1.,2.]);s=np.array([1])',
            'call': 'branch_trajectory(H.copy(),L.copy(),s.copy(),1,1,1,1)',
            'gold_call': '_oracle_branch_trajectory(H.copy(),L.copy(),s.copy(),1,1,1,1)',
        },
        {
            'setup': 'import numpy as np\nH=np.array([[1,1,1,1,1,0],[0,1,1,1,1,1],[1,0,1,1,1,1]]);L=np.array([1.,-.25,.5,2.,-.75,1.25]);s=np.array([1,0,1])',
            'call': 'branch_trajectory(H.copy(),L.copy(),s.copy(),5,4,3,4)',
            'gold_call': '_oracle_branch_trajectory(H.copy(),L.copy(),s.copy(),5,4,3,4)',
        },
        {
            'setup': 'import numpy as np\ndef run_model():\n    try: branch_trajectory([[1,1,1]],[1,2,3],[0],1,1,2,1)\n    except ValueError: return 1\n    except Exception: return 2\n    return 0\ndef run_gold():\n    try: _oracle_branch_trajectory([[1,1,1]],[1,2,3],[0],1,1,2,1)\n    except ValueError: return 1\n    except Exception: return 2\n    return 0',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
    ]
