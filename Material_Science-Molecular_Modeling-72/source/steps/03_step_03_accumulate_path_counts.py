"""
Accumulate the raw conditional path-weighted transition counts and their response.

Each ensemble contributes every contiguous window of lag integration steps, including overlaps. Windows are confined to their own recorded ensemble. Microstates are position intervals separated by cuts; an exact cut belongs to the interval on its right. Use the empirical starting ensemble in the separated-density kinetic estimator. The one-step log density increments define the conditional probability of each full path.

Returns
-------
counts : np.ndarray: Float shape (2,L,L). Entry 0 contains raw sums of conditional path weights; entry 1 their alpha derivatives. Rows are starting states and columns are endpoint states, both ordered from left to right. Units are 1 and inverse energy. Each recorded path instance has unit multiplicity before conditional path weighting.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def accumulate_path_counts(records: "np.ndarray", action: "np.ndarray", cuts: "np.ndarray", lag: int) -> "np.ndarray":
    """Accumulate the raw conditional path-weighted transition counts and their response.

    Parameters
    ----------
    records : np.ndarray
        Float shape (E,N,6); columns as in simulate_biased_paths.
    action : np.ndarray
        Float shape (E,N,2), storing one-step log density ratios and their
        alpha derivatives in that order. Window log ratios are finite
        with representable exponentials.
    cuts : np.ndarray
        Finite strictly increasing one-dimensional cut positions;
        an empty array defines one state. L=len(cuts)+1.
    lag : int
        Window length, 1 <= lag <= N; both endpoint frames define a transition.

    Returns
    -------
    counts : np.ndarray
        Float shape (2,L,L). Entry 0 contains raw sums of conditional path
        weights; entry 1 their alpha derivatives. Rows are starting states
        and columns are endpoint states, both ordered from left to right.
        Units are 1 and inverse energy. Each recorded path instance has
        unit multiplicity before conditional path weighting."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_accumulate_path_counts(records: "np.ndarray", action: "np.ndarray", cuts: "np.ndarray", lag: int) -> "np.ndarray":
    k=len(cuts)+1
    C=np.zeros((k,k));dC=C.copy()
    logs=np.concatenate([np.zeros((len(records),1,2)),np.cumsum(action,axis=1)],axis=1)
    windows=logs[:,lag:]-logs[:,:-lag]
    start=np.searchsorted(cuts,records[:,:-lag+1 if lag>1 else None,0],side='right')
    end=np.searchsorted(cuts,records[:,lag-1:,1],side='right')
    weight=np.exp(windows[:,:,0])
    np.add.at(C,(start.ravel(),end.ravel()),weight.ravel())
    np.add.at(dC,(start.ravel(),end.ravel()),(weight*windows[:,:,1]).ravel())
    return np.stack([C,dC])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Independent scientific configurations, derived from the cited method."""
    return [{'setup': '# Case: overlapping_windows\n'
               'import numpy as np\n'
               'q=np.array([[-1.2,-.5,.0,.7,1.2,.4,-.9],[1.3,.2,-.7,-1.,-.1,.5,1.]])\n'
               'records=np.zeros((2,6,6));records[:,:,0]=q[:,:-1];records[:,:,1]=q[:,1:]\n'
               'action=np.array([[[.2,.1],[-.1,.3],[.3,-.2],[.0,.4],[-.2,-.3],[.1,.2]],[[.1,-.4],[.2,.2],[-.3,.1],[.4,-.1],[.2,.0],[-.1,.3]]])\n'
               'cuts=np.array([-.5,.5]);lag=2\n',
      'call': 'accumulate_path_counts(records.copy(), action.copy(), cuts.copy(), lag)',
      'gold_call': '_oracle_accumulate_path_counts(records.copy(), action.copy(), cuts.copy(), lag)',
      'tol': 2e-08},
     {'setup': '# Case: single_step_lag\n'
               'import numpy as np\n'
               'q=np.array([[-1.2,-.5,.0,.7,1.2,.4,-.9],[1.3,.2,-.7,-1.,-.1,.5,1.]])\n'
               'records=np.zeros((2,6,6));records[:,:,0]=q[:,:-1];records[:,:,1]=q[:,1:]\n'
               'action=np.array([[[.2,.1],[-.1,.3],[.3,-.2],[.0,.4],[-.2,-.3],[.1,.2]],[[.1,-.4],[.2,.2],[-.3,.1],[.4,-.1],[.2,.0],[-.1,.3]]])\n'
               'cuts=np.array([-.5,.5]);lag=2\n'
               'lag=1\n',
      'call': 'accumulate_path_counts(records.copy(), action.copy(), cuts.copy(), lag)',
      'gold_call': '_oracle_accumulate_path_counts(records.copy(), action.copy(), cuts.copy(), lag)',
      'tol': 2e-08},
     {'setup': '# Case: full_record_lag\n'
               'import numpy as np\n'
               'q=np.array([[-1.2,-.5,.0,.7,1.2,.4,-.9],[1.3,.2,-.7,-1.,-.1,.5,1.]])\n'
               'records=np.zeros((2,6,6));records[:,:,0]=q[:,:-1];records[:,:,1]=q[:,1:]\n'
               'action=np.array([[[.2,.1],[-.1,.3],[.3,-.2],[.0,.4],[-.2,-.3],[.1,.2]],[[.1,-.4],[.2,.2],[-.3,.1],[.4,-.1],[.2,.0],[-.1,.3]]])\n'
               'cuts=np.array([-.5,.5]);lag=2\n'
               'lag=6\n',
      'call': 'accumulate_path_counts(records.copy(), action.copy(), cuts.copy(), lag)',
      'gold_call': '_oracle_accumulate_path_counts(records.copy(), action.copy(), cuts.copy(), lag)',
      'tol': 2e-08},
     {'setup': '# Case: exact_cut_on_right\n'
               'import numpy as np\n'
               'q=np.array([[-1.2,-.5,.0,.7,1.2,.4,-.9],[1.3,.2,-.7,-1.,-.1,.5,1.]])\n'
               'records=np.zeros((2,6,6));records[:,:,0]=q[:,:-1];records[:,:,1]=q[:,1:]\n'
               'action=np.array([[[.2,.1],[-.1,.3],[.3,-.2],[.0,.4],[-.2,-.3],[.1,.2]],[[.1,-.4],[.2,.2],[-.3,.1],[.4,-.1],[.2,.0],[-.1,.3]]])\n'
               'cuts=np.array([-.5,.5]);lag=2\n'
               'cuts=np.array([-.5,0.,.5,1.]);lag=1\n',
      'call': 'accumulate_path_counts(records.copy(), action.copy(), cuts.copy(), lag)',
      'gold_call': '_oracle_accumulate_path_counts(records.copy(), action.copy(), cuts.copy(), lag)',
      'tol': 2e-08},
     {'setup': '# Case: independent_ensemble_boundaries\n'
               'import numpy as np\n'
               'q=np.array([[-1.2,-.5,.0,.7,1.2,.4,-.9],[1.3,.2,-.7,-1.,-.1,.5,1.]])\n'
               'records=np.zeros((2,6,6));records[:,:,0]=q[:,:-1];records[:,:,1]=q[:,1:]\n'
               'action=np.array([[[.2,.1],[-.1,.3],[.3,-.2],[.0,.4],[-.2,-.3],[.1,.2]],[[.1,-.4],[.2,.2],[-.3,.1],[.4,-.1],[.2,.0],[-.1,.3]]])\n'
               'cuts=np.array([-.5,.5]);lag=2\n'
               'records[0,:,0]=-1.;records[0,:,1]=-1.;records[1,:,0]=1.;records[1,:,1]=1.;lag=3\n',
      'call': 'accumulate_path_counts(records.copy(), action.copy(), cuts.copy(), lag)',
      'gold_call': '_oracle_accumulate_path_counts(records.copy(), action.copy(), cuts.copy(), lag)',
      'tol': 2e-08},
     {'setup': '# Case: cancelling_log_increments\n'
               'import numpy as np\n'
               'q=np.array([[-1.2,-.5,.0,.7,1.2,.4,-.9],[1.3,.2,-.7,-1.,-.1,.5,1.]])\n'
               'records=np.zeros((2,6,6));records[:,:,0]=q[:,:-1];records[:,:,1]=q[:,1:]\n'
               'action=np.array([[[.2,.1],[-.1,.3],[.3,-.2],[.0,.4],[-.2,-.3],[.1,.2]],[[.1,-.4],[.2,.2],[-.3,.1],[.4,-.1],[.2,.0],[-.1,.3]]])\n'
               'cuts=np.array([-.5,.5]);lag=2\n'
               'action[:,:,0]=np.array([20.,-20.,20.,-20.,20.,-20.]);lag=2\n',
      'call': 'accumulate_path_counts(records.copy(), action.copy(), cuts.copy(), lag)',
      'gold_call': '_oracle_accumulate_path_counts(records.copy(), action.copy(), cuts.copy(), lag)',
      'tol': 2e-08},
     {'setup': '# Case: dynamic_range_path_weights\n'
               'import numpy as np\n'
               'q=np.array([[-1.2,-.5,.0,.7,1.2,.4,-.9],[1.3,.2,-.7,-1.,-.1,.5,1.]])\n'
               'records=np.zeros((2,6,6));records[:,:,0]=q[:,:-1];records[:,:,1]=q[:,1:]\n'
               'action=np.array([[[.2,.1],[-.1,.3],[.3,-.2],[.0,.4],[-.2,-.3],[.1,.2]],[[.1,-.4],[.2,.2],[-.3,.1],[.4,-.1],[.2,.0],[-.1,.3]]])\n'
               'cuts=np.array([-.5,.5]);lag=2\n'
               'action[:,:,0]*=15.;lag=4\n',
      'call': 'accumulate_path_counts(records.copy(), action.copy(), cuts.copy(), lag)',
      'gold_call': '_oracle_accumulate_path_counts(records.copy(), action.copy(), cuts.copy(), lag)',
      'tol': 2e-06},
     {'setup': '# Case: single_microstate\n'
               'import numpy as np\n'
               'q=np.array([[-1.2,-.5,.0,.7,1.2,.4,-.9],[1.3,.2,-.7,-1.,-.1,.5,1.]])\n'
               'records=np.zeros((2,6,6));records[:,:,0]=q[:,:-1];records[:,:,1]=q[:,1:]\n'
               'action=np.array([[[.2,.1],[-.1,.3],[.3,-.2],[.0,.4],[-.2,-.3],[.1,.2]],[[.1,-.4],[.2,.2],[-.3,.1],[.4,-.1],[.2,.0],[-.1,.3]]])\n'
               'cuts=np.array([-.5,.5]);lag=2\n'
               'cuts=np.array([]);lag=3\n',
      'call': 'accumulate_path_counts(records.copy(), action.copy(), cuts.copy(), lag)',
      'gold_call': '_oracle_accumulate_path_counts(records.copy(), action.copy(), cuts.copy(), lag)',
      'tol': 2e-08}]
