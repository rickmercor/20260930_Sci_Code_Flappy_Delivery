"""
Evaluate the complete interaction-response localization design.

The complete finite-ensemble observable combines selected-bond Hamiltonians, adjacent-sector addition responses, transverse modular spectral responses, separately fitted batch lengths and paired-size design selection. Its construction and numerical conventions are those of the problem statement.

Returns
-------
return certificate
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def localization_design(seed: int = 49273, disorder: float = 1.9, couplings: 'np.ndarray | list | tuple' = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0), lengths: 'np.ndarray | list | tuple' = (6, 8), batches: int = 4, samples: int = 2, drift_limit: float = 1.0, fit_limit: float = 0.2, penalty: float = 1.0, radius: float = 0.1) -> 'np.ndarray':
    """Parameters
    ----------
    seed : int
        Nonnegative random seed.
    disorder : float
        Finite nonnegative binary-disorder magnitude.
    couplings : array_like
        Nonempty finite real interactions.
    lengths : array_like
        The two integer lengths (6,8).
    batches, samples : int
        At least two batches and at least one sample each. Generate a local default_rng(seed) uniform array on [-1,1) of shape (batches,samples,8); signs select +/-disorder. All couplings share it and length 6 uses its first six bonds.
    drift_limit, fit_limit, penalty, radius : float
        Finite nonnegative design bounds, uncertainty penalty and response radius.
    
    Returns
    -------
    certificate : real ndarray, shape (10,)
        [J,V,mu_long,se_long,se_delta,D,max_residual,gamma,gamma'_+,gamma'_-]. The last three entries concern batch 0, L=8, separation 2, at the selected coupling. Empty eligibility returns [-1,0,0,0,0,0,0,0,0,0]. Use the problem's ordered matching, N=floor(2L/3), two-site slices, zero trim, separations 1,...,4, frequency 1/3 and stated rate dictionaries. Call all preceding scientific functions. Defaults specify the main task. Invalid domains raise ValueError."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_localization_design(seed: int = 49273, disorder: float = 1.9, couplings: 'np.ndarray | list | tuple' = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0), lengths: 'np.ndarray | list | tuple' = (6, 8), batches: int = 4, samples: int = 2, drift_limit: float = 1.0, fit_limit: float = 0.2, penalty: float = 1.0, radius: float = 0.1) -> 'np.ndarray':
    if not isinstance(seed,(int,np.integer)) or seed<0 or not np.isfinite(disorder) or disorder<0:
        raise ValueError('nonnegative integer seed and finite nonnegative disorder required')
    if len(lengths)!=2 or any(not isinstance(x,(int,np.integer)) or x not in (6,8) for x in lengths) or lengths[0]>=lengths[1]:
        raise ValueError('lengths must be (6,8)')
    if not isinstance(batches,(int,np.integer)) or batches<2 or not isinstance(samples,(int,np.integer)) or samples<1:
        raise ValueError('integer batches>=2 and samples>=1 required')
    couplings=np.asarray(couplings,dtype=float)
    if couplings.ndim!=1 or len(couplings)<1 or not np.isfinite(couplings).all():
        raise ValueError('nonempty finite couplings required')
    potentials=disorder*np.where(np.random.default_rng(seed).uniform(-1,1,(batches,samples,lengths[1]))<0,-1.,1.)
    x=np.arange(1,5);rates=np.arange(1,41,dtype=float)/40
    oscillation_rates=np.array([.025,.05,.075,.1,.15,.2,.3,.4,.6,.8,1.,1.4,2.,3.])
    if not np.isfinite(radius) or radius<0:raise ValueError('nonnegative finite radius required')
    statistics=[];diagnostics=[]
    for V in couplings:
        fitted=[]
        for L in lengths:
            bonds=[]
            for i in range(L):
                if i%4==0:bonds.extend(((2*i,2*i+2),(2*i+1,2*i+3)))
                elif i%4 in (2,3):bonds.append((2*i,2*i+1))
            deltas=np.empty((batches,samples,2,2*L,2*L),dtype=complex)
            for b in range(batches):
                for s in range(samples):
                    model=_oracle_bond_ladder_model(potentials[b,s,:L],L,2,bonds,1.,V)
                    pair=_oracle_sector_density_pair(model,2*L//3)
                    deltas[b,s]=_oracle_addition_density(pair)
            curves=_oracle_modular_batch_curves(deltas,x,0,2)
            fitted.append(np.array([_oracle_fit_decay_dictionary(x,curve,1/3,rates,oscillation_rates,radius) for curve in curves]))
        diagnostics.append(curves[0,:,1])
        statistics.append(_oracle_paired_batch_statistics(*fitted))
    certificate=_oracle_select_localization_certificate(couplings,np.array(statistics),drift_limit,fit_limit,penalty)
    if certificate[0]<0:return np.r_[certificate,np.zeros(3)]
    index=int(np.flatnonzero(couplings==certificate[1])[0])
    return np.r_[certificate,diagnostics[index]]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n# canonical paired disordered design\nargs=()\nargs=tuple(np.asarray(value) if i in [2, 3] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'localization_design(*deepcopy(args))', 'gold_call': '_oracle_localization_design(*deepcopy(args))'}, {'setup': 'import numpy as np\n# weak disorder finite-size rejection\nargs=(61, 0.4, (0.0, 0.8, 2.0), (6, 8), 2, 2, 5.0, 0.08, 1.0)\nargs=tuple(np.asarray(value) if i in [2, 3] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nargs=(*args,.1)', 'call': 'localization_design(*deepcopy(args))', 'gold_call': '_oracle_localization_design(*deepcopy(args))'}, {'setup': 'import numpy as np\n# strong disorder shorter localization\nargs=(82, 3.0, (0.0, 0.6, 1.5), (6, 8), 3, 2, 10.0, 0.3, 0.5)\nargs=tuple(np.asarray(value) if i in [2, 3] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nargs=(*args,.1)', 'call': 'localization_design(*deepcopy(args))', 'gold_call': '_oracle_localization_design(*deepcopy(args))'}, {'setup': 'import numpy as np\n# attractive sector rearrangement\nargs=(93, 1.2, (-1.0, -0.4, 0.8), (6, 8), 2, 2, 40.0, 0.5, 0.0)\nargs=tuple(np.asarray(value) if i in [2, 3] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nargs=(*args,.1)', 'call': 'localization_design(*deepcopy(args))', 'gold_call': '_oracle_localization_design(*deepcopy(args))'}, {'setup': 'import numpy as np\n# strict design constraints yield empty feasible set\nargs=(104, 1.7, (0.2, 1.2), (6, 8), 2, 1, 0.0, 0.0, 1.0)\nargs=tuple(np.asarray(value) if i in [2, 3] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nargs=(*args,.1)', 'call': 'localization_design(*deepcopy(args))', 'gold_call': '_oracle_localization_design(*deepcopy(args))'}]
