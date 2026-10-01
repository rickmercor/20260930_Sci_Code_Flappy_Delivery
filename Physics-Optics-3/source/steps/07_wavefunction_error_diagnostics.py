"""
Measure complex, modulus, overlap, norm, and phase-aligned wavefunction errors.

Compute the paper-style full-wavefunction and modulus L2 comparisons using rectangular quadrature. Return both norms, the relative complex L2 error, the relative modulus L2 error, normalized overlap magnitude, and the relative complex error after removing only the single global overlap phase. The required feasibility error is the unaligned complex L2 value.

Returns
-------
One length-6 diagnostic array in the documented order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def wavefunction_error_diagnostics(tgwp_field, reference_field, dx, dy):
    """Return a length-6 float array.

    Order is [TGWP_norm,reference_norm,relative_L2,relative_abs_L2,
    overlap_magnitude,global_phase_aligned_relative_L2].

    Raises
    ------
    ValueError
        If packed fields, spacings, shapes, or norms are invalid.
    """
    return np.zeros(6, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_wavefunction_error_diagnostics(tgwp_field, reference_field, dx, dy):
    import numpy as np
    tgwp = tgwp_field
    grid = reference_field
    tgwp=np.asarray(tgwp,float); grid=np.asarray(grid,float)
    if tgwp.shape != grid.shape or tgwp.ndim != 3 or tgwp.shape[2] != 2 or not np.all(np.isfinite(tgwp)) or not np.all(np.isfinite(grid)):
        raise ValueError('fields must be equal finite arrays of shape (Ny,Nx,2)')
    if not np.all(np.isfinite([dx,dy])) or dx <= 0 or dy <= 0:
        raise ValueError('grid spacings must be finite and positive')
    tgwp=tgwp[...,0]+1j*tgwp[...,1]; grid=grid[...,0]+1j*grid[...,1]
    area=dx*dy
    ng=np.sqrt(area*np.sum(np.abs(grid)**2))
    nt=np.sqrt(area*np.sum(np.abs(tgwp)**2))
    if ng == 0 or nt == 0:
        raise ValueError('both fields must have nonzero norm')
    l2=np.sqrt(area*np.sum(np.abs(tgwp-grid)**2))/ng
    ab=np.sqrt(area*np.sum((np.abs(tgwp)-np.abs(grid))**2))/ng
    overlap=area*np.sum(np.conj(grid)*tgwp)/(ng*nt)
    aligned=tgwp*np.exp(-1j*np.angle(overlap))
    aligned_l2=np.sqrt(area*np.sum(np.abs(aligned-grid)**2))/ng
    return np.array([nt,ng,l2,ab,abs(overlap),aligned_l2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\na=np.zeros((3,4,2));a[...,0]=1.;b=a.copy();b[...,0]*=.9", "call":"wavefunction_error_diagnostics(a,b,.2,.3)", "gold_call":"_oracle_wavefunction_error_diagnostics(a,b,.2,.3)"},
        {"setup":"import numpy as np\nr=np.random.default_rng(4);z=r.normal(size=(4,5))+1j*r.normal(size=(4,5));a=np.stack((z.real,z.imag),-1);q=z*np.exp(1j*.7);b=np.stack((q.real,q.imag),-1)", "call":"wavefunction_error_diagnostics(a,b,.1,.15)", "gold_call":"_oracle_wavefunction_error_diagnostics(a,b,.1,.15)"},
        {"setup":"import numpy as np\nx=np.linspace(-4,4,16,endpoint=False);y=np.linspace(-3,3,12,endpoint=False);p=_oracle_phase_space_sobol_packets(3,3,np.array([-1.,0.]),np.array([1.,0.]),np.ones(2),2*np.ones(2));v=np.array([.3,0.,1.,1.,1.2,.4,2.1,.2]);tr=_oracle_stormer_verlet_centers_actions(p,1.,12,v);w=_oracle_hagedorn_width_evolution(tr,1.,v,2*np.ones(2));a=_oracle_coherent_tgwp_reconstruction(x,y,p,tr[-1],w);b=_oracle_strang_split_step_reference(x,y,np.array([-1.,0.]),np.array([1.,0.]),np.ones(2),1.,24,v)", "call":"wavefunction_error_diagnostics(a,b,x[1]-x[0],y[1]-y[0])", "gold_call":"_oracle_wavefunction_error_diagnostics(a,b,x[1]-x[0],y[1]-y[0])"}
    ]
