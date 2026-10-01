"""
Compose all preceding scientific operations into the requested robust production fraction

Transport-species reconstruction, buffered balancing, expression-based inference and simultaneous energetic calibration jointly determine the selected design.

Returns
-------
One finite dimensionless float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def resolve_transport_panel(species_energies: "np.ndarray", weights: "np.ndarray",
                            assay_intervals: "np.ndarray", radius: float,
                            limit: float, model: dict) -> float:
    """
    species_energies is finite (4,2,4,2), weights positive finite (C,14),
    and assay_intervals finite ordered (4,3,2). They use the main problem
    axes and units. radius>=0 is the shared intrinsic-factor norm bound;
    limit>=0 is the calibration eligibility limit in kJ/mol. model is a dictionary with keys s, transports, conditions, factors,
    loadings, h, z, assay, demand, oneway, x_bounds, rt, faraday, activity
    and drive. These are the aligned inputs of the preceding public
    functions: s is organic stoichiometry; h and z are species properties.
    The model uses four conditions, eight organic pools and two intrinsic
    factors. All required numerical inputs are explicit arguments. Compose
    the preceding public functions and use every returned scientific
    result. Return the selected robust fraction as a finite float. Invalid
    inputs, infeasible phenotype or no eligible design raise ValueError.
    Unresolved numerical failures propagate as RuntimeError. The main
    benchmark uses radius=0.6, limit=3.0. Absolute accuracy 0.000002 suffices.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_resolve_transport_panel(species_energies: "np.ndarray", weights: "np.ndarray",
                            assay_intervals: "np.ndarray", radius: float,
                            limit: float, model: dict) -> float:
    import numpy as np
    fixed = model
    required = {'s','transports','conditions','factors','loadings','h','z','assay',
                'demand','oneway','x_bounds','rt','faraday','activity','drive'}
    if not isinstance(fixed, dict) or set(fixed) != required:
        raise ValueError('Model keys do not match the stated contract')
    e, w, bands = map(lambda x: np.asarray(x, dtype=float), (species_energies, weights, assay_intervals))
    if e.shape != (4, 2, 4, 2) or w.ndim != 2 or w.shape[1] != 14 or bands.shape != (4, 3, 2):
        raise ValueError('Benchmark-shaped species, weight and assay inputs required')
    a = _oracle_reactant_state(e, fixed['h'], fixed['z'], fixed['rt'])
    t = _oracle_transport_state(a, fixed['transports'], fixed['h'], fixed['z'])
    b = _oracle_balanced_reactions(fixed['s'], a, fixed['transports'], t)
    op = _oracle_energy_operator(fixed['s'], a, b, fixed['transports'],
                                fixed['conditions'], fixed['loadings'], fixed['rt'], fixed['faraday'])
    n = _oracle_phenotype_panel(b, fixed['demand'], w, fixed['factors'], fixed['oneway'])
    directional = _oracle_directional_state(n, w, fixed['factors'], fixed['rt'])
    radii = []
    for c in range(w.shape[0]):
        system = _oracle_calibration_system(op, n[c], directional[c, :, :, 2],
                    fixed['assay'], bands, fixed['x_bounds'], fixed['activity'], fixed['drive'])
        radii.append(_oracle_joint_energy_radius(system, 32, 2, radius))
    selected = _oracle_robust_delivery(n, radii, 10, 8., limit, 1e-10)
    return float(selected[0])


def _fixed_transport_inputs():
    import numpy as np
    edges = [(0,4), (0,4), (1,5), (6,2), (7,3), (4,5), (4,6),
             (5,6), (5,7), (6,7), (2,3), (1,2), (0,1)]
    s = np.zeros((8,14))
    for j, (src, dst) in enumerate(edges):
        s[src,j], s[dst,j] = -1., 1.
    tr = np.array([[0,0,0,1,0,-1], [1,0,0,1,1,-1], [2,1,0,1,0,-1],
                   [3,2,1,0,0,-1], [4,3,1,0,0,1], [13,-1,0,1,1,-1]])
    cond = np.array([[[6.,0.],[6.8,.038]], [[6.4,0.],[7.3,.044]],
                     [[6.2,0.],[7.,.040]], [[6.5,0.],[7.6,.057]]])
    fac = np.ones((4,14))
    fac[1,[0,8]], fac[2,[3,13]], fac[3,[5,10]] = [.31,1.55], [.37,1.8], [.42,1.65]
    load = np.tile([[1.6,-.8], [-1.2,.8], [.8,1.8], [-1.,-1.2]], (2,1))
    h = np.zeros((3,8))
    h[0,[4,5]], h[1,[6,7]], h[2,[1,5,2,6]] = [1.,-1.], [1.,-1.], [1.,-1.,-1.,1.]
    return dict(s=s, transports=tr, conditions=cond, factors=fac, loadings=load,
                h=np.array([0.,1.]), z=np.array([-1.,0.]), assay=h,
                demand=np.array([-10.,0.,0.,8.,0.,0.,0.,2.]),
                oneway=np.array([7,10,12,13]), x_bounds=np.tile([-10.5,-4.5], (4,8,1)),
                rt=.00831446261815324*298.15, faraday=96.48533212,
                activity=1e-7, drive=.001)


def _panel_inputs():
    import numpy as np
    energies = np.array([[[[0.0, -1.712403], [-2.0, -8.27881], [-1.0, -4.995607],
       [-4.0, -13.132815]],
      [[0.0, 2.854005], [-2.0, -3.712403], [-1.0, -0.429199],
       [-4.0, -8.566408]]],
     [[[0.0, 0.570801], [-2.0, -5.995607], [-1.0, -2.712403],
       [-4.0, -10.849611]],
      [[0.0, 5.70801], [-2.0, -0.858398], [-1.0, 2.424806],
       [-4.0, -5.712403]]],
     [[[0.0, -0.570801], [-2.0, -7.137209], [-1.0, -3.854005],
       [-4.0, -11.991213]],
      [[0.0, 3.995607], [-2.0, -2.570801], [-1.0, 0.712403],
       [-4.0, -7.424806]]],
     [[[0.0, 1.141602], [-2.0, -5.424806], [-1.0, -2.141602],
       [-4.0, -10.27881]],
      [[0.0, 7.420412], [-2.0, 0.854005], [-1.0, 4.137209], [-4.0, -4.0]]]], dtype=float)
    weights = np.array([[5.32, 9.35, 14.2, 16.73, 12.15, 11.54, 12.67, 7.33, 5.78, 6.68, 13.78,
      9.64, 17.74, 0.04],
     [12.4, 11.31, 10.31, 14.13, 11.86, 10.59, 12.43, 14.05, 17.7, 15.02,
      16.57, 11.48, 4.84, 0.08],
     [7.7, 13.65, 15.06, 6.44, 6.92, 9.54, 6.51, 10.02, 9.02, 9.07, 17.85,
      2.42, 7.25, 0.025],
     [2.87, 12.67, 15.81, 16.62, 4.19, 8.77, 9.63, 9.24, 17.8, 13.36, 14.85,
      9.08, 7.47, 0.12],
     [9.83, 9.45, 9.87, 9.71, 10.31, 2.23, 17.36, 9.88, 15.8, 9.29, 15.28,
      13.85, 11.52, 0.055],
     [17.22, 8.89, 5.97, 3.19, 7.9, 4.59, 4.2, 4.63, 14.57, 9.47, 17.93,
      10.34, 17.0, 0.035]], dtype=float)
    bands = np.array([[[0.707, 1.067], [-0.433, -0.073], [1.421, 1.821]],
     [[0.437, 0.797], [0.146, 0.506], [0.625, 1.025]],
     [[-0.496, -0.136], [-0.327, 0.033], [1.604, 2.004]],
     [[-0.66, -0.3], [-0.817, -0.457], [1.197, 1.597]]], dtype=float)
    return energies, weights, bands

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\ne,w,b=_panel_inputs(); model=_fixed_transport_inputs()',
      'call': 'resolve_transport_panel(e,w,b,.6,3.,model)',
      'gold_call': '_oracle_resolve_transport_panel(e,w,b,.6,3.,model)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'e,w,b=_panel_inputs(); w=w[4:5].copy(); model=_fixed_transport_inputs()',
      'call': 'resolve_transport_panel(e,w,b,.6,3.,model)',
      'gold_call': '_oracle_resolve_transport_panel(e,w,b,.6,3.,model)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\ne,w,b=_panel_inputs(); model=_fixed_transport_inputs()',
      'call': 'resolve_transport_panel(e,w,b,.6,4.5,model)',
      'gold_call': '_oracle_resolve_transport_panel(e,w,b,.6,4.5,model)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'e,w,b=_panel_inputs(); model=_fixed_transport_inputs()\n'
               '\n'
               'def _raises_value_error(fn, arguments):\n'
               '    try:\n'
               '        fn(*arguments)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': '_raises_value_error(resolve_transport_panel, (e,w,b,.6,0.,model,))',
      'gold_call': '_raises_value_error(_oracle_resolve_transport_panel, '
                   '(e,w,b,.6,0.,model,))',
      'tol': 0.0}]
