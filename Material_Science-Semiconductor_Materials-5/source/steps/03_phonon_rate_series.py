"""
Expand the paper’s joint-system phonon rate operator in the probe coupling.

The noninvasive limit retains the coupling dependence of the bath-resolved joint transitions.

Returns
-------
result : np.ndarray     Complex shape (order+1,D,D), the ordinary power-series coefficients     Z_k in Z(lambda)=sum_k lambda**k Z_k about zero. Here Z(lambda) is     integral_0^infinity C(t)A(-t;lambda)dt, and A(-t;lambda) evolves     backward under the complete H(lambda). Coefficient k has units     ps**(k-1); it includes the derivative factorial denominator.     Use the same causal bath response as bath_response.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def phonon_rate_series(hamiltonian0: "np.ndarray", exchange: "np.ndarray", coupling_operator: "np.ndarray", alpha: float, cutoff: float, temperature: float, order: int) -> "np.ndarray":
    """Compute Taylor coefficients of the complete joint phonon rate operator.

    Parameters
    ----------
    hamiltonian0 : np.ndarray
        Hermitian shape (D,D), in ps^-1, with spectral span <=4*cutoff.
    exchange : np.ndarray
        Dimensionless Hermitian shape (D,D), defining
        H(lambda)=hamiltonian0+lambda*exchange for real lambda in ps^-1.
    coupling_operator : np.ndarray
        Dimensionless Hermitian shape (D,D), the phonon coupling A.
    alpha : float
        Nonnegative acoustic strength in ps^2.
    cutoff : float
        Cutoff in ps^-1, 0.8<=cutoff<=3.
    temperature : float
        Temperature in K, 1<=temperature<=10, with the bath_response units.
    order : int
        Highest Taylor power, 0<=order<=6. Degenerate eigenvalues are allowed.

    Returns
    -------
    result : np.ndarray
        Complex shape (order+1,D,D), the ordinary power-series coefficients
        Z_k in Z(lambda)=sum_k lambda**k Z_k about zero. Here Z(lambda) is
        integral_0^infinity C(t)A(-t;lambda)dt, and A(-t;lambda) evolves
        backward under the complete H(lambda). Coefficient k has units
        ps**(k-1); it includes the derivative factorial denominator.
        Use the same causal bath response as bath_response.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_phonon_rate_series(hamiltonian0: "np.ndarray", exchange: "np.ndarray", coupling_operator: "np.ndarray", alpha: float, cutoff: float, temperature: float, order: int) -> "np.ndarray":
    def _evaluate(h):
        energies,vectors=np.linalg.eig(h)
        if np.linalg.cond(vectors)>1e6:
            raise np.linalg.LinAlgError('Ill-conditioned contour eigensystem')
        inverse=np.linalg.inv(vectors)
        f=_oracle_bath_response(energies[None,:]-energies[:,None],alpha,cutoff,temperature)
        return vectors@((inverse@coupling_operator@vectors)*f)@inverse
    norm=np.linalg.norm(exchange,2)
    if norm==0:
        result=np.zeros((order+1,*hamiltonian0.shape),complex)
        result[0]=_evaluate(hamiltonian0)
        return result
    span=np.ptp(np.linalg.eigvalsh(hamiltonian0))
    radius=min(4.,np.pi*.1309203391*temperature/(2*norm),(6*cutoff-span)/(2*norm))
    samples=64
    while True:
        try:
            values=np.array([_evaluate(hamiltonian0+radius*np.exp(2j*np.pi*j/samples)*exchange) for j in range(samples)])
            break
        except np.linalg.LinAlgError:
            radius*=.8
    result=np.fft.fft(values,axis=0)[:order+1]/samples
    result/=radius**np.arange(order+1)[:,None,None]
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independently initialized scientific cases."""
    return [{'setup': 'import numpy as np\n'
               '## Joint rate Taylor geometry 1\n'
               'hamiltonian0 = np.array([[(-0.27899586358888495+0j), '
               '(0.17172009239381136+0.4059235538985941j)], [(0.17172009239381134-0.4059235538985941j), '
               '(0.3789958635888854-3.469446951953614e-18j)]], dtype=complex)\n'
               'exchange = np.array([[(0.6420418858424637+0j), (-0.3012475380460721-0.10617673694831624j)], '
               '[(-0.3012475380460721+0.10617673694831624j), (0.15410992216543978+0j)]], dtype=complex)\n'
               'coupling_operator = np.array([[(0.7095431794212317+8.40583686802162e-18j), '
               '(0.41169038585456685+0.19131827448772232j)], [(0.41169038585456685-0.19131827448772232j), '
               '(0.29045682057876854+1.6855850730762757e-18j)]], dtype=complex)\n'
               'alpha = 0.012\n'
               'cutoff = 1.6\n'
               'temperature = 1.0\n'
               'order = 6\n',
      'call': 'phonon_rate_series(hamiltonian0.copy(), exchange.copy(), coupling_operator.copy(), alpha, '
              'cutoff, temperature, order)',
      'gold_call': '_oracle_phonon_rate_series(hamiltonian0.copy(), exchange.copy(), '
                   'coupling_operator.copy(), alpha, cutoff, temperature, order)',
      'tol': 2e-08},
     {'setup': 'import numpy as np\n'
               '## Joint rate Taylor geometry 2\n'
               'hamiltonian0 = np.array([[(0.27745368510275686+2.7755575615628914e-17j), '
               '(-0.035052005660467754+0.14263918462738623j), (-0.3134876350189312-0.3618228070912975j)], '
               '[(-0.03505200566046776-0.14263918462738623j), (-0.47224969087498064+0j), '
               '(-0.05224972317232687+0.07382849529426586j)], [(-0.3134876350189312+0.3618228070912975j), '
               '(-0.05224972317232687-0.07382849529426584j), (-0.20520399422777655+0j)]], dtype=complex)\n'
               'exchange = np.array([[(-0.11464258908554922+0j), '
               '(-0.21857406449713238-0.23054597601653817j), (-0.33395561256380163-0.13675658321621098j)], '
               '[(-0.21857406449713238+0.23054597601653817j), (0.2382687552560023+0j), '
               '(0.2568965791077918-0.06786211056426524j)], [(-0.33395561256380163+0.13675658321621098j), '
               '(0.2568965791077918+0.06786211056426524j), (-0.597753538832849+0j)]], dtype=complex)\n'
               'coupling_operator = np.array([[(0.6740915729300143-6.926779532794839e-18j), '
               '(0.21662712810334697-0.4062382098913058j), (-0.085915143969049-0.01881266768787397j)], '
               '[(0.21662712810334697+0.4062382098913059j), (0.3144332377346043-4.8211517204171154e-18j), '
               '(-0.0162724574735747-0.05782203787581509j)], [(-0.085915143969049+0.018812667687873975j), '
               '(-0.0162724574735747+0.05782203787581509j), (0.011475189335381133+4.20688104364354e-19j)]], '
               'dtype=complex)\n'
               'alpha = 0.014\n'
               'cutoff = 1.6700000000000002\n'
               'temperature = 1.7\n'
               'order = 5\n',
      'call': 'phonon_rate_series(hamiltonian0.copy(), exchange.copy(), coupling_operator.copy(), alpha, '
              'cutoff, temperature, order)',
      'gold_call': '_oracle_phonon_rate_series(hamiltonian0.copy(), exchange.copy(), '
                   'coupling_operator.copy(), alpha, cutoff, temperature, order)',
      'tol': 2e-08},
     {'setup': 'import numpy as np\n'
               '## Joint rate Taylor geometry 3\n'
               'hamiltonian0 = np.array([[(0.09999999999999992+6.505213034913027e-19j), '
               '(3.469446951953614e-17-3.469446951953614e-18j)], '
               '[(3.469446951953614e-17-3.0357660829594124e-18j), (0.09999999999999998+0j)]], '
               'dtype=complex)\n'
               'exchange = np.array([[(0.21810986467915747+0j), '
               '(-0.34030055631515643-0.12521163090599322j)], [(-0.34030055631515643+0.12521163090599322j), '
               '(0.5740425328399031+0j)]], dtype=complex)\n'
               'coupling_operator = np.array([[(0.773831208769909-5.236576819904093e-18j), '
               '(-0.090860389311485-0.40836363544979876j)], [(-0.090860389311485+0.40836363544979876j), '
               '(0.2261687912300913+4.06808741892631e-18j)]], dtype=complex)\n'
               'alpha = 0.016\n'
               'cutoff = 1.7400000000000002\n'
               'temperature = 2.4\n'
               'order = 6\n',
      'call': 'phonon_rate_series(hamiltonian0.copy(), exchange.copy(), coupling_operator.copy(), alpha, '
              'cutoff, temperature, order)',
      'gold_call': '_oracle_phonon_rate_series(hamiltonian0.copy(), exchange.copy(), '
                   'coupling_operator.copy(), alpha, cutoff, temperature, order)',
      'tol': 2e-08},
     {'setup': 'import numpy as np\n'
               '## Joint rate Taylor geometry 4\n'
               'hamiltonian0 = np.array([[(0.11370328571552787+0j), '
               '(0.14505891433323953-0.12069830471649257j), (-0.27321744480881777-0.2813754472025598j)], '
               '[(0.14505891433323953+0.12069830471649257j), (-0.19779498143115498+0j), '
               '(-0.17574952821353168+0.18029673478743274j)], [(-0.27321744480881777+0.28137544720255986j), '
               '(-0.17574952821353168-0.18029673478743274j), (0.2340916957156272+0j)]], dtype=complex)\n'
               'exchange = np.array([[0j, 0j, 0j], [0j, 0j, 0j], [0j, 0j, 0j]], dtype=complex)\n'
               'coupling_operator = np.array([[(0.6382989553864424+2.3294695897450762e-17j), '
               '(0.13764586233290318-0.44039144152884824j), (-0.12981794088682036-0.03361095020524352j)], '
               '[(0.13764586233290318+0.44039144152884824j), (0.33352867554097104+7.798695826104223e-18j), '
               '(-0.004804845099190098-0.09681532113877436j)], '
               '[(-0.12981794088682036+0.033610950205243524j), '
               '(-0.004804845099190098+0.09681532113877438j), (0.0281723690725866+4.54221533574121e-19j)]], '
               'dtype=complex)\n'
               'alpha = 0.018000000000000002\n'
               'cutoff = 1.81\n'
               'temperature = 3.0999999999999996\n'
               'order = 4\n',
      'call': 'phonon_rate_series(hamiltonian0.copy(), exchange.copy(), coupling_operator.copy(), alpha, '
              'cutoff, temperature, order)',
      'gold_call': '_oracle_phonon_rate_series(hamiltonian0.copy(), exchange.copy(), '
                   'coupling_operator.copy(), alpha, cutoff, temperature, order)',
      'tol': 2e-08},
     {'setup': 'import numpy as np\n'
               '## Joint rate Taylor geometry 5\n'
               'hamiltonian0 = np.array([[(0.09948985077404925+0j), '
               '(-0.1353489793611132-0.17099615076111954j), (0.2973301670874804-0.2243599742676734j), '
               '(0.04446456579274431+0.05435314880890481j)], [(-0.1353489793611132+0.17099615076111951j), '
               '(0.09023021172717167+0j), (-0.14369274172174362-0.009176909164205549j), '
               '(-0.13244809169988786+0.25651471896293176j)], [(0.29733016708748045+0.2243599742676734j), '
               '(-0.14369274172174362+0.009176909164205549j), (-0.005687662975347063+0j), '
               '(0.025785912234040836+0.1896722013681054j)], [(0.044464565792744305-0.05435314880890481j), '
               '(-0.13244809169988786-0.2565147189629318j), (0.025785912234040823-0.18967220136810534j), '
               '(0.01596760047412578+0j)]], dtype=complex)\n'
               'exchange = np.array([[0.7, 0.0, 0.0, 0.0], [0.0, 0.7, 0.0, 0.0], [0.0, 0.0, 0.7, 0.0], '
               '[0.0, 0.0, 0.0, 0.7]], dtype=float)\n'
               'coupling_operator = np.array([[(0.3225741501351186-6.116381654621283e-19j), '
               '(0.2752047812645607+0.11333140609023472j), (-0.21705739104505042+0.21028098254582614j), '
               '(-0.17243055008861702+0.09420239556866437j)], [(0.2752047812645607-0.1133314060902347j), '
               '(0.27460873476736947+5.751535047940324e-18j), (-0.11130399749457762+0.25566137615332746j), '
               '(-0.11401292341061808+0.14094975168494617j)], [(-0.21705739104505042-0.21028098254582614j), '
               '(-0.11130399749457762-0.25566137615332746j), (0.2831349089487337-1.0909581179202485e-17j), '
               '(0.17743609527630314+0.04901675870369802j)], [(-0.17243055008861702-0.09420239556866437j), '
               '(-0.11401292341061808-0.14094975168494617j), (0.17743609527630314-0.04901675870369802j), '
               '(0.11968220614877816+2.2939754342883695e-18j)]], dtype=complex)\n'
               'alpha = 0.02\n'
               'cutoff = 1.8800000000000001\n'
               'temperature = 3.8\n'
               'order = 6\n',
      'call': 'phonon_rate_series(hamiltonian0.copy(), exchange.copy(), coupling_operator.copy(), alpha, '
              'cutoff, temperature, order)',
      'gold_call': '_oracle_phonon_rate_series(hamiltonian0.copy(), exchange.copy(), '
                   'coupling_operator.copy(), alpha, cutoff, temperature, order)',
      'tol': 2e-08},
     {'setup': 'import numpy as np\n'
               '## Joint rate Taylor geometry 6\n'
               'hamiltonian0 = np.array([[(0.5809877385158565+0j), '
               '(0.12948334982734666-0.06153116010043127j)], [(0.12948334982734666+0.06153116010043127j), '
               '(-0.48098773851585663+0j)]], dtype=complex)\n'
               'exchange = np.array([[(-0.12375228252992444+0j), (0.494474082071411-0.5818116805995619j)], '
               '[(0.494474082071411+0.5818116805995619j), (0.16886819058705288+0j)]], dtype=complex)\n'
               'coupling_operator = np.array([[(0.16984370935525367+6.46863735932472e-18j), '
               '(0.1046972470911807+0.36060409065792653j)], [(0.1046972470911807-0.36060409065792653j), '
               '(0.8301562906447466+2.1279674458062512e-17j)]], dtype=complex)\n'
               'alpha = 0.0\n'
               'cutoff = 1.9500000000000002\n'
               'temperature = 4.5\n'
               'order = 3\n',
      'call': 'phonon_rate_series(hamiltonian0.copy(), exchange.copy(), coupling_operator.copy(), alpha, '
              'cutoff, temperature, order)',
      'gold_call': '_oracle_phonon_rate_series(hamiltonian0.copy(), exchange.copy(), '
                   'coupling_operator.copy(), alpha, cutoff, temperature, order)',
      'tol': 2e-08},
     {'setup': 'import numpy as np\n'
               '## Joint rate Taylor geometry 7\n'
               'hamiltonian0 = np.array([[(-0.2519175629337952+0j), '
               '(0.011773514573679533+0.06177851441492248j), (0.37489448093483896+0.10481496520357664j)], '
               '[(0.011773514573679526-0.06177851441492248j), (0.18785561797418676+0j), '
               '(0.2783407433104774+0.031651656499807834j)], [(0.37489448093483896-0.10481496520357664j), '
               '(0.2783407433104774-0.031651656499807834j), (0.21406194495960884+6.938893903907228e-18j)]], '
               'dtype=complex)\n'
               'exchange = np.array([[(-0.14032388727719827+0j), (0.3982915863658459-0.09977153195453546j), '
               '(0.20332835055085616-0.21534090928997432j)], [(0.3982915863658459+0.09977153195453546j), '
               '(0.11869332065865981+0j), (-0.39474165928803523-0.03924044732680244j)], '
               '[(0.20332835055085616+0.21534090928997432j), (-0.39474165928803523+0.03924044732680244j), '
               '(-0.20530757124179108+0j)]], dtype=complex)\n'
               'coupling_operator = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], '
               'dtype=float)\n'
               'alpha = 0.024\n'
               'cutoff = 2.02\n'
               'temperature = 5.199999999999999\n'
               'order = 6\n',
      'call': 'phonon_rate_series(hamiltonian0.copy(), exchange.copy(), coupling_operator.copy(), alpha, '
              'cutoff, temperature, order)',
      'gold_call': '_oracle_phonon_rate_series(hamiltonian0.copy(), exchange.copy(), '
                   'coupling_operator.copy(), alpha, cutoff, temperature, order)',
      'tol': 2e-08},
     {'setup': 'import numpy as np\n'
               '## Joint rate Taylor geometry 8\n'
               'hamiltonian0 = np.array([[(0.08928786423887114-3.469446951953614e-18j), '
               '(-0.18145043480341994-0.08799585630293438j), (0.3013065740924674-0.060562667298790285j), '
               '(0.09865132199473123-0.07166913459739849j)], [(-0.18145043480341994+0.08799585630293438j), '
               '(0.34171695658301815+0j), (0.27926596719616054+0.1570710635939911j), '
               '(-0.08039688431415214+0.05591464466474997j)], [(0.3013065740924674+0.06056266729879026j), '
               '(0.27926596719616054-0.1570710635939911j), (-0.14053769784457337+0j), '
               '(-0.049502451094051686-0.009009553212475641j)], '
               '[(0.09865132199473123+0.07166913459739849j), (-0.08039688431415212-0.05591464466474996j), '
               '(-0.04950245109405168+0.009009553212475645j), (-0.09046712297731563+0j)]], dtype=complex)\n'
               'exchange = np.array([[(0.044349206915732034+0j), '
               '(-0.14835462865856636-0.06874284008777783j), (0.056054929854711597-0.06015217114924256j), '
               '(-0.29675477932483246-0.19283260266340085j)], [(-0.14835462865856636+0.06874284008777783j), '
               '(-0.1567680563978551+0j), (-0.10738747321369968+0.024283534388808824j), '
               '(0.007553330166874936+0.2221636575436955j)], [(0.056054929854711597+0.06015217114924256j), '
               '(-0.10738747321369968-0.024283534388808824j), (-0.27660962598723515+0j), '
               '(0.29839658428747223+0.36082650227910423j)], [(-0.29675477932483246+0.19283260266340085j), '
               '(0.007553330166874936-0.2221636575436955j), (0.29839658428747223-0.36082650227910423j), '
               '(-0.03154330565293832+0j)]], dtype=complex)\n'
               'coupling_operator = np.array([[0j, 0j, 0j, 0j], [0j, 0j, 0j, 0j], [0j, 0j, 0j, 0j], [0j, '
               '0j, 0j, 0j]], dtype=complex)\n'
               'alpha = 0.026000000000000002\n'
               'cutoff = 2.0900000000000003\n'
               'temperature = 5.8999999999999995\n'
               'order = 2\n',
      'call': 'phonon_rate_series(hamiltonian0.copy(), exchange.copy(), coupling_operator.copy(), alpha, '
              'cutoff, temperature, order)',
      'gold_call': '_oracle_phonon_rate_series(hamiltonian0.copy(), exchange.copy(), '
                   'coupling_operator.copy(), alpha, cutoff, temperature, order)',
      'tol': 2e-08},
     {'setup': 'import numpy as np\n'
               '## Joint rate Taylor geometry 9\n'
               'hamiltonian0 = np.array([[(0.11385656856929506+0j), '
               '(0.1491536816151687-0.05518590964066791j), (0.386116556749726-0.049189872971196126j)], '
               '[(0.1491536816151687+0.0551859096406679j), (0.13150433447579543-3.469446951953614e-18j), '
               '(0.19786284283766017-0.26575768295921315j)], [(0.386116556749726+0.04918987297119611j), '
               '(0.19786284283766015+0.26575768295921315j), (-0.09536090304509033+0j)]], dtype=complex)\n'
               'exchange = np.array([[(-0.14660220983472447+0j), '
               '(-0.3511292222718858+0.10715377524203093j), (-0.32858492734498557+0.11097652618282945j)], '
               '[(-0.3511292222718858-0.10715377524203093j), (-0.13134621436003638+0j), '
               '(-0.05776944766614998-0.25072821127522843j)], [(-0.32858492734498557-0.11097652618282945j), '
               '(-0.05776944766614998+0.25072821127522843j), (-0.34818334958378094+0j)]], dtype=complex)\n'
               'coupling_operator = np.array([[(0.7021742791711955+3.0232598033628757e-18j), '
               '(0.003485080379529275-0.20354409881712016j), (-0.03232645752086131-0.4082134429893682j)], '
               '[(0.003485080379529275+0.20354409881712013j), (0.05902002847133823+1.330956421446535e-18j), '
               '(0.11817119985283425-0.011396766532521904j)], [(-0.03232645752086131+0.4082134429893682j), '
               '(0.11817119985283425+0.0113967665325219j), (0.2388056923574664+3.7394962002735966e-19j)]], '
               'dtype=complex)\n'
               'alpha = 0.028\n'
               'cutoff = 2.16\n'
               'temperature = 6.6\n'
               'order = 0\n',
      'call': 'phonon_rate_series(hamiltonian0.copy(), exchange.copy(), coupling_operator.copy(), alpha, '
              'cutoff, temperature, order)',
      'gold_call': '_oracle_phonon_rate_series(hamiltonian0.copy(), exchange.copy(), '
                   'coupling_operator.copy(), alpha, cutoff, temperature, order)',
      'tol': 2e-08},
     {'setup': 'import numpy as np\n'
               '## Joint rate Taylor geometry 10\n'
               'hamiltonian0 = np.array([[(-0.8225967510440891+0j), 0j], [0j, (0.8225967510440891+0j)]], '
               'dtype=complex)\n'
               'exchange = np.array([[0j, (1+0j)], [(1+0j), 0j]], dtype=complex)\n'
               'coupling_operator = np.array([[0j, 0j], [0j, (1+0j)]], dtype=complex)\n'
               'alpha = 0.027\n'
               'cutoff = 2.2\n'
               'temperature = 4.0\n'
               'order = 6\n',
      'call': 'phonon_rate_series(hamiltonian0.copy(), exchange.copy(), coupling_operator.copy(), alpha, '
              'cutoff, temperature, order)',
      'gold_call': '_oracle_phonon_rate_series(hamiltonian0.copy(), exchange.copy(), '
                   'coupling_operator.copy(), alpha, cutoff, temperature, order)',
      'tol': 2e-08}]
