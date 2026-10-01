# Material_Science-Molecular_Modeling-41

## Background

A molecular probe transfers momentum to unresolved bath coordinates. Eliminating those coordinates yields a vector memory force and a correlated random force. When the bath and probe time scales overlap, replacing the memory by its integral changes finite-time velocity statistics.

The task uses a stationary Gaussian model with gyroscopic bath couplings. It does not impose detailed balance at fixed bath parameters or symmetrize positive-lag cross correlations. The fitted normalized correlation must satisfy C(0)=I, C′(0)=0 and symmetric negative C″(0); individual perturbed samples need not satisfy a noiseless finite-order relation.

The computational core builds a finite-dimensional Markovian embedding of the vector memory dynamics directly from the correlation data. Exact discretization and the Gaussian likelihood provide a trajectory-level validation of the reconstructed molecular model. Large atomistic simulations and reproduction of the article's published trajectory are outside this task.

## Problem

A planar probe in a thermostatted molecular environment has coupled velocity fluctuations whose memory cannot be represented by an instantaneous drag coefficient. Using the supplied correlation matrices and held-out trajectory, use the source's matrix-rational reconstruction to construct a real finite-dimensional stochastic model while retaining inertial short-time behavior. Recover the decaying modes, fit their matrix amplitudes with the required short-time constraints, and determine a compatible stationary covariance and thermal noise. The synthetic bath permits gyroscopic auxiliary couplings, so positive-lag cross correlations need not be symmetric. Propagate the resulting stochastic model exactly between the irregular observation times and compute the stationary negative log likelihood per scalar normalized velocity, including independent measurement noise. In the reasoning, report the continuous mode pairs, the number of state variables, the zero-time normalized memory matrix and the final score, and explain the realness, positive-realness and stationarity conditions used in the reconstruction. The singular reduction and thermal covariance completion certify that this correlation model admits a stationary stochastic realization; the observed Gaussian likelihood is invariant under compatible hidden-coordinate and noise-factor choices.

Inputs

The literal arrays below define the complete benchmark. The 128 two-by-two correlation samples are synthetic data with small decaying perturbations at positive lag; they are not exact samples of a finite-order model and are not measurements from the source article. The 24 two-component held-out velocities are independent of those perturbations. Fit the supplied samples; do not replace them by a generating model. Rows of each lag matrix have orientation ⟨v(t)v(0)ᵀ⟩. The zero-lag matrix fixes thermal normalization, and the fitted correlation must retain the inertial short-time conditions.

The scalar mass is 5, thermal energy is 1, correlation spacing is 0.4, and the 23 intervals are in the same reduced time units. Measurement covariance is already in normalized velocity units and is excluded from the correlation samples. Evaluate the density of y = v√(m/(k_BT)), include the first stationary observation and all Gaussian constants, and divide the total negative log likelihood by 48. Use zero mean without centering the finite trajectory or adding a units Jacobian.

Numerical conventions

Use the matrix-rational branch on a counterclockwise 64-point complex circle of radius 1.4, with initial real support indices 0 and 32, at most 18 support points, and a maximum Frobenius grid residual of 0.001. Nonreal support additions and weights are conjugate paired. For deterministic greedy selection, score a nonsupport conjugate orbit by its mean Frobenius residual; ties within 1e-12·max(1,largest score) select the smallest representative index from 0 through 32, inserted before its partner.

Use the principal logarithm and the strictly open alias band (−π/τ,π/τ). Retain discrete poles only for 1e-12 < |z| < 1−1e-10, discarding negative real poles; a pole is real within imaginary tolerance 1e-8 and conjugate pairing uses 1e-7. Average conjugate pairs before the logarithm. Order real rates by increasing value, then conjugate pairs by the positive member's real and imaginary parts, with the positive member first.

Use unweighted constrained least squares over every supplied lag and matrix entry. Use relative null-space rank threshold 1e-12 and retain residue singular values strictly above 1e-7·max(1,s_max). Use the stabilizing branch of the regular covariance-completion problem for the admissibility certificate. Equivalent hidden-coordinate bases and noise-factor rotations within that branch are acceptable. The default parameters of the final scoring function are rho=1.4, grid_size=64, fit_tolerance=0.001, max_support=18 and rank_tolerance=1e-7.

Report the continuous mode pairs and zero-time normalized memory matrix to at least four decimal places; the final scalar has absolute tolerance 2e-6.

Input arrays

```python
import numpy as np
data = {
    'cvv': np.array([[[0.2, 0.0], [0.0, 0.2]],
     [[0.16945275355461512, -0.0031061281066597324], [0.006318887056165058, 0.17215784336029188]],
     [[0.10228698581513448, -0.02294982511747461], [0.032306787674341315, 0.11678141426416602]],
     [[0.03535322878232866, -0.05537457355094818], [0.06718706424579374, 0.0664658578601601]],
     [[-0.012070503948953008, -0.08250120144868953], [0.09182743610774419, 0.03442788237776536]],
     [[-0.036567907394091034, -0.08967786383321005], [0.09367154682294565, 0.01629099765430479]],
     [[-0.04805115675008539, -0.07612552988109926], [0.07524453360736394, 0.0013469836284512177]],
     [[-0.052980487861631634, -0.04976334460550338], [0.04744033651063496, -0.014399023985480316]],
     [[-0.053573674685929565, -0.021844282384545315], [0.02084220030053809, -0.029663465237752137]],
     [[-0.04940535547554183, -1.4279943422836207e-05], [0.00035011170274973304, -0.03935188077887899]],
     [[-0.03915422712825922, 0.013814255171149876], [-0.01236497699941597, -0.04059198783712678]],
     [[-0.02496308971383378, 0.021156445919811882], [-0.020382843812545563, -0.03462103679445504]],
     [[-0.009249846254845411, 0.024686491046292593], [-0.024208678295770638, -0.023564731597636782]],
     [[0.004365438477382304, 0.025193834506066343], [-0.02508201946078254, -0.012288311204434854]],
     [[0.01358990676609818, 0.023031162770754537], [-0.02327059893817632, -0.003142777389224482]],
     [[0.018122885841297225, 0.01836972622396861], [-0.0184703043130576, 0.003931289913838741]],
     [[0.01873215766456137, 0.01183830314128188], [-0.01192020618640197, 0.008058969618742067]],
     [[0.016716952695203933, 0.004982808835726663], [-0.005172428744607777, 0.01061720087710268]],
     [[0.013218619742006825, -0.0009571182041697382], [0.0006421196033715314, 0.011361386530192908]],
     [[0.009086035078043791, -0.005229656083456315], [0.004915736183000062, 0.010582115084983463]],
     [[0.004830636209681205, -0.007547881529941231], [0.007565730075036082, 0.00851557517510024]],
     [[0.0007359987473980553, -0.008467173841718117], [0.007831263449807235, 0.005906063360175109]],
     [[-0.0022509390924260733, -0.007806776358392653], [0.007599086011296447, 0.002895564879486651]],
     [[-0.004721944583818917, -0.006203691780404964], [0.006293874655016793, 0.00017913119095960882]],
     [[-0.00563401891759615, -0.004506168716168017], [0.004532862513282481, -0.0018762386088536347]],
     [[-0.005623675093181799, -0.0026182951153545355], [0.002659209082206636, -0.002925842514936929]],
     [[-0.004788167623683006, -0.0005524811008411267], [0.0008970014639894886, -0.0033943354542712305]],
     [[-0.003616693700975458, 0.0006469192711624382], [-0.0007522478167353058, -0.0033516718421862162]],
     [[-0.002188476128449707, 0.0019439810271163801], [-0.0018834007125281215, -0.0027859983711293447]],
     [[-0.0007038256616400821, 0.0024633839224239553], [-0.002480797344634457, -0.0022709111605531012]],
     [[0.00020349630111010425, 0.002615696077141865], [-0.0024982768937633917, -0.001307052561986828]],
     [[0.0010832116729455527, 0.0023144301388396396], [-0.0021782710761860244, -0.0006492020796481774]],
     [[0.001704387842520097, 0.0016631006641346282], [-0.0017135325449676266, 0.00022722265144433808]],
     [[0.001624242152888081, 0.001021942449047799], [-0.0011219081065389103, 0.0006523458305635018]],
     [[0.0013264597869771804, 0.0005174761288357546], [-0.0005212118783658866, 0.0010132293824609802]],
     [[0.0013492189942925154, -0.000192073875027788], [1.4237017069651186e-05, 0.0012086499590651793]],
     [[0.0009909357975836257, -0.0005332406949934581], [0.00042307402351765555, 0.0009771426260838099]],
     [[0.000643101226689506, -0.0005553599790969417], [0.0006929626647005008, 0.0008231842460808193]],
     [[-1.0662035861166535e-05, -0.0006297788738246896], [0.00066564486982282, 0.0006112383690790388]],
     [[-8.74821614196258e-05, -0.000716593450257982], [0.0007616126615676449, 0.0004374551791716873]],
     [[-0.00033061558141287046, -0.0007187465600888608],
      [0.0005826897811688773, -6.257120921320456e-05]],
     [[-0.0005061538341057612, -0.0004606460547027496], [0.00053767976402595, -5.492366477692096e-05]],
     [[-0.0005010055318834733, -0.00025829400952672365],
      [0.0002890113954020254, -0.00020581486787535043]],
     [[-0.00047366194329932873, -2.8981293198352755e-05],
      [0.00014968437858105997, -0.00042411977425599117]],
     [[-0.00035024813171045244, 1.6434569195682753e-05],
      [-5.501974776150669e-05, -0.00039362309952963053]],
     [[-0.00013892955296638274, 4.686758526523353e-05],
      [-0.00020568669053105287, -0.00028285054789021217]],
     [[-0.0002003103943955653, 0.00039531752712794065],
      [-7.316636950648031e-05, -0.0002731729693580255]],
     [[2.2286696831127793e-05, 0.00015437860236684287],
      [-0.0001544354845687408, -9.563983878576827e-05]],
     [[0.0001440272157608188, 0.0001939925799713992],
      [-0.00019804613463913988, -3.690729057163481e-05]],
     [[0.00010252431489315893, 0.0001400003979238398],
      [-0.0001385116171799568, 2.4140381372999342e-05]],
     [[0.0001363407089332727, 0.00021272843964391242], [-6.406958715635323e-05, 7.958828852769995e-05]],
     [[0.00011630478588597312, -6.936128147111542e-05],
      [-1.8557028207100188e-05, 0.00025453608482457274]],
     [[6.22604773256754e-05, -4.1767845117361375e-05], [0.00010027246529721863, 0.0001165975462666937]],
     [[0.0001605395181351869, -5.408087416581433e-05], [0.00010328705484075705, 8.427168474534932e-05]],
     [[0.00010008097506184958, -0.00010731245895343017],
      [9.447450128903084e-05, 0.00014116820291291575]],
     [[-4.9150276985951924e-05, -0.00010609248660087736],
      [5.044545510272651e-05, 1.412408325984388e-05]],
     [[4.830971150547375e-05, -5.2118054146629344e-05],
      [0.00014691153985706433, 8.147231055826554e-05]],
     [[3.5216990260324576e-05, -5.740398182903633e-05],
      [2.992906091353924e-05, -0.0001678243295094388]],
     [[-2.142839148494098e-05, -3.844705509572901e-05],
      [-0.00011402473901201629, -0.0001409368712290601]],
     [[2.5578362040290127e-05, -8.660659896232096e-05],
      [0.00014428477844329399, 2.9892099298629904e-05]],
     [[-4.518713570257886e-05, 8.658337816386911e-05],
      [-1.7130929539254325e-06, -3.55135403653932e-05]],
     [[-1.5743338701147348e-05, 1.715407116781407e-05],
      [-2.2317017717851454e-05, -7.321969621187974e-05]],
     [[-6.24064550504235e-05, 3.0034007594653694e-05],
      [1.0421647162102535e-05, -7.322195707014262e-05]],
     [[-3.0135586111838027e-05, 2.333347818979117e-05],
      [-8.960611284407689e-05, -0.0001427388512389588]],
     [[3.763187750506885e-05, -3.089800736402216e-05],
      [-6.514831619996491e-05, -3.288379083309858e-05]],
     [[-1.2309791989966141e-05, -5.9158697383808875e-06],
      [-0.0001051774794279742, 9.423545818588685e-05]],
     [[-7.12066443649222e-05, -6.392464320962927e-05],
      [-9.312783442542194e-05, -7.521094145958501e-05]],
     [[1.4747072008726796e-05, 4.2474303883829376e-05], [6.40441282655833e-06, 1.981232320425809e-05]],
     [[2.505086805117769e-05, -1.0083372167592046e-05],
      [-2.137116682327815e-05, 3.157126405865931e-05]],
     [[-1.4363204030243868e-05, -1.4181274490354373e-05],
      [-9.024570388196097e-06, 8.234475658919713e-05]],
     [[4.358272132169892e-06, 5.972696569630652e-05], [1.880384036766271e-05, -4.7629632721311394e-05]],
     [[9.021311663896799e-05, 4.10473002886764e-05], [-6.411202634197987e-06, -8.595663954717196e-06]],
     [[-2.6222302165776125e-05, 1.189316154983247e-05],
      [-5.186911866133515e-05, 5.8073649290162005e-05]],
     [[-3.754797458576825e-05, -4.337158583213995e-05],
      [-3.0575129026285615e-07, 1.5043052756945144e-05]],
     [[-5.230600888614886e-07, -1.962725504434252e-05],
      [-7.295639387778146e-05, -2.446771720031154e-05]],
     [[-4.140288344963852e-05, 8.139750654482538e-05],
      [-3.601760617374465e-05, 1.5595674761334436e-05]],
     [[-3.571381856937219e-05, -5.506392012621399e-05],
      [-2.068400262475203e-05, 2.6654262625442626e-05]],
     [[8.486360858553457e-06, 2.2695217986930107e-05],
      [2.2944169412917867e-05, -8.503129724927262e-06]],
     [[8.910145697119443e-06, -2.7502930323898968e-05],
      [-2.6268555576725443e-05, -1.764460509942499e-05]],
     [[3.6714412863273056e-06, 4.358859443419548e-06],
      [-6.653291827063059e-06, 0.00010942534406397968]],
     [[-4.218862305083614e-06, 4.6782236827428836e-05],
      [-1.3941027638269417e-05, 4.7602860091795876e-05]],
     [[3.857757687181387e-05, -5.942653235718807e-06], [-1.19333109701094e-05, -7.395522538611696e-06]],
     [[3.8516849227630255e-07, -1.469311634794489e-05],
      [1.1469265810564096e-05, 6.542695560985904e-06]],
     [[8.785228140505579e-05, 6.138228884729674e-05],
      [-3.1628770201456747e-05, -5.917858506993576e-06]],
     [[7.471261312927948e-05, -2.192725750417227e-05],
      [-1.5544526056373188e-05, 3.489277101879881e-05]],
     [[3.79386352000514e-05, 2.7010485591287206e-05],
      [-2.2889964009117745e-05, -3.6083213836026665e-05]],
     [[8.156261071118911e-05, -1.6272000363119138e-05], [4.627895823582638e-05, 7.847118446969751e-05]],
     [[2.2860974538761207e-05, -1.4141919688006165e-05],
      [-4.001706628272636e-05, -7.302413299342011e-05]],
     [[1.8229104686196623e-05, 2.9479897141023017e-05],
      [1.2212558950728554e-05, 3.9128968995768535e-05]],
     [[-2.251559122461778e-06, 1.1918302253010489e-05],
      [3.4608254906484577e-05, -1.4480563031320118e-05]],
     [[2.222461157843098e-05, 4.602814137559041e-05],
      [-1.4863227706421655e-05, -3.464693079977688e-05]],
     [[1.1408027860383027e-05, -4.185027164871985e-05], [4.304678293009884e-05, 5.964120822948625e-06]],
     [[6.12494353188849e-05, -2.6007808053192416e-05],
      [-5.845231823022385e-05, -1.9566610694596033e-05]],
     [[3.2566842710366366e-05, 1.6777430826097137e-05],
      [-1.372275924821406e-05, -2.188898074738862e-05]],
     [[-5.637250310627042e-05, -2.62470658955132e-05],
      [3.071051220982115e-05, -1.3089120040882971e-05]],
     [[3.642534188890838e-05, 1.7572595380545352e-05], [3.9120027499985e-05, -1.4423170846364087e-05]],
     [[-1.4299087744479837e-05, 1.8535549680679776e-05],
      [-2.3288970298525078e-05, 9.515129395076562e-06]],
     [[-1.1357724123202616e-05, 5.88640823503951e-05], [1.0164899664124045e-05, 6.402909322379169e-06]],
     [[-1.3551733885690897e-05, -3.578453849905521e-05],
      [-3.1283674702305404e-05, -3.9374639050833186e-05]],
     [[6.417052270730976e-05, 2.1310466076685957e-05],
      [1.5464319676770254e-05, 1.3563275313497464e-05]],
     [[-2.201953653610577e-05, 1.178232746244161e-05], [2.1098045659187664e-05, -2.83176147528608e-05]],
     [[3.962249887041102e-05, 5.396108986690356e-05],
      [-2.2024136435024297e-05, -1.720105690887811e-06]],
     [[4.155651892090912e-06, -7.5818866508571404e-06],
      [-2.7133422342654934e-05, 8.10468392302828e-06]],
     [[1.1676037453984336e-05, -9.138721886691081e-06],
      [-2.7882086532147694e-05, 5.513964350417517e-06]],
     [[3.2002852338094223e-06, -1.7140538399323085e-05],
      [-8.094542689575464e-06, 1.340120869283062e-06]],
     [[-2.980684627323806e-07, 1.1716419130637584e-05],
      [-1.6206632817093632e-05, -4.731342504399433e-05]],
     [[-2.3158355626351732e-05, 2.710589318325983e-05], [5.264225007809326e-05, 4.124338615847594e-05]],
     [[1.0290761526286932e-05, 4.1515553887657316e-05],
      [-6.294504971748244e-05, 4.3643154408683365e-05]],
     [[-2.1409151236450712e-05, -1.0235777121399537e-05],
      [2.7635894154029304e-05, 1.4742028843041657e-05]],
     [[2.0741323336249596e-06, 1.5700713707866398e-06],
      [2.1680618667031612e-05, 2.994048005784835e-05]],
     [[1.6294243331539282e-06, -2.0826359928999633e-05],
      [2.7100375812963935e-06, -1.4626823832537979e-05]],
     [[-1.3361057533918406e-06, -1.5739266376055273e-05],
      [-1.4299336095268268e-05, 1.2189101049520563e-05]],
     [[7.25167944028748e-06, 2.6829799182446905e-05],
      [-2.7048571531943143e-05, 2.6521897562414144e-06]],
     [[-3.17044162624041e-06, 2.225123591869783e-05], [4.001806111426734e-06, 1.907243215885327e-05]],
     [[2.8881753603962484e-05, -8.298154271469826e-07],
      [-4.8265089210566464e-05, -2.823407490766972e-05]],
     [[2.0379687915479157e-05, 3.6854661440476254e-06],
      [-4.50252676606985e-06, 4.5976385947080476e-05]],
     [[-3.5366085249669806e-06, -3.539479228542985e-05],
      [2.052162066992743e-05, 3.4151067360658445e-05]],
     [[2.731714049776435e-06, -1.826350473882273e-05], [3.396541345137203e-05, 3.087081083574963e-05]],
     [[1.0614363222310238e-05, -3.155146282661812e-06],
      [-1.630762736184292e-05, -8.542453939331966e-06]],
     [[-2.2440704597324105e-05, -1.530636449161752e-05],
      [-6.772118654896633e-06, 8.10273719522035e-06]],
     [[1.19228659488092e-06, -2.027235404168733e-05], [-3.565112117731127e-06, 2.843439320956975e-06]],
     [[-1.2229911092714656e-05, -6.764466860102689e-08],
      [2.0824019026878105e-05, -2.0378801397439018e-05]],
     [[2.601326098229826e-05, -1.6327422493467892e-05],
      [-4.09096423632767e-06, -1.2796899034336425e-05]],
     [[1.5045833127684287e-05, 1.6666491039754603e-05],
      [-2.680538849628366e-07, 5.4337728578602895e-06]],
     [[3.0219458135166468e-05, -3.51967441406466e-05], [3.441366515601803e-05, 1.5197801936123761e-06]],
     [[4.495594750668476e-06, 1.890943851237289e-06], [6.626018485508577e-06, -5.510260973659996e-06]],
     [[-1.2083128833352775e-05, 3.8258955624314664e-05],
      [-1.103947572401589e-05, -6.207780531760683e-06]],
     [[1.2839974366087394e-05, 2.170310522139568e-06],
      [-1.4386189273414595e-05, -1.6968770273904225e-05]]], dtype=float),
    'tau': 0.4,
    'mass': 5.0,
    'thermal_energy': 1.0,
    'observations': np.array([[0.5164846654625908, 0.20130516002576324], [0.6337259009390972, 0.4720131379597723],
     [0.7452604227057262, 0.5077125414823482], [0.8147575451481152, 0.6061955494637903],
     [0.5627002676258996, 0.596849741670902], [0.2918826919485984, 0.6393957137990919],
     [0.14818510601615853, 0.6250724811303531], [-0.021659149255744966, 0.6248065703643814],
     [-0.3003608695936576, 0.6863868156319217], [-0.48138316031851064, 0.5710454338354477],
     [-0.674287706174415, 0.6204914835116102], [-0.6499272013346309, 0.4377196418488011],
     [-0.6509116948331924, 0.14082401859968596], [-0.723928169933927, -0.06439161054128857],
     [-0.6057395715165842, -0.061074791738953534], [-0.40782199086916904, -0.15086016695850402],
     [-0.35625064929434114, -0.149774759721444], [-0.3741949223688179, -0.18022312151511505],
     [-0.2601212184770717, -0.3610384397980052], [0.011395460334694988, -0.6431715418321573],
     [0.17692853767991987, -0.6953602419050365], [0.2337240870973197, -0.6835101741386617],
     [0.3397446404640645, -0.7515667486087267], [0.43028097402357973, -0.6803542182364358]], dtype=float),
    'intervals': np.array([0.22, 0.26, 0.3, 0.33999999999999997, 0.38, 0.22, 0.26, 0.3, 0.33999999999999997, 0.38, 0.22, 0.26,
     0.3, 0.33999999999999997, 0.38, 0.22, 0.26, 0.3, 0.33999999999999997, 0.38, 0.22, 0.26, 0.3], dtype=float),
    'observation_covariance': np.array([[0.008, 0.0024], [0.0024, 0.0136]], dtype=float),
    'rho': 1.4,
    'grid_size': 64,
    'fit_tolerance': 0.001,
    'max_support': 18,
    'rank_tolerance': 1e-07,
}
```

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 10 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_correlation_transform

Goal
----
Normalize a vector velocity correlation and evaluate its generating sum

```python
def correlation_transform(cvv: "np.ndarray", mass: float, thermal_energy: float, rho: float, grid_size: int) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Normalize a vector velocity correlation and evaluate its generating sum.

    Parameters
    ----------
    cvv : real array (n,d,d)
        n >= 4, d >= 1. cvv[k] = <v(k*tau) v(0)^T>, in squared
        velocity units. Positive and negative lags are not symmetrized.
    mass, thermal_energy : positive finite real scalars
        Common probe mass and k_B*T in compatible reduced units.
    rho : finite real scalar > 1
    grid_size : even integer >= 8

    Contract
    --------
    Return the thermally normalized correlation sequence, the complex
    sampling grid and the one-sided matrix generating function
    F(z) = sum over k = 0..n-1 of Y[k] * z**(-(k+1)), evaluated at every
    grid point using every supplied lag, so that a mode
    Y[k] = Gamma * exp(lambda*tau*k) contributes Gamma / (z - exp(lambda*tau)).
    The normalized zero-lag matrix must equal the identity within maximum
    entry error 1e-8. Grid points are uniformly spaced counterclockwise on
    the circle of radius rho, beginning at the positive real axis. Preserve
    the orientation of each cross-correlation matrix.

    Returns
    -------
    result
        Tuple (Y,z,F): real (n,d,d), complex (grid_size,), and complex
        (grid_size,d,d) arrays, in that order. Y and F are dimensionless.

    Raises
    ------
    ValueError
        Complex/nonfinite real input, incompatible shape, n<4, d<1,
        nonpositive mass or thermal_energy, rho<=1, invalid grid_size,
        or Y[0] outside the stated tolerance.
    """
    return result
```

### Step 2

02_shared_rational_fit

Goal
----
Fit matrix samples using a common scalar barycentric denominator.

```python
def shared_rational_fit(z: "np.ndarray", values: "np.ndarray", tolerance: float, max_support: int) -> "tuple[np.ndarray, np.ndarray]":
    """Fit matrix samples using a common scalar barycentric denominator.

    Parameters
    ----------
    z : complex array (g,)
        Counterclockwise uniform circle rho*exp(2*pi*i*l/g), even g>=8,
        rho>1. Circle agreement tolerance is 1e-10 in absolute entries.
    values : complex array (g,d,d), d>=1
        Conjugate symmetry values[(-l)%g]=conj(values[l]), tolerance 1e-9.
    tolerance : finite real scalar > 0
        Maximum Frobenius residual allowed over the entire grid.
    max_support : integer, 2<=max_support<=g-2
        Maximum number of support points, including conjugate partners.

    Contract
    --------
    Use the source's matrix-valued rational fit with a shared denominator.
    Initial support indices are [0,g//2]. Support and weights are
    conjugate paired, and real supports have real weights. Normalize
    the weight vector to Euclidean norm one; a common sign is immaterial.
    The convergence measure is the largest Frobenius residual on the grid.
    Nonsupport conjugate orbits are scored by their mean residual. Among
    orbits within 1e-12*max(1,largest_score), choose the smallest index
    in 0,...,g//2, inserted before its distinct conjugate partner.
    Any minimizing weights are accepted through the fitted function.

    Returns
    -------
    result
        Tuple (indices,weights): integer (p,) and complex (p,) arrays in
        support insertion order. 2<=p<=max_support, ||weights||_2=1.

    Raises
    ------
    ValueError
        Nonfinite inputs; invalid shapes, circle, conjugacy, tolerance or
        support budget; or residual tolerance not reached within budget.
    """
    return result
```

### Step 3

03_continuous_poles

Goal
----
Extract stable continuous exponents of a barycentric denominator.

```python
def continuous_poles(support: "np.ndarray", weights: "np.ndarray", tau: float) -> "np.ndarray":
    """Extract stable continuous exponents of a barycentric denominator.

    Parameters
    ----------
    support, weights : finite complex arrays (p,), p>=2
        Distinct support points and nonzero weights; same shape.
    tau : positive finite real scalar
        Correlation sampling interval in reduced time units.

    Contract
    --------
    Return the stable continuous exponents of the supplied denominator.
    Retain discrete poles only when 1e-12<abs(z)<1-1e-10; discard negative
    real poles. Real means |Im(z)|<=1e-8; conjugate pairing tolerance is
    1e-7. Average each pair with the conjugate of its partner.
    Use the principal logarithm and the open band |Im(lambda)|<pi/tau.
    Return real rates first, in increasing real part; then conjugate
    pairs sorted by the positive member's (real part, imaginary part),
    with the positive member first. Rates have inverse-time units.

    Returns
    -------
    result
        Complex array (r,) of stable rates in the specified order, r>=1.

    Raises
    ------
    ValueError
        Invalid shape, nonfinite values, duplicate support within 1e-12,
        zero weight, invalid tau, no retained poles, or unpaired pole.
    """
    return result
```

### Step 4

04_constrained_residues

Goal
----
Fit matrix exponential residues with inertial short-time constraints.

```python
def constrained_residues(Y: "np.ndarray", tau: float, rates: "np.ndarray") -> "np.ndarray":
    """Fit matrix exponential residues with inertial short-time constraints.

    Parameters
    ----------
    Y : finite real array (n,d,d), n>=4, d>=1
    tau : positive finite real scalar
    rates : finite complex array (r,), r>=1
        Distinct stable rates, conjugate closed, with |Im(rate)|<pi/tau.
        Real means |Im(rate)|<=1e-8. Pair and distinctness tolerance 1e-7.

    Contract
    --------
    Return the least-squares matrix amplitudes of the source's inertial
    exponential reconstruction, using every supplied lag and matrix
    entry with equal weight. The fitted curve has identity zero-lag
    covariance, zero initial slope and symmetric curvature at zero.
    Real rates have real residues and paired rates conjugate residues.
    The relative null-space rank threshold is 1e-12. Individual residue
    matrices need not be symmetric or positive.

    Returns
    -------
    result
        Complex array (r,d,d), ordered exactly as rates. Dimensionless.
        On exact-data fixtures, tests compare the sampled curve with Y
        (max abs error 2e-7) and check all three constraints
        (max abs residual 1e-7). Noisy-data fits compare the minimizer
        with rtol=2e-8 and atol=2e-8.

    Raises
    ------
    ValueError
        Complex Y, invalid shapes or nonfinite data, invalid tau, unstable,
        repeated, unpaired or out-of-band rates; inconsistent constraints
        (maximum constraint residual >1e-8); or a rank-deficient constrained fit.
    """
    return result
```

### Step 5

05_real_velocity_embedding

Goal
----
Construct a real minimal residue realization with velocity first.

```python
def real_velocity_embedding(rates: "np.ndarray", residues: "np.ndarray", rank_tolerance: float) -> "np.ndarray":
    """Construct a real minimal residue realization with velocity first.

    Parameters
    ----------
    rates : finite complex array (r,), stable and conjugate closed
    residues : finite complex array (r,d,d)
        Conjugate closed with rates, maximum entry tolerance 1e-7.
        Real rates have real residues. sum residues=I within 1e-7.
    rank_tolerance : finite real scalar in (0,1)

    Contract
    --------
    Return a real residue realization with the normalized velocities
    first and the retained auxiliary coordinates following them.
    Retain each residue singular value s strictly above
    rank_tolerance*max(1,s_max). Conjugate pairs use real rotation blocks
    with the positive-imaginary representative first; equivalent real
    hidden-coordinate bases are accepted. The retained zero-lag
    correlation must agree with the identity within 1e-6.
    Tests compare the correlation and basis identities, not individual
    entries in an arbitrary hidden-coordinate basis.

    Returns
    -------
    result
        Real drift matrix A, shape (q,q), q>d, inverse-time units.
        Its correlation is exp(A*t)[:d,:d].

    Raises
    ------
    ValueError
        Invalid shapes, nonfinite inputs, unstable/unpaired/repeated
        rates (pair tolerance 1e-7), inconsistent residue conjugacy or
        normalization, invalid rank_tolerance, retained q<=d, or singular X.
    """
    return result
```

### Step 6

06_singular_memory_reduction

Goal
----
Reduce a purely inertial embedding to a regular auxiliary noise problem.

```python
def singular_memory_reduction(A: "np.ndarray", dimension: int) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Reduce a purely inertial embedding to a regular auxiliary noise problem.

    Parameters
    ----------
    A : finite real drift array (q,q), q>=2*d
    dimension : integer d>=1
        Number of leading normalized velocity coordinates.

    Contract
    --------
    Return the singular-to-regular auxiliary reduction of an inertial
    realization. Partition A = [[0, B^T], [-C, A0]] with the first d
    coordinates the normalized velocities, so B = A[:d, d:].T,
    C = -A[d:, :d] and A0 = A[d:, d:]. The leading velocity block must be
    zero within 1e-7; all drift eigenvalues have real part below -1e-10.
    The induced zero-time memory S = B^T C must be symmetric within 1e-7
    and positive definite with minimum eigenvalue above 1e-10; let S^(1/2)
    be its symmetric positive square root. Build the deflation X, of size
    (q-d, q-d), by stacking S^(-1/2) B^T on top of the transpose of an
    orthonormal basis of the null space of C^T, so that X C = [S^(1/2); 0].
    Transform the auxiliary drift as V = X A0 X^(-1) and read the outputs
    from its blocks as V = [[-D1, B1^T], [-C1, A1]], with D1 the leading
    d x d block. The symmetric part D1+D1.T must have minimum eigenvalue
    >1e-10. Any orthonormal null-space basis is accepted; the result is
    checked through X C = [S^(1/2); 0], the first d rows of X equalling
    S^(-1/2) B^T, and X A0 = V X.

    Returns
    -------
    result
        Tuple (A1,B1,C1,D1,X). With h=q-2*d and a=q-d, the shapes are
        (h,h),(h,d),(h,d),(d,d),(a,a). All arrays are real; h may be zero.
        Hidden bases may differ, provided the defining identities hold.

    Raises
    ------
    ValueError
        Complex/nonfinite A, invalid square shape or dimension, q<2*d,
        nonzero velocity block, unstable A, nonsymmetric or nonpositive S,
        singular X, or nonpositive R1 under the stated tolerances.
    """
    return result
```

### Step 7

07_regular_lure

Goal
----
Solve the stabilizing regular Lur'e covariance completion.

```python
def regular_lure(A0: "np.ndarray", B: "np.ndarray", C: "np.ndarray", D: "np.ndarray") -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Solve the stabilizing regular Lur'e covariance completion.

    Parameters
    ----------
    A0 : finite real array (h,h), h>=0
    B, C : finite real arrays (h,d), d>=1
    D : finite real array (d,d)

    Contract
    --------
    Return the positive stationary covariance and shared-noise factors
    of the source's regular auxiliary problem, selecting its stabilizing
    covariance branch. The symmetric direct-damping part must have
    minimum eigenvalue >1e-10. K uses the lower-Cholesky convention.
    Stability means eigenvalue real parts below -1e-10. Reject Hamiltonian
    eigenvalues within 1e-10 of the imaginary axis, covariance minimum
    eigenvalue <=1e-10, or Riccati Frobenius residual greater than
    1e-7*max(1,||S||_F). For h=0 return empty S and L, together with K.

    Returns
    -------
    result
        Tuple (S,L,K): real arrays (h,h),(h,d),(d,d). S is dimensionless;
        L,K have inverse-square-root-time units in the rescaled state.

    Raises
    ------
    ValueError
        Complex/nonfinite input, incompatible shapes, nonpositive R,
        missing stable h-dimensional graph, singular U, imaginary-axis
        Hamiltonian spectrum, nonpositive covariance, unstable completed
        Riccati branch, or excessive residual under the stated thresholds.
    """
    return result
```

### Step 8

08_complete_discrete_embedding

Goal
----
Complete the stationary covariance and integrate the embedded noise.

```python
def complete_discrete_embedding(A: "np.ndarray", dimension: int, X: "np.ndarray", S1: "np.ndarray", L1: "np.ndarray", K1: "np.ndarray", intervals: "np.ndarray") -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Complete the stationary covariance and integrate the embedded noise.

    Parameters
    ----------
    A : finite real stable drift (q,q), q>=2*d
    dimension : integer d>=1
    X : finite real invertible auxiliary transformation (q-d,q-d)
    S1, L1, K1 : finite real arrays (q-2*d,q-2*d),(q-2*d,d),(d,d)
        The covariance and noise factors from the reduced Lur'e problem.
    intervals : finite real array (j,), j>=0, each entry in (0,3]
        Time increments between successive held-out measurements.

    Contract
    --------
    Return the stationary covariance in the original state coordinates
    and exact finite-interval transition and process-noise covariances.
    Keep all coupled noise contributions. Sigma is symmetric within
    1e-7 and has minimum eigenvalue >1e-10; drift eigenvalue real parts
    are below -1e-10. The continuous stationarity residual has Frobenius
    norm at most 1e-6*max(1,||Sigma||_F). Return symmetric process-noise
    covariances with minimum eigenvalue no smaller than -1e-8.
    Exact covariance identities or exact integration are acceptable;
    no diagonal regularizer is part of the model.

    Returns
    -------
    result
        Tuple (Sigma,T,Q): real (q,q), (j,q,q), (j,q,q) arrays. Sigma
        and Q are covariances of the dimensionless state. For j=0, the
        last two arrays retain shape (0,q,q).

    Raises
    ------
    ValueError
        Complex/nonfinite input, incompatible shapes, invalid dimension
        or intervals, singular X, nonpositive/nonsymmetric covariance,
        unstable A, failed stationary Lyapunov identity, or a transition
        covariance with minimum eigenvalue below -1e-8.
    """
    return result
```

### Step 9

09_velocity_innovation_score

Goal
----
Evaluate the stationary Gaussian innovation likelihood of probe velocities.

```python
def velocity_innovation_score(Sigma: "np.ndarray", transitions: "np.ndarray", noise_covariances: "np.ndarray", observations: "np.ndarray", observation_covariance: "np.ndarray") -> float:
    """Evaluate the stationary Gaussian innovation likelihood of probe velocities.

    Parameters
    ----------
    Sigma : finite real stationary covariance (q,q), positive definite
    transitions, noise_covariances : finite real arrays (m-1,q,q)
    observations : finite real array (m,d), m>=1, 1<=d<=q
        Dimensionless velocities v*sqrt(mass/(k_B*T)).
    observation_covariance : finite real symmetric positive definite (d,d)
        Independent measurement-error covariance in those same units.

    Contract
    --------
    Evaluate the complete joint stationary Gaussian likelihood of all
    measurements of the first d state coordinates, with zero mean and
    initial state covariance Sigma. Use each supplied transition and
    process covariance for its corresponding interval and include the
    independent measurement covariance, the initial observation and
    Gaussian normalization constants. Divide by m*d. Equivalent dense
    joint-covariance and sequential-conditioning calculations are valid.
    Symmetry tolerance is 1e-7; positive definite means minimum >1e-10;
    process covariances may be semidefinite down to -1e-8.

    Returns
    -------
    result
        Finite Python float, negative log likelihood in nats per scalar
        normalized velocity. Negative values are allowed for a density.

    Raises
    ------
    ValueError
        Complex/nonfinite input, incompatible shapes, nonsymmetric or
        nonpositive Sigma/R, nonsymmetric or non-semidefinite Q under
        the stated tolerances, or a nonpositive innovation covariance.
    """
    return result
```

### Step 10

10_molecular_memory_score

Goal
----
Reconstruct an inertial vector memory model and score a held-out trajectory.

```python
def molecular_memory_score(cvv: "np.ndarray", tau: float, mass: float, thermal_energy: float, observations: "np.ndarray", intervals: "np.ndarray", observation_covariance: "np.ndarray", rho: float = 1.4, grid_size: int = 64, fit_tolerance: float = 0.001, max_support: int = 18, rank_tolerance: float = 1e-07) -> float:
    """Reconstruct an inertial vector memory model and score a held-out trajectory.

    Parameters
    ----------
    cvv : real (n,d,d), n>=4, normalized zero-lag mass*cvv[0]/k_B*T=I
    tau : positive correlation sample spacing
    mass, thermal_energy : positive mass and k_B*T
    observations : real (m,d), m>=1, physical velocities
    intervals : real (m-1,), increments in (0,3] between observations
    observation_covariance : real positive definite (d,d)
        Measurement covariance of normalized velocities, already scaled.
    rho=1.4, grid_size=64, fit_tolerance=0.001, max_support=18,
    rank_tolerance=1e-7 : numerical parameters
        Their domains and conventions are defined in the earlier steps.

    Definition
    ----------
    Execute correlation_transform, shared_rational_fit, continuous_poles,
    constrained_residues, real_velocity_embedding, singular_memory_reduction,
    regular_lure, complete_discrete_embedding, and velocity_innovation_score,
    in that order. Multiply observations by sqrt(mass/thermal_energy).
    The likelihood is a density in normalized velocity coordinates; do
    not add a change-of-units Jacobian. Preserve every cross correlation.
    Inputs must admit the stable positive-real inertial realization with
    q>=2*d specified by those steps. All their rejection conditions apply.
    Final score comparisons use absolute tolerance 2e-6. A numerically
    equivalent realization is accepted. The fitted parameters, covariance,
    and noise must be derived from these inputs, without stored answers.
    Import required packages inside the function.

    Returns
    -------
    result
        Finite Python float: stationary negative log likelihood per scalar
        normalized velocity, in nats.

    Raises
    ------
    ValueError
        Any invalid input or failed admissibility condition declared by
        an earlier step; in addition, observation dimension must equal the
        cvv dimension and intervals must have length len(observations)-1.
    """
    return result
```
