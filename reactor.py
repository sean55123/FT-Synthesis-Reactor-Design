import numpy as np

def safe_exp(x):
    return np.exp(np.minimum(x, 700))

def compute_derivatives(Y, constants):
    # Extract constants
    To = constants['To']
    Ta0 = constants['Ta0']
    z = constants['z']
    Nt = constants['Nt']
    mc = constants['mc']
    H2in = constants['H2in']
    alpha = constants['alpha']
    k1 = constants['k1']
    k6m = constants['k6m']
    K2 = constants['K2']
    K3 = constants['K3']
    K4 = constants['K4']
    PT = constants['PT']
    Ac_1 = constants['Ac_1']
    Ao_1 = constants['Ao_1']
    a = constants['a']
    bd = constants['bd']
    Sc = constants['Sc']
    Ut = constants['Ut']
    Us = constants['Us']

    T = Y[53]  # Process temperature
    Ta = Y[54]  # Coolant temperature
    init = Y.copy()
    
    T = max(T, 1e-6)
    Ta = max(Ta, 1e-6)

    R_gas = 8.314  # J/mol-K

    k5m = 1.4e3 * safe_exp(-92890 / (R_gas * T))
    k5 = 2.74e2 * safe_exp(-87010 / (R_gas * T))
    k6 = 1.5e6 * safe_exp(-111040 / (R_gas * T))
    k6e = k6
    k5e = k5
    kv = 1.57e1 * safe_exp(-45080 / (R_gas * T))

    Kp = safe_exp(
        5078.0045 / T - 5.8972089
        + 1.3958689e-3 * T
        - 2.7592844e-6 * T ** 2
    )

    # Kinetics calculations
    FT = np.sum(init[:-2])
    P = init[:-2] * (PT / FT) * (T / Ta0)
    P_CO, P_H2, P_H2O, P_CO2 = P[0], P[1], P[2], P[3]

    A = P_CO * P_H2O / np.sqrt(P_H2)
    B = P_CO2 * np.sqrt(P_H2) / Kp
    A1 = (1 / (K2 * K3 * K4)) * (P_H2O / (P_H2 ** 2))
    A2 = (1 / (K3 * K4)) * (1 / P_H2)
    A3 = 1 / K4

    # Reaction rate constants sum
    RRR = k1 * P_CO + k5 * P_H2 + k6

    upper_1 = (k1 * P_CO) / RRR
    
    # Compute beta factors
    i_vals = np.arange(1, 26)
    betaf = alpha ** i_vals * upper_1

    beta = np.ones_like(betaf)
    var = init[5:55:2].flatten()

    # Avoid division by zero
    denom_beta = betaf + k6m / k6 * var * (PT / FT) * (T / Ta0)
    denom_beta = np.where(denom_beta == 0, 1e-6, denom_beta)
    beta = (k6m / k6) * var * (PT / FT) * (T / Ta0) / denom_beta
    alpha_prob = i_vals * (1 - alpha) ** 2 * alpha ** (i_vals - 1)
    Deno = 1 + (1 + A1 + A2 + A3) * np.sum(alpha_prob)

    # Reaction rates
    r_olef = (k5e * P_H2 * alpha_prob[1:]) / Deno
    r_olef = r_olef.flatten()
    r_paraf = (k6e * (1 - beta[1:]) * alpha_prob[1:]) / Deno
    r_paraf = r_paraf.flatten()

    r_co2 = (kv * (A - B)) / (1 + kv * A)
    r_ch4 = (k5m * P_H2 * alpha_prob[0]) / Deno

    # Species rates
    r_co = -r_co2 - r_ch4 - np.sum((i_vals[1:] + 1) * (r_olef + r_paraf))
    r_h2 = (
        r_co2
        - 3 * r_ch4
        - np.sum((2 * (i_vals[1:] + 1) + 1) * r_olef + 2 * (i_vals[1:] + 1) * r_paraf)
    )
    r_h2o = -r_co2 + r_ch4 + np.sum((i_vals[1:] + 1) * (r_olef + r_paraf))

    R = np.zeros(53)
    R[0] = r_co
    R[1] = r_h2
    R[2] = r_h2o
    R[3] = r_co2
    R[4] = r_ch4
    R[5::2] = r_paraf  # Even indices for paraffins
    R[6::2] = r_olef   # Odd indices for olefins

    # Energy balance
    # Enthalpy changes (dHr) and heat capacities (CP)
    dHr = np.array([
         4.10953e4, -7.4399e4, -2.09725e5, -8.3684e4, -4.83377e5, -1.0451e5,
        -7.54025e5, -1.25586e5, -1.01496e6, -1.46522e5, -1.27689e6, 
        -1.66669e5, -1.54408e6, -1.91349e5, -1.80075e6, -2.55233e5, 
        -2.15835e6, -2.35357e5, -2.42959e6, -2.65858e5, -2.61318e6, 
        -2.69991e5, -2.85974e6, -2.90248e5, -3.10372315e6, -3.112637e5,
        -3.36571e6, -3.31899942e5, -3.6283056e6, -3.525366e5, -3.891285e6,
        -3.7356242e5, -4.152388592e6, -3.93809504e5, -4.41448e6, -4.144457e5,
        -4.67657e6, -4.35084e5, -4.938562992e6, -4.5572032e5, -5.199596468e6,
        -4.7702578e5, -5.46185e6, -4.976917e5, -5.7235398e6, -4.12429e5, 
        -5.9859305e6, -5.3912386e5, -6.4840277e6, -5.5978978e5
    ])

    CP = []
    try:  # Ideal gas thermodynamic model for cp evaluation #cal/mol-K CP[-1]: N2
        CP.append(6.95233 + 2.09540*(((3085.1/T)/np.sinh(3085.1/T))**2) + 2.01951*(((1538.2/T)/np.cosh(1538.2/T))**2)) # CO
        CP.append(6.59621 + 2.28337*(((2466/T)/np.sinh(2466/T))**2) + 0.89806*(((567.6/T)/np.cosh(567.6/T))**2)) # H2
        CP.append(7.96862 + 6.39868*(((2610.5/T)/np.sinh(2610.5/T))**2) + 2.12477*(((1169/T)/np.cosh(1169/T))**2)) # H2O
        CP.append(7.014904 + 8.249737*(((1428/T)/np.sinh(1428/T))**2)+ 6.305532*(((588/T)/np.cosh(588/T))**2)) # CO2
        CP.append(7.953091 + 19.09167*(((2086.9/T)/np.sinh(2086.9/T))**2) + 9.936467*(((991.96/T)/np.cosh(991.96/T))**2)) # CH4
        CP.append(7.972676 + 22.6402*(((1596/T)/np.sinh(1596/T))**2) + 13.16041*(((740.8/T)/np.cosh(740.8/T))**2)) # C2H4
        CP.append(10.57036 + 20.23908*(((872.24/T)/np.sinh(872.24/T))**2) + 16.03373*(((2430.4/T)/np.cosh(2430.4/T))**2))
        CP.append(10.47387 + 35.97019*(((1398.8/T)/np.sinh(1398.8/T))**2) + 17.85469*(((616.46/T)/np.cosh(616.46/T))**2))
        CP.append(14.20512 + 30.24028*(((844.31/T)/np.sinh(844.31/T))**2) + 20.58016*(((2482.7/T)/np.cosh(2482.7/T))**2))
        CP.append(15.34752 + 49.24525*(((1676.8/T)/np.sinh(1676.8/T))**2) + 31.8238*(((757.06/T)/np.cosh(757.06/T))**2))
        CP.append(19.14445 + 38.79335*(((841.49/T)/np.sinh(841.49/T))**2) + 25.25795*(((2476.1/T)/np.cosh(2476.1/T))**2))
        CP.append(19.71028+ 61.96379*(((1729.1/T)/np.sinh(1729.1/T))**2) + 42.228*(((778.7/T)/np.cosh(778.7/T))**2))
        CP.append(21.03038 + 71.91650*(((1650.2/T)/np.sinh(1650.2/T))**2) + 45.18964*(((747.6/T)/np.cosh(747.6/T))**2))
        CP.append(24.92118 + 73.44272*(((1745.9/T)/np.sinh(1745.9/T))**2) + 49.50798*(((793.53/T)/np.cosh(793.53/T))**2))
        CP.append(24.93551 + 84.14541*(((1694.6/T)/np.sinh(1694.6/T))**2) + 56.58259*(((761.6/T)/np.cosh(761.6/T))**2))
        CP.append(28.30563 + 86.84914*(((1735.9/T)/np.sinh(1735.9/T))**2) + 59.82612*(((785.73/T)/np.cosh(785.73/T))**2))
        CP.append(28.69733 + 95.56224*(((1676.6/T)/np.sinh(1676.6/T))**2) + 65.44378*(((756.4/T)/np.cosh(756.4/T))**2))
        CP.append(32.48065 + 99.37184*(((1731.7/T)/np.sinh(1731.7/T))**2) + 68.48906*(((784.47/T)/np.cosh(784.47/T))**2))
        CP.append(32.37317 + 105.8326*(((1635.6/T)/np.sinh(1635.6/T))**2) + 72.94354*(((746.4/T)/np.cosh(746.4/T))**2))
        CP.append(36.66762 + 111.885*(((1728.8/T)/np.sinh(1728.8/T))**2) + 77.15678*(((783.67/T)/np.cosh(783.67/T))**2))
        CP.append(36.24486 + 117.3928*(((1644.8/T)/np.sinh(1644.8/T))**2) + 82.87953*(((749.6/T)/np.cosh(749.6/T))**2))
        CP.append(40.84504 + 124.4124*(((1726.5/T)/np.sinh(1726.5/T))**2) + 85.82927*(((782.92/T)/np.cosh(782.92/T))**2))
        CP.append(39.93503 + 127.8542*(((1614.1/T)/np.sinh(1614.1/T))**2) + 90.33152*(((742/T)/np.cosh(742/T))**2))
        CP.append(124.1163 + 57.65262*(((1834.6/T)/np.sinh(1834.6/T))**2) -161.8276 *(((278.28/T)/np.cosh(278.28/T))**2))
        CP.append(46.64422 + 145.6912*(((1708.7/T)/np.sinh(1708.7/T))**2) + 98.64813*(((775.4/T)/np.cosh(775.4/T))**2))
        CP.append(51.69342 + 102.0971*(((815.94/T)/np.sinh(815.94/T))**2) + 63.27028*(((2417.3/T)/np.cosh(2417.3/T))**2))
        CP.append(50.86223 + 158.4265*(((1715.5/T)/np.sinh(1715.5/T))**2) + 107.8652*(((777.5/T)/np.cosh(777.5/T))**2))            
        CP.append(53.3725 + 161.9853*(((1721.1/T)/np.sinh(1721.1/T))**2) + 111.8181*(((781.22/T)/np.cosh(781.22/T))**2))
        CP.append(51.34231 + 174.465*(((1669.5/T)/np.sinh(1669.5/T))**2) + 119.4182*(((741.02/T)/np.cosh(741.02/T))**2))
        CP.append(57.55231+ 174.508*(((1719.9/T)/np.sinh(1719.9/T))**2) + 120.4834*(((780.87/T)/np.cosh(780.87/T))**2))
        CP.append(55.13041 + 187.9192*(((1682.3/T)/np.sinh(1682.3/T))**2) + 130.1376*(((743.1/T)/np.cosh(743.1/T))**2))
        CP.append(56.08102 + 172.8193*(((1553.7/T)/np.sinh(1553.7/T))**2) + 127.8733*(((723.17/T)/np.cosh(723.17/T))**2))
        CP.append(58.94478 + 201.1369*(((1656.5/T)/np.sinh(1656.5/T))**2) + 139.8132*(((743.6/T)/np.cosh(743.6/T))**2))
        CP.append(59.85717 + 184.0212*(((1551.8/T)/np.sinh(1551.8/T))**2) + 136.5721*(((723.07/T)/np.cosh(723.07/T))**2))
        CP.append(62.77587 + 214.3236*(((1691.2/T)/np.sinh(1691.2/T))**2) + 149.6131*(((744.41/T)/np.cosh(744.41/T))**2))
        CP.append(63.67632 + 195.2064*(((1552.5/T)/np.sinh(1552.5/T))**2) + 145.5527*(((723.89/T)/np.cosh(723.89/T))**2))
        CP.append(66.58546 + 227.4936*(((1693.5/T)/np.sinh(1693.5/T))**2) + 159.1932*(((744.57/T)/np.cosh(744.57/T))**2))
        CP.append(67.50502 + 206.4894*(((1554.5/T)/np.sinh(1554.5/T))**2) + 154.6694*(((724.9/T)/np.cosh(724.9/T))**2))
        CP.append(70.46432 + 239.658*(((771.07/T)/np.sinh(771.07/T))**2) -102.7324 *(((916.73/T)/np.cosh(916.73/T))**2))
        CP.append(71.22146 + 217.3665*(((1546.3/T)/np.sinh(1546.3/T))**2) + 162.5752*(((723.07/T)/np.cosh(723.07/T))**2))
        CP.append(74.19031 + 252.5795*(((767.91/T)/np.sinh(767.91/T))**2) -109.0594 *(((912.03/T)/np.cosh(912.03/T))**2))
        CP.append(75.18152 + 229.3565*(((747.85/T)/np.sinh(747.85/T))**2) -70.4476 *(((864.56/T)/np.cosh(864.56/T))**2))
        CP.append(77.57954 + 264.8801*(((1636/T)/np.sinh(1636/T))**2) + 177.9402*(((726.27/T)/np.cosh(726.27/T))**2))
        CP.append(91.43499 + 184.1669*(((801.08/T)/np.sinh(801.08/T))**2) + 119.2032*(((2361.6/T)/np.cosh(2361.6/T))**2))
        CP.append(91.43499 + 184.1669*(((801.08/T)/np.sinh(801.08/T))**2) + 119.2032*(((2361.6/T)/np.cosh(2361.6/T))**2))
        CP.append(91.08388 + 274.6011*(((1715.3/T)/np.sinh(1715.3/T))**2) + 189.7774*(((779.78/T)/np.cosh(778.78/T))**2))
        CP.append(93.76135 + 282.3158*(((1723.4/T)/np.sinh(1723.4/T))**2) + 194.8457*(((785.13/T)/np.cosh(785.13/T))**2))
        CP.append(97.96264 + 294.7836*(((1723.1/T)/np.sinh(1723.1/T))**2) + 203.5636*(((784.97/T)/np.cosh(784.97/T))**2))
        CP.append(97.96264 + 294.7836*(((1723.1/T)/np.sinh(1723.1/T))**2) + 203.5636*(((784.97/T)/np.cosh(784.97/T))**2))
        CP.append(99.44827 + 299.6561*(((1714.4/T)/np.sinh(1714.4/T))**2) +207.1033*(((779.51/T)/np.cosh(779.51/T))**2))
        CP.append(102.0302 + 307.8962*(((815.29/T)/np.sinh(815.29/T))**2) -120.4213 *(((944.98/T)/np.cosh(944.98/T))**2))
        CP.append(106.3223+ 319.7908*(((1721.5/T)/np.sinh(1721.5/T))**2) + 220.8799*(((784.28/T)/np.cosh(784.28/T))**2)) # C25H50
        CP.append(106.3223+ 319.7908*(((1721.5/T)/np.sinh(1721.5/T))**2) + 220.8799*(((784.28/T)/np.cosh(784.28/T))**2)) # C25H52
        CP.append(6.95161 + 2.05763*(((1701.6/T)/np.sinh(1701.6/T))**2) + 0.0247*(((909.79/T)/np.cosh(909.79/T))**2)) # N2
        CP = np.array(CP)
    except OverflowError:
        CP = np.zeros(54)

    b = init[:-2].flatten()
    sumFiCpi = np.dot(b, CP[:53])

    dcp = np.zeros_like(dHr) 
    dH = dHr + dcp * (T - 298)

    # Heat flow
    Q = -R[3] * dH[0] + np.sum(R[4:] * dH[1:])

    # Reactor equations
    dFdz = R * Ac_1 * bd * Nt

    dTtdz = ((Ut * a * (Ta - T) - Q * Sc * bd) / sumFiCpi) * Ac_1  # Process temp derivative
    CP_oil = 0.4725 * T + 122.1  # J/kg-K, thermal oil CP
    dTsdz = (Nt * Us * Ao_1 * (Ta - T)) / (CP_oil * mc * z)

    dYdW = np.zeros_like(Y).flatten()
    dYdW[:-2] = dFdz
    dYdW[53] = dTtdz
    dYdW[54] = dTsdz     
    return dYdW