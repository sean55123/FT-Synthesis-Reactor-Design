import math
import numpy as np
from scipy.integrate import odeint

class FT_PBR:
    def __init__(self, init, To, Ta0, z_1, Nt_1, mc_1, H2in, alpha):
        self.init = init # [1, 2, ..., 25, 26, ..., T_1(53), T_a(54)]
        self.To = To
        self.Ta0 = Ta0
        self.z_1 = z_1
        self.Nt_1 = Nt_1
        self.mc_1 = mc_1
        self.H2in = H2in
        self.alpha = alpha
        
        self.k1_1 = 2.33 * 1e-5
        self.k1 = 0.06 * 1e-5
        self.k6m = 2.74 * 1e-3
        self.K2 = 0.0025 * 1e-2
        self.K3 = 4.68 * 1e-2
        self.K4 = 0.8
        self.Kv = 1.13 * 1e-3
        self.PT = 20.92
        self.NH3 = 0 
        self.N2 = 0
        self.R = 8.314
        self.bd = 1.64 * 1e6
        
        try:
            self.k5m = 1.4*(10**3)*math.exp(-92890/(8.314*init[53]))#92890 4.65 1.4
            self.k5 = 2.74*(10**2)*math.exp(-87010/(8.314*init[53]))  #2.74 87010
            self.k5e = 2.74*(10**2)*math.exp(-87010/(8.314*init[53]))  #2.74 87010
            self.k6 = 1.5*(10**6)*math.exp(-111040/(8.314*init[53]))  #2.66 111040 0.5
            self.k6e = 1.5*(10**6)*math.exp(-111040/(8.314*init[53]))
            self.kv = 1.57*(10)*math.exp(-45080/(8.314*init[53]))
        except OverflowError:
            self.k5m = 0
            self.k5 = 0
            self.k5e = 0
            self.k6 = 0
            self.k6e = 0
            self.kv = 0
        
        try:
            self.Kp = (math.exp(5078.0045/init[53] - 5.8972089 + (13.958689*(10**(-4))*init[53])-(27.592844*(10**(-8))*(init[53]**2))))
        except OverflowError:
            self.Kp = 0
        
    def kinetics(self):
        FT = sum(self.init[-2])
        A = (self.init[0]*(self.PT/FT)*(self.init[53]/self.init[54]))*(self.init[2]*(self.PT/FT)*(self.init[53]/self.init[54]))/((self.init[1]*(self.PT/FT)*(self.init[53]/self.init[54]))**0.5)
        B = (self.init[3]*(self.PT/FT)*(self.init[53]/self.init[54]))*((self.init[1]*(self.PT/FT)*(self.init[53]/self.init[54]))**0.5)/self.Kp
        A1 = (1/(self.K2*self.K3*self.K4))*((self.init[2]*(self.PT/FT)*(self.init[53]/self.init[54]))/((self.init[1]*(self.PT/FT)*(self.init[53]/self.init[54]))**2)) 
        A2 = (1/(self.K3*self.K4))*(1/(self.init[1]*(self.PT/FT)*(self.init[53]/self.init[54])))
        A3 = 1/self.K4 
        
        # RRR = k1Pco + k5PH2 + k6
        RRR = (self.k1*(self.init[0]*(self.PT/FT)*(self.init[53]/self.init[54]))+self.k5*(self.init[1]*(self.PT/FT)*(self.init[53]/self.init[54]))+self.k6)
        # k1Pco/(k1Pco+k5PH2)
        upper_1 = ((self.k1*(self.init[0]*(self.PT/FT)*(self.init[53]/self.init[54])))/(self.k1*(self.init[0]*(self.PT/FT)*(self.init[53]/self.init[54]))+self.k5*(self.init[1]*(self.PT/FT)*(self.init[53]/self.init[54]))))
        
        betaf = []
        for i in range(23):
            betaf.append(self.alpha**(i+1))*upper_1
        betaf = np.array(betaf)
        
        betai = np.ones(len(self.init) - 7)
        for i in range(2, 26):
            b_sum = 0
            for j in range(i, 1, -1):
                b_sum = (self.alpha ** (i-j)) * self.init[2*i+1] * (self.PT/FT) * (self.init[53] - self.To)
            
            betaf[i] = self.k6m * b_sum / RRR
        
        try:
            betas = betai + betaf
            beta = np.ones(len(betas))
            for i in range(len(betas)):
                beta[i] = (self.k6m/self.k6) * (self.init[(i+2)*2+1] * (self.PT/self.FT)*(self.init[53]/self.To))/betas[i]
        except ZeroDivisionError:
            beta = np.zeros(len(betas))
        
        alpha_prob = np.ones(25)
        for i in range(alpha_prob):
            alpha_prob[i] = (i+1) * ((1-self.alpha)**2) * (self.alpha**i)
        
        Chigh = 0
        for i in range(11, 26):
            product = 0
            for j in range(i):
                product *= alpha_prob[j]
            Chigh += product
        
        Deno = 0
        for i in range(1, 11):
            product = 0
            for j in range(j):
                product *= alpha_prob[j]
            Deno += product
        
        Deno = 1 + (1+A1+A2+A3) * (Deno+Chigh)
        
        r_olef = (self.k5e * (self.init[1] * (self.PT/self.FT)) * alpha_prob[1:]) / Deno
        r_paraf = (self.k6e * (1-beta) * alpha_prob[1:]) / Deno
        rCO2_1 = (self.kv*(A-B))/(1 + self.kv*A)
        r_ch4 = (self.k5m * (self.init[1] * (self.PT/self.FT) * alpha_prob[0])) / Deno
        
        r_co = -rCO2_1 - r_ch4
        for i in range(len(r_olef)):
            r_co -= ((i+2)*r_olef[i] + (i+2)*r_paraf)
        
        r_h2 = rCO2_1 - 3*r_ch4
        for i in range(1, 26):
            j = i + 1
            r_h2 -= (2*j*r_paraf[i-1] + (2*j+1)*r_olef[i-1])
        
        r_h2o = -rCO2_1 + r_ch4
        for i in range(r_olef):
            r_h2o += ((i+2)*r_olef[i] + (i+2)*r_paraf)
        return r_olef, r_paraf, rCO2_1, r_ch4, r_co, r_h2, r_h2o
                
    def energy_balance(self):
        # Enthalpy for specific component
        dHr = []
        dHr.append(41.0953 *1000) #J/mol
        dHr.append(-74.399 *1000)
        dHr.append(-209.725*1000)
        dHr.append(-83.684*1000)
        dHr.append(-483.377*1000)
        dHr.append(-104.51*1000)
        dHr.append(-754.025*1000)
        dHr.append(-125.586*1000)
        dHr.append(-1014.96*1000)
        dHr.append(-146.522*1000)
        dHr.append(-1276.89*1000)
        dHr.append(-166.669*1000)
        dHr.append(-1544.08*1000)
        dHr.append(-191.349*1000)
        dHr.append(-1800.75*1000)
        dHr.append(-255.233*1000)
        dHr.append(-2158.35*1000)
        dHr.append(-235.357*1000)
        dHr.append(-2429.59*1000)
        dHr.append(-265.858*1000)
        dHr.append(-2613.18*1000)
        dHr.append(-269.991*1000)
        dHr.append(-2859.74*1000)
        dHr.append(-290.248*1000)
        dHr.append(-3103.72315*1000)
        dHr.append(-311.2637*1000)
        dHr.append(-3365.71*1000)
        dHr.append(-331.899942*1000)
        dHr.append(-3628.3056*1000)
        dHr.append(-352.5366*1000)
        dHr.append(-3891.285*1000)
        dHr.append(-373.56242*1000)
        dHr.append(-4152.388592*1000)
        dHr.append(-393.809504*1000)
        dHr.append(-4414.48*1000)
        dHr.append(-414.4457*1000)
        dHr.append(-4676.57*1000)
        dHr.append(-435.084*1000)
        dHr.append(-4938.562992*1000)
        dHr.append(-455.72032*1000)
        dHr.append(-5199.596468*1000)
        dHr.append(-477.02578*1000)
        dHr.append(-5461.85*1000)
        dHr.append(-497.6917*1000)
        dHr.append(-5723.5398*1000)
        dHr.append(-412.429*1000)
        dHr.append(-5985.9305*1000)
        dHr.append(-539.12386*1000)
        dHr.append(-6484.0277*1000)
        dHr.append(-559.78978*1000)
        dHr = np.array(dHr)
        
        CP = []
        try:  # ideal gas for cp #cal/mol-K CP[-1]: N2
            CP.append(6.95233 + 2.09540*(((3085.1/self.init[53])/math.sinh(3085.1/self.init[53]))**2) + 2.01951*(((1538.2/self.init[53])/math.cosh(1538.2/self.init[53]))**2))
            CP.append(6.59621 + 2.28337*(((2466/self.init[53])/math.sinh(2466/self.init[53]))**2) + 0.89806*(((567.6/self.init[53])/math.cosh(567.6/self.init[53]))**2))
            CP.append(7.96862 + 6.39868*(((2610.5/self.init[53])/math.sinh(2610.5/self.init[53]))**2) + 2.12477*(((1169/self.init[53])/math.cosh(1169/self.init[53]))**2))
            CP.append(7.014904 + 8.249737*(((1428/self.init[53])/math.sinh(1428/self.init[53]))**2)+ 6.305532*(((588/self.init[53])/math.cosh(588/self.init[53]))**2))
            CP.append(7.953091 + 19.09167*(((2086.9/self.init[53])/math.sinh(2086.9/self.init[53]))**2) + 9.936467*(((991.96/self.init[53])/math.cosh(991.96/self.init[53]))**2))
            CP.append(7.972676 + 22.6402*(((1596/self.init[53])/math.sinh(1596/self.init[53]))**2) + 13.16041*(((740.8/self.init[53])/math.cosh(740.8/self.init[53]))**2))
            CP.append(10.57036 + 20.23908*(((872.24/self.init[53])/math.sinh(872.24/self.init[53]))**2) + 16.03373*(((2430.4/self.init[53])/math.cosh(2430.4/self.init[53]))**2))
            CP.append(10.47387 + 35.97019*(((1398.8/self.init[53])/math.sinh(1398.8/self.init[53]))**2) + 17.85469*(((616.46/self.init[53])/math.cosh(616.46/self.init[53]))**2))
            CP.append(14.20512 + 30.24028*(((844.31/self.init[53])/math.sinh(844.31/self.init[53]))**2) + 20.58016*(((2482.7/self.init[53])/math.cosh(2482.7/self.init[53]))**2))
            CP.append(15.34752 + 49.24525*(((1676.8/self.init[53])/math.sinh(1676.8/self.init[53]))**2) + 31.8238*(((757.06/self.init[53])/math.cosh(757.06/self.init[53]))**2))
            CP.append(19.14445 + 38.79335*(((841.49/self.init[53])/math.sinh(841.49/self.init[53]))**2) + 25.25795*(((2476.1/self.init[53])/math.cosh(2476.1/self.init[53]))**2))
            CP.append(19.71028+ 61.96379*(((1729.1/self.init[53])/math.sinh(1729.1/self.init[53]))**2) + 42.228*(((778.7/self.init[53])/math.cosh(778.7/self.init[53]))**2))
            CP.append(21.03038 + 71.91650*(((1650.2/self.init[53])/math.sinh(1650.2/self.init[53]))**2) + 45.18964*(((747.6/self.init[53])/math.cosh(747.6/self.init[53]))**2))
            CP.append(24.92118 + 73.44272*(((1745.9/self.init[53])/math.sinh(1745.9/self.init[53]))**2) + 49.50798*(((793.53/self.init[53])/math.cosh(793.53/self.init[53]))**2))
            CP.append(24.93551 + 84.14541*(((1694.6/self.init[53])/math.sinh(1694.6/self.init[53]))**2) + 56.58259*(((761.6/self.init[53])/math.cosh(761.6/self.init[53]))**2))
            CP.append(28.30563 + 86.84914*(((1735.9/self.init[53])/math.sinh(1735.9/self.init[53]))**2) + 59.82612*(((785.73/self.init[53])/math.cosh(785.73/self.init[53]))**2))
            CP.append(28.69733 + 95.56224*(((1676.6/self.init[53])/math.sinh(1676.6/self.init[53]))**2) + 65.44378*(((756.4/self.init[53])/math.cosh(756.4/self.init[53]))**2))
            CP.append(32.48065 + 99.37184*(((1731.7/self.init[53])/math.sinh(1731.7/self.init[53]))**2) + 68.48906*(((784.47/self.init[53])/math.cosh(784.47/self.init[53]))**2))
            CP.append(32.37317 + 105.8326*(((1635.6/self.init[53])/math.sinh(1635.6/self.init[53]))**2) + 72.94354*(((746.4/self.init[53])/math.cosh(746.4/self.init[53]))**2))
            CP.append(36.66762 + 111.885*(((1728.8/self.init[53])/math.sinh(1728.8/self.init[53]))**2) + 77.15678*(((783.67/self.init[53])/math.cosh(783.67/self.init[53]))**2))
            CP.append(36.24486 + 117.3928*(((1644.8/self.init[53])/math.sinh(1644.8/self.init[53]))**2) + 82.87953*(((749.6/self.init[53])/math.cosh(749.6/self.init[53]))**2))
            CP.append(40.84504 + 124.4124*(((1726.5/self.init[53])/math.sinh(1726.5/self.init[53]))**2) + 85.82927*(((782.92/self.init[53])/math.cosh(782.92/self.init[53]))**2))
            CP.append(39.93503 + 127.8542*(((1614.1/self.init[53])/math.sinh(1614.1/self.init[53]))**2) + 90.33152*(((742/self.init[53])/math.cosh(742/self.init[53]))**2))
            CP.append(124.1163 + 57.65262*(((1834.6/self.init[53])/math.sinh(1834.6/self.init[53]))**2) -161.8276 *(((278.28/self.init[53])/math.cosh(278.28/self.init[53]))**2))
            CP.append(46.64422 + 145.6912*(((1708.7/self.init[53])/math.sinh(1708.7/self.init[53]))**2) + 98.64813*(((775.4/self.init[53])/math.cosh(775.4/self.init[53]))**2))
            CP.append(51.69342 + 102.0971*(((815.94/self.init[53])/math.sinh(815.94/self.init[53]))**2) + 63.27028*(((2417.3/self.init[53])/math.cosh(2417.3/self.init[53]))**2))
            CP.append(50.86223 + 158.4265*(((1715.5/self.init[53])/math.sinh(1715.5/self.init[53]))**2) + 107.8652*(((777.5/self.init[53])/math.cosh(777.5/self.init[53]))**2))            
            CP.append(53.3725 + 161.9853*(((1721.1/self.init[53])/math.sinh(1721.1/self.init[53]))**2) + 111.8181*(((781.22/self.init[53])/math.cosh(781.22/self.init[53]))**2))
            CP.append(51.34231 + 174.465*(((1669.5/self.init[53])/math.sinh(1669.5/self.init[53]))**2) + 119.4182*(((741.02/self.init[53])/math.cosh(741.02/self.init[53]))**2))
            CP.append(57.55231+ 174.508*(((1719.9/self.init[53])/math.sinh(1719.9/self.init[53]))**2) + 120.4834*(((780.87/self.init[53])/math.cosh(780.87/self.init[53]))**2))
            CP.append(55.13041 + 187.9192*(((1682.3/self.init[53])/math.sinh(1682.3/self.init[53]))**2) + 130.1376*(((743.1/self.init[53])/math.cosh(743.1/self.init[53]))**2))
            CP.append(56.08102 + 172.8193*(((1553.7/self.init[53])/math.sinh(1553.7/self.init[53]))**2) + 127.8733*(((723.17/self.init[53])/math.cosh(723.17/self.init[53]))**2))
            CP.append(58.94478 + 201.1369*(((1656.5/self.init[53])/math.sinh(1656.5/self.init[53]))**2) + 139.8132*(((743.6/self.init[53])/math.cosh(743.6/self.init[53]))**2))
            CP.append(59.85717 + 184.0212*(((1551.8/self.init[53])/math.sinh(1551.8/self.init[53]))**2) + 136.5721*(((723.07/self.init[53])/math.cosh(723.07/self.init[53]))**2))
            CP.append(62.77587 + 214.3236*(((1691.2/self.init[53])/math.sinh(1691.2/self.init[53]))**2) + 149.6131*(((744.41/self.init[53])/math.cosh(744.41/self.init[53]))**2))
            CP.append(63.67632 + 195.2064*(((1552.5/self.init[53])/math.sinh(1552.5/self.init[53]))**2) + 145.5527*(((723.89/self.init[53])/math.cosh(723.89/self.init[53]))**2))
            CP.append(66.58546 + 227.4936*(((1693.5/self.init[53])/math.sinh(1693.5/self.init[53]))**2) + 159.1932*(((744.57/self.init[53])/math.cosh(744.57/self.init[53]))**2))
            CP.append(67.50502 + 206.4894*(((1554.5/self.init[53])/math.sinh(1554.5/self.init[53]))**2) + 154.6694*(((724.9/self.init[53])/math.cosh(724.9/self.init[53]))**2))
            CP.append(70.46432 + 239.658*(((771.07/self.init[53])/math.sinh(771.07/self.init[53]))**2) -102.7324 *(((916.73/self.init[53])/math.cosh(916.73/self.init[53]))**2))
            CP.append(71.22146 + 217.3665*(((1546.3/self.init[53])/math.sinh(1546.3/self.init[53]))**2) + 162.5752*(((723.07/self.init[53])/math.cosh(723.07/self.init[53]))**2))
            CP.append(74.19031 + 252.5795*(((767.91/self.init[53])/math.sinh(767.91/self.init[53]))**2) -109.0594 *(((912.03/self.init[53])/math.cosh(912.03/self.init[53]))**2))
            CP.append(75.18152 + 229.3565*(((747.85/self.init[53])/math.sinh(747.85/self.init[53]))**2) -70.4476 *(((864.56/self.init[53])/math.cosh(864.56/self.init[53]))**2))
            CP.append(77.57954 + 264.8801*(((1636/self.init[53])/math.sinh(1636/self.init[53]))**2) + 177.9402*(((726.27/self.init[53])/math.cosh(726.27/self.init[53]))**2))
            CP.append(91.43499 + 184.1669*(((801.08/self.init[53])/math.sinh(801.08/self.init[53]))**2) + 119.2032*(((2361.6/self.init[53])/math.cosh(2361.6/self.init[53]))**2))
            CP.append(91.43499 + 184.1669*(((801.08/self.init[53])/math.sinh(801.08/self.init[53]))**2) + 119.2032*(((2361.6/self.init[53])/math.cosh(2361.6/self.init[53]))**2))
            CP.append(91.08388 + 274.6011*(((1715.3/self.init[53])/math.sinh(1715.3/self.init[53]))**2) + 189.7774*(((779.78/self.init[53])/math.cosh(778.78/self.init[53]))**2))
            CP.append(93.76135 + 282.3158*(((1723.4/self.init[53])/math.sinh(1723.4/self.init[53]))**2) + 194.8457*(((785.13/self.init[53])/math.cosh(785.13/self.init[53]))**2))
            CP.append(97.96264 + 294.7836*(((1723.1/self.init[53])/math.sinh(1723.1/self.init[53]))**2) + 203.5636*(((784.97/self.init[53])/math.cosh(784.97/self.init[53]))**2))
            CP.append(97.96264 + 294.7836*(((1723.1/self.init[53])/math.sinh(1723.1/self.init[53]))**2) + 203.5636*(((784.97/self.init[53])/math.cosh(784.97/self.init[53]))**2))
            CP.append(99.44827 + 299.6561*(((1714.4/self.init[53])/math.sinh(1714.4/self.init[53]))**2) +207.1033*(((779.51/self.init[53])/math.cosh(779.51/self.init[53]))**2))
            CP.append(102.0302 + 307.8962*(((815.29/self.init[53])/math.sinh(815.29/self.init[53]))**2) -120.4213 *(((944.98/self.init[53])/math.cosh(944.98/self.init[53]))**2))
            CP.append(106.3223+ 319.7908*(((1721.5/self.init[53])/math.sinh(1721.5/self.init[53]))**2) + 220.8799*(((784.28/self.init[53])/math.cosh(784.28/self.init[53]))**2))
            CP.append(106.3223+ 319.7908*(((1721.5/self.init[53])/math.sinh(1721.5/self.init[53]))**2) + 220.8799*(((784.28/self.init[53])/math.cosh(784.28/self.init[53]))**2))
            CP.append(6.95161 + 2.05763*(((1701.6/self.init[53])/math.sinh(1701.6/self.init[53]))**2) + 0.0247*(((909.79/self.init[53])/math.cosh(909.79/self.init[53]))**2))
            CP = np.array(CP)
        except OverflowError:
            CP = np.zeros(52)
        
        sumSiFi = sum(self.init[4:53] * CP[4:53] * 4.18)
        
        dcp = []
        num = 0
        for i in range(5, 26, 2):
            if i == 5:
                dcp_co2 = (CP[0] + CP[2]) - (CP[3] + CP[1])
                dcp_ch4 = (CP[4] + CP[2]) - (CP[0] + 3*CP[1])
                dcp.append(dcp_co2)
                dcp.append(dcp_ch4)
            
            j = i - (3+num)
            dcp_paraf = (CP[i] + (j)*CP[2]) - ((j)*CP[0] + 2*j*CP[1])
            dcp_olef = (CP[i+1] + (j)*CP[2]) - ((j)*CP[0] + (2*j+1)*CP[1])
            dcp.append(dcp_paraf)
            dcp.append(dcp_olef)
            num += 1
        
        dcp = np.array(dcp)
        dH = dHr + dcp * 4.18 * (self.init[53] - 298)
        return sumSiFi, dH
        
    def reactor(self):
        pass