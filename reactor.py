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
        
        rco = -rCO2_1 - r_ch4
        for i in range(len(r_olef)):
            rco -= ((i+2)*r_olef[i] + (i+2)*r_paraf)
        
        rh2 = rCO2_1 - 3*r_ch4
        for i in range(1, 26):
            j = i + 1
            rh2 -= (2*j*r_paraf[i-1] + (2*j+1)*r_olef[i-1])
        
        rh2o = -rCO2_1 + r_ch4
        for i in range(r_olef):
            rh2o += ((i+2)*r_olef[i] + (i+2)*r_paraf)
                
    def energy_balance(self):
        # Enthalpy for specific component
        DHCO2R_1 = 41.0953 *1000 #J/mol
        DHCH4R_1 = -74.399 *1000   
        DHC2H4R_1 = -209.725*1000
        DHC2H6R_1 = -83.684*1000
        DHC3H6R_1 = -483.377*1000
        DHC3H8R_1 = -104.51*1000
        DHC4H8R_1 = -754.025*1000
        DHC4H10R_1 = -125.586*1000
        DHC5H10R_1 = -1014.96*1000
        DHC5H12R_1 = -146.522*1000
        DHC6H12R_1 = -1276.89*1000
        DHC6H14R_1 = -166.669*1000
        DHC7H14R_1= -1544.08*1000
        DHC7H16R_1 = -191.349*1000
        DHC8H16R_1 = -1800.75*1000
        DHC8H18R_1 = -255.233*1000
        DHC9H18R_1 = -2158.35*1000
        DHC9H20R_1 = -235.357*1000
        DHC10H20R_1 = -2429.59*1000
        DHC10H22R_1 = -265.858*1000
        DHC11H22R_1 = -2613.18*1000
        DHC11H24R_1 = -269.991*1000
        DHC12H24R_1 = -2859.74*1000
        DHC12H26R_1 = -290.248*1000
        DHC13H26R_1 = -3103.72315*1000
        DHC13H28R_1 = -311.2637*1000
        DHC14H28R_1 = -3365.71*1000
        DHC14H30R_1 = -331.899942*1000
        DHC15H30R_1 = -3628.3056*1000
        DHC15H32R_1 = -352.5366*1000
        DHC16H32R_1 = -3891.285*1000
        DHC16H34R_1 = -373.56242*1000
        DHC17H34R_1 = -4152.388592*1000
        DHC17H36R_1 = -393.809504*1000
        DHC18H36R_1 = -4414.48*1000
        DHC18H38R_1 = -414.4457*1000
        DHC19H38R_1 = -4676.57*1000
        DHC19H40R_1 = -435.084*1000
        DHC20H40R_1 = -4938.562992*1000
        DHC20H42R_1 = -455.72032*1000
        DHC21H42R_1 = -5199.596468*1000
        DHC21H44R_1 = -477.02578*1000
        DHC22H44R_1 = -5461.85*1000
        DHC22H46R_1 = -497.6917*1000
        DHC23H46R_1 = -5723.5398*1000
        DHC23H48R_1 = -412.429*1000
        DHC24H48R_1 = -5985.9305*1000
        DHC24H50R_1 = -539.12386*1000
        DHC25H50R_1 = -6484.0277*1000
        DHC25H52R_1 = -559.78978*1000
        
        
    
    def reactor(self):
        pass