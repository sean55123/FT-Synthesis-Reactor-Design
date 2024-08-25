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
        rCO2_1 = (self.kv*(A-B))/(1 + self.kv*A)
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
        
        req = []
        for i in range(len(r_olef)):
            if i == 0:
                r_ch4 = (self.k5m * (self.init[1] * (self.PT/self.FT) * alpha_prob[0])) / Deno
                req.append(r_ch4)
            req.append(r_olef[i])
            req.append(r_paraf[i])
        
        req = np.array(req)
                
            
                
            
        
                
        
        
    
    def energy_balance(self):
        pass
    
    def reactor(self):
        pass