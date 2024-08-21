import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 13, 'lines.linewidth': 2.5})
from matplotlib.widgets import Slider, Button
import sys

#%% mol,g,s,bar
k1_1 = 2.33*(10**(-5))  
k1 =  0.06*(10**(-5)) 
k6m = 2.74*(10**(-3))  
K2 = 0.0025*(10**(-2))  
K3 = 4.68*(10**(-2))
K4 = 0.8
Kv = 1.13*(10**(-3))
PT = 20.92  #20.92 #bar
NH3 = 0.397708
N2 = 0 #67.2057

# z = 10

# mc = 500
  #coolant flow rate mol/s
# To = 533
   # inital temperature for input 
R = 8.314
  #ideal gas constant 
#U = 0.0116  #J/s/m2/K   38.0762
#a = 78.74                                                  
#Ua = 0.01 #J/s/m3/K
#L = 1
  #number of tube 

# Ta0= 530 #K coolant initial temperature
bd = 1640000    #g/m3 particle density for Fu-Cu-K catalyst 
def reactor(To, Ta0, z_1, Nt_1, mc_1,H2in):
    global conv, T_1, Ta_1,Tout,Taout,C9_,Cyield,CO2,para,ole,C1_C4,CO2,H
    def ODEfun(Yfuncvec, W,bd,PT,mc_1,To,R,Ta0,k1_1,k1,k6m,K2,K3,K4,Kv):
        FCO_1 =Yfuncvec[0]
        FH2_1 =Yfuncvec[1]
        FH2O_1 =Yfuncvec[2]
        FCO2_1 = Yfuncvec[3]
        FCH4_1 =Yfuncvec[4]
        FC2H4_1 =Yfuncvec[5]
        FC2H6_1 = Yfuncvec[6]
        FC3H6_1 = Yfuncvec[7]
        FC3H8_1 = Yfuncvec[8]
        FC4H8_1 =Yfuncvec[9]
        FC4H10_1 = Yfuncvec[10]
        FC5H10_1 =Yfuncvec[11]
        FC5H12_1 = Yfuncvec[12]
        FC6H12_1= Yfuncvec[13]
        FC6H14_1 = Yfuncvec[14]
        FC7H14_1= Yfuncvec[15]
        FC7H16_1 = Yfuncvec[16]
        FC8H16_1 = Yfuncvec[17]
        FC8H18_1 = Yfuncvec[18]
        FC9H18_1= Yfuncvec[19]
        FC9H20_1 = Yfuncvec[20]
        FC10H20_1 = Yfuncvec[21]
        FC10H22_1 = Yfuncvec[22]
        FC11H22_1 = Yfuncvec[23]
        FC11H24_1 = Yfuncvec[24]
        FC12H24_1 = Yfuncvec[25]
        FC12H26_1 = Yfuncvec[26]
        T_1 = Yfuncvec[27]
        Ta_1 = Yfuncvec[28]
        try:
            k5m = 1.4*(10**3)*math.exp(-92890/(8.314*T_1))#92890 4.65 1.4
            k5 = 2.74*(10**2)*math.exp(-87010/(8.314*T_1))  #2.74 87010
            k5e = 2.74*(10**2)*math.exp(-87010/(8.314*T_1))  #2.74 87010
            k6 = 1.5*(10**6)*math.exp(-111040/(8.314*T_1))  #2.66 111040 0.5
            k6e = 1.5*(10**6)*math.exp(-111040/(8.314*T_1))
            kv = 1.57*(10)*math.exp(-45080/(8.314*T_1))
        except OverflowError:
            k5m = 0
            k5 = 0
            k5e = 0
            k6 = 0
            k6e = 0
            kv = 0
         #1.57
        
        try:
            Kp = (math.exp(5078.0045/T_1 - 5.8972089 + (13.958689*(10**(-4))*T_1)-(27.592844*(10**(-8))*(T_1**2))))
        except OverflowError:
            Kp = 0
        FT = (FCO_1 + FH2_1 + FH2O_1 + FCO2_1 + FCH4_1 + FC2H4_1 + FC2H6_1 + FC3H6_1 + FC3H8_1 + FC4H8_1 + FC4H10_1 + FC5H10_1 + FC5H12_1 + 
              FC6H12_1 + FC6H14_1 + FC7H14_1 + FC7H16_1 + FC8H16_1 + FC8H18_1 + FC9H18_1 + FC9H20_1 + FC10H20_1 + FC10H22_1 +
              FC11H22_1 + FC11H24_1 + FC12H24_1 + FC12H26_1 + N2)
        
        A = (FCO_1*(PT/FT)*(T_1/To))*(FH2O_1*(PT/FT)*(T_1/To))/((FH2_1*(PT/FT)*(T_1/To))**0.5)
        B = (FCO2_1*(PT/FT)*(T_1/To))*((FH2_1*(PT/FT)*(T_1/To))**0.5)/Kp
        rCO2_1 = (kv*(A-B))/(1 + Kv*A)
        A1 = (1/(K2*K3*K4))*((FH2O_1*(PT/FT)*(T_1/To))/((FH2_1*(PT/FT)*(T_1/To))**2)) 
        A2 = (1/(K3*K4))*(1/(FH2_1*(PT/FT)*(T_1/To)))
        A3 = 1/K4 
        alphA = 0.65
        
        
        RRR = (k1*(FCO_1*(PT/FT)*(T_1/To))+k5*(FH2_1*(PT/FT)*(T_1/To))+k6)
        upper_1 = ((k1*(FCO_1*(PT/FT)*(T_1/To)))/(k1*(FCO_1*(PT/FT)*(T_1/To))+k5*(FH2_1*(PT/FT)*(T_1/To))))
        alph1_1 = (k1_1*(FCO_1*(PT/FT)*(T_1/To)))/(k1_1*(FCO_1*(PT/FT)*(T_1/To))+k5m*(FH2_1*(PT/FT)*(T_1/To)))
        
        #C2
        bataf2_1 = (alphA)*upper_1 
        #C3 
        bataf3_1 = (alphA**2)*upper_1
        #C4
        bataf4_1 = (alphA**3)*upper_1 
        #C5
        bataf5_1 = (alphA**4)*upper_1 
        #C6
        bataf6_1 = (alphA**5)*upper_1 
        #C7
        bataf7_1 = (alphA**6)*upper_1 
        #C8
        bataf8_1 = (alphA**7)*upper_1 
        #C9
        bataf9_1 = (alphA**8)*upper_1 
        #C10
        bataf10_1 = (alphA**9)*upper_1 
        #C11
        bataf11_1 = (alphA**10)*upper_1 
        #C12
        bataf12_1 = (alphA**11)*upper_1 
    
    
        batai2_1 = k6m*(FC2H4_1*(PT/FT)*(T_1/To)) / RRR
        batai3_1 = k6m*( (FC3H6_1*(PT/FT)*(T_1/To))+alphA*(FC2H4_1*(PT/FT)*(T_1/To))) / RRR
        batai4_1 = k6m*((FC4H8_1*(PT/FT)*(T_1/To))+alphA*(FC3H6_1*(PT/FT)*(T_1/To))+(alphA**2)*(FC2H4_1*(PT/FT)*(T_1/To))) / RRR
        batai5_1 = k6m*((FC5H10_1*(PT/FT)*(T_1/To))+alphA*(FC4H8_1*(PT/FT)*(T_1/To))+(alphA**2)*(FC3H6_1*(PT/FT)*(T_1/To))+(alphA**3)*(FC2H4_1*(PT/FT)*(T_1/To))) / RRR
        batai6_1 = k6m*((FC6H12_1*(PT/FT)*(T_1/To))+alphA*(FC5H10_1*(PT/FT)*(T_1/To))+(alphA**2)*(FC4H8_1*(PT/FT)*(T_1/To))+(alphA**3)*(FC3H6_1*(PT/FT)*(T_1/To))+(alphA**4)*(FC2H4_1*(PT/FT)*(T_1/To))) / RRR
        batai7_1 = k6m*((FC7H14_1*(PT/FT)*(T_1/To))+alphA*(FC6H12_1*(PT/FT)*(T_1/To))+(alphA**2)*(FC5H10_1*(PT/FT)*(T_1/To))+(alphA**3)*(FC4H8_1*(PT/FT)*(T_1/To))+(alphA**4)*(FC3H6_1*(PT/FT)*(T_1/To))+(alphA**5)*(FC2H4_1*(PT/FT)*(T_1/To))) / RRR
        batai8_1 = k6m*((FC8H16_1*(PT/FT)*(T_1/To))+alphA*(FC7H14_1*(PT/FT)*(T_1/To))+(alphA**2)*(FC6H12_1*(PT/FT)*(T_1/To))+(alphA**3)*(FC5H10_1*(PT/FT)*(T_1/To))+(alphA**4)*(FC4H8_1*(PT/FT)*(T_1/To))+(alphA**5)*(FC3H6_1*(PT/FT)*(T_1/To))+(alphA**6)*(FC2H4_1*(PT/FT)*(T_1/To))) / RRR
        batai9_1 = k6m*((FC9H18_1*(PT/FT)*(T_1/To))+alphA*(FC8H16_1*(PT/FT)*(T_1/To))+(alphA**2)*(FC7H14_1*(PT/FT)*(T_1/To))+(alphA**3)*(FC6H12_1*(PT/FT)*(T_1/To))+(alphA**4)*(FC5H10_1*(PT/FT)*(T_1/To))+(alphA**5)*(FC4H8_1*(PT/FT)*(T_1/To))+(alphA**6)*(FC3H6_1*(PT/FT)*(T_1/To))+(alphA**7)*(FC2H4_1*(PT/FT)*(T_1/To))) / RRR
        batai10_1 = k6m*((FC10H20_1*(PT/FT)*(T_1/To))+alphA*(FC9H18_1*(PT/FT)*(T_1/To))+(alphA**2)*(FC8H16_1*(PT/FT)*(T_1/To))+(alphA**3)*(FC7H14_1*(PT/FT)*(T_1/To))+(alphA**4)*(FC6H12_1*(PT/FT)*(T_1/To))+(alphA**5)*(FC5H10_1*(PT/FT)*(T_1/To))+(alphA**6)*(FC4H8_1*(PT/FT)*(T_1/To))+(alphA**7)*(FC3H6_1*(PT/FT)*(T_1/To))+(alphA**8)*(FC2H4_1*(PT/FT)*(T_1/To))) / RRR
        batai11_1 = k6m*((FC11H22_1*(PT/FT)*(T_1/To))+alphA*(FC10H20_1*(PT/FT)*(T_1/To))+(alphA**2)*(FC9H18_1*(PT/FT)*(T_1/To))+(alphA**3)*(FC8H16_1*(PT/FT)*(T_1/To))+(alphA**4)*(FC7H14_1*(PT/FT)*(T_1/To))+(alphA**5)*(FC6H12_1*(PT/FT)*(T_1/To))+(alphA**6)*(FC5H10_1*(PT/FT)*(T_1/To))+(alphA**7)*(FC4H8_1*(PT/FT)*(T_1/To))+(alphA**8)*(FC3H6_1*(PT/FT)*(T_1/To))+(alphA**9)*(FC2H4_1*(PT/FT)*(T_1/To))) / RRR
        batai12_1 = k6m*((FC12H24_1*(PT/FT)*(T_1/To))+alphA*(FC11H22_1*(PT/FT)*(T_1/To))+(alphA**2)*(FC10H20_1*(PT/FT)*(T_1/To))+(alphA**3)*(FC9H18_1*(PT/FT)*(T_1/To))+(alphA**4)*(FC8H16_1*(PT/FT)*(T_1/To))+(alphA**5)*(FC7H14_1*(PT/FT)*(T_1/To))+(alphA**6)*(FC6H12_1*(PT/FT)*(T_1/To))+(alphA**7)*(FC5H10_1*(PT/FT)*(T_1/To))+(alphA**8)*(FC4H8_1*(PT/FT)*(T_1/To))+(alphA**9)*(FC3H6_1*(PT/FT)*(T_1/To))+(alphA**10)*(FC2H4_1*(PT/FT)*(T_1/To))) / RRR
        
        try:
            bata2_1 = (k6m/k6)*(FC2H4_1*(PT/FT)*(T_1/To))/(bataf2_1+batai2_1)
            bata3_1 = (k6m/k6)*(FC3H6_1*(PT/FT)*(T_1/To))/(bataf3_1+batai3_1) 
            bata4_1 = (k6m/k6)*(FC4H8_1*(PT/FT)*(T_1/To))/(bataf4_1+batai4_1)
            bata5_1 = (k6m/k6)*(FC5H10_1*(PT/FT)*(T_1/To))/(bataf5_1+batai5_1)
            bata6_1 = (k6m/k6)*(FC6H12_1*(PT/FT)*(T_1/To))/(bataf6_1+batai6_1)
            bata7_1 = (k6m/k6)*(FC7H14_1*(PT/FT)*(T_1/To))/(bataf7_1+batai7_1)
            bata8_1 = (k6m/k6)*(FC8H16_1*(PT/FT)*(T_1/To))/(bataf8_1+batai8_1)
            bata9_1 = (k6m/k6)*(FC9H18_1*(PT/FT)*(T_1/To))/(bataf9_1+batai9_1)
            bata10_1 = (k6m/k6)*(FC10H20_1*(PT/FT)*(T_1/To))/(bataf10_1+batai10_1)
            bata11_1 = (k6m/k6)*(FC11H22_1*(PT/FT)*(T_1/To))/(bataf11_1+batai11_1)
            bata12_1 = (k6m/k6)*(FC12H24_1*(PT/FT)*(T_1/To))/(bataf12_1+batai12_1) 
        except ZeroDivisionError:
            bata2_1 = 0
            bata3_1 = 0
            bata4_1 = 0
            bata5_1 = 0
            bata6_1 = 0
            bata7_1 = 0
            bata8_1 = 0
            bata9_1 = 0
            bata10_1 = 0
            bata11_1 = 0
            bata12_1 = 0
            
        alph1_1 = 1*((1-alphA)**2)*((alphA)**0)
        alph2_1 = 2*((1-alphA)**2)*((alphA)**1)
        alph3_1 = 3*((1-alphA)**2)*((alphA)**2)
        alph4_1 = 4*((1-alphA)**2)*((alphA)**3)
        alph5_1 = 5*((1-alphA)**2)*((alphA)**4)
        alph6_1 = 6*((1-alphA)**2)*((alphA)**5)
        alph7_1 = 7*((1-alphA)**2)*((alphA)**6)
        alph8_1 = 8*((1-alphA)**2)*((alphA)**7)
        alph9_1 = 9*((1-alphA)**2)*((alphA)**8)
        alph10_1 = 10*((1-alphA)**2)*((alphA)**9)
        alph11_1 = 11*((1-alphA)**2)*((alphA)**10)
        alph12_1 = 12*((1-alphA)**2)*((alphA)**11)
        
        
        Chigh = (alph1_1*alph2_1*alph3_1*alph4_1*alph5_1*alph6_1*alph7_1*alph8_1*alph9_1*alph10_1*alph11_1
                 +alph1_1*alph2_1*alph3_1*alph4_1*alph5_1*alph6_1*alph7_1*alph8_1*alph9_1*alph10_1*alph11_1*alph12_1)
        adsor = 1+(1+A1+A2+A3)*(alph1_1+alph1_1*alph2_1+alph1_1*alph2_1*alph3_1+alph1_1*alph2_1*alph3_1*alph4_1+
                                alph1_1*alph2_1*alph3_1*alph4_1*alph5_1+
                                alph1_1*alph2_1*alph3_1*alph4_1*alph5_1*alph6_1+
                                alph1_1*alph2_1*alph3_1*alph4_1*alph5_1*alph6_1*alph7_1+
                                alph1_1*alph2_1*alph3_1*alph4_1*alph5_1*alph6_1*alph7_1*alph8_1+
                                alph1_1*alph2_1*alph3_1*alph4_1*alph5_1*alph6_1*alph7_1*alph8_1*alph9_1+
                                alph1_1*alph2_1*alph3_1*alph4_1*alph5_1*alph6_1*alph7_1*alph8_1*alph9_1*alph10_1+Chigh)
        
        
        rCH4_1 = (k5m*(FH2_1*(PT/FT)*(T_1/To))*alph1_1)/adsor 
        rC2H6_1 = (k5e*(FH2_1*(PT/FT)*(T_1/To))*(alph2_1))/adsor  
        rC2H4_1 =( k6e*(1-bata2_1)*(alph2_1))/adsor
        rC3H8_1 = (k5e*(FH2_1*(PT/FT)*(T_1/To))*(alph3_1))/adsor 
        rC3H6_1 = (k6e*(1-bata3_1)*(alph3_1))/adsor
        rC4H10_1 = (k5e*(FH2_1*(PT/FT)*(T_1/To))*(alph4_1))/adsor
        rC4H8_1 = (k6e*(1-bata4_1)*(alph4_1))/adsor
        rC5H12_1 =  (k5e*(FH2_1*(PT/FT)*(T_1/To))*(alph5_1))/adsor
        rC5H10_1 = (k6e*(1-bata5_1)*(alph5_1))/adsor
        rC6H14_1 = (k5e*(FH2_1*(PT/FT)*(T_1/To))*(alph6_1))/adsor
        rC6H12_1 = (k6e*(1-bata6_1)*(alph6_1))/adsor    
        rC7H16_1 = (k5e*(FH2_1*(PT/FT)*(T_1/To))*(alph7_1))/adsor
        rC7H14_1 = (k6e*(1-bata7_1)*(alph7_1))/adsor    
        rC8H18_1 = (k5e*(FH2_1*(PT/FT)*(T_1/To))*(alph8_1))/adsor
        rC8H16_1 = (k6e*(1-bata8_1)*(alph8_1))/adsor   
        rC9H20_1 = (k5e*(FH2_1*(PT/FT)*(T_1/To))*(alph9_1))/adsor
        rC9H18_1 = (k6e*(1-bata9_1)*(alph9_1))/adsor    
        rC10H22_1 = (k5e*(FH2_1*(PT/FT)*(T_1/To))*(alph10_1))/adsor
        rC10H20_1 = (k6e*(1-bata10_1)*(alph10_1))/adsor     
        rC11H24_1 = (k5e*(FH2_1*(PT/FT)*(T_1/To))*(alph11_1))/adsor
        rC11H22_1 = (k6e*(1-bata11_1)*(alph11_1))/adsor
        rC12H26_1 = (k5e*(FH2_1*(PT/FT)*(T_1/To))*(alph12_1))/adsor
        rC12H24_1 = (k6e*(1-bata12_1)*(alph12_1))/adsor
        
        rCO_1 = -rCO2_1-rCH4_1-2*rC2H4_1-2*rC2H6_1-3*rC3H6_1-3*rC3H8_1-4*rC4H8_1-4*rC4H10_1-5*rC5H10_1-5*rC5H12_1-6*rC6H12_1-6*rC6H14_1-7*rC7H14_1-7*rC7H16_1-8*rC8H16_1-8*rC8H18_1-9*rC9H18_1-9*rC9H20_1-10*rC10H20_1-10*rC10H22_1-11*rC11H22_1-11*rC11H24_1-12*rC12H24_1-12*rC12H26_1
        rH2_1 = rCO2_1-3*rCH4_1-4*rC2H4_1-5*rC2H6_1-6*rC3H6_1-7*rC3H8_1-8*rC4H8_1-9*rC4H10_1-10*rC5H10_1-11*rC5H12_1-12*rC6H12_1-13*rC6H14_1-14*rC7H14_1-15*rC7H16_1-16*rC8H16_1-17*rC8H18_1-18*rC9H18_1-19*rC9H20_1-20*rC10H20_1-21*rC10H22_1-22*rC11H22_1-23*rC11H24_1-24*rC12H24_1-25*rC12H26_1
        rH2O_1 = -rCO2_1+rCH4_1+2*rC2H4_1+2*rC2H6_1+3*rC3H6_1+3*rC3H8_1+4*rC4H8_1+4*rC4H10_1+5*rC5H10_1+5*rC5H12_1+6*rC6H12_1+6*rC6H14_1+7*rC7H14_1+7*rC7H16_1+8*rC8H16_1+8*rC8H18_1+9*rC9H18_1+9*rC9H20_1+10*rC10H20_1+10*rC10H22_1+11*rC11H22_1+11*rC11H24_1+12*rC12H24_1+12*rC12H26_1
        #%%
        DHCO2R_1 = 41.0953 *1000
        DHCH4R_1 = -74.399 *1000   #J/mol
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
        try:
            CPCO2_1 = 7.014904 + 8.249737*(((1428/T_1)/math.sinh(1428/T_1))**2)+ 6.305532*(((588/T_1)/math.cosh(588/T_1))**2)
            CPH2_1 = 6.59621 + 2.28337*(((2466/T_1)/math.sinh(2466/T_1))**2) + 0.89806*(((567.6/T_1)/math.cosh(567.6/T_1))**2)
            CPCO_1 = 6.95233 + 2.09540*(((3085.1/T_1)/math.sinh(3085.1/T_1))**2) + 2.01951*(((1538.2/T_1)/math.cosh(1538.2/T_1))**2)
            CPH2O_1 = 7.96862 + 6.39868*(((2610.5/T_1)/math.sinh(2610.5/T_1))**2) + 2.12477*(((1169/T_1)/math.cosh(1169/T_1))**2)
            CPCH4_1 = 7.953091 + 19.09167*(((2086.9/T_1)/math.sinh(2086.9/T_1))**2) + 9.936467*(((991.96/T_1)/math.cosh(991.96/T_1))**2) #cal/mol-K
            CPC2H4_1 = 7.972676 + 22.6402*(((1596/T_1)/math.sinh(1596/T_1))**2) + 13.16041*(((740.8/T_1)/math.cosh(740.8/T_1))**2)
            CPC2H6_1 = 10.57036 + 20.23908*(((872.24/T_1)/math.sinh(872.24/T_1))**2) + 16.03373*(((2430.4/T_1)/math.cosh(2430.4/T_1))**2)
            CPC3H6_1 = 10.47387 + 35.97019*(((1398.8/T_1)/math.sinh(1398.8/T_1))**2) + 17.85469*(((616.46/T_1)/math.cosh(616.46/T_1))**2)
            CPC3H8_1 = 14.20512 + 30.24028*(((844.31/T_1)/math.sinh(844.31/T_1))**2) + 20.58016*(((2482.7/T_1)/math.cosh(2482.7/T_1))**2)
            CPC4H8_1 = 11.83563 + 49.89945*(((1262.47/T_1)/math.sinh(1262.47/T_1))**2) + 24.10385*(((551.653/T_1)/math.cosh(551.653/T_1))**2)
            CPC4H10_1 = 19.14445 + 38.79335*(((841.49/T_1)/math.sinh(841.49/T_1))**2) + 25.25795*(((2476.1/T_1)/math.cosh(2476.1/T_1))**2)
            CPC5H10_1 = 14.77281 + 62.29316*(((1234.21/T_1)/math.sinh(1234.21/T_1))**2) + 30.29020*(((540.923/T_1)/math.cosh(540.923/T_1))**2)
            CPC5H12_1 = 21.03038 + 71.91650*(((1650.2/T_1)/math.sinh(1650.2/T_1))**2) + 45.18964*(((747.6/T_1)/math.cosh(747.6/T_1))**2)
            CPC6H12_1 = 18.82048 + 38.52083*(((579.16/T_1)/math.sinh(579.16/T_1))**2) + 48.13557*(((1591.22/T_1)/math.cosh(1591.22/T_1))**2)
            CPC6H14_1 = 24.93551 + 84.14541*(((1694.6/T_1)/math.sinh(1694.6/T_1))**2) + 56.58259*(((761.6/T_1)/math.cosh(761.6/T_1))**2)
            CPC7H14_1 = 22.01665 + 45.25604*(((576.454/T_1)/math.sinh(576.454/T_1))**2) + 55.78819*(((1580.37/T_1)/math.cosh(1580.37/T_1))**2)
            CPC7H16_1 = 21.02269 + 97.12430*(((579.558/T_1)/math.sinh(579.558/T_1))**2) - 54.97970*(((654.179/T_1)/math.cosh(654.179/T_1))**2)
            CPC8H16_1 = 23.57567 + 99.77501*((((-564.429)/T_1)/math.sinh((-564.429)/T_1))**2) - 59.22041*(((630.567/T_1)/math.cosh(630.567/T_1))**2)
            CPC8H18_1 = 27.11379 + 134.54428*(((1621.1/T_1)/math.sinh(1621.1/T_1))**2) + 80.79918*(((681.9/T_1)/math.cosh(681.9/T_1))**2)
            CPC9H18_1 = 28.19815 + 125.92195*(((1609.2/T_1)/math.sinh(1609.2/T_1))**2) + 86.38578*(((749.71/T_1)/math.cosh(749.71/T_1))**2)
            CPC9H20_1 = 20.19616 + 130.41249*((((-1197.98)/T_1)/math.sinh((-1197.98)/T_1))**2) + 77.96288*(((507.226/T_1)/math.cosh(507.226/T_1))**2)
            CPC10H20_1 = 21.42398 + 138.33907*((((-1198.95)/T_1)/math.sinh((-1198.95)/T_1))**2) + 70.82521*(((528.485/T_1)/math.cosh(528.485/T_1))**2)
            CPC10H22_1 = 26.68100 + 142.25399*((((-574.498)/T_1)/math.sinh((-574.498)/T_1))**2) - 77.12931*(((658.909/T_1)/math.cosh(658.909/T_1))**2)
            CPC11H22_1 = 21.25337 + 158.65601*((((-1163.83)/T_1)/math.sinh((-1163.83)/T_1))**2) + 72.73694*(((509.777/T_1)/math.cosh(509.777/T_1))**2)
            CPC11H24_1 = 46.64422 + 145.69122*(((1708.7/T_1)/math.sinh(1708.7/T_1))**2) + 98.64813*(((775.4/T_1)/math.cosh(775.4/T_1))**2)
            CPC12H24_1 = 32.86854 + 160.43589*((((-1268.72)/T_1)/math.sinh((-1268.72)/T_1))**2) + 92.97124*(((556.052/T_1)/math.cosh(556.052/T_1))**2)
            CPC12H26_1 = 50.86223 + 158.42648*(((1715.5/T_1)/math.sinh(1715.5/T_1))**2) + 107.86520*(((777.5/T_1)/math.cosh(777.5/T_1))**2)
            CPH2_1 = 6.59621 + 2.28337*(((2466/T_1)/math.sinh(2466/T_1))**2) + 0.89806*(((567.6/T_1)/math.cosh(567.6/T_1))**2)
            CPCO2_1 = 7.01490 + 8.24974*(((1428/T_1)/math.sinh(1428/T_1))**2) + 6.30553*(((588/T_1)/math.cosh(588/T_1))**2)
            CPCO_1 = 6.95233 + 2.09540*(((3085.1/T_1)/math.sinh(3085.1/T_1))**2) + 2.01951*(((1538.2/T_1)/math.cosh(1538.2/T_1))**2)
            CPH2O_1 = 7.96862 + 6.39868*(((2610.5/T_1)/math.sinh(2610.5/T_1))**2) + 2.12477*(((1169/T_1)/math.cosh(1169/T_1))**2)
            CPN2_1 = 6.95161 + 2.05763*(((1701.6/T_1)/math.sinh(1701.6/T_1))**2) + 0.0247*(((909.79/T_1)/math.cosh(909.79/T_1))**2)
        except OverflowError:
            CPCO2_1 = 0
            CPH2_1 = 0
            CPCO_1 = 0
            CPH2O_1 =0
            CPCH4_1 = 0 #cal/mol-K
            CPC2H4_1 = 0
            CPC2H6_1 = 0
            CPC3H6_1 = 0
            CPC3H8_1 = 0
            CPC4H8_1 = 0
            CPC4H10_1 = 0
            CPC5H10_1 = 0
            CPC5H12_1 = 0
            CPC6H12_1 = 0
            CPC6H14_1 = 0
            CPC7H14_1 =0
            CPC7H16_1 = 0
            CPC8H16_1 = 0
            CPC8H18_1 = 0
            CPC9H18_1 =0
            CPC9H20_1 = 0
            CPC10H20_1 =0
            CPC10H22_1 = 0
            CPC11H22_1 =0
            CPC11H24_1 = 0
            CPC12H24_1 =0
            CPC12H26_1 = 0
            CPH2_1 = 0
            CPCO2_1 = 0
            CPCO_1 = 0
            CPH2O_1 = 0
            CPN2_1 = 0
        sumFiCpi_1 = (CPCH4_1*4.18*FCH4_1 + CPC2H4_1*4.18*FC2H4_1 + CPC2H6_1*4.18*FC2H6_1 + CPC3H6_1*4.18*FC3H6_1 + CPC3H8_1*4.18*FC3H8_1 + CPC4H8_1*4.18*FC4H8_1 + 
                      CPC4H10_1*4.18*FC4H10_1 + CPC5H10_1*4.18*FC5H10_1 + CPC5H12_1*4.18*FC5H12_1 + CPC6H12_1*4.18*FC6H12_1 + CPC6H14_1*4.18*FC6H14_1 + 
                      CPC7H14_1*4.18*FC7H16_1 + CPC7H16_1*4.18*FC7H16_1 + CPC8H16_1*4.18*FC8H16_1 + CPC8H18_1*4.18*FC8H18_1 + CPC9H18_1*4.18*FC9H18_1 + 
                      CPC9H20_1*4.18*FC9H20_1 + CPC10H20_1*4.18*FC10H20_1 + CPC10H22_1*4.18*FC10H22_1 + CPC11H22_1*4.18*FC11H22_1 + 
                      CPC11H24_1*4.18*FC11H24_1 + CPC12H24_1*4.18*FC12H24_1 + CPC12H26_1*4.18*FC12H26_1 + CPH2_1*4.18*FH2_1 + CPCO2_1*4.18*FCO2_1 + CPCO_1*4.18*FCO_1
                      +CPN2_1 * 4.48 *N2)
       #%% 
        DCPCO2_1 = (CPCO_1+CPH2O_1)-(CPCO2_1+CPH2_1)
        DCPCH4_1 =  (CPCH4_1+CPH2O_1)-(CPCO_1 + 3*CPH2_1)
        DCPC2H4_1 = (CPC2H4_1 + 2*CPH2O_1)-(2*CPCO_1 + 4*CPH2_1)
        DCPC2H6_1 = (CPC2H6_1 + 2*CPH2O_1)-(2*CPCO_1 + 5*CPH2_1)
        DCPC3H6_1 = (CPC3H6_1 + 3*CPH2O_1)-(3*CPCO_1 + 6*CPH2_1)
        DCPC3H8_1 = (CPC3H8_1 + 3*CPH2O_1)-(3*CPCO_1 + 7*CPH2_1)
        DCPC4H8_1 = (CPC4H8_1 + 4*CPH2O_1)-(4*CPCO_1 + 8*CPH2_1)
        DCPC4H10_1 = (CPC4H10_1 + 4*CPH2O_1)-(4*CPCO_1 + 9*CPH2_1)
        DCPC5H10_1 = (CPC5H10_1 + 5*CPH2O_1)-(5*CPCO_1 + 10*CPH2_1)
        DCPC5H12_1 = (CPC5H12_1 + 5*CPH2O_1)-(5*CPCO_1 + 11*CPH2_1)
        DCPC6H12_1 = (CPC6H12_1 + 6*CPH2O_1)-(6*CPCO_1 + 12*CPH2_1)
        #DCPC6H12 = (CPC6H12 + 6*CPCO_1)-(6*CPCO_1 + 12*CPH2_1)
        DCPC6H14_1 = (CPC6H14_1 + 6*CPH2O_1)-(6*CPCO_1 + 13*CPH2_1)
        DCPC7H14_1 = (CPC7H14_1 + 7*CPH2O_1)-(7*CPCO_1 + 14*CPH2_1)
        DCPC7H16_1 = (CPC7H16_1 + 7*CPH2O_1)-(7*CPCO_1 + 15*CPH2_1)
        DCPC8H16_1 = (CPC8H16_1 + 8*CPH2O_1)-(8*CPCO_1 + 16*CPH2_1)
        DCPC8H18_1 = (CPC8H18_1 + 8*CPH2O_1)-(8*CPCO_1 + 17*CPH2_1)
        DCPC9H18_1 = (CPC9H18_1 + 9*CPH2O_1)-(9*CPCO_1 + 18*CPH2_1)
        DCPC9H20_1 = (CPC9H20_1 + 9*CPH2O_1)-(9*CPCO_1 + 19*CPH2_1)
        DCPC10H20_1 = (CPC10H20_1 + 10*CPH2O_1)-(10*CPCO_1 + 20*CPH2_1)
        DCPC10H22_1 = (CPC10H22_1 + 10*CPH2O_1)-(10*CPCO_1 + 21*CPH2_1)
        DCPC11H22_1 = (CPC11H22_1 + 11*CPH2O_1)-(11*CPCO_1 + 22*CPH2_1)
        DCPC11H24_1 = (CPC11H24_1 + 11*CPH2O_1)-(11*CPCO_1 + 23*CPH2_1)
        DCPC12H24_1 = (CPC12H24_1 + 12*CPH2O_1)-(12*CPCO_1 + 24*CPH2_1)
        DCPC12H26_1 = (CPC12H26_1 + 12*CPH2O_1)-(12*CPCO_1 + 25*CPH2_1)
        
        TR = 298
        DHCO2_1 = DHCO2R_1 + DCPCO2_1*4.18*(T_1-TR)
        DHCH4_1 =  DHCH4R_1 + DCPCH4_1*4.18*(T_1-TR)  #J/mol
        DHC2H4_1 = DHC2H4R_1 + DCPC2H4_1*4.18*(T_1-TR)
        DHC2H6_1 = DHC2H6R_1 + DCPC2H6_1*4.18*(T_1-TR)
        DHC3H6_1 = DHC3H6R_1 + DCPC3H6_1*4.18*(T_1-TR)
        DHC3H8_1 = DHC3H8R_1 + DCPC3H8_1*4.18*(T_1-TR)
        DHC4H8_1 = DHC4H8R_1 + DCPC4H8_1*4.18*(T_1-TR)
        DHC4H10_1 = DHC4H10R_1 + DCPC4H10_1*4.18*(T_1-TR)
        DHC5H10_1 = DHC5H10R_1 + DCPC5H10_1*4.18*(T_1-TR)
        DHC5H12_1 = DHC5H12R_1 + DCPC5H12_1*4.18*(T_1-TR)
        DHC6H12_1 = DHC6H12R_1 + DCPC6H12_1*4.18*(T_1-TR)
        DHC6H14_1 = DHC6H14R_1 + DCPC6H14_1*4.18*(T_1-TR)
        DHC7H14_1 = DHC7H14R_1 + DCPC7H14_1*4.18*(T_1-TR)
        DHC7H16_1 = DHC7H16R_1 + DCPC7H16_1*4.18*(T_1-TR)
        DHC8H16_1 = DHC8H16R_1 + DCPC8H16_1*4.18*(T_1-TR) 
        DHC8H18_1 = DHC8H18R_1 + DCPC8H18_1*4.18*(T_1-TR)
        DHC9H18_1 = DHC9H18R_1 + DCPC9H18_1*4.18*(T_1-TR)
        DHC9H20_1 = DHC9H20R_1 + DCPC9H20_1*4.18*(T_1-TR)
        DHC10H20_1 = DHC10H20R_1 + DCPC10H20_1*4.18*(T_1-TR)
        DHC10H22_1 = DHC10H22R_1 + DCPC10H22_1*4.18*(T_1-TR)
        DHC11H22_1 = DHC11H22R_1 + DCPC11H22_1*4.18*(T_1-TR)
        DHC11H24_1 = DHC11H24R_1 + DCPC11H24_1*4.18*(T_1-TR)
        DHC12H24_1 = DHC12H24R_1 + DCPC12H24_1*4.18*(T_1-TR)
        DHC12H26_1 = DHC12H26R_1 + DCPC12H26_1*4.18*(T_1-TR)
      #%%  
        #print(DHCO2R,DHCH4,DHC2H4,DHC2H6)
        Ac_1 = 0.159592907*z_1  #m2
        Ao_1 = 0.18315*z_1
        a = 4/0.0508 
        bd = 1640000 #g/m3
        Sc = 24 #m2/g
        Ut = 38.8 #8.4277#38.8 #W/m2-K32.9
        Us = 39.9 #9.6126#39.9 #W/m2-K
        L_1 = z_1
        dFCOdz_1 = rCO_1*Ac_1*bd*Nt_1
        dFH2dz_1 = rH2_1*Ac_1*bd*Nt_1
        dFH2Odz_1 = rH2O_1*Ac_1*bd*Nt_1
        dFCO2dz_1 = rCO2_1*Ac_1*bd*Nt_1
        dFCH4dz_1 = rCH4_1*Ac_1*bd*Nt_1
        dFC2H4dz_1 = rC2H4_1*Ac_1*bd*Nt_1
        dFC2H6dz_1 = rC2H6_1*Ac_1*bd*Nt_1
        dFC3H6dz_1 = rC3H6_1*Ac_1*bd*Nt_1
        dFC3H8dz_1 = rC3H8_1*Ac_1*bd*Nt_1
        dFC4H8dz_1 = rC4H8_1*Ac_1*bd*Nt_1
        dFC4H10dz_1 = rC4H10_1*Ac_1*bd*Nt_1
        dFC5H10dz_1 = rC5H10_1*Ac_1*bd*Nt_1
        dFC5H12dz_1 = rC5H12_1*Ac_1*bd*Nt_1
        dFC6H12dz_1 = rC6H12_1*Ac_1*bd*Nt_1
        dFC6H14dz_1 = rC6H14_1*Ac_1*bd*Nt_1
        dFC7H14dz_1 = rC7H14_1*Ac_1*bd*Nt_1
        dFC7H16dz_1 = rC7H16_1*Ac_1*bd*Nt_1
        dFC8H16dz_1 = rC8H16_1*Ac_1*bd*Nt_1
        dFC8H18dz_1 = rC8H18_1*Ac_1*bd*Nt_1
        dFC9H18dz_1 = rC9H18_1*Ac_1*bd*Nt_1
        dFC9H20dz_1 = rC9H20_1*Ac_1*bd*Nt_1
        dFC10H20dz_1 = rC10H20_1*Ac_1*bd*Nt_1
        dFC10H22dz_1 = rC10H22_1*Ac_1*bd*Nt_1
        dFC11H22dz_1 = rC11H22_1*Ac_1*bd*Nt_1
        dFC11H24dz_1 = rC11H24_1*Ac_1*bd*Nt_1
        dFC12H24dz_1 = rC12H24_1*Ac_1*bd*Nt_1
        dFC12H26dz_1 = rC12H26_1*Ac_1*bd*Nt_1
        Qg1_1 = rCO2_1*DHCO2_1
        #print(DHCO2)
        # print("**",rCO2,DHCO2)
        Qg_1 = (rCH4_1*DHCH4_1 + rC2H4_1*DHC2H4_1 + rC2H6_1*DHC2H6_1 + 
              rC3H6_1*DHC3H6_1 + rC3H8_1*DHC3H8_1 + rC4H8_1*DHC4H8_1 + 
              rC4H10_1*DHC4H10_1 + rC5H10_1*DHC5H10_1 + rC5H12_1*DHC5H12_1 + 
              rC6H12_1*DHC6H12_1 + rC6H14_1*DHC6H14_1 + rC7H14_1*DHC7H14_1 + 
              rC7H16_1*DHC7H16_1 + rC8H16_1*DHC8H16_1 + rC8H18_1*DHC8H18_1 + 
              rC9H18_1*DHC9H18_1 + rC9H20_1*DHC9H20_1 + rC10H20_1*DHC10H20_1 + 
              rC10H22_1*DHC10H22_1 + rC11H22_1*DHC11H22_1 + rC11H24_1*DHC11H24_1 + 
              rC12H24_1*DHC12H24_1 + rC12H26_1*DHC12H26_1 )
        Q_1 = (-Qg1_1+Qg_1)#*30000000
        #print(rCH4,DHCH4)
        #print(rC2H4,DHC2H4)
        #print(rC2H6,DHC2H6)
        #print(rC3H6,DHC3H6)
        #print(rC3H8,DHC3H8)
        #print(Qg1,Qg,Q)
        # print("**",Qg,Qg1)
        #Qg = ((rCO2)*(-DHCO2)+rCH4*(-DHCH4)+rC2H4*(-DHC2H4)+rC2H6*(-DHC2H6))
        dTtdz_1 = ((Ut*a*(Ta_1-T_1) - Q_1*Sc*bd)/sumFiCpi_1)*Ac_1
        #print(Ut*a*(Ta-T),Q*Sc*bd)
        #dTdW = (( -Q - L*(Ua/bd)*(T-Ta)) / sumFiCpi);
        CPco_1 = 0.4725*T_1 + 122.1 #J/mol-K
        #dTadW = L*(Ua/bd)*(T-Ta)/(CPco*mc)
        dTsdz_1 = (Nt_1*Us*Ao_1*(Ta_1-T_1))/ (CPco_1*mc_1*z_1)
        # print("2.",(CPco*mc)) 
        #print("temp",dTtdz,dTsdz)
        return np.array([dFCOdz_1, dFH2dz_1, dFH2Odz_1,dFCO2dz_1,dFCH4dz_1,dFC2H4dz_1,dFC2H6dz_1,dFC3H6dz_1,dFC3H8dz_1,dFC4H8dz_1,dFC4H10dz_1,
                         dFC5H10dz_1,dFC5H12dz_1,dFC6H12dz_1,dFC6H14dz_1,dFC7H14dz_1,dFC7H16dz_1,dFC8H16dz_1,dFC8H18dz_1,
                         dFC9H18dz_1,dFC9H20dz_1,dFC10H20dz_1,dFC10H22dz_1,dFC11H22dz_1,dFC11H24dz_1,dFC12H24dz_1,dFC12H26dz_1,dTtdz_1,dTsdz_1,N2])
    
    Vspan = np.linspace(0,z_1, 20000) # Range for the independent variable
    y0 = np.array([0.0001,H2in,0,83.33,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,To,Ta0,N2])
    #y0 = np.array([4.945563837538746,H2in,37.523800004441185,62.09536807901022,5.324943944,0.001852,2.3467433,0.0016346,1.0560345,0.0014441,0.4224138,0.001278,0.1584052,0.0011332,0.0570259,0.0009993,0.0199591,0.0008316,0.0068431,0.0005808,0.0023095,0.000323,0.0007698,0.0001497,0.0002541,6.15e-5,8.31e-5,To,Ta0,0])
    #y0 = np.array([3.13,H2in,47.29,58.12,7.360782,0.001916,3.141525,0.002068,1.413686,0.002234,0.565474,0.00242,0.212053,0.002627,0.076339,0.002755,0.026719,0.002372,0.009161,0.001499,0.003092,0.000741,0.001031,0.000315,0.00034,0.000122,0.000111,To,Ta0,0])
    # 處理100kmol/hr = 27.78 mole/s
    H = H2in
    #y0 = np.array([0.08293056113398942, 63.85873006510399, 87.62983309943472, 32.218618219714855, 1.129119622777935, 0.0019918457991118716, 1.100365369995191, 0.013437296770826475, 1.0728562357453113, 0.05161880879556583, 0.9298087376459361, 0.11606303696971669, 0.7554695993373218, 0.1819853521880655, 0.5892662874831119, 0.22893454188762058, 0.44686026800802625, 0.2524205187002632, 0.3319533419488196, 0.2520072327814364, 0.24274088130007446, 0.22854301148923734, 0.17531285871672006, 0.1915142506093172, 0.12534869398245524, 0.1517426129758328, 0.08888361936937736,To,Ta0,N2])
    # y0 = np.array([0.16957757730050035, 73.03851750660758, 81.73156041785515, 35.124431052422814, 1.0733749888990676, 0.001750248937813851, 1.0377656974429768, 0.009523126260704578, 1.0118215550069023, 0.037793887281816506, 0.876912014339316, 0.0935635672966858, 0.7124910116506965, 0.15774949986386083, 0.5557429890875412, 0.20762369952091544, 0.421438433391385, 0.23401961816620045, 0.31306855051931537, 0.23381186365982318, 0.22893137756724927, 0.21048030655419006, 0.1653393282430134, 0.1751884336132478, 0.11821761969375455, 0.13817352096586907, 0.08382703941920773,To,Ta0,N2])
    # # y0 = np.array([0.0312897937571806, H2in, 37.90388811306181, 8.812411096590147, 0.5270606958747274, 0.0002754509530413545, 0.502689217507806, 0.0014605295208502426, 0.4901219870701109, 0.006595225466386207, 0.42477238879409657, 0.02066153501963699, 0.3451275658952027, 0.0437327767015207, 0.2691995013982587, 0.07006955874987086, 0.20414295522701298, 0.0942409554333302, 0.1516490524543525, 0.10554763528588684, 0.11089336960724497, 0.10000310326411477, 0.0800896558274547, 0.0849165268348754, 0.05726410391663019, 0.06748294138784974, 0.04060545550451959,To,Ta0,N2])
    # fig,(ax2) = plt.subplots(1, 1)
    # plt.subplots_adjust(left  = 0.1)
    sol = odeint(ODEfun, y0, Vspan, (bd,PT,mc_1,To,R,Ta0,k1_1,k1,k6m,K2,K3,K4,Kv))
    FCO = sol[:, 0]
    FH2 = sol[:, 1]
    FH2O = sol[:, 2]
    FCO2 = sol[:,3]
    FCH4 = sol[:,4]
    FC2H4 = sol[:,5]
    FC2H6 = sol[:,6]
    FC3H6 = sol[:,7]
    FC3H8 = sol[:,8]
    FC4H8 = sol[:,9]
    FC4H10 = sol[:,10]
    FC5H10 = sol[:,11]
    FC5H12 = sol[:,12]
    FC6H12 = sol[:,13]
    FC6H14 = sol[:,14]
    FC7H14 = sol[:,15]
    FC7H16 = sol[:,16]
    FC8H16 = sol[:,17]
    FC8H18 = sol[:,18]
    FC9H18 = sol[:,19]
    FC9H20 = sol[:,20]
    FC10H20 = sol[:,21]
    FC10H22 = sol[:,22]
    FC11H22 = sol[:,23]
    FC11H24 = sol[:,24]
    FC12H24 = sol[:,25]
    FC12H26 = sol[:,26]
    T_1 = sol[:,27]
    Ta_1 = sol[:,28]
        
    CO2 = FCO2[19999]
    CO = FCO[19999]
    H2 = FH2[19999]
    H2O = FH2O[19999]
    CH4 = FCH4[19999] #19999 catalyst 3g
    C2H4 = FC2H4[19999]
    C2H6 = FC2H6[19999]
    C3H6 = FC3H6[19999]
    C3H8 = FC3H8[19999]
    C4H8 = FC4H8[19999]
    C4H10 = FC4H10[19999]
    C5H10 = FC5H10[19999]
    C5H12 = FC5H12[19999]
    C6H12 = FC6H12[19999]
    C6H14 = FC6H14[19999]
    C7H14 = FC7H14[19999]
    C7H16 = FC7H16[19999]
    C8H16 = FC8H16[19999]
    C8H18 = FC8H18[19999]
    C9H18 = FC9H18[19999]
    C9H20 = FC9H20[19999]
    C10H20 = FC10H20[19999]
    C10H22 = FC10H22[19999]
    C11H22 = FC11H22[19999]
    C11H24 = FC11H24[19999]
    C12H24 = FC12H24[19999]
    C12H26 = FC12H26[19999]
    Tout = T_1[19999]
    Taout = Ta_1[19999]
    #conv = (27.78-CO2)/27.78 *100
    conv = (58.12-CO2)/58.12 * 100
    # conv = (7.977678456366703-CO2)/7.977678456366703 *100
    ConvT = (83.33-CO2)/83.33 *100
    print("total conversion = " , ConvT)
    # p1,p2= ax2.plot(Vspan,T_1,Vspan,Ta_1[::-1])
    # ax2.legend(['T','$T_a$'], loc='best')
    # ax2.set_xlabel(r'$length  {(m)}$', fontsize='medium')
    # ax2.set_ylabel('Temperature (K)', fontsize='medium')
    # ax2.set_ylim(500,550)
    # ax2.set_xlim(0,z_1) #100000
    # plt.savefig('R+F CO', dpi=800,bbox_inches='tight')
    # ax2.grid()
    print(CO2)
    print("conversion = ",ConvT)
    print("CO = ",CO)
    print("H2 = ",H2)
    print("H2O = ",H2O)
    x = [1,2,3,4,5,6,7,8,9,10,11,12]
    para = [CH4,C2H6,C3H8,C4H10,C5H12,C6H14,C7H16,C8H18,C9H20,C10H22,C11H24,C12H26]
    ole = [C2H4,C3H6,C4H8,C5H10,C6H12,C7H14,C8H16,C9H18,C10H20,C11H22,C12H24]
    C1 = CH4
    C2 = C2H4+C2H6
    C3 = C3H6+C3H8 
    C4 = C4H8+C4H10
    C5 = C5H10+C5H12
    C6 = C6H12+C6H14
    C7 = C7H14+C7H16
    C8 = C8H16+C8H18
    C9 = C9H18+C9H20
    C10 = C10H20+C10H22
    C11 = C11H22+C11H24
    C12 = C12H24+C12H26
    Carbon = [C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12]
    C1_C4 = (C1*1)/58.12 + (C2*2)/58.12 + (C3*3)/58.12
    # print(C1_C4)
    #C9_ = (C9*9)/27.78 + (C10*10)/27.78 + (C11*11)/27.78 + (C12*12)/27.78
    #C6_C8 = (C6*6)/27.78 + (C7*7)/27.78 + (C8*8)/27.78
    # y02 = [CO,H2,H2O,CO2,CH4,C2H4,C2H6,C3H6,C3H8,C4H8,C4H10,C5H10,C5H12,
    #           C6H12,C6H14,C7H14,C7H16,C8H16,C8H18,C9H18,C9H20,C10H20,C10H22,C11H22,C11H24,C12H24,C12H26]
    # print(y02)
    # plt.figure(2)
    # plt.scatter(x, y,edgecolors="red",color="red")
    # plt.xticks(range(1, len(x)+1, 1))
    # plt.plot(x,Carbon)
    # plt.xlabel('Carbon number')
    # plt.ylabel('mole flow(mole/s)')
    # plt.title('FT reaction')
    # plt.savefig('FT', dpi=800,bbox_inches='tight')
    # plt.show()
        # # print(Tout,Taout)
        # # print(Tout-Taout)
    
    # Cyield = np.zeros(12)
    # for i in range(12):
    #     Cyield[i] = (Carbon[i]*(i+1))/(83.33-FCO[19999])*100
        
    
    # print(sum(Cyield))
    # plt.figure(3)
    # plt.plot(x,Cyield)
    # plt.xlabel('Carbon number')
    # plt.ylabel('Yield')
    # plt.title('FT reaction')
    # plt.savefig('FTy', dpi=800,bbox_inches='tight')
    # plt.show()
#print(reactor(534.34,534.74,6,300,62.56,68.76))
#print(reactor(511.4,509.88,4.3,184,69.2,82.73))
#print(reactor(511.48,524.63,4.32,177,25.44,68.2))
# print(reactor(503.91,538.55,10,167,269.53,246.04))
# print(reactor(537.15,544.3,7.22,137,68.3,224.53))
# print(reactor(512.04,544.6,11.46,66,151.85,247.68))
# print(reactor(520.14,537.58,9.73,219,401.66,245.25)) #0.7 count
#print(reactor(512.04,544.6,11.46,66,151.85,247.68)) #0.1 co
#print(reactor(503.02,544.54,9.29,154,462.37,243.17)) #0.3 co
#print(reactor(536.45,538.22,7.36,184,431.6,183.44)) #0.6co
#print(reactor(502.65,540.58,10.98,129,83.81,228.57)) #0.65 co
#print(reactor(524.77,542.47,10.07,169,299.1,213.85)) #0.7 co
# print(reactor(512.41,537.49,11.91,162,227.5,229.25)) #0.75 co
# print(reactor(522.44,541.73,10.65,230,403.75,222.84)) #0.1 count
# print(reactor(508.55,544.96,11.47,154,19.44,209.16))  #0.3 count 
# print(reactor(522.44,539.98,9.72,185,416.71,152.89))  #0.6 count
# print(reactor(522.55,544.325,9.46,235,403.75,247.5))  #0.65 count  
print(reactor(520.14,537.58,9.73,219,401.66,245.25))   #0.7 count
#print(reactor(503.27,526.43,11.02,88,350.98,244.68))   #0.75 count
# print(reactor(528.53,544.5,10.21,250,157.64,169.26))
#print(reactor(535.24,543.63,10.1,221,141.78,170.41))
# print(reactor(539.84,532.59,6,2000,1000,181.54))
#print(reactor(539.89,538.72,4.42,1153,744.35,153.67)) #0.3 coco
#print(reactor(508.55,544.96,11.47,154,19.44,209.16))


# print(Cyield)
# print(CO2)
#12.55 62.56
#Ta_1 = Ta_1[::-1]









