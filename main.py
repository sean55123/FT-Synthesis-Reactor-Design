from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from reactor import FT_PBR

H2in = 234.19 # H2 inlet flowrate
To = 522.55 # Initital process temp (oC)
Ta0 = 535.53 # Initital coolent temp (oC)
z = 12.45 # Reactor length
Nt = 96 # Number of tubes packed 
mc = 598.05 # Coolent mass flow rate (kg/hr)
alpha = 0.3 # Chain factor
Y_init = np.array([0.0001, H2in, 0, 83.33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, To, Ta0])
Vspan = np.linspace(0, z, 20000)

def ODEs(t, Y):
    reac = FT_PBR(Y, To, Ta0, z, Nt, mc, H2in, alpha)
    dFdz, dTtdz, dTsdz = reac.reactor()
    dYdW = np.concatenate((dFdz, [dTtdz, dTsdz])) 
    return dYdW

sol = solve_ivp(ODEs, [Vspan[0], Vspan[-1]], Y_init, t_eval=Vspan, method='BDF')

plt.plot(sol.t, sol.y[53,:], label='Process Temp')
plt.plot(sol.t, sol.y[54,:], label='Coolent Temp')
plt.legend()
plt.show()