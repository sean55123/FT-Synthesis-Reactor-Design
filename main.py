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

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
ax1 = axs[0, 0]
ax2 = axs[0, 1]
ax3 = axs[1, 0]
ax4 = axs[1, 1]
ax1.plot(sol.t, sol.y[53,:], label='Process Temp')
ax1.plot(sol.t, sol.y[54,:], label='Coolent Temp')
ax1.set_ylabel('Temp. ($^o$C)')
ax1.legend()

ax2.plot(sol.t, sol.y[0,:], label='CO flowrate')
ax2.set_ylabel('Flowrate (kg/hr)')
ax2.legend()

ax3.plot(sol.t, sol.y[1,:], label='H$_2$ flowrate')
ax3.set_xlabel('Reactor length (m)')
ax3.set_ylabel('Flowrate (kg/hr)')
ax3.legend()

ax4.plot(sol.t, sol.y[3,:], label='CO$_2$ flowrate')
ax4.set_xlabel('Reactor length (m)')
ax4.set_ylabel('Flowrate (kg/hr)')
ax4.legend()
plt.show()