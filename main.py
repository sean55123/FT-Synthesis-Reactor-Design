from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from reactor import compute_derivatives

# Input parameters
H2in = 234.19  # H2 inlet flowrate
To = 522.55  # Initial process temp (K)
Ta0 = 535.53  # Initial coolant temp (K)
z = 12.45  # Reactor length (m)
Nt = 96  # Number of tubes packed
mc = 598.05  # Coolant mass flow rate (kg/hr)
alpha = 0.3  # Chain factor

# Constants
k1 = 0.06e-5
k6m = 2.74e-3
K2 = 0.0025e-2
K3 = 4.68e-2
K4 = 0.8
PT = 20.92

Ac_1 = 0.159592907 * z  # m^2
Ao_1 = 0.18315 * z
a = 4 / 0.0508  # Heat exchanging area with 2-inch tube
bd = 1.64e6  # g/m^3
Sc = 24  # m^2/g catalyst surface area
Ut = 38.8  # W/m^2-K Tube-side U
Us = 39.9  # W/m^2-K Shell-side U

Y_init = np.zeros(55)
Y_init[0] = 0.0001  # CO
Y_init[1] = H2in  # H2
Y_init[3] = 83.33  # CO2
Y_init[53] = To  # Process temperature
Y_init[54] = Ta0  # Coolant temperature

Vspan = np.linspace(0, z, 20000)

def ODEs(t, Y):
    return compute_derivatives(
        Y, To, Ta0, z, Nt, mc, H2in, alpha, k1, k6m,
        K2, K3, K4, PT, Ac_1, Ao_1, a, bd, Sc, Ut, Us
    )

sol = solve_ivp(
    ODEs,
    [Vspan[0], Vspan[-1]],
    Y_init,
    t_eval=Vspan,
    method='BDF',
)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
ax1 = axs[0, 0]
ax2 = axs[0, 1]
ax3 = axs[1, 0]
ax4 = axs[1, 1]

ax1.plot(sol.t, sol.y[53, :], label='Process Temp')
ax1.plot(sol.t, sol.y[54, :], label='Coolant Temp')
ax1.set_ylabel('Temperature (K)')
ax1.legend()

ax2.plot(sol.t, sol.y[0, :], label='CO flowrate')
ax2.set_ylabel('Flowrate (kg/hr)')
ax2.legend()

ax3.plot(sol.t, sol.y[1, :], label='H$_2$ flowrate')
ax3.set_xlabel('Reactor length (m)')
ax3.set_ylabel('Flowrate (kg/hr)')
ax3.legend()

ax4.plot(sol.t, sol.y[3, :], label='CO$_2$ flowrate')
ax4.set_xlabel('Reactor length (m)')
ax4.set_ylabel('Flowrate (kg/hr)')
ax4.legend()

plt.tight_layout()
plt.show()