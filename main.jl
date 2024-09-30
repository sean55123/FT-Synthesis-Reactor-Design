using DifferentialEquations
using Sundials
using Plots
using LaTeXStrings 

# Input parameters
H2in = 234.19       # H2 inlet flowrate (kg/hr)
To = 522.55         # Initial process temperature (K)
Ta0 = 535.53        # Initial coolant temperature (K)
z = 12.45           # Reactor length (m)
Nt = 96             # Number of tubes packed
mc = 598.05         # Coolant mass flow rate (kg/hr)
alpha = 0.3         # Chain growth factor

k1 = 0.06e-5
k6m = 2.74e-3
K2 = 0.0025e-2
K3 = 4.68e-2
K4 = 0.8
PT = 20.92          # Total pressure (bar)

Ac_1 = 0.159592907 * z  # Cross-sectional area (m^2)
Ao_1 = 0.18315 * z      # Outer area (m^2)
a = 4 / 0.0508          # Heat exchanging area per unit volume (m^-1)
bd = 1.64e6             # Bulk density (g/m^3)
Sc = 24                 # Catalyst surface area (m^2/g)
Ut = 38.8               # Tube-side heat transfer coefficient (W/m^2-K)
Us = 39.9               # Shell-side heat transfer coefficient (W/m^2-K)

Y_init = zeros(55)
Y_init[1] = 0.0001      # CO
Y_init[2] = H2in        # H2
Y_init[4] = 83.33       # CO2
Y_init[54] = To         # Process temperature
Y_init[55] = Ta0        # Coolant temperature

Vspan = LinRange(0, z, 20000)

struct Parameters
    To::Float64
    Ta0::Float64
    z::Float64
    Nt::Int
    mc::Float64
    H2in::Float64
    alpha::Float64
    k1::Float64
    k6m::Float64
    K2::Float64
    K3::Float64
    K4::Float64
    PT::Float64
    Ac_1::Float64
    Ao_1::Float64
    a::Float64
    bd::Float64
    Sc::Float64
    Ut::Float64
    Us::Float64
end

params = Parameters(
    To, Ta0, z, Nt, mc, H2in, alpha, k1, k6m,
    K2, K3, K4, PT, Ac_1, Ao_1, a, bd, Sc, Ut, Us
)

include("reactor.jl")

function ODEs!(dYdW, Y, params, z)
    dYdW .= compute_derivatives(Y, params)
end

zspan = (0.0, z)

prob = ODEProblem(ODEs!, Y_init, zspan, params; isoutofdomain=(u,p,t)->any(u .< 0))
sol = solve(prob, Rodas5(autodiff=true), reltol=1e-6, abstol=1e-6, saveat=Vspan)

Ysol = hcat(sol.u...)  # Convert solution to a matrix (55 x N)

plot1 = plot(sol.t, Ysol[54, :], label="Process Temp")
plot!(sol.t, Ysol[55, :], label="Coolant Temp")
xlabel!("Reactor length (m)")
ylabel!("Temperature (K)")
title!("Temperature Profiles")

plot2 = plot(sol.t, Ysol[1, :], label="CO flowrate")
xlabel!("Reactor length (m)")
ylabel!("Flowrate (kg/hr)")
title!("CO Flowrate")

plot3 = plot(sol.t, Ysol[2, :], label="H₂ flowrate")
xlabel!("Reactor length (m)")
ylabel!("Flowrate (kg/hr)")
title!("H₂ Flowrate")

plot4 = plot(sol.t, Ysol[4, :], label="CO₂ flowrate")
xlabel!("Reactor length (m)")
ylabel!("Flowrate (kg/hr)")
title!("CO₂ Flowrate")

plot5 = plot(sol.t, Ysol[5, :], label= "CH₄ flowrate")
xlabel!("Reactor length (m)")
ylabel!("Flowrate (kg/hr)")
title!("CH₄ Flowrate")

plot6 = plot(sol.t, Ysol[6, :], label= "C₂H₄ flowrate")
xlabel!("Reactor length (m)")
ylabel!("Flowrate (kg/hr)")
title!("C₂H₄ Flowrate")

plot(plot1, plot2, plot3, plot4, plot5, plot6, layout=(3,3), size=(1000,800))