using DifferentialEquations, Plots, Sundials
using LinearAlgebra

# Assuming the FT_PBR struct and its methods have already been defined in another file
include("reactor.jl")

H2in = 234.19  # H2 inlet flowrate
To = 522.55    # Initial process temp (°C)
Ta0 = 535.53   # Initial coolant temp (°C)
z = 12.45      # Reactor length
Nt = 96        # Number of tubes packed 
mc = 598.05    # Coolant mass flow rate (kg/hr)
alpha = 0.3    # Chain factor

Y_init = [0.0001, H2in, 0.0, 83.33, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, To, Ta0]

Vspan = LinRange(0, z, 20000)

function ODEs!(dYdW, Y, t, params)
    reac = FT_PBR(Y, To, Ta0, z, Nt, mc, H2in, alpha)
    dFdz, dTtdz, dTsdz = reactor(reac)
    dYdW .= vcat(dFdz, dTtdz, dTsdz)
end

prob = ODEProblem(ODEs!, Y_init, (Vspan[1], Vspan[end]))
sol = solve(prob, CVODE_BDF(), saveat=Vspan)

# Plotting
fig = plot(layout = (2, 2), size=(800, 600))

plot!(fig[1, 1], sol.t, sol[53, :], label="Process Temp", ylabel="Temp. (°C)")
plot!(fig[1, 1], sol.t, sol[54, :], label="Coolant Temp")

plot!(fig[1, 2], sol.t, sol[1, :], label="CO flowrate", ylabel="Flowrate (kg/hr)")

plot!(fig[2, 1], sol.t, sol[2, :], label="H₂ flowrate", xlabel="Reactor length (m)", ylabel="Flowrate (kg/hr)")

plot!(fig[2, 2], sol.t, sol[4, :], label="CO₂ flowrate", xlabel="Reactor length (m)", ylabel="Flowrate (kg/hr)")

display(fig)