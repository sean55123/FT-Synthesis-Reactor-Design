using LinearAlgebra, Base.MathConstants
using DifferentialEquations, Plots

mutable struct FT_PBR
    init::Vector{Float64}
    To::Float64
    Ta0::Float64
    z::Float64
    Nt::Int
    mc::Float64
    H2in::Float64
    alpha::Float64
end

function kinetics(pbr::FT_PBR)
    k1 = 0.06 * 1e-5
    k6m = 2.74 * 1e-3
    K2 = 0.0025 * 1e-2
    K3 = 4.68 * 1e-2
    K4 = 0.8
    PT = 20.92

    try
        k5m = 1.4*(10^3)*exp(-92890/(R*init[53]))
        k5 = 2.74*(10^2)*exp(-87010/(R*init[53]))
        k5e = 2.74*(10^2)*exp(-87010/(R*init[53]))
        k6 = 1.5*(10^6)*exp(-111040/(R*init[53]))
        k6e = 1.5*(10^6)*exp(-111040/(R*init[53]))
        kv = 1.57*(10)*exp(-45080/(R*init[53]))
    catch OverflowError
        k5m = 0.0
        k5 = 0.0
        k5e = 0.0
        k6 = 0.0
        k6e = 0.0
        kv = 0.0
    end

    try
        Kp = exp(5078.0045/init[53] - 5.8972089 + (13.958689*(10^(-4))*init[53]) - (27.592844*(10^(-8))*(init[53]^2)))
    catch OverflowError
        Kp = 0.0
    end

    FT = sum(pbr.init[1:end-2])
    A = (pbr.init[1]*(PT/FT)*(pbr.init[53]/pbr.init[54]))*(pbr.init[3]*(PT/FT)*(pbr.init[53]/pbr.init[54])) / ((pbr.init[2]*(PT/FT)*(pbr.init[53]/pbr.init[54]))^0.5)
    B = (pbr.init[4]*(PT/FT)*(pbr.init[53]/pbr.init[54]))*((pbr.init[2]*(PT/FT)*(pbr.init[53]/pbr.init[54]))^0.5) / Kp
    A1 = (1/(K2*K3*K4)) * ((pbr.init[3]*(PT/FT)*(pbr.init[53]/pbr.init[54])) / ((pbr.init[2]*(PT/FT)*(pbr.init[53]/pbr.init[54]))^2))
    A2 = (1/(K3*K4)) * (1/(pbr.init[2]*(PT/FT)*(pbr.init[53]/pbr.init[54])))
    A3 = 1/K4

    RRR = (k1*(pbr.init[1]*(PT/FT)*(pbr.init[53]/pbr.init[54])) + k5*(pbr.init[2]*(PT/FT)*(pbr.init[53]/pbr.init[54])) + k6)
    upper_1 = ((k1*(pbr.init[1]*(PT/FT)*(pbr.init[53]/pbr.init[54])))/(k1*(pbr.init[1]*(PT/FT)*(pbr.init[53]/pbr.init[54])) + k5*(pbr.init[2]*(PT/FT)*(pbr.init[53]/pbr.init[54]))))

    betaf = [pbr.alpha^(i+1) * upper_1 for i in 0:23]
    betai = ones(length(betaf))
    for i in 2:25
        x = i - 2
        b_sum = 0.0
        for j in i:-1:2
            b_sum += (pbr.alpha^(i-j)) * pbr.init[2*j+1] * (PT/FT) * (pbr.init[53] - pbr.To)
        end
        betai[x] = k6m * b_sum / RRR
    end

    try
        betas = betai .+ betaf
        beta = ones(length(betas))
        for i in 1:length(betas)
            beta[i] = (k6m/k6) * (pbr.init[(i+2)*2+1] * (PT/FT)*(pbr.init[53]/pbr.To)) / betas[i]
        end
    catch ZeroDivisionError
        beta = zeros(length(betas))
    end

    alpha_prob = [(i+1) * ((1-pbr.alpha)^2) * (pbr.alpha^i) for i in 0:24]

    Chigh = 0.0
    for i in 11:25
        product = 1.0
        for j in 1:i
            product *= alpha_prob[j]
        end
        Chigh += product
    end

    Deno = 0.0
    for i in 1:10
        product = 1.0
        for j in 1:i
            product *= alpha_prob[j]
        end
        Deno += product
    end

    Deno = 1 + (1 + A1 + A2 + A3) * (Deno + Chigh)

    r_olef = (k5e * (pbr.init[2] * (PT/FT)) * alpha_prob[2:end]) / Deno
    r_paraf = (k6e * (1 .- beta) .* alpha_prob[2:end]) / Deno
    r_co2 = (kv*(A-B))/(1 + kv*A)
    r_ch4 = (k5m * (pbr.init[2] * (PT/FT) * alpha_prob[1])) / Deno

    r_co = -r_co2 - r_ch4
    for i in 1:length(r_olef)
        r_co -= ((i+2)*r_olef[i] + (i+2)*r_paraf[i])
    end

    r_h2 = r_co2 - 3*r_ch4
    for i in 1:24
        j = i + 1
        r_h2 -= (2*j*r_paraf[i] + (2*j+1)*r_olef[i])
    end

    r_h2o = -r_co2 + r_ch4
    for i in 1:length(r_olef)
        r_h2o += ((i+2)*r_olef[i] + (i+2)*r_paraf[i])
    end

    return r_olef, r_paraf, r_co2, r_ch4, r_co, r_h2, r_h2o
end

function energy_balance(pbr::FT_PBR)
    dHr = [41.0953 *1000, -74.399 *1000, -209.725*1000, -83.684*1000, -483.377*1000, -104.51*1000, -754.025*1000, -125.586*1000, -1014.96*1000, -146.522*1000, 
           -1276.89*1000, -166.669*1000, -1544.08*1000, -191.349*1000, -1800.75*1000, -255.233*1000, -2158.35*1000, -235.357*1000, -2429.59*1000, -265.858*1000, 
           -2613.18*1000, -269.991*1000, -2859.74*1000, -290.248*1000, -3103.72315*1000, -311.2637*1000, -3365.71*1000, -331.899942*1000, -3628.3056*1000, -352.5366*1000, 
           -3891.285*1000, -373.56242*1000, -4152.388592*1000, -393.809504*1000, -4414.48*1000, -414.4457*1000, -4676.57*1000, -435.084*1000, -4938.562992*1000, -455.72032*1000, 
           -5199.596468*1000, -477.02578*1000, -5461.85*1000, -497.6917*1000, -5723.5398*1000, -412.429*1000, -5985.9305*1000, -539.12386*1000, -6484.0277*1000, -559.78978*1000]

    try
        CP = [
            6.95233 + 2.09540*((3085.1/pbr.init[53])/sinh(3085.1/pbr.init[53]))^2 + 2.01951*((1538.2/pbr.init[53])/cosh(1538.2/pbr.init[53]))^2,  # CO
            6.59621 + 2.28337*((2466/pbr.init[53])/sinh(2466/pbr.init[53]))^2 + 0.89806*((567.6/pbr.init[53])/cosh(567.6/pbr.init[53]))^2,      # H2
            7.96862 + 6.39868*((2610.5/pbr.init[53])/sinh(2610.5/pbr.init[53]))^2 + 2.12477*((1169/pbr.init[53])/cosh(1169/pbr.init[53]))^2,      # H2O
            7.014904 + 8.249737*((1428/pbr.init[53])/sinh(1428/pbr.init[53]))^2 + 6.305532*((588/pbr.init[53])/cosh(588/pbr.init[53]))^2,        # CO2
            7.953091 + 19.09167*((2086.9/pbr.init[53])/sinh(2086.9/pbr.init[53]))^2 + 9.936467*((991.96/pbr.init[53])/cosh(991.96/pbr.init[53]))^2, # CH4
            7.972676 + 22.6402*((1596/pbr.init[53])/sinh(1596/pbr.init[53]))^2 + 13.16041*((740.8/pbr.init[53])/cosh(740.8/pbr.init[53]))^2,     # C2H4
            10.57036 + 20.23908*((872.24/pbr.init[53])/sinh(872.24/pbr.init[53]))^2 + 16.03373*((2430.4/pbr.init[53])/cosh(2430.4/pbr.init[53]))^2,  # other species...
        ]
    catch OverflowError
        CP = zeros(52)
    end

    sumFiCpi = dot(pbr.init[1:52], CP[1:52]) * 4.18

    dcp = Float64[]
    num = 0
    for i in 6:2:52
        if i == 6
            dcp_co2 = (CP[1] + CP[3]) - (CP[4] + CP[2])
            dcp_ch4 = (CP[5] + CP[3]) - (CP[1] + 3*CP[2])
            push!(dcp, dcp_co2, dcp_ch4)
        end
        j = i - (3 + num)
        dcp_paraf = (CP[i] + (j)*CP[3]) - ((j)*CP[1] + 2*j*CP[2])
        dcp_olef = (CP[i+1] + (j)*CP[3]) - ((j)*CP[1] + (2*j+1)*CP[2])
        push!(dcp, dcp_paraf, dcp_olef)
        num += 1
    end

    dH = dHr + dcp * 4.18 * (pbr.init[53] - 298)
    return sumFiCpi, dH
end

function reactor(pbr::FT_PBR)
    Ac_1 = 0.159592907 * pbr.z
    Ao_1 = 0.18315 * pbr.z
    a = 4 / 0.0508
    bd = 1640000.0
    Sc = 24.0
    Ut = 38.8
    Us = 39.9

    r_olef, r_paraf, r_co2, r_ch4, r_co, r_h2, r_h2o = kinetics(pbr)
    sumFiCpi, dH = energy_balance(pbr)
    R = vcat(r_co, r_h2, r_h2o, r_co2, r_ch4, r_paraf, r_olef)

    dFdz = R * Ac_1 * bd * pbr.Nt
    Q = -R[4]*dH[1] + sum(R[5:end] .* dH[2:end])

    dTtdz = ((Ut*a*(pbr.init[54]-pbr.init[53]) - Q*Sc*bd) / sumFiCpi) * Ac_1
    CP_oil = 0.4725 * pbr.init[53] + 122.1
    dTsdz = (pbr.Nt * Us * Ao_1 * (pbr.init[54] - pbr.init[53])) / (CP_oil * pbr.mc * pbr.z)

    return dFdz, dTtdz, dTsdz
end