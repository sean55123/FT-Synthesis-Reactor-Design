function compute_derivatives(Y, params)
    try
        etype = eltype(Y)

        # Extract parameters
        To = params.To
        Ta0 = params.Ta0
        z = params.z
        Nt = params.Nt
        mc = params.mc
        H2in = params.H2in
        alpha = params.alpha
        k1 = params.k1
        k6m = params.k6m
        K2 = params.K2
        K3 = params.K3
        K4 = params.K4
        PT = params.PT
        Ac_1 = params.Ac_1
        Ao_1 = params.Ao_1
        a = params.a
        bd = params.bd
        Sc = params.Sc
        Ut = params.Ut
        Us = params.Us

        T = Y[54]  # Process temperature
        Ta = Y[55]  # Coolant temperature
        init = copy(Y)

        T = max(T, etype(1e-6))
        Ta = max(Ta, etype(1e-6))

        R_gas = etype(8.314) # J/mol-K

        # Reaction rate constants
        k5m = etype(1.4e3) * exp(-etype(92890) / (R_gas * T))
        k5 = etype(2.74e2) * exp(-etype(87010) / (R_gas * T))
        k6 = etype(1.5e6) * exp(-etype(111040) / (R_gas * T))
        k6e = k6
        k5e = k5
        kv = etype(1.57e1) * exp(-etype(45080) / (R_gas * T))

        # Equilibrium constant Kp
        Kp = exp(
            5078.0045 / T - 5.8972089
            + 1.3958689e-3 * T
            - 2.7592844e-6 * T^2
        )

        # Kinetics calculations
        FT = sum(init[1:end-2])
        P = init[1:end-2] * (PT / FT) * (T / Ta0)
        P_CO = P[1]
        P_H2 = P[2]
        P_H2O = P[3]
        P_CO2 = P[4]

        P_H2_sqrt = sqrt(P_H2)
        P_H2_inv = 1.0 / P_H2

        A = P_CO * P_H2O / P_H2_sqrt
        B = P_CO2 * P_H2_sqrt / Kp
        A1 = (etype(1.0) / (K2 * K3 * K4)) * (P_H2O * P_H2_inv^2)
        A2 = (etype(1.0) / (K3 * K4)) * P_H2_inv
        A3 = etype(1.0) / K4

        # Reaction rate constants sum
        RRR = k1 * P_CO + k5 * P_H2 + k6

        upper_1 = (k1 * P_CO) / RRR

        # Compute beta factors
        # Compute beta factors
        i_vals = collect(1:25)
        betaf = alpha .^ i_vals * upper_1

        var = init[6:2:54]  # Adjusted indices for 1-based indexing
        denom_beta = betaf .+ (k6m / k6) .* var .* (PT / FT) .* (T / Ta0)
        denom_beta = [x == 0 ? etype(1e-6) : x for x in denom_beta]
        beta = ((k6m / k6) .* var .* (PT / FT) .* (T / Ta0)) ./ denom_beta
        alpha_prob = i_vals .* (etype(1.0) - alpha)^2 .* alpha .^ (i_vals .- 1)
        Deno = etype(1.0) + (etype(1.0) + A1 + A2 + A3) * sum(alpha_prob)

        # Reaction rates
        r_olef = (k5e * P_H2 .* alpha_prob[2:end]) / Deno
        r_paraf = (k6e .* (etype(1.0) .- beta[2:end]) .* alpha_prob[2:end]) / Deno

        r_co2 = (kv * (A - B)) / (etype(1.0) + kv * A)
        r_ch4 = (k5m * P_H2 * alpha_prob[1]) / Deno

        # Species rates
        i_vals_shifted = i_vals[2:end] .+ 1

        r_co = -r_co2 - r_ch4 - sum(i_vals_shifted .* (r_olef + r_paraf))
        r_h2 = r_co2 - etype(3.0) * r_ch4 - sum((etype(2.0) .* i_vals_shifted .+ etype(1.0)) .* r_olef + etype(2.0) .* i_vals_shifted .* r_paraf)
        r_h2o = -r_co2 + r_ch4 + sum(i_vals_shifted .* (r_olef + r_paraf))

        R = zeros(etype, 53)
        R[1] = r_co
        R[2] = r_h2
        R[3] = r_h2o
        R[4] = r_co2
        R[5] = r_ch4
        R[6:2:end] = r_paraf  # Even indices for paraffins
        R[7:2:end] = r_olef   # Odd indices for olefins

        # Energy balance
        # Enthalpy changes (dHr) and heat capacities (CP)
        dHr_raw = [
            4.10953e4, -7.4399e4, -2.09725e5, -8.3684e4, -4.83377e5, -1.0451e5,
            -7.54025e5, -1.25586e5, -1.01496e6, -1.46522e5, -1.27689e6,
            -1.66669e5, -1.54408e6, -1.91349e5, -1.80075e6, -2.55233e5,
            -2.15835e6, -2.35357e5, -2.42959e6, -2.65858e5, -2.61318e6,
            -2.69991e5, -2.85974e6, -2.90248e5, -3.10372315e6, -3.112637e5,
            -3.36571e6, -3.31899942e5, -3.6283056e6, -3.525366e5, -3.891285e6,
            -3.7356242e5, -4.152388592e6, -3.93809504e5, -4.41448e6, -4.144457e5,
            -4.67657e6, -4.35084e5, -4.938562992e6, -4.5572032e5, -5.199596468e6,
            -4.7702578e5, -5.46185e6, -4.976917e5, -5.7235398e6, -4.12429e5,
            -5.9859305e6, -5.3912386e5, -6.4840277e6, -5.5978978e5
        ]

        dHr = map(etype, dHr_raw)
        CP = zeros(etype, 54)

        # Ideal gas parameters for CP calculation
        ideal_gas_params_raw = [
            (6.95233, 2.09540, 3085.1, 2.01951, 1538.2),  # CO
            (6.59621, 2.28337, 2466.0, 0.89806, 567.6),   # H2
            (7.96862, 6.39868, 2610.5, 2.12477, 1169.0),  # H2O
            (7.014904, 8.249737, 1428.0, 6.305532, 588.0),  # CO2
            (7.953091, 19.09167, 2086.9, 9.936467, 991.96),  # CH4
            (7.972676, 22.6402, 1596.0, 13.16041, 740.8),  # C2H4
            (10.57036, 20.23908, 872.24, 16.03373, 2430.4),
            (10.47387, 35.97019, 1398.8, 17.85469, 616.46),
            (14.20512, 30.24028, 844.31, 20.58016, 2482.7),
            (15.34752, 49.24525, 1676.8, 31.8238, 757.06),
            (19.14445, 38.79335, 841.49, 25.25795, 2476.1),
            (19.71028, 61.96379, 1729.1, 42.228, 778.7),
            (21.03038, 71.91650, 1650.2, 45.18964, 747.6),
            (24.92118, 73.44272, 1745.9, 49.50798, 793.53),
            (24.93551, 84.14541, 1694.6, 56.58259, 761.6),
            (28.30563, 86.84914, 1735.9, 59.82612, 785.73),
            (28.69733, 95.56224, 1676.6, 65.44378, 756.4),
            (32.48065, 99.37184, 1731.7, 68.48906, 784.47),
            (32.37317, 105.8326, 1635.6, 72.94354, 746.4),
            (36.66762, 111.885, 1728.8, 77.15678, 783.67),
            (36.24486, 117.3928, 1644.8, 82.87953, 749.6),
            (40.84504, 124.4124, 1726.5, 85.82927, 782.92),
            (39.93503, 127.8542, 1614.1, 90.33152, 742.0),
            (124.1163, 57.65262, 1834.6, -161.8276, 278.28),
            (46.64422, 145.6912, 1708.7, 98.64813, 775.4),
            (51.69342, 102.0971, 815.94, 63.27028, 2417.3),
            (50.86223, 158.4265, 1715.5, 107.8652, 777.5),
            (53.3725, 161.9853, 1721.1, 111.8181, 781.22),
            (51.34231, 174.465, 1669.5, 119.4182, 741.02),
            (57.55231, 174.508, 1719.9, 120.4834, 780.87),
            (55.13041, 187.9192, 1682.3, 130.1376, 743.1),
            (56.08102, 172.8193, 1553.7, 127.8733, 723.17),
            (58.94478, 201.1369, 1656.5, 139.8132, 743.6),
            (59.85717, 184.0212, 1551.8, 136.5721, 723.07),
            (62.77587, 214.3236, 1691.2, 149.6131, 744.41),
            (63.67632, 195.2064, 1552.5, 145.5527, 723.89),
            (66.58546, 227.4936, 1693.5, 159.1932, 744.57),
            (67.50502, 206.4894, 1554.5, 154.6694, 724.9),
            (70.46432, 239.658, 771.07, -102.7324, 916.73),
            (71.22146, 217.3665, 1546.3, 162.5752, 723.07),
            (74.19031, 252.5795, 767.91, -109.0594, 912.03),
            (75.18152, 229.3565, 747.85, -70.4476, 864.56),
            (77.57954, 264.8801, 1636.0, 177.9402, 726.27),
            (91.43499, 184.1669, 801.08, 119.2032, 2361.6),
            (91.43499, 184.1669, 801.08, 119.2032, 2361.6),
            (91.08388, 274.6011, 1715.3, 189.7774, 779.78),
            (93.76135, 282.3158, 1723.4, 194.8457, 785.13),
            (97.96264, 294.7836, 1723.1, 203.5636, 784.97),
            (97.96264, 294.7836, 1723.1, 203.5636, 784.97),
            (99.44827, 299.6561, 1714.4, 207.1033, 779.51),
            (102.0302, 307.8962, 815.29, -120.4213, 944.98),
            (106.3223, 319.7908, 1721.5, 220.8799, 784.28),  # C25H50
            (106.3223, 319.7908, 1721.5, 220.8799, 784.28),  # C25H52
            (6.95161, 2.05763, 1701.6, 0.0247, 909.79), # N2
        ]

        ideal_gas_params = [map(etype, params) for params in ideal_gas_params_raw]

        for idx in 1:length(ideal_gas_params)
            A, B, C, D, E = ideal_gas_params[idx]
            theta_C = C * (etype(1.0) / T)
            theta_E = E * (etype(1.0) / T)
            sinh_C = sinh(theta_C)
            cosh_E = cosh(theta_E)

            # Avoid division by zero
            sinh_C = sinh_C == etype(0) ? etype(1e-6) : sinh_C
            cosh_E = cosh_E == etype(0) ? etype(1e-6) : cosh_E

            CP[idx] = A + B * (theta_C / sinh_C)^2 + D * (theta_E / cosh_E)^2
        end

        # Convert CP from cal/mol-K to J/mol-K
        CP .= CP .* etype(4.184)

        # For species without parameters, set a default CP value
        default_CP = etype(29.0 * 4.184)  # J/mol-K
        CP[length(ideal_gas_params)+1:end] .= default_CP

        b = init[1:end-2]
        sumFiCpi = sum(b .* CP[1:53])

        dcp = zeros(length(dHr))
        dH = dHr .+ dcp .* (T - etype(298.0))

        # Heat flow
        Q = -R[4] * dH[1] + sum(R[5:end] .* dH[2:end])

        # Reactor equations
        dFdz = R .* Ac_1 .* bd .* Nt

        dTtdz = ((Ut * a * (Ta - T) - Q * Sc * bd) / sumFiCpi) * Ac_1  # Process temp derivative
        CP_oil = etype(0.4725) * T + etype(122.1)  # J/kg-K, thermal oil CP
        dTsdz = (Nt * Us * Ao_1 * (Ta - T)) / (CP_oil * mc * z)

        dYdW = zeros(etype, 55)
        dYdW[1:end-2] .= dFdz
        dYdW[54] = dTtdz
        dYdW[55] = dTsdz
        if any(isnan.(dYdW)) || any(isinf.(dYdW))
            error("dYdW contains NaN or Inf values")
        end
        
        return dYdW
    catch e
        println("Error in compute_derivatives: ", e)
        rethrow(e)
    end
end