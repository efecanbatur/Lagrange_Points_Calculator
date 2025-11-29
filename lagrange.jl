# ####################################################################
# Author: Efe Can Batur                                              #
#                                                                    #
# based on Periodic Orbits For the Perturbed                         #
# Planar Circular Restricted 3-Body Problem                          #
# (Abouelmagd et al., 2019)                                          #
#                                                                    #
#  Calculation of Lagrange points                                    #
# ####################################################################

using Printf
using NLsolve
using LinearAlgebra

# Define the main function
function main()

    # Gravitational Parameters km^3/s^2
    GM_earth = 398600.432897
    GM_moon  = 4902.800582

    # Radii km
    R_earth  = 6378.1363
    R_moon   = 1737.4

    # Mean Distance km
    Dist_EM  = 384400.0

    # J2 Coefficients
    J2_earth = 1.08262668e-3
    J2_moon  = 2.027e-4

    # Normalized System
    mu = GM_moon / (GM_earth + GM_moon)

    # Scaling Factors for A (Dimensionless)
    Re_norm = R_earth / Dist_EM
    Rm_norm = R_moon / Dist_EM
    scale = Dist_EM

    A_earth = J2_earth * (Re_norm^2)
    A_moon  = J2_moon  * (Rm_norm^2)

    # ---------------------------------------------------------
    # IMPLEMENTATION OF MEAN MOTION (n) FORMULATION (Eq. 37)
    # ---------------------------------------------------------
    # Formula: n^2 = 1 + (3/2) * [ (1-mu)*J2p*R1^2 + mu*J2s*R2^2 ]
    # Note: J2 * R^2 is equivalent to the dimensionless 'A' parameters

    term_earth = (1 - mu) * A_earth
    term_moon = mu * A_moon

    n_sq = 1.0 + (3.0 / 2.0) * (term_earth + term_moon)
    n = sqrt(n_sq)

    @printf("--- Model: Abouelmagd et al. (2019) + Eq 37 for n ---\n")
    @printf("Mass Ratio (mu):      %.10f\n", mu)
    @printf("Earth Parameter (A1): %.10e\n", A_earth)
    @printf("Moon Parameter (A2):  %.10e\n", A_moon)
    @printf("Perturbed Mean Motion n: %.10f\n", n)
    println("Coordinates: Earth at x=mu, Moon at x=mu-1")

    # Equations function
    function equations(vars, include_pert::Bool)
        x = vars[1]
        y = vars[2]

        dx1 = x - mu
        dx2 = x - mu + 1

        r1_sq = dx1^2 + y^2
        r2_sq = dx2^2 + y^2

        r1 = sqrt(r1_sq)
        r2 = sqrt(r2_sq)

        f = (1 - mu) / (r1^3)
        g = mu / (r2^3)

        # Determine rotation rate for centrifugal force
        # If perturbed, use calculated n^2. If unperturbed, n=1.
        current_n_sq = include_pert ? n_sq : 1.0

        # Point Mass Acceleration with dynamic mean motion
        # Centrifugal force is n^2 * r
        ax_pm = current_n_sq * x - (dx1 * f) - (dx2 * g)
        ay_pm = current_n_sq * y - y * (f + g)

        # Perturbed Forces
        Fx = 0.0
        Fy = 0.0

        if include_pert
            # Earth Perturbation (A_earth)
            h_e = (3 * (1 - mu)) / (2 * r1^5)
            ax_pert_e = A_earth * dx1 * h_e
            ay_pert_e = A_earth * y * h_e

            # Moon Perturbation (A_moon) - Symmetric
            h_m = (3 * mu) / (2 * r2^5)
            ax_pert_m = A_moon * dx2 * h_m
            ay_pert_m = A_moon * y * h_m

            Fx = ax_pm + ax_pert_e + ax_pert_m
            Fy = ay_pm + ay_pert_e + ay_pert_m
        else
            Fx = ax_pm
            Fy = ay_pm
        end

        return [Fx, Fy]
    end

    # Notation:
    # L1: Between Moon and Earth
    # L2: Left of Moon
    # L3: Right of Earth

    g_L1 = [(mu - 1) + (mu/3)^(1/3), 0.0]
    g_L2 = [(mu - 1) - (mu/3)^(1/3), 0.0]
    g_L3 = [mu + 1, 0.0]

    # Triangle points - standard location relative to barycenter
    g_L4 = [mu - 0.5, sqrt(3)/2]
    g_L5 = [mu - 0.5, -sqrt(3)/2]

    labels = ["L1", "L2", "L3", "L4", "L5"]
    guesses = [g_L1, g_L2, g_L3, g_L4, g_L5]

    pm_pts = Vector{Vector{Float64}}()
    j2_pts = Vector{Vector{Float64}}()

    # Solve
    for guess in guesses
        # Solve Point Mass (Unperturbed)
        res_pm = nlsolve(v -> equations(v, false), guess, ftol=1e-13, method=:trust_region)
        push!(pm_pts, res_pm.zero)

        # Solve J2 Perturbed
        res_j2 = nlsolve(v -> equations(v, true), guess, ftol=1e-13, method=:trust_region)
        push!(j2_pts, res_j2.zero)
    end

    # Data Collection for Table 1
    t1_data = []
    for i in 1:length(labels)
        x = pm_pts[i][1]
        y = pm_pts[i][2]
        x_km = x * scale
        y_km = y * scale
        # Formatting similar to Python table row
        push!(t1_data, (labels[i], @sprintf("%.10f", x), @sprintf("%.10f", y), @sprintf("%.10f", x_km), @sprintf("%.10f", y_km)))
    end

    # Data Collection for Table 2
    t2_data = []
    for i in 1:length(labels)
        x_p = j2_pts[i][1]
        y_p = j2_pts[i][2]
        
        dx = (x_p - pm_pts[i][1]) * scale * 1000
        dy = (y_p - pm_pts[i][2]) * scale * 1000
        total_shift = sqrt(dx^2 + dy^2)
        
        push!(t2_data, (labels[i], @sprintf("%.10f", x_p), @sprintf("%.10f", y_p), @sprintf("%.10f", total_shift)))
    end

    # Print to terminal
    println("Table 1: Unperturbed Coordinates")
    println("Point | X (ND)         | Y (ND)         | X (km)             | Y (km)")
    println("---------------------------------------------------------------------------")
    for row in t1_data
        @printf("%-5s | %-14s | %-14s | %-18s | %-18s\n", row[1], row[2], row[3], row[4], row[5])
    end

    println("\nTable 2: Perturbed Coordinates & Shift")
    println("Point | J2 X (ND)      | J2 Y (ND)      | Shift (m)")
    println("-----------------------------------------------------------")
    for row in t2_data
        @printf("%-5s | %-14s | %-14s | %-14s\n", row[1], row[2], row[3], row[4])
    end
end

# Run main function
main()