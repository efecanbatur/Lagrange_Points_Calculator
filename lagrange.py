# ####################################################################
# Author: Efe Can Batur                                              #
#                                                                    #
# based on Periodic Orbits For the Perturbed                         #
# Planar Circular Restricted 3-Body Problem                          #
# (Abouelmagd et al., 2019)                                          #
#                                                                    #
#  Calculation of Lagrange points                                    #
# ####################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def main():

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

    A_earth = J2_earth * (Re_norm**2)
    A_moon  = J2_moon  * (Rm_norm**2)

    # ---------------------------------------------------------
    # IMPLEMENTATION OF MEAN MOTION (n) FORMULATION (Eq. 37)
    # ---------------------------------------------------------
    # Formula: n^2 = 1 + (3/2) * [ (1-mu)*J2p*R1^2 + mu*J2s*R2^2 ]
    # Note: J2 * R^2 is equivalent to the dimensionless 'A' parameters
    
    term_earth = (1 - mu) * A_earth
    term_moon = mu * A_moon
    
    n_sq = 1.0 + (3.0 / 2.0) * (term_earth + term_moon)
    n = np.sqrt(n_sq)

    print(f"--- Model: Abouelmagd et al. (2019) + Eq 37 for n ---")
    print(f"Mass Ratio (mu):      {mu:.10f}")
    print(f"Earth Parameter (A1): {A_earth:.10e}")
    print(f"Moon Parameter (A2):  {A_moon:.10e}")
    print(f"Perturbed Mean Motion n: {n:.10f}")
    print(f"Coordinates: Earth at x=mu, Moon at x=mu-1")

    # eq.
    def equations(vars, include_pert=False):
        x, y = vars
        
        dx1 = x - mu
        dx2 = x - mu + 1
        
        r1_sq = dx1**2 + y**2
        r2_sq = dx2**2 + y**2
        
        r1 = np.sqrt(r1_sq)
        r2 = np.sqrt(r2_sq)
        
        f = (1 - mu) / (r1**3)
        g = mu / (r2**3)
        
        # Determine rotation rate for centrifugal force
        # If perturbed, use calculated n^2. If unperturbed, n=1.
        current_n_sq = n_sq if include_pert else 1.0

        # Point Mass Acceleration with dynamic mean motion
        # Centrifugal force is n^2 * r
        ax_pm = current_n_sq * x - (dx1 * f) - (dx2 * g)
        ay_pm = current_n_sq * y - y * (f + g)
        
        # Perturbed Forces 
        
        if include_pert:
            # Earth Perturbation (A_earth)
            h_e = (3 * (1 - mu)) / (2 * r1**5)
            ax_pert_e = A_earth * dx1 * h_e
            ay_pert_e = A_earth * y * h_e
            
            # Moon Perturbation (A_moon) - Symmetric
            h_m = (3 * mu) / (2 * r2**5)
            ax_pert_m = A_moon * dx2 * h_m
            ay_pert_m = A_moon * y * h_m
            
            Fx = ax_pm + ax_pert_e + ax_pert_m
            Fy = ay_pm + ay_pert_e + ay_pert_m
        else:
            Fx = ax_pm
            Fy = ay_pm
            
        return [Fx, Fy]
    
    # Notation:
    # L1: Between Moon and Earth
    # L2: Left of Moon
    # L3: Right of Earth
    
    g_L1 = [(mu - 1) + (mu/3)**(1/3), 0] 
    g_L2 = [(mu - 1) - (mu/3)**(1/3), 0]
    g_L3 = [mu + 1, 0]
    
    # Triangle points - standard location relative to barycenter
    g_L4 = [mu - 0.5, np.sqrt(3)/2]
    g_L5 = [mu - 0.5, -np.sqrt(3)/2]
    
    labels = ['L1', 'L2', 'L3', 'L4', 'L5']
    guesses = [g_L1, g_L2, g_L3, g_L4, g_L5]
    
    pm_pts = [] 
    j2_pts = [] 
    
    # Solve
    for guess in guesses:
        pm_pts.append(fsolve(lambda v: equations(v, False), guess, xtol=1e-13))
        j2_pts.append(fsolve(lambda v: equations(v, True), guess, xtol=1e-13))

    # Plot

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.0, 1.0], hspace=0.4)
    
    ax_plot = fig.add_subplot(gs[0])
    ax_t1 = fig.add_subplot(gs[1])
    ax_t2 = fig.add_subplot(gs[2])
    
    ax_t1.axis('off')
    ax_t2.axis('off')

    ax_plot.plot(mu*scale, 0, 'ko', markersize=12, label='Earth')
    ax_plot.plot((mu-1)*scale, 0, 'o', color='gray', markersize=8, markeredgecolor='k', label='Moon')

    for i, pt_code in enumerate(labels):
        x_pm, y_pm = pm_pts[i] * scale
        x_j2, y_j2 = j2_pts[i] * scale
        
        ax_plot.plot(x_pm, y_pm, 'bo', markersize=6, label='Point-Mass' if i==0 else "")
        ax_plot.plot(x_j2, y_j2, 'r*', markersize=10, label='J2-Perturbed' if i==0 else "")
        ax_plot.plot([x_pm, x_j2], [y_pm, y_j2], 'k--', linewidth=0.8)
        
        y_offset = 25000 if y_pm >= 0 else -30000
        ax_plot.text(x_pm, y_pm + y_offset, pt_code, ha='center', fontweight='bold')

    ax_plot.set_title('Lagrange Points')
    ax_plot.set_xlabel('X (km)')
    ax_plot.set_ylabel('Y (km)')
    ax_plot.legend(loc='upper right')
    ax_plot.grid(True, alpha=0.3)
    ax_plot.axis('equal')

    t1_data = []
    for i, lab in enumerate(labels):
        x, y = pm_pts[i]
        x_km = x * scale
        y_km = y * scale
        t1_data.append([lab, f"{x:.10f}", f"{y:.10f}", f"{x_km:.10f}", f"{y_km:.10f}"])
        
    tab1 = ax_t1.table(cellText=t1_data, 
                       colLabels=['Point', 'X (ND)', 'Y (ND)', 'X (km)', 'Y (km)'],
                       loc='center', cellLoc='center')
    tab1.scale(1, 1.5)
    tab1.auto_set_font_size(False)
    tab1.set_fontsize(9)
    ax_t1.set_title("Table 1: Reference Point-Mass Coordinates")

    t2_data = []
    for i, lab in enumerate(labels):
        x_p, y_p = j2_pts[i]
        dx = (x_p - pm_pts[i][0]) * scale * 1000
        dy = (y_p - pm_pts[i][1]) * scale * 1000
        total_shift = np.sqrt(dx**2 + dy**2)
        t2_data.append([lab, f"{x_p:.10f}", f"{y_p:.10f}", f"{total_shift:.10f}"])
        
    tab2 = ax_t2.table(cellText=t2_data,
                       colLabels=['Point', 'J2 X (ND)', 'J2 Y (ND)', 'Shift (m)'],
                       loc='center', cellLoc='center')
    tab2.scale(1, 1.5)
    tab2.auto_set_font_size(False)
    tab2.set_fontsize(9)
    ax_t2.set_title("Table 2: J2-Perturbed Coordinates")

    # Print to terminal
    print("Table 1: Unperturbed Coordinates")
    for row in t1_data: print(row)
    print("Table 2: Perturbed Coordinates & Shift")
    for row in t2_data: print(row)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()