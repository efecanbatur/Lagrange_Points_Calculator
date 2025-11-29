################################################################
# Author: Efe Can Batur                                        #
#                                                              #
# Based on Sharma & Rao, 1976:                                 #
#                                                              #
# "Stationary solutions and their characteristic exponents     #
# in the restricted three-body problem when the more           #
# massive primary is an oblate spheroid"                       #
#                                                              #
# Calculates Lagrange points' coordinates and shifts.          #
################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def main():

    # Planetary constants & ref: NASA
    GM_earth = 3.986004418e14
    GM_moon  = 4.9048695e12
    R_earth  = 6378.137     
    R_moon   = 1737.4       
    Dist_EM  = 384400.0     
    
    # Oblateness J2
    J2_earth = 1.08263e-3   
    J2_moon  = 0.0          # Set to 2.027e-4 to include Moon effects
                            # Set to 0 for benchmark
    # Normalized System
    mu = GM_moon / (GM_earth + GM_moon)
    Re_norm = R_earth / Dist_EM
    Rm_norm = R_moon / Dist_EM
    scale = Dist_EM

    # An the dimensionless oblateness parameter respectively
    A1 = J2_earth * (Re_norm**2)
    A2 = J2_moon  * (Rm_norm**2)

    # Perturbed Mean Motion n
    # Equation: n^2 = 1 + 3/2 * A1
    n_sq_perturbed = 1.0 + 1.5 * A1 + 1.5 * A2
    n_perturbed = np.sqrt(n_sq_perturbed)

    print(f"Model: Sharma & Subba Rao")
    print(f"Mass Ratio (mu): {mu:.9f}")
    print(f"Earth Oblateness (A1): {A1:.9e}")
    print(f"Moon Oblateness (A2):  {A2:.9e}")
    print(f"Perturbed n: {n_perturbed:.9f}")

    def equations(vars, include_j2=False):
        x, y = vars
        r1 = np.sqrt((x + mu)**2 + y**2)       
        r2 = np.sqrt((x - (1 - mu))**2 + y**2) 
        
        # Rotation Rate n^2
        n_sq = n_sq_perturbed if include_j2 else 1.0

        # Centrifugal Force n^2 * r
        f_cent_x = n_sq * x
        f_cent_y = n_sq * y
        
        # Gravity - Point Mass assumed
        fx_grav = -(1 - mu) * (x + mu) / r1**3 - mu * (x - (1 - mu)) / r2**3
        fy_grav = -(1 - mu) * y / r1**3        - mu * y / r2**3
        
        Fx = f_cent_x + fx_grav
        Fy = f_cent_y + fy_grav
        
        # J2 Perturbation
        if include_j2:
            # Earth - A1
            factor_e = -1.5 * (1 - mu) * A1 / (r1**5)
            fx_j2_e = factor_e * (x + mu)
            fy_j2_e = factor_e * y
            
            # Moon - A2
            factor_m = -1.5 * mu * A2 / (r2**5)
            fx_j2_m = factor_m * (x - (1 - mu))
            fy_j2_m = factor_m * y
            
            Fx += fx_j2_e + fx_j2_m
            Fy += fy_j2_e + fy_j2_m
            
        return [Fx, Fy]

    # Solve for points
    
    # initial guess
    g_L1 = [(1 - mu) - (mu/3)**(1/3), 0]
    g_L2 = [(1 - mu) + (mu/3)**(1/3), 0]
    g_L3 = [-(1 + 5*mu/12), 0]
    g_L4 = [0.5 - mu, np.sqrt(3)/2]
    g_L5 = [0.5 - mu, -np.sqrt(3)/2]
    
    labels = ['L1', 'L2', 'L3', 'L4', 'L5']
    guesses = [g_L1, g_L2, g_L3, g_L4, g_L5]
    
    pm_pts = [] 
    j2_pts = [] 
    
    for guess in guesses:
        pm_pts.append(fsolve(lambda v: equations(v, False), guess, xtol=1e-12))
        j2_pts.append(fsolve(lambda v: equations(v, True), guess, xtol=1e-12))

    # Plot
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 0.8, 1.2], hspace=0.3)
    
    ax_plot = fig.add_subplot(gs[0])
    ax_t1 = fig.add_subplot(gs[1])
    ax_t2 = fig.add_subplot(gs[2])
    
    ax_t1.axis('off')
    ax_t2.axis('off')

    ax_plot.plot(-mu*scale, 0, 'ko', markersize=10, label='Earth (Oblate)')
    ax_plot.plot((1-mu)*scale, 0, 'o', color='gray', markersize=8, markeredgecolor='k', label='Moon')

    for i, pt_code in enumerate(labels):
        x_pm, y_pm = pm_pts[i] * scale
        x_j2, y_j2 = j2_pts[i] * scale
        
        ax_plot.plot(x_pm, y_pm, 'bo', markersize=6, label='Point-Mass' if i==0 else "")
        ax_plot.plot(x_j2, y_j2, 'r*', markersize=10, label='J2-Perturbed' if i==0 else "")
        
        # shift
        ax_plot.plot([x_pm, x_j2], [y_pm, y_j2], 'k--', linewidth=0.8)
        
        y_offset = 20000 if y_pm >= 0 else -25000
        ax_plot.text(x_pm, y_pm + y_offset, pt_code, ha='center', fontweight='bold')

    ax_plot.set_title('Earth-Moon Lagrangian Points: Validation')
    ax_plot.set_xlabel('X (km)')
    ax_plot.set_ylabel('Y (km)')
    ax_plot.legend(loc='upper right')
    ax_plot.grid(True, alpha=0.3)
    ax_plot.axis('equal')

    # unperturbed
    t1_data = []
    for i, lab in enumerate(labels):
        x, y = pm_pts[i]
        t1_data.append([lab, f"{x:.8f}", f"{y:.8f}"])
        
    tab1 = ax_t1.table(cellText=t1_data, 
                       colLabels=['Point', 'Unperturbed X (ND)', 'Unperturbed Y (ND)'],
                       loc='center', cellLoc='center')
    tab1.scale(1, 1.3)
    ax_t1.set_title("Table 1: Reference Point-Mass Coordinates")

    # perturbed
    t2_data = []
    for i, lab in enumerate(labels):
        x_p, y_p = j2_pts[i]
        
        # Shift in meters
        dx = (x_p - pm_pts[i][0]) * scale * 1000
        dy = (y_p - pm_pts[i][1]) * scale * 1000
        total = np.sqrt(dx**2 + dy**2)
        
        t2_data.append([lab, f"{x_p:.8f}", f"{y_p:.8f}", f"{total:.2f}"])
        
    tab2 = ax_t2.table(cellText=t2_data,
                       colLabels=['Point', 'Perturbed X (ND)', 'Perturbed Y (ND)', 'Total Shift (m)'],
                       loc='center', cellLoc='center')
    tab2.scale(1, 1.3)
    ax_t2.set_title("Table 2: J2-Perturbed Coordinates & Shift Magnitude")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()