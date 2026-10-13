# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:08:51 2026

@author: sarah
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. FILE CLEANING AND LOADING (use the RAW file, not the "cal" one)
cleaned_rows = []
with open("Test_7_data.txt", "r") as f:          # your raw filename
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace("- ", "-").split()
        if len(parts) == 3:
            cleaned_rows.append([float(p) for p in parts])

data = pd.DataFrame(cleaned_rows, columns=["t", "x", "y"])
t = data["t"].to_numpy()

# CALIBRATION: paste the "Metres per pixel" value your calibration script printed
M_PER_PX = 9.1e-8
positions = np.column_stack((data["x"], data["y"])) * M_PER_PX

dt_frame = t[1] - t[0]
n_frames = len(t)
max_lags = int(n_frames * 0.25)

# 2. TIME-LAG MSD AND ERROR BARS
msd, dt, tamsd_error = [], [], []

for lag in range(1, max_lags):
    sq_distances = np.sum((positions[lag:] - positions[:-lag])**2, axis=1)
    current_msd = np.mean(sq_distances)

    # FIX: error for overlapping windows (Qian et al. 1991)
    frac_err = np.sqrt((2 * lag**2 + 1) / (3 * lag * (n_frames - lag + 1)))

    dt.append(lag * dt_frame)
    msd.append(current_msd)
    tamsd_error.append(current_msd * frac_err)

dt = np.array(dt)
msd = np.array(msd)
tamsd_error = np.array(tamsd_error)

# 3. LINEAR FIT WITH SLOPE UNCERTAINTY
# FIX: fit only the first few lags and get the error on the slope
n_fit = 12
(coeffs, cov) = np.polyfit(dt[:n_fit], msd[:n_fit], 1, cov=True)
slope, intercept = coeffs
slope_err = np.sqrt(cov[0, 0])

print("FIT RESULTS")
print(f"Slope: {slope:.3e} +/- {slope_err:.1e} m^2/s")
print(f"Intercept: {intercept:.3e}")

fit_line = slope * dt[:n_fit] + intercept

# 4. PLOT
plt.figure(figsize=(7, 5))
plt.errorbar(dt, msd, yerr=tamsd_error, fmt='o', color='#1f77b4',
             ecolor='red', capsize=3, markersize=4, label='Data with Error Bars')
plt.plot(dt[:n_fit], fit_line, '--', color='black',
         label=f'Linear Fit (slope = {slope:.2e})')
plt.title('Mean Squared Displacement (MSD) vs Time Lag')
plt.xlabel(r'Time Lag, $\tau$ (s)')
plt.ylabel(r'MSD (m$^2$)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# 5. VISCOSITY WITH UNCERTAINTY
k = 1.38e-23
T, dT = 293, 1          # K, your temperature uncertainty
r, dr = 0.53e-6, 0.05e-6   # m, bead radius and its uncertainty

vis = (2 * k * T) / (3 * np.pi * slope * r)

# FIX: errors add in quadrature
vis_err = vis * np.sqrt((slope_err / slope)**2 + (dT / T)**2 + (dr / r)**2)

print(f"Viscosity = {vis:.3e} +/- {vis_err:.1e} Pa s")


# n_fit sweep
etas = []
for n in [5, 8, 11]:
    s = np.polyfit(dt[:n], msd[:n], 1)[0]
    e = (2 * k * T) / (3 * np.pi * s * r)
    etas.append(e)
    print(f"n_fit = {n:2d}: slope = {s:.3e}, viscosity = {e:.3e} Pa s")

fit_spread = (max(etas) - min(etas)) / 2
print(f"Fit-range spread = +/- {fit_spread:.1e} Pa s")

# combine with your existing error
total_err = np.sqrt(vis_err**2 + fit_spread**2)
print(f"Final: eta = {vis:.3e} +/- {total_err:.1e} Pa s")
