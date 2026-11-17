# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 07:27:33 2026

@author: Celes
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm

# CONSTANTS
k_B = 1.380649e-23
T = 293
CONVERSION = 9.0498e-09
MAX_LAG_FRACTION = 0.25
TRACK_ID = 0
START_FRAME = None
END_FRAME = None

def load_data(filename, FPS):
    data = pd.read_csv(filename)
    data.columns = data.columns.str.strip()

    required = [
        "FRAME",
        "POSITION_X",
        "POSITION_Y"
    ]

    if not all(c in data.columns for c in required):
        with open(filename, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()

        header_row = None

        for i, line in enumerate(lines):
            if all(c in line for c in required):
                header_row = i
                break

        if header_row is None:
            raise ValueError(
                "Cannot find FRAME, POSITION_X "
                "and POSITION_Y columns."
            )

        data = pd.read_csv(
            filename,
            skiprows=header_row
        )
        data.columns = data.columns.str.strip()

    for col in required:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    if "TRACK_ID" in data.columns:

        data["TRACK_ID"] = pd.to_numeric(
            data["TRACK_ID"], errors="coerce"
        )

        data = data[
            data["TRACK_ID"] == TRACK_ID
        ]

    data = data.dropna(subset=required)
    if START_FRAME is not None:
        data = data[
            data["FRAME"] >= START_FRAME
        ]
    if END_FRAME is not None:
        data = data[
            data["FRAME"] <= END_FRAME
        ]
    data = data.sort_values("FRAME")
    if data["FRAME"].duplicated().any(): # duplicates removed
        raise ValueError(
            "Multiple detections in one frame. "
            "Check the selected TRACK_ID."
        )

    if len(data) < 10:
        raise ValueError(
            "Not enough valid tracked positions."
        )
    frames = data["FRAME"].to_numpy().astype(int)
    t = (frames - frames[0]) / FPS
    x = data["POSITION_X"].to_numpy() * CONVERSION # pixel > metres 
    y = data["POSITION_Y"].to_numpy() * CONVERSION
    return frames, t, x, y

def plot_positions(t, x, y):

    x_nm = (x - np.mean(x)) * 1e9
    y_nm = (y - np.mean(y)) * 1e9

    plt.figure(figsize=(9, 4))
    plt.plot(t, x_nm, linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("X displacement (nm)")
    plt.title("X Position vs Time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(9, 4))
    plt.plot(t, y_nm, linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Y displacement (nm)")
    plt.title("Y Position vs Time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_trajectory(x, y):
    x_nm = (x - np.mean(x)) * 1e9
    y_nm = (y - np.mean(y)) * 1e9

    plt.figure(figsize=(6, 6))
    plt.plot(
        x_nm, y_nm,
        linewidth=0.6,
        label="Bead trajectory"
    )
    plt.scatter(
        0, 0,
        color="red",
        marker="x",
        s=70,
        label="Mean position"
    )
    plt.xlabel("X displacement (nm)")
    plt.ylabel("Y displacement (nm)")
    plt.title("Trapped Bead Trajectory")
    plt.axis("equal")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def probability_distributions(x, y):
    for position, direction in [(x, "X"), (y, "Y")]:
        displacement_nm = (
            position - np.mean(position)
        ) * 1e9
        mu, sigma = norm.fit(displacement_nm)
        fitted_x = np.linspace(
            displacement_nm.min(),
            displacement_nm.max(),
            400
        )

        plt.figure(figsize=(7, 5))

        plt.hist(
            displacement_nm,
            bins=30,
            density=True,
            alpha=0.6,
            label="Experimental data"
        )

        plt.plot(
            fitted_x,
            norm.pdf(fitted_x, mu, sigma),
            "r-",
            linewidth=2,
            label="Gaussian fit"
        )

        plt.xlabel(
            f"{direction} displacement (nm)"
        )
        plt.ylabel("Probability density (nm$^{-1}$)")
        plt.title(
            f"{direction} Probability Distribution"
        )
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(
            f"{direction} standard deviation:",
            f"{sigma:.4f} nm"
        )


def calculate_msd(frames, x, y, FPS):
    max_lag = max(
        2,
        int((frames[-1] - frames[0]) *
            MAX_LAG_FRACTION)
    )
    frame_lookup = {
        frame: i
        for i, frame in enumerate(frames)
    }
    lag_times = []
    msd_x_values = []
    msd_y_values = []
    msd_2d_values = []
    pair_counts = []
    for lag in range(1, max_lag + 1):
        dx_squared = []
        dy_squared = []
        for i, frame in enumerate(frames):
            next_index = frame_lookup.get(
                frame + lag
            )
            if next_index is not None:
                dx = x[next_index] - x[i]
                dy = y[next_index] - y[i]
                dx_squared.append(dx**2)
                dy_squared.append(dy**2)
        if len(dx_squared) < 10:
            continue

        msd_x = np.mean(dx_squared)
        msd_y = np.mean(dy_squared)
        lag_times.append(lag / FPS)
        msd_x_values.append(msd_x)
        msd_y_values.append(msd_y)
        msd_2d_values.append(msd_x + msd_y)
        pair_counts.append(len(dx_squared))

    return (
        np.array(lag_times),
        np.array(msd_x_values),
        np.array(msd_y_values),
        np.array(msd_2d_values),
        np.array(pair_counts)
    )

def trapped_particle(tau, A, tau_c):
    return A * (
        1 - np.exp(-tau / tau_c)
    )

def fit_msd_yeast(tau, msd_x, msd_y):
    
    results = {}
    for direction, msd in [
        ("X", msd_x),
        ("Y", msd_y)
    ]:

        msd_nm2 = msd * 1e18

        A_guess = np.max(msd_nm2)
        tau_guess = max(tau[len(tau)//4], 1e-9)

        params, covariance = curve_fit(
            trapped_particle,
            tau,
            msd_nm2,
            p0=[A_guess, tau_guess],
            bounds=(0, np.inf),
            maxfev=20000
        )

        A_nm2, tau_c = params
        errors = np.sqrt(np.diag(covariance))
        A_error = errors[0]
        tau_error = errors[1]

        A_m2 = A_nm2 * 1e-18
        k = 2 * k_B * T / A_m2

        k_error = k * A_error / A_nm2

        results[direction] = {
            "A": A_nm2,
            "tau_c": tau_c,
            "k": k,
            "k_error": k_error
        }

        plt.figure(figsize=(8, 5))
        plt.plot(
            tau,
            msd_nm2,
            "o",
            markersize=4,
            label=f"Experimental {direction} MSD"
        )

        plt.xlabel("Time lag (ms)")
        plt.ylabel("MSD (nm$^2$)")
        plt.title(f"Yeast cell: {direction} MSD")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"\n{direction} direction:")
        print(f"MSD plateau: {A_nm2:.3f} nm²")
        print(f"Relaxation time: {tau_c:.6f} s")
        print(f"Relaxation time error: {tau_error:.6f} s")
        print(f"Trap stiffness: {k * 1e6:.6f} pN/µm")
        print(f"Stiffness error: {k_error * 1e6:.6f} pN/µm")
    return results

# final vals + graphs
def final_vals(data, FPS):
    frames, t, x, y = load_data(data, FPS)
    plot_positions(t, x, y)
    plot_trajectory(x, y)
    probability_distributions(x, y)
    tau, msd_x, msd_y, msd_2d, pair_counts = calculate_msd(
        frames, x, y, FPS)
    results = fit_msd_yeast(
        tau, msd_x, msd_y
    )
#final_vals("yeast 65ma_spots.csv", 93.458) # yikes
#final_vals("yeast 70 mA_spots.csv", 93.458)
#final_vals("yeast 75 mA Trap_spots.csv", 93.458) # YIPPEEEEE
#final_vals("yeast 80 mA Trap_spots.csv", 93.458) # YIPPEE
#final_vals("yeast 85 mA Trap_spots.csv", 93.458)
final_vals("yeast 90 mA_spots.csv", 93.458)



