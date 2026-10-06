import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read data
data = pd.read_csv(
    "test 7 data.txt",
    sep="\t",
    header=None,
    names=["t", "x", "y"],
    engine="python"
)

t = data["t"].to_numpy()
x = data["x"].to_numpy()
y = data["y"].to_numpy()

msd = []
dt = []
tamsd_error = []

# MSD for lags
for lag in range(1, len(t)):
    dx = x[lag:] - x[:-lag]
    dy = y[lag:] - y[:-lag]

    squared_displacement = dx**2 + dy**2
    msd.append(np.mean(squared_displacement))
    dt.append(np.mean(t[lag:] - t[:-lag]))

    error = np.std(squared_displacement, ddof=1)
    tamsd_error.append(error)

msd = np.array(msd)
dt = np.array(dt)
tamsd_error = np.array(tamsd_error)

# MAY REMOVE SOME VALUES FROM TRAJECTORY
frames_per_average = 5
n_groups = len(msd) // frames_per_average

# trims groups for binning
msd_trimmed = msd[:n_groups * frames_per_average]
dt_trimmed = dt[:n_groups * frames_per_average]
error_trimmed = tamsd_error[:n_groups * frames_per_average]

msd_groups = msd_trimmed.reshape(n_groups, frames_per_average)
dt_groups = dt_trimmed.reshape(n_groups, frames_per_average)
error_groups = error_trimmed.reshape(n_groups, frames_per_average)

# average each group
msd_avg = np.mean(msd_groups, axis=1)
dt_avg = np.mean(dt_groups, axis=1)

# combined errors for each group
error_avg = np.sqrt(
    np.mean(error_groups**2, axis=1)
)

slope, intercept = np.polyfit(dt_avg, msd_avg, 1)

# Generate fitted line
fit_msd = slope * dt_avg + intercept

print(f"Slope = {slope}")
print(f"Intercept = {intercept}")


plt.figure(figsize=(8, 6))

plt.errorbar(
    dt_avg,
    msd_avg,
    yerr=error_avg,
    fmt='o',
    capsize=3,
    label="Averaged MSD"
)

plt.plot(
    dt_avg,
    fit_msd,
    '-',
    label=f"Linear fit: MSD = {slope:.3g}t + {intercept:.3g}"
)

plt.xlabel("Time lag (s)")
plt.ylabel("MSD (units)")
plt.title("Mean Squared Displacement")
plt.grid()
plt.legend()

plt.show()