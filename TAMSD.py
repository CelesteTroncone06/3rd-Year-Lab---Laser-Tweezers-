import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(
    "Test 1 data.txt",
    sep="\t",  
    header=None,
    names=["t", "x", "y"],
    engine="python"
)

print(data.head())

t = data["t"].to_numpy()
x = data["x"].to_numpy()
y = data["y"].to_numpy()

msd = [] # msd array
dt = [] # time diff array
tamsd_error = []


for lag in range(1, len(t)):
    dx = x[lag:] - x[:-lag]
    dy = y[lag:] - y[:-lag]

    squared_displacement = dx**2 + dy**2

    msd.append(np.mean(squared_displacement))
    dt.append(np.mean(t[lag:] - t[:-lag]))

    error = np.std(squared_displacement, ddof=3)
    tamsd_error.append(error)

# Plot MSD
plt.figure()
plt.errorbar(dt, msd, yerr=tamsd_error, fmt='o', capsize=3)
plt.xlabel("Time lag (s)")
plt.ylabel("MSD")
plt.title("Mean Squared Displacement")
plt.grid()
plt.show()