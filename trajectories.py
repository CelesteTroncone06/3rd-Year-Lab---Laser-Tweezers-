# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 23:56:59 2026

@author: Celes
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv(
    "test 7 data.txt",
    sep="\t",
    header=None,
    names=["t", "x", "y"],
    engine="python"
)

CONVERSION = 9.043E-8
t = data["t"].to_numpy()
x = data["x"].to_numpy()
y = data["y"].to_numpy()

# Plot trajectory
plt.figure(figsize=(8, 8))

plt.plot(x, y, '-', linewidth=1, label="Particle trajectory")

plt.scatter(x[0], y[0], color='green', s=80, label="Start", zorder=3)
plt.scatter(x[-1], y[-1], color='red', s=80, label="End", zorder=3)

plt.xlabel("x position (µm)")
plt.ylabel("y position (µm)")
plt.title("Brownian Motion Particle Trajectory")

plt.axis('equal')
plt.legend()
plt.grid()
plt.show()