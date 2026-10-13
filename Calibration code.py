import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy.signal import find_peaks

# metres between neighbouring lines on the slide (check the label!)
SPACING = 1e-6

img = imread("cal 3.jpg")
if img.ndim == 3:                 # colour -> grey
    img = img.mean(axis=2)

# Average a strip of rows to reduce noise. Assumes the lines are vertical.
# If they are horizontal, use img.T instead.
r0, r1 = 100, 200                 # CHANGE: choose rows that cross clean lines
profile = img[r0:r1, :].mean(axis=0)
profile = profile - profile.mean()

# Find the lines (use -profile if the lines are dark on a bright background)
peaks, _ = find_peaks(-profile, distance=5, prominence=0.2 * np.ptp(profile))

# Fit line position vs line number
n = np.arange(len(peaks))
(px_per_div, offset), cov = np.polyfit(n, peaks, 1, cov=True)
px_err = np.sqrt(cov[0, 0])

m_per_px = SPACING / px_per_div
m_per_px_err = m_per_px * px_err / px_per_div

print(f"Lines found: {len(peaks)}")
print(f"Pixels per division = {px_per_div:.3f} +/- {px_err:.3f}")
print(f"Metres per pixel    = {m_per_px:.4e} +/- {m_per_px_err:.1e}")
print(f"Length uncertainty dL/L = {px_err/px_per_div:.2%}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(profile)
ax[0].plot(peaks, profile[peaks], 'rx')
ax[0].set_xlabel("Pixel")
ax[0].set_ylabel("Intensity")
ax[0].set_title("Profile across lines")
ax[1].plot(n, peaks, 'o')
ax[1].plot(n, px_per_div * n + offset, 'k--')
ax[1].set_xlabel("Line number")
ax[1].set_ylabel("Position (pixels)")
ax[1].set_title("Line position fit")
plt.tight_layout()
plt.show()
