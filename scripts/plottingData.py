import csv
import numpy as np
import matplotlib.pyplot as plt
from math import isnan
# Read the data
t, angle, phase, rep, fps = [], [], [], [], []

with open("session_log1.csv", newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        t.append(float(row["t"]))
        val = row["knee_angle"] or "nan"
        angle.append(float(val) if val != "nan" else float('nan'))
        phase.append(row["phase"])
        rep.append(int(row["rep"]))
        fps.append(float(row["fps"]))

# Filter valid angle values for axis limits
valid_angles = [a for a in angle if not np.isnan(a)]
if not valid_angles:
    print("No valid knee_angle data found. Check if your session used a lower-body exercise.")
    exit()

ymin, ymax = min(valid_angles), max(valid_angles)

# Find transition times where rep increases
rep_times = [t[i] for i in range(1, len(rep)) if rep[i] > rep[i-1]]

# Boolean mask for "down" phase (or use "flex", adjust per your phase labels)
down_phase = [p == "down" for p in phase]

# Plot
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(t, angle, label="Knee Angle", color="b")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Knee Angle (deg)", color="b")
ax1.set_ylim(ymin-10, ymax+10)
ax1.set_title("Knee Angle vs Time with Phase and Rep Events")

# Shade "down" phase regions
ax1.fill_between(t, ymin-10, ymax+10, where=down_phase, alpha=0.14, color="coral", label="Down Phase")

# Mark points where rep increments
for rt in rep_times:
    ax1.axvline(rt, linestyle="--", color="green", alpha=0.66)

ax1.legend(loc="upper left")

# FPS on secondary y-axis
ax2 = ax1.twinx()
ax2.plot(t, fps, color="gray", alpha=0.32, label="FPS", linewidth=1)
ax2.set_ylabel("FPS", color="gray")
ax2.set_ylim(0, max(fps) + 2)

fig.tight_layout()
plt.savefig("performance_knee_angle.png", dpi=220)
plt.show()
