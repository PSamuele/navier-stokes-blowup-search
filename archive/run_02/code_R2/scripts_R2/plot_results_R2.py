import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

# Leggi i dati dal CSV
df = pd.read_csv("results/blowup_data.csv")
t = df['t'].values
vort = df['max_vorticity'].values

# Vogliamo fittare il tratto di esplosione, diciamo da t=0.30 in poi
mask = t >= 0.30
t_fit = t[mask]
vort_fit = vort[mask]

# Se il blow-up è ~ 1/(T* - t), allora 1/vorticità è lineare: 1/vort = a*t + b
# T* sarà l'intercetta sull'asse x, ovvero quando 1/vort = 0 -> t = -b/a
inv_vort = 1.0 / vort_fit

slope, intercept, r_value, p_value, std_err = linregress(t_fit, inv_vort)
T_star = -intercept / slope

print("=======================================")
print(f"BLOW-UP SCALING ANALYSIS (t >= 0.30)")
print("=======================================")
print(f"Correlation Coefficient (R^2): {r_value**2:.5f}")
print(f"Estimated Singularity Time (T*): {T_star:.5f} seconds")

# Creazione del grafico
plt.figure(figsize=(10, 5))

# Plot 1: Vorticity over time
plt.subplot(1, 2, 1)
plt.plot(t, vort, 'r-', linewidth=2)
plt.axvline(T_star, color='k', linestyle='--', label=f'T* ~ {T_star:.3f}')
plt.title("Maximum Vorticity over Time")
plt.xlabel("Time (t)")
plt.ylabel("Max Vorticity")
plt.yscale('log')
plt.legend()
plt.grid(True)

# Plot 2: 1/Vorticity (should be a straight line towards zero)
plt.subplot(1, 2, 2)
plt.plot(t_fit, inv_vort, 'b-', linewidth=2, label='1 / MaxVort')
# Fit line
t_ext = np.linspace(0.3, T_star + 0.05, 50)
plt.plot(t_ext, slope*t_ext + intercept, 'k--', label='Linear Fit')
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(T_star, color='r', linestyle='--', label=f'T* ~ {T_star:.3f}')
plt.title("Inverse Vorticity (1/\u03c9)")
plt.xlabel("Time (t)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("docs/images/vorticity_growth.png", dpi=300)
print("\nPlot saved in: docs/images/vorticity_growth.png")
