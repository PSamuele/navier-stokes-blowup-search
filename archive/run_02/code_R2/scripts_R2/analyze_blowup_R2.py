import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create results folder if not exists
os.makedirs('results', exist_ok=True)

# Load data
df = pd.read_csv('blowup_data_RUN_AWS.csv')

# Remove NaNs just in case
df = df.dropna()

# We only care about positive time and realistic values before complete numerical crash
# Sometimes the last row might have NaNs or inf if it completely crashed, dropna handles some.

print(f"Loaded {len(df)} rows. Max time: {df['t'].max()}")

# 1. Vorticity over time (semi-log scale)
plt.figure(figsize=(10,6))
plt.semilogy(df['t'], df['max_vorticity'], label='Max Vorticity', color='blue')
plt.xlabel('Time (t)')
plt.ylabel('Max Vorticity (log scale)')
plt.title('Maximum Vorticity vs Time')
plt.grid(True)
plt.savefig('results/vorticity_time.png', dpi=150)
plt.close()

# 2. Beale-Kato-Majda (BKM) analysis
df['inv_vorticity'] = 1.0 / df['max_vorticity']

# Let's zoom in on the late stage for the BKM plot to see the linear collapse clearly
plt.figure(figsize=(10,6))
plt.plot(df['t'], df['inv_vorticity'], label='1 / Max Vorticity', color='red')
plt.xlabel('Time (t)')
plt.ylabel('1 / ||ω||_inf')
plt.title('BKM Analysis: Inverse Max Vorticity vs Time')
plt.grid(True)
plt.savefig('results/bkm_analysis.png', dpi=150)
plt.close()

# 3. Velocity over time
plt.figure(figsize=(10,6))
plt.semilogy(df['t'], df['max_velocity'], label='Max Velocity', color='green')
plt.xlabel('Time (t)')
plt.ylabel('Max Velocity (log scale)')
plt.title('Maximum Velocity vs Time')
plt.grid(True)
plt.savefig('results/velocity_time.png', dpi=150)
plt.close()

print("Plots saved in results/")
