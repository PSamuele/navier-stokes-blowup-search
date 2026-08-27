import pandas as pd
import matplotlib.pyplot as plt

try:
    df = pd.read_csv('blowup_data.csv')
    print(df.columns.tolist())
    print(df.head(2))
except Exception as e:
    print(f"Error: {e}")
df = pd.read_csv('blowup_data.csv')

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

ax1.plot(df['t'], df['max_velocity'], color='black', linewidth=1.2)
ax1.set_title('Max Velocity vs Time', fontsize=12, pad=10)
ax1.set_xlabel('Time (t)', fontsize=10)
ax1.set_ylabel('Max Velocity', fontsize=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.plot(df['t'], df['max_vorticity'], color='black', linewidth=1.2)
ax2.set_title('Max Vorticity vs Time', fontsize=12, pad=10)
ax2.set_xlabel('Time (t)', fontsize=10)
ax2.set_ylabel('Max Vorticity', fontsize=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('Vel_Vor_Plot.png', dpi=300, bbox_inches='tight')
plt.show()