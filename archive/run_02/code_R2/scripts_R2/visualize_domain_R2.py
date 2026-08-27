import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(z, R0=1.0, H=2.0, k=0.5):
    return R0 * np.cos(np.pi * z / (2 * H)) * np.exp(-k * z**2)

# Generate z values
z = np.linspace(-2.0, 2.0, 100)
# Generate theta values for revolution
theta = np.linspace(0, 2 * np.pi, 100)

Z, Theta = np.meshgrid(z, theta)
R = f(Z)

X = R * np.cos(Theta)
Y = R * np.sin(Theta)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')

ax.set_title("La 'Mela' Cuspidale - Dominio dell'Esplosione")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Asse di rotazione e spinta)')

# Imposta i limiti per mantenere le proporzioni
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-2, 2)

plt.savefig('apple_domain.png')
print("Image saved as apple_domain.png")
