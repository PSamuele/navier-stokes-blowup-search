import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import gmsh as gmshio

def plot_initial_field():
    comm = MPI.COMM_WORLD
    print("Reading mesh...")
    mesh_data = gmshio.read_from_msh("assets/apple_domain.msh", comm, 0, gdim=2)
    domain = mesh_data.mesh

    # Spazio delle funzioni
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    u_init = fem.Function(V)

    print("Calculating ICs...")
    r_sym, z_sym = sp.symbols('r z')
    R0, H, k = 1.0, 2.0, 0.5
    J, A, S, Rv, sigma = 10.0, 5.0, 20.0, 0.5, 0.2

    f_z = R0 * sp.cos(sp.pi * z_sym / (2 * H)) * sp.exp(-k * z_sym**2)
    psi_jet = J * r_sym**2 * (f_z**2 - r_sym**2)**2
    psi_ring = A * r_sym**2 * sp.exp(-((r_sym - Rv)**2 + z_sym**2) / sigma**2)
    psi_tot = psi_jet + psi_ring

    Gamma_0 = S * r_sym**2 * sp.exp(-((r_sym - Rv)**2 + z_sym**2) / sigma**2)

    ur_sym = - (1/sp.Max(r_sym, 1e-12)) * sp.diff(psi_tot, z_sym)
    uz_sym = (1/sp.Max(r_sym, 1e-12)) * sp.diff(psi_tot, r_sym)
    ut_sym = Gamma_0 / sp.Max(r_sym, 1e-12)

    ur_num = sp.lambdify((r_sym, z_sym), ur_sym, modules=['numpy'])
    uz_num = sp.lambdify((r_sym, z_sym), uz_sym, modules=['numpy'])
    ut_num = sp.lambdify((r_sym, z_sym), ut_sym, modules=['numpy'])

    def initial_velocity(x):
        r_arr, z_arr = x[0], x[1]
        r_arr = np.maximum(r_arr, 1e-12)
        return (ur_num(r_arr, z_arr), uz_num(r_arr, z_arr), ut_num(r_arr, z_arr))

    u_init.interpolate(initial_velocity)

    # Extract data for Matplotlib
    import dolfinx.plot
    topology, cell_types, x_dolfinx = dolfinx.plot.vtk_mesh(V)
    # The topology array is returned as a 1D flat array, reshape it for matplotlib
    triangles = topology.reshape(-1, 3)
    triangulation = mtri.Triangulation(x_dolfinx[:, 0], x_dolfinx[:, 1], triangles)

    # Calculate magnitude
    u_val = u_init.x.array.reshape(-1, 3)
    magnitude = np.sqrt(u_val[:, 0]**2 + u_val[:, 1]**2 + u_val[:, 2]**2)

    fig, ax = plt.subplots(figsize=(8, 10))
    # Colormap
    cmap = plt.get_cmap("turbo")
    
    # Plot contour
    tpc = ax.tripcolor(triangulation, magnitude, cmap=cmap, shading='gouraud')
    
    # Formatting
    ax.set_aspect('equal')
    ax.set_title("Vortex Ring Initial Velocity Magnitude", fontsize=16)
    ax.set_xlabel("Radius (r)", fontsize=12)
    ax.set_ylabel("Height (z)", fontsize=12)
    
    # Add elegant colorbar
    cbar = plt.colorbar(tpc, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label("Velocity Magnitude", fontsize=12)

    plt.tight_layout()
    plt.savefig("docs/images/initial_vortex.png", dpi=300, bbox_inches='tight')
    print("Professional plot saved in docs/images/initial_vortex.png")

if __name__ == "__main__":
    plot_initial_field()
