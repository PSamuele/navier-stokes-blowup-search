import numpy as np
import json
try:
    import ufl
    from dolfinx import mesh, fem
    from dolfinx.io import gmsh as gmshio
    from mpi4py import MPI
except:
    pass

comm = MPI.COMM_WORLD
mesh_data = gmshio.read_from_msh("apple_domain.msh", comm, 0, gdim=2)
domain = mesh_data.mesh

V = fem.functionspace(domain, ("Lagrange", 2, (3,)))
u = fem.Function(V)
u.x.array[:] = np.load("results/checkpoint_un.npy")

x = ufl.SpatialCoordinate(domain)
r_safe = x[0] + 1e-14

vel_sq = u[0]**2 + u[1]**2 + u[2]**2
omega_r = - u[2].dx(1)
omega_z = u[2].dx(0) + u[2] / r_safe
omega_t = u[0].dx(1) - u[1].dx(0)
vort_sq = omega_r**2 + omega_z**2 + omega_t**2

W = fem.functionspace(domain, ("DG", 1))
vel_expr = fem.Expression(vel_sq, W.element.interpolation_points())
vort_expr = fem.Expression(vort_sq, W.element.interpolation_points())

vel_func = fem.Function(W)
vort_func = fem.Function(W)

vel_func.interpolate(vel_expr)
vort_func.interpolate(vort_expr)

max_vel = np.sqrt(np.max(vel_func.x.array))
max_vort = np.sqrt(np.max(vort_func.x.array))

with open("results/checkpoint_meta.json", "r") as f:
    meta = json.load(f)

print(f"--- FINAL STATE ANALYSIS (Step {meta['step']}, t={meta['t']:.4f}) ---")
print(f"Final Max Velocity:  {max_vel:.2f}")
print(f"Final Max Vorticity: {max_vort:.2f}")

# Calcoliamo anche i valori iniziali per confronto
import sympy as sp
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

u_init = fem.Function(V)
u_init.interpolate(initial_velocity)
vel_func.interpolate(fem.Expression(u_init[0]**2 + u_init[1]**2 + u_init[2]**2, W.element.interpolation_points()))
omega_r_i = - u_init[2].dx(1)
omega_z_i = u_init[2].dx(0) + u_init[2] / r_safe
omega_t_i = u_init[0].dx(1) - u_init[1].dx(0)
vort_func.interpolate(fem.Expression(omega_r_i**2 + omega_z_i**2 + omega_t_i**2, W.element.interpolation_points()))

print(f"Initial Max Velocity (t=0):  {np.sqrt(np.max(vel_func.x.array)):.2f}")
print(f"Initial Max Vorticity (t=0): {np.sqrt(np.max(vort_func.x.array)):.2f}")
