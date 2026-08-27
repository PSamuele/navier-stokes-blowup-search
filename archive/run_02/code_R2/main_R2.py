import numpy as np
import sys
import sympy as sp
import time
import json
import os
from mpi4py import MPI
from petsc4py import PETSc

try:
    import ufl
    import basix.ufl
    from dolfinx import mesh, fem, io
    from dolfinx.io import gmsh as gmshio
    from dolfinx.io import XDMFFile
    from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_vector
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def run_solver(mesh_file="assets/apple_domain.msh", T=0.55, num_steps=1100):
    comm = MPI.COMM_WORLD
    print(f"--- Starting Optimized Navier-Stokes Solver (Dolfinx 0.10.0) ---")
    
    # 1. Lettura della Mesh
    mesh_data = gmshio.read_from_msh(mesh_file, comm, 0, gdim=2)
    domain = mesh_data.mesh
    
    # 2. Spazi Funzionali
    V = fem.functionspace(domain, ("Lagrange", 2, (3,)))
    Q = fem.functionspace(domain, ("Lagrange", 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    p = ufl.TrialFunction(Q)
    q = ufl.TestFunction(Q)

    u_n = fem.Function(V)
    u_s = fem.Function(V)
    p_n = fem.Function(Q)
    
    u_n.name = "Velocity"
    p_n.name = "Pressure"

    # --- IMPOSTAZIONI CHECKPOINT / RESTART ---
    out_dir = "results"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    meta_file = f"{out_dir}/checkpoint_meta.json"
    un_file = f"{out_dir}/checkpoint_un.npy"
    pn_file = f"{out_dir}/checkpoint_pn.npy"
    
    # Set to True to resume from where you left off
    restart = False 
    start_step = 0
    t = 0.0

    if restart and os.path.exists(meta_file):
        print("Loading checkpoint...")
        with open(meta_file, "r") as f:
            meta = json.load(f)
        t = meta["t"]
        start_step = meta["step"]
        u_n.x.array[:] = np.load(f"{out_dir}/checkpoint_un_rank{comm.rank}.npy")
        p_n.x.array[:] = np.load(f"{out_dir}/checkpoint_pn_rank{comm.rank}.npy")
        print(f"Restart complete. Resuming from step {start_step} (t={t:.4f})")
    else:
        # --- CONDIZIONI INIZIALI (SYMPY) ---
        print("Performing symbolic IC calculation...")
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

        u_n.interpolate(initial_velocity)
        print("Initial conditions calculated.")

    # --- FASE 4: CONDIZIONI AL CONTORNO ---
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    def is_axis(x): return np.isclose(x[0], 0.0)
    axis_facets = mesh.locate_entities_boundary(domain, 1, is_axis)
    wall_facets = np.setdiff1d(boundary_facets, axis_facets)
    
    # Parete Mela: No-slip
    wall_dofs_V = fem.locate_dofs_topological(V, 1, wall_facets)
    u_bc_wall = fem.Function(V)
    bc_wall = fem.dirichletbc(u_bc_wall, wall_dofs_V)

    # Asse: u_r = 0, u_theta = 0
    V_r, _ = V.sub(0).collapse()
    V_t, _ = V.sub(2).collapse()
    axis_dofs_r = fem.locate_dofs_topological((V.sub(0), V_r), 1, axis_facets)
    axis_dofs_t = fem.locate_dofs_topological((V.sub(2), V_t), 1, axis_facets)
    zero_r = fem.Function(V_r)
    zero_t = fem.Function(V_t)
    bc_axis_r = fem.dirichletbc(zero_r, axis_dofs_r, V.sub(0))
    bc_axis_t = fem.dirichletbc(zero_t, axis_dofs_t, V.sub(2))
    bcs_u = [bc_wall, bc_axis_r, bc_axis_t]
    
    # Pressione (Pin nell'origine)
    def origin_point(x): return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))
    origin_dofs = fem.locate_dofs_geometrical(Q, origin_point)
    zero_p = fem.Function(Q)
    bc_p = fem.dirichletbc(zero_p, origin_dofs)
    bcs_p = [bc_p]

    # --- FORMA DEBOLE IPCS ---
    dt_val = T / num_steps
    dt = fem.Constant(domain, PETSc.ScalarType(dt_val))
    nu = fem.Constant(domain, PETSc.ScalarType(1e-3))
    
    x = ufl.SpatialCoordinate(domain)
    r, z = x[0], x[1]
    r_safe = r + 1e-14
    dx = ufl.Measure("dx", domain=domain)
    
    ur, uz, ut = u[0], u[1], u[2]
    vr, vz, vt = v[0], v[1], v[2]
    un_r, un_z, un_t = u_n[0], u_n[1], u_n[2]
    
    F1 = (ufl.dot(u - u_n, v) / dt) * r * dx
    adv_r = un_r * ur.dx(0) + un_z * ur.dx(1)
    adv_z = un_r * uz.dx(0) + un_z * uz.dx(1)
    adv_t = un_r * ut.dx(0) + un_z * ut.dx(1)
    F1 += (adv_r * vr + adv_z * vz + adv_t * vt) * r * dx
    F1 += (- (un_t * ut) / r_safe * vr + (un_r * ut) / r_safe * vt) * r * dx
    F1 += nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * r * dx
    F1 += nu * (ur * vr + ut * vt) / r_safe * dx
    a1, L1 = ufl.lhs(F1), ufl.rhs(F1)
    
    div_u_s = u_s[0].dx(0) + u_s[0] / r_safe + u_s[1].dx(1)
    a2 = ufl.inner(ufl.grad(p), ufl.grad(q)) * r * dx
    L2 = - (1.0 / dt) * div_u_s * q * r * dx
    
    a3 = ufl.dot(u, v) * r * dx
    L3 = ufl.dot(u_s, v) * r * dx - dt * (p_n.dx(0) * vr + p_n.dx(1) * vz) * r * dx

    bilinear_form1 = fem.form(a1)
    linear_form1 = fem.form(L1)
    bilinear_form2 = fem.form(a2)
    linear_form2 = fem.form(L2)
    bilinear_form3 = fem.form(a3)
    linear_form3 = fem.form(L3)

    print("Assembling matrices...")
    A1 = assemble_matrix(bilinear_form1, bcs=bcs_u)
    A1.assemble()
    A2 = assemble_matrix(bilinear_form2, bcs=bcs_p)
    A2.assemble()
    A3 = assemble_matrix(bilinear_form3)
    A3.assemble()

    # --- OTTIMIZZAZIONE SOLUTORI PETSC ---
    # Step 1: Matrice non-simmetrica (Advezione + Diffusione) -> BCGS + BJACOBI (per parallelo)
    solver1 = PETSc.KSP().create(comm)
    solver1.setOperators(A1)
    solver1.setType(PETSc.KSP.Type.BCGS)
    solver1.getPC().setType(PETSc.PC.Type.BJACOBI) # BJACOBI usa ILU localmente ed è compatibile con MPI
    solver1.setTolerances(rtol=1e-6, atol=1e-10)

    # Step 2: Equazione di Poisson (Simmetrica Def. Positiva) -> CG + GAMG/HYPRE
    solver2 = PETSc.KSP().create(comm)
    solver2.setOperators(A2)
    solver2.setType(PETSc.KSP.Type.CG)
    try:
        solver2.getPC().setType(PETSc.PC.Type.HYPRE)
    except:
        solver2.getPC().setType(PETSc.PC.Type.GAMG) # GAMG è il multigrid nativo di PETSc
    solver2.setTolerances(rtol=1e-6, atol=1e-10)
    
    # Step 3: Matrice di Massa (Diagonale dominante) -> CG + SOR/JACOBI
    solver3 = PETSc.KSP().create(comm)
    solver3.setOperators(A3)
    solver3.setType(PETSc.KSP.Type.CG)
    solver3.getPC().setType(PETSc.PC.Type.SOR)
    solver3.setTolerances(rtol=1e-6, atol=1e-10)

    b1 = create_vector(V)
    b2 = create_vector(Q)
    b3 = create_vector(V)

    # XDMF Writer per ParaView (richiede interpolazione a grado 1)
    V_out = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    u_out = fem.Function(V_out)
    u_out.name = "Velocity"
    
    xdmf_file = XDMFFile(comm, f"{out_dir}/velocity.xdmf", "w")
    xdmf_file.write_mesh(domain)
    if not restart:
        u_out.interpolate(u_n)
        xdmf_file.write_function(u_out, t)

    # --- RADAR DEL BLOW-UP (Vorticità e Velocità) ---
    print("Initializing Blow-Up radar (Vorticity Tracking)...")
    # Componenti della Vorticità in coordinate cilindriche
    omega_r = - u_n[2].dx(1)
    omega_z = u_n[2].dx(0) + u_n[2] / r_safe
    omega_t = u_n[0].dx(1) - u_n[1].dx(0)
    
    # Magnitudo quadrate per evitare radici costose in UFL
    vorticity_sq = omega_r**2 + omega_z**2 + omega_t**2
    vel_sq = u_n[0]**2 + u_n[1]**2 + u_n[2]**2
    
    # Spazio Discontinuo Galerkin (DG) per valutare le derivate senza risolvere matrici
    W_DG = fem.functionspace(domain, ("DG", 1))
    vort_expr = fem.Expression(vorticity_sq, W_DG.element.interpolation_points)
    vel_expr = fem.Expression(vel_sq, W_DG.element.interpolation_points)
    
    vort_func = fem.Function(W_DG)
    vel_func = fem.Function(W_DG)

    # File CSV per i dati
    csv_file = f"{out_dir}/blowup_data.csv"
    if not restart:
        with open(csv_file, "w") as f:
            f.write("t,max_velocity,max_vorticity\n")

    # --- TIME LOOP (ADAPTIVE) ---
    print(f"Starting Adaptive Time-Stepping (initial dt={dt.value:.5f}, T={T})")
    
    i = start_step
    CFL_target = 0.5
    dx_min = 0.0001
    
    while t < T:
        # 1. Calcolo max vel globale (MPI-safe) per CFL
        vel_func.interpolate(vel_expr)
        local_max_vel = np.sqrt(np.max(vel_func.x.array))
        global_max_vel = comm.allreduce(local_max_vel, op=MPI.MAX)
        
        if global_max_vel > 1e-6:
            dt_new = CFL_target * dx_min / global_max_vel
            # Limiti per evitare salti o stalli
            dt_new = max(min(dt_new, 0.005), 1e-6)
        else:
            dt_new = 0.001
            
        if t + dt_new > T:
            dt_new = T - t
            
        # 2. Aggiornamento dt e ri-assemblaggio matrice A1
        if not np.isclose(dt.value, dt_new):
            dt.value = dt_new
            A1.zeroEntries()
            assemble_matrix(A1, bilinear_form1, bcs=bcs_u)
            A1.assemble()
            # Non serve rifare solver1.setOperators(A1) se l'oggetto è lo stesso
            
        t += dt.value
        step_start_time = time.time()
        
        # Step 1 (Predictor)
        with b1.localForm() as loc: loc.set(0)
        assemble_vector(b1, linear_form1)
        apply_lifting(b1, [bilinear_form1], [bcs_u])
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b1, bcs_u)
        solver1.solve(b1, u_s.x.petsc_vec)
        u_s.x.scatter_forward()
        iters1 = solver1.getIterationNumber()
        
        # Step 2 (Pressure)
        with b2.localForm() as loc: loc.set(0)
        assemble_vector(b2, linear_form2)
        apply_lifting(b2, [bilinear_form2], [bcs_p])
        b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b2, bcs_p)
        solver2.solve(b2, p_n.x.petsc_vec)
        p_n.x.scatter_forward()
        iters2 = solver2.getIterationNumber()
        
        # Step 3 (Corrector)
        with b3.localForm() as loc: loc.set(0)
        assemble_vector(b3, linear_form3)
        b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solver3.solve(b3, u_n.x.petsc_vec)
        u_n.x.scatter_forward()
        iters3 = solver3.getIterationNumber()
        
        elapsed = time.time() - step_start_time
        
        # Stampa diagnostica ogni 10 step
        if (i+1) % 10 == 0 or i == 0:
            # Calcolo Vorticità globale (MPI-safe)
            vort_func.interpolate(vort_expr)
            local_max_vort = np.sqrt(np.max(vort_func.x.array))
            global_max_vort = comm.allreduce(local_max_vort, op=MPI.MAX)
            
            print(f"Step {i+1:04d} | t={t:.5f} | dt={dt.value:.5f} | MaxVort: {global_max_vort:.2f} | MaxVel: {global_max_vel:.2f} | Iters(S1:{iters1}, S2:{iters2}, S3:{iters3}) | T/Step: {elapsed:.3f}s")
            u_out.interpolate(u_n)
            xdmf_file.write_function(u_out, t)
            
            # Write to CSV (Solo il rank 0 per non duplicare)
            if comm.rank == 0:
                with open(csv_file, "a") as f:
                    f.write(f"{t},{global_max_vel},{global_max_vort}\n")
                with open(meta_file, "w") as f:
                    json.dump({"step": i+1, "t": t}, f)
            
            # Ultrafast checkpoint (MPI-safe: ogni rank salva il suo pezzo)
            np.save(f"{out_dir}/checkpoint_un_rank{comm.rank}.npy", u_n.x.array)
            np.save(f"{out_dir}/checkpoint_pn_rank{comm.rank}.npy", p_n.x.array)

        i += 1

    xdmf_file.close()
    print("Simulation successfully completed!")

if __name__ == "__main__":
    run_solver()
