import gmsh
import numpy as np
import sys

def generate_mesh(R0=1.0, H=2.0, k=0.5, num_points=400, lc_boundary=0.05, lc_pole=0.002):
    gmsh.initialize()
    gmsh.model.add("AppleDomain")

    # Funzione del bordo
    def f(z):
        return R0 * np.cos(np.pi * z / (2 * H)) * np.exp(-k * z**2)

    # Parametri z dal polo sud al polo nord
    z_vals = np.linspace(-H, H, num_points)
    
    # 1. Punti sull'asse di simmetria (r=0)
    # Polo Sud
    p_south = gmsh.model.geo.addPoint(0.0, -H, 0.0, lc_pole)
    # Polo Nord
    p_north = gmsh.model.geo.addPoint(0.0, H, 0.0, lc_pole)

    # 2. Punti sul bordo curvo r = f(z)
    curve_pts = []
    # Aggiungiamo punti intermedi, escludendo i poli esatti per evitare duplicati
    for i in range(1, num_points - 1):
        z = z_vals[i]
        r = f(z)
        # La dimensione caratteristica della mesh può decrescere verso i poli
        lc = lc_pole + (lc_boundary - lc_pole) * (1.0 - abs(z)/H)
        curve_pts.append(gmsh.model.geo.addPoint(r, z, 0.0, lc))

    # 3. Creazione delle curve
    # Asse Z (da Polo Sud a Polo Nord)
    axis_line = gmsh.model.geo.addLine(p_south, p_north)
    
    # Bordo curvo (Spline da Polo Sud a Polo Nord)
    spline_pts = [p_south] + curve_pts + [p_north]
    apple_curve = gmsh.model.geo.addSpline(spline_pts)

    # 4. Creazione della superficie
    curve_loop = gmsh.model.geo.addCurveLoop([axis_line, -apple_curve])
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])

    # 5. Gruppi Fisici (Physical Groups) per FEniCS
    axis_group = gmsh.model.geo.addPhysicalGroup(1, [axis_line], name="SymmetryAxis")
    wall_group = gmsh.model.geo.addPhysicalGroup(1, [apple_curve], name="AppleWall")
    domain_group = gmsh.model.geo.addPhysicalGroup(2, [surface], name="FluidDomain")

    gmsh.model.geo.synchronize()

    # --- CAMPO DI RAFFINAMENTO MATEMATICO (MathEval Field) ---
    gmsh.model.mesh.field.add("MathEval", 1)
    # Rendi la mesh fittissima (0.0001) ai poli (z = +-2) e fitta (0.015) al centro
    gmsh.model.mesh.field.setString(1, "F", "0.0001 + 0.0149 * (1.0 - abs(z)/2.0)^3")
    gmsh.model.mesh.field.setAsBackgroundMesh(1)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    
    # Limiti assoluti di sicurezza
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.015)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0001)

    # Genera mesh 2D
    gmsh.model.mesh.generate(2)
    
    # Salva la mesh (formato msh, può essere convertito in xdmf per dolfinx se necessario)
    gmsh.write("apple_domain.msh")
    
    print("Mesh generata e salvata come apple_domain.msh")
    
    # Decommentare per vedere la GUI di gmsh: gmsh.fltk.run()
    # Se passiamo un flag --gui dalla riga di comando:
    if '--gui' in sys.argv:
        gmsh.fltk.run()

    gmsh.finalize()

if __name__ == "__main__":
    generate_mesh()
