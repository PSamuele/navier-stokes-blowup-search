import dolfinx
import dolfinx.plot
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def plot_mesh():
    comm = MPI.COMM_WORLD
    print("Reading mesh...")
    mesh_data = gmshio.read_from_msh("assets/apple_domain.msh", comm, 0, gdim=2)
    domain = mesh_data.mesh

    print("Extracting topology...")
    # We want to visualize the domain mesh
    topology, cell_types, x_dolfinx = dolfinx.plot.vtk_mesh(domain, domain.topology.dim)
    
    # Reshape the topology for triangles
    triangles = topology.reshape(-1, 3)
    
    # Create matplotlib triangulation
    triangulation = mtri.Triangulation(x_dolfinx[:, 0], x_dolfinx[:, 1], triangles)

    print("Drawing contour (outline)...")
    fig, ax = plt.subplots(figsize=(6, 12))
    
    # To draw ONLY the contour, we extract the exterior facets
    domain.topology.create_connectivity(1, 0)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    facet_to_vertex = domain.topology.connectivity(1, 0)
    
    lines_r = []
    lines_z = []
    
    for f in boundary_facets:
        vertices = facet_to_vertex.links(f)
        p0 = domain.geometry.x[vertices[0]]
        p1 = domain.geometry.x[vertices[1]]
        lines_r.extend([p0[0], p1[0], None])
        lines_z.extend([p0[1], p1[1], None])
        
    ax.plot(lines_r, lines_z, color='black', linewidth=1.5)
    
    ax.set_aspect('equal')
    ax.set_title("Axisymmetric Cusp Domain Mesh", fontsize=16)
    ax.set_xlabel("r", fontsize=12)
    ax.set_ylabel("z", fontsize=12)
    
    # Remove unnecessary plot spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    out_file = "docs/images/apple_domain.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved in {out_file}")

if __name__ == "__main__":
    plot_mesh()
