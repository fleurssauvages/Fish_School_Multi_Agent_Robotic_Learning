import numpy as np
from vispy import app, scene
from vispy.scene import visuals
import trimesh

def build_trimesh_obstacles(obs_list):
    """
    obs_list items look like (verts, faces). Returns list of trimesh.Trimesh.
    """
    meshes = []
    for verts, faces in obs_list:
        m = trimesh.Trimesh(vertices=np.asarray(verts),
                            faces=np.asarray(faces),
                            process=False)  # keep as-is (faster, preserves indexing)
        meshes.append(m)
    return meshes

def sphere_collides_mesh_robust(center, radius, mesh):
    """
    Robust sphere-vs-triangle-mesh test.
    - Uses closest-point distance (works for any mesh).
    - Uses contains() only when watertight (optional "inside" check).
    """
    c = np.asarray(center, dtype=np.float64).reshape(1, 3)

    # Distance to surface (always meaningful)
    _, dist, _ = trimesh.proximity.closest_point(mesh, c)
    dist = float(dist[0])

    # Intersects surface (sphere overlaps mesh surface)
    if dist < radius:
        return True, dist, False

    # Optional: if you also want to prevent being strictly "inside" even when radius is tiny
    inside = False
    if mesh.is_watertight:
        inside = bool(mesh.contains(c)[0])
        if inside:
            return True, dist, True

    return False, dist, inside


def apply_collision_revert(x_prev, x_new, user_radius, obstacle_meshes, debug=False):
    for i, m in enumerate(obstacle_meshes):
        hit, dist, inside = sphere_collides_mesh_robust(x_new, user_radius, m)
        if debug:
            print(f"[obs {i}] dist={dist:.4f}  inside={inside}  hit={hit}")
        if hit:
            return x_prev, True
    return x_new, False


def sphere_wireframe_lines(center, radius, n_u=24, n_v=24, stride_u=4, stride_v=3):
    """
    Build polyline segments approximating the same 'lat/long' wireframe you drew in Matplotlib.
    Returns a list of (N, 3) arrays, each one is a polyline.
    """
    cx, cy, cz = center
    u = np.linspace(0, 2*np.pi, n_u, endpoint=True)
    v = np.linspace(0, np.pi, n_v, endpoint=True)

    lines = []

    # parallels (vary u, fixed v)
    for j in range(0, n_v, stride_v):
        vv = v[j]
        x = cx + radius * np.cos(u) * np.sin(vv)
        y = cy + radius * np.sin(u) * np.sin(vv)
        z = cz + radius * np.cos(vv) * np.ones_like(u)
        lines.append(np.c_[x, y, z])

    # meridians (vary v, fixed u)
    for i in range(0, n_u, stride_u):
        uu = u[i]
        x = cx + radius * np.cos(uu) * np.sin(v)
        y = cy + radius * np.sin(uu) * np.sin(v)
        z = cz + radius * np.cos(v)
        lines.append(np.c_[x, y, z])

    return lines

from vispy.geometry import create_sphere
def add_transparent_sphere(view, center, radius, color_rgba, rows=12, cols=12):
    sphere = create_sphere(rows=rows, cols=cols, radius=radius)
    verts = sphere.get_vertices().astype(np.float32)
    faces = sphere.get_faces()
    colors = np.tile(np.array(color_rgba, dtype=np.float32), (verts.shape[0], 1))

    mesh = visuals.Mesh(
        vertices=verts,
        faces=faces,
        vertex_colors=colors,
        parent=view.scene
    )
    mesh.transform = scene.transforms.STTransform(translate=tuple(center))
    mesh.set_gl_state(
        blend=True,
        depth_test=True,
        depth_mask=False,
        blend_func=('src_alpha', 'one_minus_src_alpha')
    )
    return mesh

def set_mesh_color(mesh, rgba):
    md = mesh.mesh_data
    n = md.n_vertices
    md.set_vertex_colors(np.tile(np.array(rgba, np.float32), (n, 1)))
    mesh.mesh_data_changed()

def add_wireframe(view, center, radius, line_color=(0, 0, 0, 0.5), width=1.0):
    for poly in sphere_wireframe_lines(center, radius):
        ln = visuals.Line(poly, color=line_color, width=width, method="gl", parent=view.scene)
        ln.set_gl_state(
            blend=True,
            depth_test=True,
            depth_mask=False,
            blend_func=('src_alpha', 'one_minus_src_alpha')
        )

def mesh_wireframe_segments(verts, faces):
    """
    verts: (V,3)
    faces: (F,3) int
    Returns:
      seg: (2*E, 3) float32 where each pair of rows is one segment endpoint.
      (usable with visuals.Line(..., connect='segments'))
    """
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    # edges per triangle: (i0,i1), (i1,i2), (i2,i0)
    e01 = faces[:, [0, 1]]
    e12 = faces[:, [1, 2]]
    e20 = faces[:, [2, 0]]
    edges = np.vstack((e01, e12, e20))

    # undirected unique edges: sort indices in each row then unique
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)  # (E,2)

    # build segment endpoints (2*E,3): [v0, v1, v0, v1, ...]
    seg = verts[edges.reshape(-1)]
    return seg

def add_transparent_mesh_with_wireframe(
    view,
    verts,
    faces,
    *,
    mesh_rgba=(0.6, 0.6, 0.6, 0.25),
    wire_rgba=(0.0, 0.0, 0.0, 0.5),
    wire_width=1.0,
    shading="flat",
):
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    # --- filled mesh ---
    colors = np.tile(np.array(mesh_rgba, dtype=np.float32), (verts.shape[0], 1))
    mesh = visuals.Mesh(
        vertices=verts,
        faces=faces,
        vertex_colors=colors,
        shading=shading,
        parent=view.scene,
    )
    mesh.set_gl_state(
        blend=True,
        depth_test=True,
        depth_mask=False,
        blend_func=('src_alpha', 'one_minus_src_alpha')
    )

    # --- wireframe as segments ---
    seg = mesh_wireframe_segments(verts, faces)  # (2*E,3)
    wire = visuals.Line(
        pos=seg,
        color=wire_rgba,
        width=wire_width,
        connect="segments",   # critical: interpret pairs as independent segments
        method="gl",
        parent=view.scene,
    )
    wire.set_gl_state(
        blend=True,
        depth_test=True,
        depth_mask=False,
        blend_func=('src_alpha', 'one_minus_src_alpha')
    )

    return mesh, wire

def add_floor_grid(view, z=0.0, xlim=(10, 40), ylim=(10, 40), step=2.0,
                color=(0, 0, 0, 0.08), width=1.0):
    xs = np.arange(xlim[0], xlim[1] + 1e-9, step)
    ys = np.arange(ylim[0], ylim[1] + 1e-9, step)

    # lines parallel to Y
    for x in xs:
        p = np.array([[x, ylim[0], z], [x, ylim[1], z]], dtype=np.float32)
        visuals.Line(p, color=color, width=width, method="gl", parent=view.scene)

    # lines parallel to X
    for y in ys:
        p = np.array([[xlim[0], y, z], [xlim[1], y, z]], dtype=np.float32)
        visuals.Line(p, color=color, width=width, method="gl", parent=view.scene)

def add_backwall_grid(view, x=0.0, zlim=(10, 40), ylim=(10, 40), step=2.0,
                color=(0, 0, 0, 0.08), width=1.0):
    zs = np.arange(zlim[0], zlim[1] + 1e-9, step)
    ys = np.arange(ylim[0], ylim[1] + 1e-9, step)

    # lines parallel to Y
    for z in zs:
        p = np.array([[x, ylim[0], z], [x, ylim[1], z]], dtype=np.float32)
        visuals.Line(p, color=color, width=width, method="gl", parent=view.scene)