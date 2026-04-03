"""Differentiable 3D spatial primitives: shape constructors, smooth SAT,
boundary-sampled SDF, and directional predicates."""
import torch
from typing import Optional
from diffspatiall.utils import smooth_max, smooth_min


# --- shape constructors ---

def make_box(lo, hi):
    """AABB from lo/hi corners -> (verts (8,3), faces (12,3))."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    verts = torch.stack([
        torch.tensor([x0, y0, z0]), torch.tensor([x1, y0, z0]),
        torch.tensor([x1, y1, z0]), torch.tensor([x0, y1, z0]),
        torch.tensor([x0, y0, z1]), torch.tensor([x1, y0, z1]),
        torch.tensor([x1, y1, z1]), torch.tensor([x0, y1, z1]),
    ]).to(lo.dtype)
    faces = torch.tensor([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [0, 4, 7], [0, 7, 3],
        [1, 2, 6], [1, 6, 5],
    ], dtype=torch.long)
    return verts, faces


def make_cone(center, radius=1.0, height=2.0, n_sides=16):
    """Cone with base at `center`, tip at center+[0,0,h]."""
    dtype = center.dtype
    angles = torch.linspace(0, 2 * 3.14159265, n_sides + 1, dtype=dtype)[:-1]
    base = torch.stack([
        center[0] + radius * torch.cos(angles),
        center[1] + radius * torch.sin(angles),
        torch.full_like(angles, center[2].item()),
    ], dim=1)
    tip = center + torch.tensor([0, 0, height], dtype=dtype)
    verts = torch.cat([base, tip.unsqueeze(0)], dim=0)

    side_faces = [[i, (i + 1) % n_sides, n_sides] for i in range(n_sides)]
    # base winding flipped so outward normal points -z
    base_faces = [[0, i + 1, i] for i in range(1, n_sides - 1)]
    faces = side_faces + base_faces
    return verts, torch.tensor(faces, dtype=torch.long)


def make_cylinder(center, radius=1.0, height=2.0, n_sides=16):
    dtype = center.dtype
    angles = torch.linspace(0, 2 * 3.14159265, n_sides + 1, dtype=dtype)[:-1]
    bottom = torch.stack([
        center[0] + radius * torch.cos(angles),
        center[1] + radius * torch.sin(angles),
        torch.full_like(angles, center[2].item()),
    ], dim=1)
    top = bottom.clone()
    top[:, 2] += height
    verts = torch.cat([bottom, top], dim=0)

    faces = []
    for i in range(n_sides):
        j = (i + 1) % n_sides
        faces.append([i, j, j + n_sides])
        faces.append([i, j + n_sides, i + n_sides])
    for i in range(1, n_sides - 1):
        faces.append([0, i + 1, i])
        faces.append([n_sides, n_sides + i, n_sides + i + 1])
    return verts, torch.tensor(faces, dtype=torch.long)


# --- geometry utilities ---

def compute_face_normals(vertices, faces):
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    n = torch.cross(v1 - v0, v2 - v0, dim=-1)
    return n / n.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def extract_unique_edges(faces):
    all_edges = torch.cat([faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]], dim=0)
    return torch.unique(torch.sort(all_edges, dim=1)[0], dim=0)


# --- smooth SAT penetration depth ---

def sat_penetration_depth_3d(verts_a, normals_a, edges_a,
                              verts_b, normals_b, edges_b, tau=1e-2):
    """Smooth SAT: overlap along face normals + edge cross-product axes."""
    face_axes = torch.cat([normals_a, normals_b], dim=0)

    # edge x edge cross products for 3D SAT
    edge_dirs_a = verts_a[edges_a[:, 1]] - verts_a[edges_a[:, 0]]
    edge_dirs_b = verts_b[edges_b[:, 1]] - verts_b[edges_b[:, 0]]
    cross_axes = torch.cross(
        edge_dirs_a.unsqueeze(1).expand(-1, edge_dirs_b.shape[0], -1),
        edge_dirs_b.unsqueeze(0).expand(edge_dirs_a.shape[0], -1, -1),
        dim=-1,
    ).reshape(-1, 3)

    cross_norms = cross_axes.norm(dim=-1)
    valid = cross_norms > 1e-8
    if valid.any():
        cross_axes = cross_axes[valid]
        cross_axes = cross_axes / cross_axes.norm(dim=-1, keepdim=True)
        all_axes = torch.cat([face_axes, cross_axes], dim=0)
    else:
        all_axes = face_axes

    proj_a = torch.einsum('ki,vi->kv', all_axes, verts_a)
    proj_b = torch.einsum('ki,vi->kv', all_axes, verts_b)

    a_max = smooth_max(proj_a, dim=-1, tau=tau)
    a_min = smooth_min(proj_a, dim=-1, tau=tau)
    b_max = smooth_max(proj_b, dim=-1, tau=tau)
    b_min = smooth_min(proj_b, dim=-1, tau=tau)

    min_max = smooth_min(torch.stack([a_max, b_max], dim=-1), dim=-1, tau=tau)
    max_min = smooth_max(torch.stack([a_min, b_min], dim=-1), dim=-1, tau=tau)
    overlaps = min_max - max_min

    return smooth_min(overlaps, dim=-1, tau=tau)


# --- point-to-polyhedron signed distance ---

def _point_to_segment_distance(p, a, b):
    ab = b - a
    t = ((p - a) * ab).sum(dim=-1) / (ab * ab).sum().clamp(min=1e-12)
    t = t.clamp(0, 1)
    return (p - (a + t.unsqueeze(-1) * ab)).norm(dim=-1)


def point_signed_distance_to_polyhedron(points, vertices, faces, face_normals, tau=1e-2):
    """Signed distance from query points to a convex polyhedron.
    Positive outside, negative inside. Uses half-space test + boundary
    distance with sigmoid blend."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    inward = -face_normals

    # half-space: positive = interior side of this face
    diff = points.unsqueeze(-2) - v0
    s = (diff * inward).sum(dim=-1)
    inside_margin = smooth_min(s, dim=-1, tau=tau)

    # exterior: closest edge distance per face
    d01 = _point_to_segment_distance(points.unsqueeze(-2), v0, v1)
    d12 = _point_to_segment_distance(points.unsqueeze(-2), v1, v2)
    d20 = _point_to_segment_distance(points.unsqueeze(-2), v2, v0)
    outside_dist = smooth_min(torch.min(torch.min(d01, d12), d20), dim=-1, tau=tau)

    w = torch.sigmoid(50.0 * inside_margin)
    return (1.0 - w) * outside_dist - w * inside_margin


# --- polyhedron-to-polyhedron signed distance ---

def polyhedron_signed_distance(verts_a, faces_a, normals_a, edges_a,
                                verts_b, faces_b, normals_b, edges_b, tau=1e-2):
    """Symmetric boundary-sampled distance blended with SAT penetration."""
    sd_a_to_b = point_signed_distance_to_polyhedron(verts_a, verts_b, faces_b, normals_b, tau)
    sd_b_to_a = point_signed_distance_to_polyhedron(verts_b, verts_a, faces_a, normals_a, tau)
    dist = smooth_min(
        torch.stack([smooth_min(sd_a_to_b, dim=-1, tau=tau),
                     smooth_min(sd_b_to_a, dim=-1, tau=tau)], dim=-1),
        dim=-1, tau=tau,
    )
    pen = sat_penetration_depth_3d(verts_a, normals_a, edges_a,
                                    verts_b, normals_b, edges_b, tau=tau)
    # separated -> +dist, overlapping -> -pen
    inter_w = torch.sigmoid(50.0 * pen)
    return (1.0 - inter_w) * dist + inter_w * (-pen)


# --- batched SD over trajectory ---

_BOX_FACES = torch.tensor([
    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
    [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
], dtype=torch.long)


def batched_polyhedron_sd(robot_verts, obs_verts, obs_faces, obs_normals,
                           tau=1e-2, robot_faces=None):
    """SD from robot to obstacle at each timestep. robot_verts: (T, V, 3)."""
    obs_edges = extract_unique_edges(obs_faces)
    if robot_faces is None:
        robot_faces = _BOX_FACES

    results = []
    for t in range(robot_verts.shape[0]):
        rv = robot_verts[t]
        rn = compute_face_normals(rv, robot_faces)
        re = extract_unique_edges(robot_faces)
        results.append(polyhedron_signed_distance(
            rv, robot_faces, rn, re,
            obs_verts, obs_faces, obs_normals, obs_edges, tau=tau))
    return torch.stack(results)


# --- directional predicates ---

def batched_left_of(v1, v2, tau=1e-2):
    return smooth_min(v2[..., 0], dim=-1, tau=tau) - smooth_max(v1[..., 0], dim=-1, tau=tau)

def batched_right_of(v1, v2, tau=1e-2):
    return smooth_min(v1[..., 0], dim=-1, tau=tau) - smooth_max(v2[..., 0], dim=-1, tau=tau)

def batched_above(v1, v2, tau=1e-2):
    return smooth_min(v1[..., 2], dim=-1, tau=tau) - smooth_max(v2[..., 2], dim=-1, tau=tau)

def batched_below(v1, v2, tau=1e-2):
    return smooth_min(v2[..., 2], dim=-1, tau=tau) - smooth_max(v1[..., 2], dim=-1, tau=tau)

def batched_in_front_of(v1, v2, tau=1e-2):
    return smooth_min(v1[..., 1], dim=-1, tau=tau) - smooth_max(v2[..., 1], dim=-1, tau=tau)

def batched_behind(v1, v2, tau=1e-2):
    return smooth_min(v2[..., 1], dim=-1, tau=tau) - smooth_max(v1[..., 1], dim=-1, tau=tau)
