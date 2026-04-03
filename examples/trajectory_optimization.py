"""Trajectory optimization demo + GIF generation.
phi = G(farFrom(robot, cone, 0.5)) /\ F(closeTo(robot, goal, 0.5))
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

from diffspatiall.spatial import (
    make_box, make_cone, compute_face_normals, batched_polyhedron_sd,
)
from diffspatiall.temporal import Always, Eventually

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)


def draw_poly(ax, v, f, color, alpha=0.15):
    polys = [[v[fi].detach().numpy() for fi in face] for face in f]
    ax.add_collection3d(Poly3DCollection(polys, alpha=alpha, facecolor=color,
                                          edgecolor=color, linewidth=0.3))


# scene
cone_v, cone_f = make_cone(torch.tensor([0., 0., 3.]), radius=2.5, height=5.0, n_sides=16)
cone_n = compute_face_normals(cone_v, cone_f)

goal_v, goal_f = make_box(torch.tensor([-0.75, -0.75, 9.25]), torch.tensor([0.75, 0.75, 10.75]))
goal_n = compute_face_normals(goal_v, goal_f)

robot_v, robot_f = make_box(torch.tensor([-0.3, -0.3, -0.3]), torch.tensor([0.3, 0.3, 0.3]))

# init trajectory: straight line through the cone with small x perturbation
T = 40
eps_far = 0.5
eps_close = 0.5
start = torch.tensor([0., 0., 0.])
end_pt = torch.tensor([0., 0., 10.])
t_interp = torch.linspace(0, 1, T).unsqueeze(1)
traj = (start * (1 - t_interp) + end_pt * t_interp).clone()
# small perturbation to break symmetry (still inside cone)
traj[6:-6, 0] += 1.0
traj = torch.nn.Parameter(traj)

optimizer = torch.optim.Adam([traj], lr=0.12)

snapshots = []
rob_history = []
phase2_start = None

n_iters = 500
save_every = 5

print(f'Optimizing trajectory ({n_iters} iterations)...')

for it in range(n_iters):
    optimizer.zero_grad()

    # pin start/end
    centers = torch.cat([start.unsqueeze(0), traj[1:-1], end_pt.unsqueeze(0)], dim=0)
    robot_traj = centers.unsqueeze(1) + robot_v.unsqueeze(0)
    sd_cone = batched_polyhedron_sd(robot_traj, cone_v, cone_f, cone_n,
                                     tau=1e-2, robot_faces=robot_f)
    sd_goal = batched_polyhedron_sd(robot_traj, goal_v, goal_f, goal_n,
                                     tau=1e-2, robot_faces=robot_f)

    far_rob = sd_cone - eps_far
    close_rob = eps_close - torch.nn.functional.relu(sd_goal)

    # phase 1: large tau to get out; phase 2: tighten
    if phase2_start is None:
        t_tau, smooth_w = 0.5, 0.05
    else:
        t_tau, smooth_w = 0.05, 0.5

    always_far = Always()(far_rob.unsqueeze(0).unsqueeze(-1), tau=t_tau)
    event_close = Eventually()(close_rob.unsqueeze(0).unsqueeze(-1), tau=t_tau)
    rob = torch.min(always_far[0, 0, 0], event_close[0, 0, 0])

    # smoothness
    vel = centers[1:] - centers[:-1]
    acc = vel[1:] - vel[:-1]
    smooth_loss = (acc ** 2).sum() + 5.0 * vel.norm(dim=-1).var()

    rob_obj = -torch.clamp(rob, max=0.05) if phase2_start else -rob
    loss = rob_obj + smooth_w * smooth_loss
    loss.backward()
    optimizer.step()

    rob_history.append(rob.item())

    if phase2_start is None and rob.item() > 0.2:
        phase2_start = it

    if it % save_every == 0 or it == n_iters - 1:
        final_centers = torch.cat([start.unsqueeze(0), traj[1:-1].detach(), end_pt.unsqueeze(0)], dim=0)
        snapshots.append((it, final_centers.numpy(), rob.item()))

    if (it + 1) % 100 == 0:
        status = 'SAT' if rob.item() >= 0 else 'UNSAT'
        print(f'  Iter {it+1:3d}: robustness = {rob.item():+.4f} [{status}]')

print(f'Final robustness: {rob_history[-1]:+.4f}')

# gif
try:
    from PIL import Image
except ImportError:
    print('PIL not available, skipping GIF generation. Install with: pip install Pillow')
    sys.exit(0)

print(f'Generating GIF ({len(snapshots)} frames)...')
frames = []

for it, traj_np, rob_val in snapshots:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=-55)

    draw_poly(ax, cone_v, cone_f, 'red', 0.12)
    draw_poly(ax, goal_v, goal_f, 'green', 0.3)

    sat = rob_val >= 0
    traj_color = '#2ca02c' if sat else '#1f77b4'
    ax.plot(traj_np[:, 0], traj_np[:, 1], traj_np[:, 2],
            '-o', color=traj_color, ms=2.5, lw=1.8, alpha=0.8)

    ax.scatter(*traj_np[0], c='blue', s=80, marker='o', zorder=10)
    ax.scatter(*traj_np[-1], c=traj_color, s=80, marker='*', zorder=10)

    # robot at a few timesteps
    for t_idx in [0, T // 4, T // 2, 3 * T // 4, T - 1]:
        rv_np = robot_v.numpy() + traj_np[t_idx]
        polys = [[rv_np[fi] for fi in face] for face in robot_f.numpy()]
        ax.add_collection3d(Poly3DCollection(polys, alpha=0.15, facecolor=traj_color,
                                              edgecolor=traj_color, linewidth=0.3))

    ax.set_xlim(-4, 6); ax.set_ylim(-3, 4); ax.set_zlim(-1, 12)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.tick_params(labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    status = 'SAT' if sat else 'UNSAT'
    bc = '#2ca02c' if sat else '#d62728'
    ax.text2D(0.02, 0.95,
              f'Iter {it:3d}/{n_iters}  |  $\\rho$ = {rob_val:+.3f}  {status}',
              transform=ax.transAxes, fontsize=12, fontweight='bold', color='white',
              bbox=dict(fc=bc, alpha=0.9, ec='none', boxstyle='round,pad=0.3'))

    ax.set_title(r'$G(\mathrm{farFrom}(\mathrm{robot}, \mathrm{cone})) \wedge '
                 r'F(\mathrm{closeTo}(\mathrm{robot}, \mathrm{goal}))$',
                 fontsize=12, pad=8)

    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    frames.append(Image.fromarray(img[:, :, :3]))
    plt.close()

# hold last frame
for _ in range(10):
    frames.append(frames[-1])

fname = os.path.join(OUT_DIR, 'traj_optimization.gif')
frames[0].save(fname, save_all=True, append_images=frames[1:], duration=120, loop=0)
print(f'Saved: {fname}')
