# Differentiable SpaTiaL

**A Differentiable Toolbox for Geometric Temporal Logic over 3D Robotic Manipulation**
> This repo contains a minimal demo implementation. Full version coming in the next few weeks — stay tuned.
---

## Overview

**Differentiable SpaTiaL** is an autograd-compatible toolbox that constructs smooth geometric primitives directly over convex polyhedral sets, enabling end-to-end differentiable mapping from high-level spatio-temporal specifications to low-level geometric configurations.

<p align="center">
  <img src="figures/traj_optimization.gif" width="600">
  <br>
  <em>Trajectory optimization: the robot learns to avoid the cone obstacle while reaching the goal, driven entirely by differentiable robustness gradients.</em>
</p>

### Key Features

- **Smooth SAT**: Differentiable penetration depth via the Separating Axis Theorem
- **Boundary-sampled SDF**: Differentiable signed distance for separated states
- **Core spatial predicates**: closeTo, farFrom, touch, overlap, enclIn, leftOf, above, ...
- **STL temporal operators**: low-level `Always`, `Eventually`, `Until`, `Next`
- **End-to-end differentiable**: Gradients flow from logical robustness to physical states

## Installation

Run all commands below from the repository root:

```bash
pip install -r requirements.txt
```

Optional, for the notebook demo:

```bash
pip install jupyter
```

## Quick Start

The fastest end-to-end smoke test is the trajectory optimization example:

```bash
python examples/trajectory_optimization.py
```

This script optimizes a collision-avoiding trajectory and writes:

- `figures/traj_optimization.gif`

## Low-Level API

Runnable example using the low-level spatial and temporal operators:

```python
import torch
from diffspatiall.spatial import make_box, make_cone, compute_face_normals, batched_polyhedron_sd
from diffspatiall.temporal import Always, Eventually

# Create geometry
robot_v, robot_f = make_box(torch.tensor([-0.2, -0.2, -0.2]),
                            torch.tensor([0.2, 0.2, 0.2]))
cone_v, cone_f = make_cone(torch.tensor([0., 0., 3.]), radius=2.5, height=5.0)
cone_n = compute_face_normals(cone_v, cone_f)
goal_v, goal_f = make_box(torch.tensor([-0.5, -0.5, 7.5]),
                          torch.tensor([0.5, 0.5, 8.5]))
goal_n = compute_face_normals(goal_v, goal_f)

# Straight-line robot trajectory: (T, V, 3)
T = 20
start = torch.tensor([0., 0., 0.])
end = torch.tensor([0., 0., 8.0])
alpha = torch.linspace(0, 1, T).unsqueeze(1)
centers = start * (1 - alpha) + end * alpha
robot_trajectory = centers.unsqueeze(1) + robot_v.unsqueeze(0)

eps_far = 0.5
eps_close = 0.5

# Compute differentiable signed distance traces along the trajectory
sd_cone = batched_polyhedron_sd(
    robot_trajectory, cone_v, cone_f, cone_n, tau=1e-2, robot_faces=robot_f
)
sd_goal = batched_polyhedron_sd(
    robot_trajectory, goal_v, goal_f, goal_n, tau=1e-2, robot_faces=robot_f
)

# Apply temporal operators
always_safe = Always()(sd_cone.unsqueeze(0).unsqueeze(-1) - eps_far)  # G(farFrom)
eventually_reach = Eventually()(
    eps_close - torch.relu(sd_goal).unsqueeze(0).unsqueeze(-1)
)  # F(closeTo)

# Robustness is differentiable!
robustness = torch.min(always_safe[0, 0, 0], eventually_reach[0, 0, 0])
print(float(robustness))
```

### High-Level Formula API

Illustrative high-level formula construction:

```python
from diffspatiall.formula import close_to, far_from, always, eventually, And, ConvexPolyhedron

# Build composable STL specification
spec = And(
    always(far_from('robot', 'cone', epsilon=0.5)),
    eventually(close_to('robot', 'goal', epsilon=0.5)),
)

# Evaluate on trajectory data
# `signals` and `geometry` follow the setup used in `demo.ipynb`
robustness = spec(signals, geometry=geometry)  # fully differentiable
```

## Interactive Demo

Launch the Jupyter notebook:
```bash
jupyter notebook demo.ipynb
```

The notebook walks through:
1. Creating 3D geometry (boxes, cones, cylinders)
2. Computing smooth signed distances
3. Verifying gradient flow
4. Spatial predicates
5. Temporal operators
6. Full trajectory optimization

## Release Scope

This demo release intentionally prioritizes clarity over completeness. It includes:

- Core smooth geometric primitives in `diffspatiall/spatial.py`
- Low-level STL temporal operators in `diffspatiall/temporal.py`
- A lightweight high-level formula API in `diffspatiall/formula.py`
- A runnable optimization example in `examples/trajectory_optimization.py`

The more optimized internal research code used for ongoing extensions is not included in this minimal release.

## Predicate Gallery

| Predicate | Robustness | Type |
|-----------|-----------|------|
| closeTo(A,B,eps) | eps - dist(A,B) | Distance |
| farFrom(A,B,eps) | dist(A,B) - eps | Distance |
| touch(A,B,eps) | eps - \|sd(A,B)\| | Distance |
| overlap(A,B) | -sd(A,B) | Topology |
| enclIn(A,B) | -delta - max(sd(v_A, B)) | Topology |
| leftOf(A,B) | min_x(B) - max_x(A) | Directional |
| rightOf(A,B) | min_x(A) - max_x(B) | Directional |
| above(A,B) | min_z(A) - max_z(B) | Directional |
| below(A,B) | min_z(B) - max_z(A) | Directional |
| inFrontOf(A,B) | min_y(A) - max_y(B) | Directional |
| behind(A,B) | min_y(B) - max_y(A) | Directional |

## Citation

```bibtex
@misc{luo2026differentiablespatialsymboliclearning,
      title={Differentiable SpaTiaL: Symbolic Learning and Reasoning with Geometric Temporal Logic for Manipulation Tasks}, 
      author={Licheng Luo and Kaier Liang and Cristian-Ioan Vasile and Mingyu Cai},
      year={2026},
      eprint={2604.02643},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2604.02643}, 
}
```

## License

MIT License
