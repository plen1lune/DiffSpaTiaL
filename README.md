# Differentiable SpaTiaL

**A Differentiable Toolbox for Geometric Temporal Logic over 3D Robotic Manipulation**

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
- **15+ spatial predicates**: closeTo, farFrom, touch, overlap, enclIn, leftOf, above, ...
- **STL temporal operators**: Always, Eventually, Until, Next
- **End-to-end differentiable**: Gradients flow from logical robustness to physical states

### Two Core Applications

1. **Trajectory Optimization**: Maximize robustness of a spatio-temporal specification
2. **Specification Learning**: Learn spatial parameters from demonstrations via backpropagation

## Quick Start

```bash
pip install torch numpy matplotlib Pillow
```

```python
import torch
from diffspatiall.spatial import make_box, make_cone, compute_face_normals, batched_polyhedron_sd
from diffspatiall.temporal import Always, Eventually

# Create geometry
cone_v, cone_f = make_cone(torch.tensor([0., 0., 3.]), radius=2.5, height=5.0)
cone_n = compute_face_normals(cone_v, cone_f)

# Compute differentiable signed distance along a trajectory
sd = batched_polyhedron_sd(robot_trajectory, cone_v, cone_f, cone_n)

# Apply temporal operators
always_safe = Always()(sd.unsqueeze(0).unsqueeze(-1) - epsilon)    # G(farFrom)
eventually_reach = Eventually()(epsilon - sd.unsqueeze(0).unsqueeze(-1))  # F(closeTo)

# Robustness is differentiable!
robustness = torch.min(always_safe[0, 0, 0], eventually_reach[0, 0, 0])
robustness.backward()  # gradients flow to robot_trajectory
```

### High-Level Formula API

```python
from diffspatiall.formula import close_to, far_from, always, eventually, And, ConvexPolyhedron

# Build composable STL specification
spec = And(
    always(far_from('robot', 'cone', epsilon=0.5)),
    eventually(close_to('robot', 'goal', epsilon=0.5)),
)

# Evaluate on trajectory data
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
@inproceedings{luo2025diffspatiall,
  title={Differentiable SpaTiaL: Symbolic Learning and Reasoning with Geometric Temporal Logic for Manipulation Tasks},
  author={Luo, Licheng and Liang, Kaier and Vasile, Cristian-Ioan and Cai, Mingyu},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2025}
}
```

## License

MIT License
