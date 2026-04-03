"""Composable STL formula API over spatial predicates.

Usage:
    spec = And(always(far_from('robot', 'obs', epsilon=0.5)),
               eventually(close_to('robot', 'goal', epsilon=0.5)))
    rob = spec(signals, geometry=geometry)
"""
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict

from diffspatiall.spatial import (
    compute_face_normals, extract_unique_edges,
    point_signed_distance_to_polyhedron,
    batched_left_of, batched_right_of, batched_above, batched_below,
    batched_in_front_of, batched_behind,
)
from diffspatiall.utils import smooth_min, smooth_max


@dataclass
class ConvexPolyhedron:
    vertices: torch.Tensor      # (V, 3)
    faces: torch.Tensor         # (F, 3)
    face_normals: torch.Tensor  # (F, 3)
    edges: torch.Tensor         # (E, 2)

    @staticmethod
    def from_vertices_faces(vertices, faces):
        return ConvexPolyhedron(vertices, faces,
                                compute_face_normals(vertices, faces),
                                extract_unique_edges(faces))


class SpatialPredicate(nn.Module):
    """Evaluates a spatial predicate on trajectory signals."""

    _DIRECTIONAL = {
        'left_of': batched_left_of, 'right_of': batched_right_of,
        'above': batched_above, 'below': batched_below,
        'in_front_of': batched_in_front_of, 'behind': batched_behind,
    }

    def __init__(self, pred_type, obj1, obj2, epsilon=1.0, tau=1e-2):
        super().__init__()
        self.pred_type, self.obj1, self.obj2 = pred_type, obj1, obj2
        self.epsilon, self.tau = epsilon, tau

    def _compute_sd_trace(self, verts1, verts2, geometry):
        B, T = verts1.shape[0], verts1.shape[1]
        poly2 = geometry.get(self.obj2)
        results = []
        for b in range(B):
            t_results = []
            for t in range(T):
                v1, v2 = verts1[b, t], verts2[b, t]
                if poly2 is not None:
                    sd = point_signed_distance_to_polyhedron(
                        v1, v2, poly2.faces, poly2.face_normals, self.tau)
                    t_results.append(smooth_min(sd, dim=-1, tau=self.tau))
                else:
                    t_results.append((v1.mean(0) - v2.mean(0)).norm())
            results.append(torch.stack(t_results))
        return torch.stack(results)

    def forward(self, signals, geometry=None, **kwargs):
        verts1, verts2 = signals[self.obj1], signals[self.obj2]

        if self.pred_type in self._DIRECTIONAL:
            fn = self._DIRECTIONAL[self.pred_type]
            B, T = verts1.shape[:2]
            results = []
            for b in range(B):
                results.append(torch.stack(
                    [fn(verts1[b, t], verts2[b, t], self.tau) for t in range(T)]))
            return torch.stack(results).unsqueeze(-1)

        sd = self._compute_sd_trace(verts1, verts2, geometry or {})
        if self.pred_type == 'close_to':
            rob = self.epsilon - torch.nn.functional.relu(sd)
        elif self.pred_type == 'far_from':
            rob = sd - self.epsilon
        elif self.pred_type == 'touch':
            rob = self.epsilon - sd.abs()
        elif self.pred_type == 'overlap':
            rob = -sd
        elif self.pred_type == 'enclosed_in':
            rob = -sd - 0.01
        else:
            raise ValueError(f"Unknown predicate: {self.pred_type}")
        return rob.unsqueeze(-1)


# temporal wrappers

class SpatialAlways(nn.Module):
    def __init__(self, formula, interval=None):
        super().__init__()
        self.formula, self.interval = formula, interval

    def forward(self, signals, tau=1e-2, **kwargs):
        rob = self.formula(signals, **kwargs).squeeze(-1)
        B, T = rob.shape
        a = self.interval[0] if self.interval else 0
        b = self.interval[-1] if self.interval else T - 1
        result = torch.empty_like(rob)
        for t in range(T):
            lo, hi = t + a, min(t + b, T - 1)
            result[:, t] = 1e9 if lo > T - 1 else smooth_min(rob[:, lo:hi+1], dim=-1, tau=tau)
        return result.unsqueeze(-1)


class SpatialEventually(nn.Module):
    def __init__(self, formula, interval=None):
        super().__init__()
        self.formula, self.interval = formula, interval

    def forward(self, signals, tau=1e-2, **kwargs):
        rob = self.formula(signals, **kwargs).squeeze(-1)
        B, T = rob.shape
        a = self.interval[0] if self.interval else 0
        b = self.interval[-1] if self.interval else T - 1
        result = torch.empty_like(rob)
        for t in range(T):
            lo, hi = t + a, min(t + b, T - 1)
            result[:, t] = -1e9 if lo > T - 1 else smooth_max(rob[:, lo:hi+1], dim=-1, tau=tau)
        return result.unsqueeze(-1)


class And(nn.Module):
    def __init__(self, *formulas):
        super().__init__()
        self.formulas = nn.ModuleList(formulas)
    def forward(self, signals, **kwargs):
        return torch.stack([f(signals, **kwargs) for f in self.formulas], dim=-1).min(dim=-1).values

class Or(nn.Module):
    def __init__(self, *formulas):
        super().__init__()
        self.formulas = nn.ModuleList(formulas)
    def forward(self, signals, **kwargs):
        return torch.stack([f(signals, **kwargs) for f in self.formulas], dim=-1).max(dim=-1).values

class Not(nn.Module):
    def __init__(self, formula):
        super().__init__()
        self.formula = formula
    def forward(self, signals, **kwargs):
        return -self.formula(signals, **kwargs)


# convenience constructors

def close_to(o1, o2, epsilon=1.0, tau=1e-2):   return SpatialPredicate('close_to', o1, o2, epsilon, tau)
def far_from(o1, o2, epsilon=1.0, tau=1e-2):   return SpatialPredicate('far_from', o1, o2, epsilon, tau)
def touch(o1, o2, epsilon=0.1, tau=1e-2):      return SpatialPredicate('touch', o1, o2, epsilon, tau)
def overlap(o1, o2, tau=1e-2):                  return SpatialPredicate('overlap', o1, o2, 0.0, tau)
def enclosed_in(o1, o2, tau=1e-2):              return SpatialPredicate('enclosed_in', o1, o2, 0.0, tau)
def left_of(o1, o2, tau=1e-2):                  return SpatialPredicate('left_of', o1, o2, tau=tau)
def right_of(o1, o2, tau=1e-2):                 return SpatialPredicate('right_of', o1, o2, tau=tau)
def above(o1, o2, tau=1e-2):                    return SpatialPredicate('above', o1, o2, tau=tau)
def below(o1, o2, tau=1e-2):                    return SpatialPredicate('below', o1, o2, tau=tau)
def in_front_of(o1, o2, tau=1e-2):              return SpatialPredicate('in_front_of', o1, o2, tau=tau)
def behind(o1, o2, tau=1e-2):                   return SpatialPredicate('behind', o1, o2, tau=tau)
def always(formula, interval=None):              return SpatialAlways(formula, interval)
def eventually(formula, interval=None):          return SpatialEventually(formula, interval)
