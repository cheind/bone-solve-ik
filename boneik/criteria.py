from typing import Callable
import torch

from . import kinematics

Criterium = Callable[[kinematics.Body], torch.FloatTensor]


def euclidean_distance_loss(
    body: kinematics.Body,
    anchors: torch.FloatTensor,
    weights: torch.FloatTensor,
) -> torch.FloatTensor:
    fkt = body.fk()
    loss = (
        torch.square(anchors - fkt[:, :3, 3]).sum(-1) * weights
    ).sum() / weights.sum()
    return loss


def parallel_segment_loss(
    body: kinematics.Body,
    anchors: torch.FloatTensor,
    weights: torch.FloatTensor,
) -> torch.FloatTensor:
    fkt = body.fk()
    loss = 0.0
    sum_weights = 0.0
    for u, v in body.bfs_edges[1:]:
        t = fkt[v, :3, 3] - fkt[u, :3, 3]
        p = anchors[v] - anchors[u]
        cos_theta = torch.dot(t, p) / (torch.norm(t) * torch.norm(p))
        w = weights[u] * weights[v]
        sum_weights = sum_weights + w
        loss = loss + (1 - cos_theta) * w
    return loss / sum_weights


class EuclideanDistanceCriterium:
    """Euclidean distance loss between anchors and joints."""

    def __init__(self, anchors: torch.FloatTensor, weights: torch.FloatTensor = None):
        self.anchors = anchors
        if weights is None:
            weights = anchors.new_ones(len(anchors))
        self.weights = weights

    def __call__(self, body: kinematics.Body) -> torch.FloatTensor:
        return euclidean_distance_loss(body, self.anchors, self.weights)


class ParallelSegmentCriterium:
    """Parallel direction loss between predicted and kinematic bones."""

    def __init__(self, anchors: torch.FloatTensor, weights: torch.FloatTensor = None):
        self.anchors = anchors
        if weights is None:
            weights = anchors.new_ones(len(anchors))
        self.weights = weights

    def __call__(self, body: kinematics.Body) -> torch.FloatTensor:
        return parallel_segment_loss(body, self.anchors, self.weights)