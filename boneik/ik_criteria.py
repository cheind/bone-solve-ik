from typing import Callable
import torch

from . import bodies

IKCriterium = Callable[[bodies.BodyKinematics, torch.FloatTensor], torch.FloatTensor]


def euclidean_distance_loss(
    kin: bodies.BodyKinematics,
    log_angles: torch.FloatTensor,
    anchors: torch.FloatTensor,
    weights: torch.FloatTensor,
) -> torch.FloatTensor:
    fkt = kin.fk(log_angles).squeeze(0)
    loss = (
        torch.square(anchors - fkt[:, :3, 3]).sum(-1) * weights
    ).sum() / weights.sum()
    return loss


def parallel_segment_loss(
    kin: bodies.BodyKinematics,
    log_angles: torch.FloatTensor,
    anchors: torch.FloatTensor,
    weights: torch.FloatTensor,
) -> torch.FloatTensor:
    fkt = kin.fk(log_angles).squeeze(0)
    loss = 0.0
    sum_weights = 0.0
    for u, v in kin.body.bfs_edges[1:]:
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

    def __call__(
        self, kin: bodies.BodyKinematics, log_angles: torch.FloatTensor
    ) -> torch.FloatTensor:
        return euclidean_distance_loss(kin, log_angles, self.anchors, self.weights)


class ParallelSegmentCriterium:
    """Parallel direction loss between predicted and kinematic bones."""

    def __init__(self, anchors: torch.FloatTensor, weights: torch.FloatTensor = None):
        self.anchors = anchors
        if weights is None:
            weights = anchors.new_ones(len(anchors))
        self.weights = weights

    def __call__(
        self, kin: bodies.BodyKinematics, log_angles: torch.FloatTensor
    ) -> torch.FloatTensor:
        return parallel_segment_loss(kin, log_angles, self.anchors, self.weights)