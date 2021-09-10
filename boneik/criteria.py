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


class EuclideanDistanceCriterium:
    """Euclidean distance loss between anchors and joints."""

    def __init__(self, anchors: torch.FloatTensor, weights: torch.FloatTensor = None):
        self.anchors = anchors
        if weights is None:
            weights = anchors.new_ones(len(anchors))
        self.weights = weights

    def __call__(self, body: kinematics.Body) -> torch.FloatTensor:
        return euclidean_distance_loss(body, self.anchors, self.weights)
