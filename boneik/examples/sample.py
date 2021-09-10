import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
from tqdm import tqdm
from boneik import kinematics, solvers, utils, draw, criteria, io
from boneik import bvh


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-body", type=Path, help="Kinematic description file", default="etc/body.json"
    )
    args = parser.parse_args()
    assert args.body.is_file()
    body = io.load_json(args.body)

    axes_ranges = [[-5, 5], [-5, 5], [-2, 5]]

    samples = body.sample(10)
    for sample in samples:
        body.set_delta(sample)
        fig, ax = draw.create_figure3d(axes_ranges=axes_ranges)
        draw.draw_kinematics(
            ax,
            body=body,
            fk=body.fk(),
            draw_vertex_labels=True,
            draw_local_frames=True,
            draw_root=False,
        )
        plt.show(block=True)


if __name__ == "__main__":
    main()
