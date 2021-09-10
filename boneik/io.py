import json
import numpy as np
from pathlib import Path

from . import kinematics
from . import utils


def load_json(jsonpath: Path) -> kinematics.Body:
    jsonpath = Path(jsonpath)
    assert jsonpath.is_file()
    data = json.load(open(jsonpath, "r"))
    b = kinematics.BodyBuilder()

    def _convert_interval(i):
        if i is None:
            return None
        else:
            return np.deg2rad(i)

    for bone in data["bones"]:
        dofs = None
        if "dofs" in bone:
            dofs = {n: _convert_interval(i) for n, i in bone["dofs"].items()}
        b.add_bone(
            bone["u"],
            bone["v"],
            tip_to_base=utils.make_tip_to_base(bone["length"], bone["axes"]),
            dofs=dofs,
        )

    order = None
    if "order" in data:
        order = data["order"]
    return b.finalize(order)


if __name__ == "__main__":
    from . import draw
    import matplotlib.pyplot as plt

    body = load_json("etc/body.json")

    fig, ax = draw.create_figure3d()
    draw.draw_kinematics(ax, body=body, draw_root=True)
    plt.show()