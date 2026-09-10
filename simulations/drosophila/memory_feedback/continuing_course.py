"""Run the existing continuing course while saving the active branch RNG.

The original runner saves its bare network, which captures the ambient RNG
instead of RuntimeBranch's advancing RNG. Current one-tick cleft delays make
that distinction behaviorally inert, but a resumable branch must retain it.
Keep the old runner immutable because its source is pinned by prior records.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from . import second_order
from ..connectome import sha256


def run(receiver, output, *, cut=False, displaced=False):
    original_load = second_order.load_checkpoint
    original_save = second_order.save_checkpoint
    branches = {}
    def load(path, **kwargs):
        branch = original_load(path, **kwargs)
        branches[id(branch.network)] = branch
        return branch
    def save(net, path, *, sources=()):
        branch = branches.pop(id(net))
        return original_save(branch, path, sources=(*sources, __file__))
    with patch.object(second_order, "load_checkpoint", load), patch.object(second_order, "save_checkpoint", save):
        result = second_order.run(receiver, output, cut=cut, displaced=displaced)
    result["branch_rng_preserved"] = True
    result["continuing_driver_sha256"] = sha256(Path(__file__))
    (Path(output)/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receiver", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--cut", action="store_true"); p.add_argument("--displaced", action="store_true")
    a=p.parse_args(); run(a.receiver, a.output, cut=a.cut, displaced=a.displaced)
