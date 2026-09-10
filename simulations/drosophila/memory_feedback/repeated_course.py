"""Increase B-before-A exposure without changing equations or parameters.

Eight pairings are the first prespecified accumulation test after the
two-pairing interface comparison. Preserve cue/gap/recovery durations and the
final retention interval. The same intact, projection-cut, displaced and
teacher-memory controls can share this protocol. Use disk-backed traces and
refuse to start below a conservative shared-volume reserve.
"""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch

from . import second_order, student_interface
from ..connectome import sha256

_original_protocol = second_order.protocol


def protocol(pairings=8, displaced=False):
    if isinstance(pairings, bool) or not isinstance(pairings, int) or not 1 <= pairings <= 32:
        raise ValueError("Need a bounded integer count of 1 to 32 pairings")
    template = _original_protocol(displaced)
    return [(f"pair{i}_"+name.split("_", 1)[1], cue, ticks)
            for i in range(pairings) for name, cue, ticks in template[:4]] + [template[-1]]


def run(receiver, output, *, pairings=8, cut=False, displaced=False):
    receiver, output = Path(receiver), Path(output)
    phases = protocol(pairings, displaced)
    # Approximately 300 MiB per eight-pairing record. Keep at least 4 GiB
    # beyond that estimate for concurrent manuscript work and filesystem use.
    estimated_bytes = int(300*1024**2*sum(t for _, _, t in phases)/11880)
    if shutil.disk_usage(receiver).free < 4*1024**3+estimated_bytes:
        raise RuntimeError("Insufficient shared-volume reserve for the bounded course")
    with patch.object(second_order, "protocol", lambda displaced=False: protocol(pairings, displaced)):
        result = student_interface.run_continuation(receiver, output, cut=cut, displaced=displaced)
    result["pairings"] = pairings
    result["repetition_driver_sha256"] = sha256(Path(__file__))
    result["limit"] = "Bounded repeated B-before-A acquisition with continuing adaptation and no nutrients. Increased exposure alone is not evidence of teacher-dependent B behavior."
    (output/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receiver", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--pairings", type=int, default=8)
    p.add_argument("--cut", action="store_true"); p.add_argument("--displaced", action="store_true")
    a=p.parse_args(); run(a.receiver, a.output, pairings=a.pairings, cut=a.cut, displaced=a.displaced)
