"""Actual movie pixels and soundtrack into PAULA; no pretrained representations.

The dog name is a human-facing description, never neural input. This first
preparation tests one recording, not semantic dog/bark understanding. The clip
also contains speech. Aligned and circularly shifted soundtrack conditions have
the same visual stream, source audio samples and repetition count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .population_hierarchy import LABELS, make_config, run, finite_cosine
from .composition_probe import encode

TICKS_PER_SECOND = 60
SOURCE_PAGE = "https://commons.wikimedia.org/wiki/File:Labrador_barking_on_command.theora.ogv"


def decode_media(source):
    """Declared physical transduction, with no fitted statistics or labels.

    96 darkness receptors sample a 12x8 image. Audio is 32 logarithmic frequency
    bands x 3 fixed dB thresholds. This is crude transduction, not a retina or
    cochlea model. A trailing 32ms Hann window avoids future audio samples.
    Three ticks of visual latency cover the source's frame-resampling lookahead.
    """
    source = Path(source).resolve()
    def ffmpeg(args):
        return subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-i", str(source), *args],
                              check=True, stdout=subprocess.PIPE).stdout
    raw = ffmpeg(["-map", "0:v:0", "-vf", "fps=60,scale=12:8:flags=area", "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"])
    pixels = np.frombuffer(raw, np.uint8).reshape(-1, 96)
    audio = np.frombuffer(ffmpeg(["-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"]), dtype="<f4")
    if not len(audio) or not len(pixels):
        raise ValueError("A real video stream and soundtrack are both required")
    count = min(len(pixels), int(len(audio)/16000*TICKS_PER_SECOND))//4*4
    # Frames, pixels, samples and transfer functions are reproducible. No
    # classifier, text embedding, recognition output or outcome label is used.
    visual = 1.-pixels[:count].astype(np.float64)/255.
    visual = np.concatenate((np.zeros((3, 96)), visual[:-3]))
    frequencies = np.fft.rfftfreq(512, 1/16000)
    boundaries = np.geomspace(80., 7600., 33)
    bands = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        selected = np.flatnonzero((frequencies >= lo) & (frequencies < hi))
        if not len(selected):
            selected = np.array([np.argmin(np.abs(frequencies-np.sqrt(lo*hi)))])
        bands.append(selected)
    auditory, rms, db_rows = [], [], []
    window = np.hanning(512)
    for tick in range(count):
        end = int(tick/TICKS_PER_SECOND*16000)
        segment = audio[max(0, end-512):end].astype(np.float64)
        segment = np.pad(segment, (512-len(segment), 0))
        spectrum = np.abs(np.fft.rfft(segment*window))/(window.sum()/2)
        energy = np.array([np.sqrt(np.mean(spectrum[ids]**2)) for ids in bands])
        db = 20*np.log10(np.maximum(energy, 1e-8))
        # Each band has three locally defined sensitivity levels. No clip-wide
        # normalization uses the future or makes quiet clips look loud.
        auditory.append(np.clip((db[:, None]-np.array([-65., -45., -25.]))/20., 0., 1.).ravel())
        rms.append(float(np.sqrt(np.mean(segment**2))))
        db_rows.append(db)
    return {"visual": visual, "auditory": np.array(auditory), "pixels": pixels[:count],
            "audio_rms": np.array(rms), "band_db": np.array(db_rows), "frequency_edges": boundaries,
            "ticks": count, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}


def media_protocol(length, condition="aligned", repeats=3, seed=11):
    if condition not in ("aligned", "shifted") or length <= 0 or length % 4:
        raise ValueError("Use aligned/shifted conditions and a positive four-tick-multiple length")
    trials, start = [], 0
    rng = np.random.default_rng(seed+4603)
    def add(phase, visual=True, auditory=True, duration=None, shift=0, still=False):
        nonlocal start
        duration = length if duration is None else duration
        trials.append({"index": len(trials), "phase": phase, "start": start, "stop": start+duration,
            "visual_enabled": visual, "audio_enabled": auditory, "audio_shift": shift,
            "still": still, "visual": None, "tactile": None, "active_ids": []})
        start += duration
    add("silent_video_before", auditory=False)
    add("audio_reference_before", visual=False)
    for _ in range(repeats):
        shift = 4*int(rng.integers(length//16, 3*length//16)) if condition == "shifted" else 0
        add("audiovisual_experience", shift=shift)
    add("media_silence", visual=False, auditory=False, duration=192)
    add("silent_video_after", auditory=False)
    add("silent_image_after", auditory=False, still=True)
    add("audio_reference_after", visual=False)
    return trials


def media_inputs(features, groups, trial, tick):
    rel = tick-trial["start"]
    vi = min(120, features["ticks"]-1) if trial["still"] else rel % features["ticks"]
    ai = (rel+trial["audio_shift"]) % features["ticks"]
    output = []
    for role, enabled, values in (("vision", trial["visual_enabled"], features["visual"][vi]),
                                   ("touch", trial["audio_enabled"], features["auditory"][ai])):
        if enabled:
            for nid, value in zip(groups[role], values):
                # Stagger receptor drive to avoid one artificial global pulse.
                if rel % 4 == nid % 4 and value > 0:
                    output.append((nid, float(2.*value)))
    return output


def analyze_media(cell, trials, groups, before, after):
    spikes = cell[:, :, 1] > 0
    rows = []
    for tr in trials:
        current = spikes[tr["start"]:tr["stop"]]
        rows.append({"index": tr["index"], "phase": tr["phase"],
            "rates": {g: float(current[:, np.array(ids)-1].mean()) for g, ids in groups.items()},
            "spikes": {g: int(current[:, np.array(ids)-1].sum()) for g, ids in groups.items()}})
    def trace(phase):
        tr = next(tr for tr in trials if tr["phase"] == phase)
        return cell[tr["start"]:tr["stop"], np.array(groups["tactile_core"])-1, 2]
    reference = trace("audio_reference_after")
    matches = []
    for phase in ("silent_video_before", "silent_video_after", "silent_image_after"):
        response = trace(phase)
        # Temporal, neuron-specific fluctuations, not just a common mean rate.
        a = response-response.mean(axis=0, keepdims=True)
        b = reference-reference.mean(axis=0, keepdims=True)
        matches.append({"phase": phase, "centered_auditory_trace_similarity": finite_cosine(a.ravel(), b.ravel()),
            "half_clip_shift_similarity": finite_cosine(a.ravel(), np.roll(b, len(b)//2, axis=0).ravel()),
            "active_auditory_neurons": int(np.count_nonzero(response.max(axis=0) > .001))})
    return {"trial_responses": rows, "auditory_reinstatement": matches,
        "changed_information_weights": int(np.count_nonzero(after != before)),
        "information_weight_change_l1": float(np.abs(after-before).sum()),
        "interpretation": "Silent auditory firing alone is not bark recall. Compare before/after, shifted-pair training, trace specificity and initial-parameter interventions. One clip with speech does not establish semantic understanding."}


def prepare(source, destination, condition="aligned", repeats=3, seed=11):
    features = decode_media(source)
    config, groups, edges = make_config(1152, seed)
    labels = {**LABELS, "touch": "Auditory input", "tactile_core": "Auditory population", "tactile_inhibition": "Auditory inhibition"}
    # The topology is shared with the earlier two-channel experiment. Stable
    # group IDs preserve comparison; labels declare that channel 2 is now sound.
    for n in config["neurons"]:
        n["metadata"]["sensory_kind"] = "audio" if n["metadata"]["role"] in ("touch", "tactile_core", "tactile_inhibition") else "vision" if n["metadata"]["role"] in ("vision", "visual_core", "visual_inhibition") else "internal"
    trials = media_protocol(features["ticks"], condition, repeats, seed)
    metadata = {"labels": labels, "media": {
        "url": "media/dog-command.mp4", "source_page": SOURCE_PAGE,
        "credit": "Sadie Campbell / Nonlinearmind, 2011, CC BY-SA 3.0. Resized/transcoded; original soundtrack.",
        "source_sha256": features["source_sha256"], "condition": condition,
        "ticks_per_second": TICKS_PER_SECOND, "clip_ticks": features["ticks"],
        "visual_latency_ticks": 3, "visual_features": features["visual"].round(6).tolist(),
        "auditory_features": features["auditory"].round(6).tolist(),
        "audio_rms": features["audio_rms"].round(6).tolist(), "frequency_edges_hz": features["frequency_edges"].tolist(),
        "encoders": "12x8 darkness receptors; 32 trailing-window frequency bands x 3 fixed dB sensitivities; staggered drive every four ticks. No learned preprocessing.",
        "confounds": "Single clip; includes human speech and background. Only 96 receptors per sense. Does not establish semantic recognition or generalization."}}
    preparation = {"network": (config, groups, edges), "protocol": ({}, trials),
        "inputs": lambda trial, tick: media_inputs(features, groups, trial, tick),
        "analyze": analyze_media, "metadata": metadata, "source_files": [Path(__file__).resolve()]}
    report = run(destination, 1152, seed, condition, repeats, condition == "aligned", preparation)
    np.savez_compressed(Path(destination)/"sensory-features.npz", **features)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=("aligned", "shifted"), default="aligned")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error("Use 1–20 repeats for this bounded preparation")
    result = prepare(args.source, args.output, args.condition, args.repeats, args.seed)
    print(encode({k: v for k, v in result.items() if k != "trial_responses"}))


if __name__ == "__main__":
    main()
