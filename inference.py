"""
Run SwinCVS inference on either the Endoscapes dataset (original pipeline,
with ground-truth metrics) or the SAFE dataset (keyframe-flag csv, no
ground-truth metrics assumed - confidences only), selected via --dataset.

Frozen vs. E2E mode and the output directory are CLI flags so one config per
dataset can be reused across runs:

    python3 inference.py --dataset endoscapes --config_path config/infer.yaml \
        --mode frozen --output_dir results/endoscapes_01

    python3 inference.py --dataset safe --config_path config/infer_safe.yaml \
        --mode frozen --output_dir results/safe_01

A single pass (batch size 1, over every video) reports per-video throughput
and resource usage in one sync mode (--sync_mode frame or video), plus
predictions unless --throughput_only, writing under --output_dir:

- result.csv (endoscapes: also metrics.json with mAP / balanced accuracy);
  not written with --throughput_only.
- throughput/framesync.csv (--sync_mode frame): the GPU (CUDA/MPS) is synchronized after
  every frame. One row per frame: vid_id, vid, frame, inference_time_ms,
  latency_time_ms.
- throughput/videosync.csv (--sync_mode video): the GPU is synchronized only
  before and after each video. One row per video with totals and
  total / num_frames.
- resources/<mode>sync/peak.csv, resources/<mode>sync/resource.csv: GPU-only
  peak and sampled memory/power over the whole run, see
  scripts/resource_monitor.py.

inference_time_ms is the model forward only (CUDA/MPS events, under
torch.inference_mode()); latency_time_ms runs from dataset loading/transform
to the end of postprocessing (sigmoid, copy to CPU) after a final sync.

Both modes read from the same VIDEO_ROOT/CSV_PATH: for SAFE this is the
1fps-sampled frames, since each prediction needs the 4 frames preceding the
keyframe to build its input sequence and only the 1fps csv marks keyframes
(`is_ds_keyframe`).
"""

print("Importing libraries...")
# Standard library imports
import argparse
import json
import time
from pathlib import Path
import warnings

# Third-party imports
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local imports
from scripts.f_environment import get_config, set_deterministic_behaviour
from scripts.f_build import build_inference_model
from scripts.f_metrics import get_map, get_balanced_accuracies
from scripts.resource_monitor import ResourceMonitor

# Endoscapes pipeline (untouched)
from scripts.f_dataset import get_inference_dataset

# SAFE pipeline
from scripts.f_dataset_safe import get_safe_inference_dataset

warnings.filterwarnings("ignore")

SYNC_MODES = ("frame", "video")


def parse_args():
    parser = argparse.ArgumentParser(description="Run SwinCVS inference")
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["endoscapes", "safe"],
        help="Which dataset/csv format this run is for.",
    )
    parser.add_argument(
        "--config_path", type=str, required=True,
        help="Path to inference config YAML (config/infer.yaml for endoscapes, config/infer_safe.yaml for safe).",
    )
    parser.add_argument(
        "--mode", type=str, required=True, choices=["e2e", "frozen"],
        help="e2e = un-frozen backbone weights (config.WEIGHTS_E2E); frozen = frozen backbone weights (config.WEIGHTS_FROZEN).",
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Override the weights file to load (defaults to config.WEIGHTS_E2E / WEIGHTS_FROZEN based on --mode). "
             "May be an absolute/relative path or a filename inside ./weights/",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to write outputs into. Created if it doesn't exist.",
    )
    parser.add_argument(
        "--warmup", type=int, default=0,
        help="Run this many forward passes (batch size 1) before the measured run, to warm up "
             "CUDA kernels and caches. Images used for warmup are NOT excluded afterwards - the full "
             "run still goes over every row. Resource sampling covers the warmup too.",
    )
    # Optional per-dataset path overrides, so the config doesn't need editing between runs
    parser.add_argument("--csv_path", type=str, default=None, help="Override config.CSV_PATH")
    parser.add_argument("--images_path", type=str, default=None, help="Endoscapes only: override config.IMAGES_PATH")
    parser.add_argument("--video_root", type=str, default=None, help="Safe only: override config.VIDEO_ROOT")
    parser.add_argument(
        "--sync_mode", "--sync-mode", choices=SYNC_MODES, default="video",
        help="Throughput sync mode: frame (synchronize after every frame, writes throughput/framesync.csv) "
             "or video (synchronize at video boundaries only, writes throughput/videosync.csv).",
    )
    parser.add_argument(
        "--throughput_only", "--throughput-only", action="store_true",
        help="Only measure throughput and resource usage; do not write result.csv (or metrics.json).",
    )
    return parser.parse_args()


def resolve_weights_path(config, args):
    if args.weights is not None:
        weights_file = args.weights
    else:
        weights_file = config.WEIGHTS_E2E if args.mode == "e2e" else config.WEIGHTS_FROZEN

    weights_path = Path(weights_file)
    if not weights_path.is_file():
        weights_path = Path("weights") / weights_file
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Could not find weights file '{weights_file}' (looked for it as given, and under ./weights/)."
        )
    return weights_path


def load_config(args):
    config = get_config(args.config_path, mode="inference")
    config.defrost()
    if args.csv_path is not None:
        config.CSV_PATH = args.csv_path
    if args.dataset == "endoscapes" and args.images_path is not None:
        config.IMAGES_PATH = args.images_path
    if args.dataset == "safe" and args.video_root is not None:
        config.VIDEO_ROOT = args.video_root

    config.MODEL.E2E = (args.mode == "e2e")
    config.MODEL.INFERENCE = True
    config.freeze()

    if config.CSV_PATH is None:
        raise ValueError("CSV_PATH must be set (via config or --csv_path).")
    if args.dataset == "endoscapes" and config.IMAGES_PATH is None:
        raise ValueError("IMAGES_PATH must be set (via config or --images_path).")
    if args.dataset == "safe" and config.VIDEO_ROOT is None:
        raise ValueError("VIDEO_ROOT must be set (via config or --video_root).")

    return config


def device_backend(device):
    """torch.cuda / torch.mps for a GPU device (event timing, synchronize), None on CPU."""
    device_type = torch.device(device).type
    return {"cuda": torch.cuda, "mps": torch.mps}.get(device_type)


def timed_forward(model, sample, backend):
    """Model forward only, bracketed by CUDA/MPS events (perf_counter on CPU), no synchronize."""
    if backend is not None:
        start = backend.Event(enable_timing=True)
        end = backend.Event(enable_timing=True)
        start.record()
        out = model(sample)
        end.record()
        return out, start, end
    start = time.perf_counter()
    out = model(sample)
    return out, start, time.perf_counter()


def elapsed_ms(start, end, backend):
    """Read a timed_forward pair; for CUDA/MPS events the end event must have completed."""
    return start.elapsed_time(end) if backend is not None else (end - start) * 1000.0


def throughput_records(vid_id, vid, frames, inference_times_ms, latency_time_ms):
    """Throughput rows for one video. Frame sync passes per-frame latencies (a
    list) and gets one row per frame; video sync passes the whole-video latency
    and gets one row with the totals and total / num_frames."""
    if isinstance(latency_time_ms, list):
        return [
            {
                "vid_id": vid_id,
                "vid": vid,
                "frame": frame,
                "inference_time_ms": round(inference_time_ms, 3),
                "latency_time_ms": round(frame_latency_time_ms, 3),
            }
            for frame, inference_time_ms, frame_latency_time_ms in zip(
                frames, inference_times_ms, latency_time_ms
            )
        ]
    num_frames = len(inference_times_ms)
    inference_total = sum(inference_times_ms)
    return [
        {
            "vid_id": vid_id,
            "vid": vid,
            "num_frames": num_frames,
            "inference_time_ms": round(inference_total, 3),
            "latency_time_ms": round(latency_time_ms, 3),
            "frame_inference_time_ms": round(inference_total / num_frames, 3),
            "frame_latency_time_ms": round(latency_time_ms / num_frames, 3),
        }
    ]


def run_video_pass(model, dataset, device, meta_df, has_targets, sync_mode):
    """
    Single pass over every video (batch size 1), timing each frame's forward
    pass with CUDA events. sync_mode "frame" synchronizes after every frame;
    "video" synchronizes only before and after each video, postprocessing
    (sigmoid + copy to CPU) once the whole video has been run. Returns (probs,
    targets, throughput_df) with probs/targets in dataset order and one
    throughput row per `vid`.
    """
    backend = device_backend(device)

    def synchronize():
        if backend is not None:
            backend.synchronize()

    def preprocess(idx):
        item = dataset[idx]  # preprocessing happens here, on the main thread
        if has_targets:
            sample, target = item
            targets[idx] = target
        else:
            sample = item
        return sample.unsqueeze(0).to(device, non_blocking=True)

    probs = [None] * len(meta_df)
    targets = [None] * len(meta_df) if has_targets else None
    rows = []
    for vid, group in tqdm(meta_df.groupby("vid", sort=False), desc=f"Processing videos ({sync_mode} sync)"):
        indices = group.index.tolist()
        vid_id = group["vid_id"].iloc[0] if "vid_id" in group.columns else None

        with torch.inference_mode():
            if sync_mode == "frame":
                inference_times_ms, latency_time_ms = [], []
                for idx in indices:
                    t_frame_start = time.perf_counter()
                    out, start, end = timed_forward(model, preprocess(idx), backend)
                    probs[idx] = torch.sigmoid(out).to("cpu")
                    synchronize()
                    latency_time_ms.append((time.perf_counter() - t_frame_start) * 1000.0)
                    inference_times_ms.append(elapsed_ms(start, end, backend))
            else:
                timings, outputs = [], []
                synchronize()
                t_video_start = time.perf_counter()
                for idx in indices:
                    out, start, end = timed_forward(model, preprocess(idx), backend)
                    timings.append((start, end))
                    outputs.append(out)
                for idx, out in zip(indices, outputs):
                    probs[idx] = torch.sigmoid(out).to("cpu")
                synchronize()
                latency_time_ms = (time.perf_counter() - t_video_start) * 1000.0
                inference_times_ms = [elapsed_ms(start, end, backend) for start, end in timings]

        rows.extend(
            throughput_records(
                vid_id, vid, group["frame"].tolist(), inference_times_ms, latency_time_ms
            )
        )

    probs = torch.cat(probs, dim=0)
    if has_targets:
        targets = torch.stack([torch.as_tensor(t) for t in targets], dim=0)
    return probs, targets, pd.DataFrame(rows)


def warmup_model(model, dataset, device, n_warmup, resource_monitor):
    """
    Run `n_warmup` forward passes (batch size 1, untimed, predictions discarded)
    before the real run, to warm up CUDA kernels/caches so the first few timed
    samples aren't penalised by one-off initialisation cost.

    Deliberately reuses samples straight out of `dataset` rather than dummy
    tensors, so the warmup forward passes see real images end-to-end. Those
    same samples are NOT dropped from the run afterwards.
    """
    if n_warmup <= 0:
        return

    n_warmup = min(n_warmup, len(dataset))
    print(f"Warming up with {n_warmup} sample(s)...")
    with torch.inference_mode(), resource_monitor.track_peak(model):
        for idx in range(n_warmup):
            item = dataset[idx]
            sample = item[0] if isinstance(item, (list, tuple)) else item
            model(sample.unsqueeze(0).to(device, non_blocking=True))
    backend = device_backend(device)
    if backend is not None:
        backend.synchronize()
    print("Warmup complete.")


def write_outputs(output_dir, throughput_df, resource_monitor, sync_mode):
    throughput_dir = output_dir / "throughput"
    throughput_dir.mkdir(parents=True, exist_ok=True)
    throughput_path = throughput_dir / f"{sync_mode}sync.csv"
    throughput_df.to_csv(throughput_path, index=False)
    print(f"Throughput ({sync_mode} sync) saved to: {throughput_path}")
    prefix = "frame_" if sync_mode == "video" else ""  # framesync rows are already per frame
    print(
        f"Mean per-frame inference time: {throughput_df[prefix + 'inference_time_ms'].mean():.2f} ms | "
        f"Mean per-frame latency: {throughput_df[prefix + 'latency_time_ms'].mean():.2f} ms"
    )

    resources_dir = output_dir / "resources" / f"{sync_mode}sync"
    resources_dir.mkdir(parents=True, exist_ok=True)
    peak_path = resources_dir / "peak.csv"
    pd.DataFrame([{
        "peak_memory_mb": resource_monitor.peak_memory_mb(),
        "peak_power_w": resource_monitor.peak_power_w(),
    }]).to_csv(peak_path, index=False)
    print(f"Peak resource usage saved to: {peak_path}")
    resource_path = resources_dir / "resource.csv"
    pd.DataFrame(resource_monitor.samples).to_csv(resource_path, index=False)
    print(f"Resource usage samples saved to: {resource_path}")


def run_endoscapes(config, args, model, device, output_dir):
    dataset = get_inference_dataset(config)

    input_df = pd.read_csv(config.CSV_PATH)
    last_index = (len(input_df) // 5) * 5
    label_rows = input_df.iloc[4:last_index:5].reset_index(drop=True)  # row index 4, 9, 14, ...

    meta_df = label_rows[["vid", "frame"]].copy()
    if "vid_id" in input_df.columns:
        meta_df["vid_id"] = input_df.iloc[4:last_index:5]["vid_id"].reset_index(drop=True)

    resource_monitor = ResourceMonitor(torch.device(device))
    resource_monitor.start()
    warmup_model(model, dataset, device, args.warmup, resource_monitor)
    probs, targets, throughput_df = run_video_pass(
        model, dataset, device, meta_df, has_targets=True, sync_mode=args.sync_mode
    )
    resource_monitor.stop()
    write_outputs(output_dir, throughput_df, resource_monitor, args.sync_mode)
    if args.throughput_only:
        return

    preds = torch.round(probs)

    (C1_bacc, C2_bacc, C3_bacc, total_bacc) = get_balanced_accuracies([targets], [preds])
    C1_ap, C2_ap, C3_ap, mAP = get_map([targets], [probs])

    metrics = {
        "avg_bal_acc": round(total_bacc, 4),
        "C1_bacc": round(C1_bacc, 4),
        "C2_bacc": round(C2_bacc, 4),
        "C3_bacc": round(C3_bacc, 4),
        "avg_map": round(mAP, 4),
        "C1_map": round(C1_ap, 4),
        "C2_map": round(C2_ap, 4),
        "C3_map": round(C3_ap, 4),
    }
    print("\nTesting results:", metrics)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    result_df = label_rows.copy()
    result_df["Conf_C1"] = probs[:, 0].tolist()
    result_df["Conf_C2"] = probs[:, 1].tolist()
    result_df["Conf_C3"] = probs[:, 2].tolist()
    result_path = output_dir / "result.csv"
    result_df.to_csv(result_path, index=False)
    print(f"Predictions saved to: {result_path}")
    print(f"Metrics saved to: {output_dir / 'metrics.json'}")


def run_safe(config, args, model, device, output_dir):
    dataset, meta_df = get_safe_inference_dataset(config)
    if len(dataset) == 0:
        raise RuntimeError("No usable 5-frame keyframe sequences were found in the csv.")

    resource_monitor = ResourceMonitor(torch.device(device))
    resource_monitor.start()
    warmup_model(model, dataset, device, args.warmup, resource_monitor)
    probs, _, throughput_df = run_video_pass(
        model, dataset, device, meta_df, has_targets=False, sync_mode=args.sync_mode
    )
    resource_monitor.stop()
    assert probs.shape[0] == len(meta_df), "Prediction / metadata row count mismatch"
    write_outputs(output_dir, throughput_df, resource_monitor, args.sync_mode)
    if args.throughput_only:
        return

    result_df = meta_df.copy()
    result_df["Conf_C1"] = probs[:, 0].tolist()
    result_df["Conf_C2"] = probs[:, 1].tolist()
    result_df["Conf_C3"] = probs[:, 2].tolist()
    result_path = output_dir / "result.csv"
    result_df.to_csv(result_path, index=False)
    print(f"Predictions saved to: {result_path}")


def main():
    args = parse_args()

    config = load_config(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_deterministic_behaviour(config.SEED)

    if torch.cuda.is_available():
        device = f"cuda:{config.CUDA_ID}"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset} | Mode: {args.mode}")

    weights_path = resolve_weights_path(config, args)
    model = build_inference_model(config)
    print("Full model initialised successfully!\n")
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    print(f"Trained SwinCVS weights loaded successfully for INFERENCE - path: {weights_path}")
    model.to(device)
    model.eval()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    task = "throughput" if args.throughput_only else "inference + throughput"
    print(f"\nRunning {task} + resource usage ({args.sync_mode} sync)")
    if args.dataset == "endoscapes":
        run_endoscapes(config, args, model, device, output_dir)
    else:
        run_safe(config, args, model, device, output_dir)


if __name__ == "__main__":
    main()