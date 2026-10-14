"""
Tests for SwinCVS. No dataset, weights or GPU needed: models use random weights and the
end-to-end tests run on a small synthetic dataset.

    python3 test.py                          # everything (a few minutes, mostly the end-to-end runs)
    python3 -m unittest test.TestMetrics     # one group
"""
import math
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, recall_score

from scripts.f_build import build_model, build_inference_model
from scripts.f_dataset import get_dataloaders
from scripts.f_environment import get_config
from scripts.f_metrics import get_balanced_accuracies, get_map, get_mean_recall
from scripts.m_swinv2 import SwinTransformerV2

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent
CONFIGS = REPO / "config"


def load(name):
    return get_config(str(CONFIGS / name))


def with_model(config, **model_settings):
    config = config.clone()
    config.defrost()
    for key, value in model_settings.items():
        config.MODEL[key] = value
    return config


class TestMetrics(unittest.TestCase):
    def random_labels(self, seed, n=50):
        rng = np.random.default_rng(seed)
        targets = rng.integers(0, 2, (n, 3)).astype(np.float32)
        probs = rng.random((n, 3)).astype(np.float32)
        return targets, probs

    def test_balanced_accuracy_matches_sklearn(self):
        for seed in range(10):
            targets, probs = self.random_labels(seed)
            preds = np.round(probs)
            # split into batches, as the training scripts pass lists of batches
            C1, C2, C3, avg = get_balanced_accuracies(np.split(targets, 5), np.split(preds, 5))
            expected = [balanced_accuracy_score(targets[:, i], preds[:, i]) for i in range(3)]
            np.testing.assert_allclose([C1, C2, C3], expected)
            self.assertAlmostEqual(avg, np.mean(expected))

    def test_mean_recall_matches_sklearn(self):
        targets, probs = self.random_labels(0)
        preds = np.round(probs)
        expected = np.mean([recall_score(targets[:, i], preds[:, i]) for i in range(3)])
        self.assertAlmostEqual(get_mean_recall([targets], [preds]), expected)

    def test_map_skips_criterion_without_positives(self):
        targets = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=torch.float32)
        probs = torch.tensor([[.9, .2, .1], [.3, .8, .2], [.6, .7, .4], [.2, .1, .3]])
        C1, C2, C3, mAP = get_map([targets], [probs])
        self.assertTrue(math.isnan(C3))
        self.assertAlmostEqual(mAP, (C1 + C2) / 2)


class TestConfigs(unittest.TestCase):
    def test_default_config(self):
        config, name = load("SwinCVS_config.yaml")
        self.assertEqual(name, "SwinCVS_E2E_MC_IMNP_sd5")
        self.assertEqual(config.INFERENCE.BATCH_SIZE, 1)

    def test_frozen_config(self):
        config, name = load("SwinCVS_frozen_config.yaml")
        self.assertEqual(name, "SwinCVS_frozen_ENDP_sd5")

    def test_inference_and_finetune_configs_load(self):
        for name in ["infer.yaml", "infer_safe.yaml", "SwinCVS_safe_finetune_e2e.yaml", "SwinCVS_safe_finetune_frozen.yaml"]:
            get_config(str(CONFIGS / name), mode="inference")

    def test_test_batch_size(self):
        dataset = torch.utils.data.TensorDataset(torch.zeros(8, 1), torch.zeros(8, 3))
        config, _ = load("SwinCVS_config.yaml")
        self.assertEqual(get_dataloaders(config, dataset, dataset, dataset)[2].batch_size, 1)
        frozen, _ = load("SwinCVS_frozen_config.yaml")
        self.assertEqual(get_dataloaders(frozen, dataset, dataset, dataset)[2].batch_size, 4)
        # configs written before INFERENCE.BATCH_SIZE existed fall back to 1
        old = config.clone()
        old.defrost()
        del old["INFERENCE"]
        self.assertEqual(get_dataloaders(old, dataset, dataset, dataset)[2].batch_size, 1)


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = load("SwinCVS_config.yaml")
        cls.x = torch.randn(1, 5, 3, 384, 384)

    def test_e2e_multiclassifier_outputs(self):
        model = build_model(self.config).eval()
        with torch.no_grad():
            outputs_swin, outputs_lstm = model(self.x)
        self.assertEqual(tuple(outputs_swin.shape), (1, 3))
        self.assertEqual(tuple(outputs_lstm.shape), (1, 3))

    def test_frozen_backbone_has_no_trainable_parameters(self):
        model = build_model(with_model(self.config, E2E=False))
        self.assertFalse(any(p.requires_grad for p in model.swinv2_model.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.lstm.parameters()))

    def test_backbone_only_outputs(self):
        model = build_model(with_model(self.config, LSTM=False)).eval()
        with torch.no_grad():
            self.assertEqual(tuple(model(self.x[0]).shape), (5, 3))

    def test_checkpoints_load_into_inference_model(self):
        # weights saved by SwinCVS.py must load strictly into inference.py's model
        inference_config = get_config(str(CONFIGS / "infer.yaml"), mode="inference")
        inference_config.defrost()
        inference_config.MODEL.INFERENCE = True
        for e2e in (True, False):
            inference_config.MODEL.E2E = e2e
            trained = build_model(with_model(self.config, E2E=e2e, INFERENCE=True))
            build_inference_model(inference_config).load_state_dict(trained.state_dict(), strict=True)

    def test_gradient_checkpointing_gives_same_gradients(self):
        x = torch.randn(2, 3, 64, 64)
        grads = []
        for use_checkpoint in (False, True):
            torch.manual_seed(0)
            model = SwinTransformerV2(img_size=64, embed_dim=16, depths=[1, 1, 2, 1], num_heads=[1, 2, 2, 4],
                                      window_size=4, num_classes=3, drop_path_rate=0.0, use_checkpoint=use_checkpoint)
            model(x).sum().backward()
            grads.append([p.grad for p in model.parameters()])
        for a, b in zip(*grads):
            torch.testing.assert_close(a, b)


class TestScripts(unittest.TestCase):
    """Runs inference.py and train_safe.py end-to-end on synthetic data."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = cls.root = Path(cls.tmp.name)
        rng = np.random.default_rng(0)

        def save_image(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 255, (120, 214, 3), dtype=np.uint8)).resize((854, 480)).save(path)

        # SAFE format: 3 videos x 12 frames, keyframes at 3 (no full 5-frame window, skipped), 5 and 10
        rows = []
        for v in range(3):
            vid = f"vid{v}"
            for f in range(1, 13):
                frame = f"{f:04d}"
                keyframe = f in (3, 5, 10)
                labels = [1.0, 0.0, 1.0] if keyframe and f == 5 else [0.0, 1.0, 0.0] if keyframe else [np.nan] * 3
                rows.append(dict(vid_id=f"id{v}", vid=vid, frame=frame, is_ds_keyframe=keyframe,
                                 C1=labels[0], C2=labels[1], C3=labels[2]))
                save_image(root / "SAFE/images/1fps" / vid / f"{vid}_{frame}.jpg")
        metadata = pd.DataFrame(rows)
        (root / "SAFE/labels/1fps/splits").mkdir(parents=True)
        metadata.to_csv(root / "SAFE/labels/1fps/metadata.csv", index=False)
        for split, vid in [("train", "vid0"), ("val", "vid1"), ("test", "vid2")]:
            metadata[metadata.vid == vid][["vid_id", "vid", "frame"]].to_csv(
                root / f"SAFE/labels/1fps/splits/{split}.csv", index=False)

        # Endoscapes-format csv: 2 videos x 10 frames in blocks of 5
        endo_rows = []
        for v in (7, 8):
            for f in range(5, 55, 5):
                endo_rows.append(dict(vid=v, frame=f, C1=float(f % 2), C2=float(f % 3 == 0), C3=float(f % 4 == 0)))
                save_image(root / "endoscapes" / f"{v}_{f:04d}.jpg")
        pd.DataFrame(endo_rows).to_csv(root / "endoscapes.csv", index=False)

        # random weights shaped like the released checkpoints
        config = get_config(str(CONFIGS / "infer.yaml"), mode="inference")
        config.defrost()
        config.MODEL.INFERENCE = False
        for mode, e2e in [("e2e", True), ("frozen", False)]:
            config.MODEL.E2E = e2e
            torch.manual_seed(0)
            torch.save(build_inference_model(config).state_dict(), root / f"{mode}.pt")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_script(self, *args):
        result = subprocess.run([sys.executable, *args], cwd=REPO, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=result.stdout[-3000:] + result.stderr[-3000:])
        return result.stdout

    def safe_inference(self, mode, weights, output_dir, *extra):
        return self.run_script(
            "inference.py", "--dataset", "safe", "--config_path", "config/infer_safe.yaml", "--mode", mode,
            "--weights", str(weights), "--output_dir", str(output_dir),
            "--csv_path", str(self.root / "SAFE/labels/1fps/metadata.csv"),
            "--video_root", str(self.root / "SAFE/images/1fps"), *extra)

    def test_inference_safe(self):
        out = self.root / "out_safe"
        self.safe_inference("e2e", self.root / "e2e.pt", out, "--warmup", "1", "--sync_mode", "video")
        result = pd.read_csv(out / "result.csv")
        self.assertEqual(len(result), 6)  # 2 usable keyframes x 3 videos
        self.assertTrue(result[["Conf_C1", "Conf_C2", "Conf_C3"]].apply(lambda c: c.between(0, 1)).all().all())
        self.assertTrue((out / "throughput/videosync.csv").is_file())
        self.assertTrue((out / "resources/videosync/peak.csv").is_file())

    def test_inference_safe_frame_sync_throughput_only(self):
        out = self.root / "out_safe_frozen"
        self.safe_inference("frozen", self.root / "frozen.pt", out, "--sync_mode", "frame", "--throughput_only")
        self.assertEqual(len(pd.read_csv(out / "throughput/framesync.csv")), 6)
        self.assertFalse((out / "result.csv").exists())

    def test_inference_endoscapes_format(self):
        out = self.root / "out_endoscapes"
        self.run_script(
            "inference.py", "--dataset", "endoscapes", "--config_path", "config/infer.yaml", "--mode", "e2e",
            "--weights", str(self.root / "e2e.pt"), "--output_dir", str(out),
            "--csv_path", str(self.root / "endoscapes.csv"), "--images_path", str(self.root / "endoscapes"))
        self.assertEqual(len(pd.read_csv(out / "result.csv")), 4)  # 20 rows / blocks of 5
        self.assertTrue((out / "metrics.json").is_file())

    def test_finetune_frozen_then_inference(self):
        out = self.root / "finetune"
        self.run_script(
            "train_safe.py", "--config", "config/SwinCVS_safe_finetune_frozen.yaml",
            "--data_root", str(self.root / "SAFE"), "--init_weights", str(self.root / "frozen.pt"),
            "--output_dir", str(out), "--device", "cpu", "--no_wandb", "--epochs", "1", "--batch_size", "1",
            "--val_batch_size", "2", "--num_workers", "0", "--eval_test")
        run_dir = next(out.iterdir())
        for name in ["best.pt", "last.pt", "results.json", "test_metrics.json", "test_result.csv"]:
            self.assertTrue((run_dir / name).is_file(), name)
        # the fine-tuned checkpoint loads back into inference.py
        self.safe_inference("frozen", run_dir / "best.pt", self.root / "out_finetuned")

    def test_finetune_e2e_last_layers(self):
        output = self.run_script(
            "train_safe.py", "--config", "config/SwinCVS_safe_finetune_e2e.yaml",
            "--data_root", str(self.root / "SAFE"), "--init_weights", str(self.root / "e2e.pt"),
            "--output_dir", str(self.root / "finetune_e2e"), "--device", "cpu", "--no_wandb", "--epochs", "1",
            "--batch_size", "1", "--val_batch_size", "2", "--num_workers", "0", "--train_last_layers", "1")
        self.assertIn("the last 1 of 28 backbone layers", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
