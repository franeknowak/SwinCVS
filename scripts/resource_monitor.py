"""Background sampler for GPU-only memory and power draw during inference.

Shared (copied) across the cvs, CVS-AdaptNet, SwinCVS and endoscapes repos so
all of them report resources the same way; see ResourceMonitor.
"""

import contextlib
import os
import re
import shutil
import subprocess
import sys
import threading
import time

import torch

try:
    import psutil
except ImportError:
    psutil = None


class ResourceMonitor:
    """Background sampler for GPU-only memory and power draw during a run.

    Memory is read from the GPU allocator's own counters, so it reflects
    only what this process actually holds on the GPU, not system-wide RAM:
    - CUDA: torch.cuda.memory_allocated / max_memory_allocated — including
      on Jetson's unified memory, where the GPU and CPU share physical DRAM
      but CUDA allocations are still tracked separately by the allocator.
    - MPS (Apple silicon, also unified memory): torch.mps.current_allocated_memory.
      MPS has no peak counter, so the peak is the max of the background
      samples and of the per-module readings taken under track_peak (used
      during warmup), which catch the activation peak inside a forward pass.
    On a CPU-only run, memory falls back to the process RSS via psutil
    (there is no GPU to isolate).

    Power backend is auto-detected:
    - tegrastats (Jetson, incl. Orin Nano / JetPack 7.2): Jetson has no
      NVML-backed nvidia-smi. Orin Nano's tegrastats exposes no rail for
      the GPU alone — VDD_IN is total module power (CPU+GPU+everything
      else) and VDD_CPU_GPU_CV lumps the GPU in with the CPU and the deep
      learning accelerator cores. VDD_CPU_GPU_CV is used here as the
      closest available proxy to GPU-only power; some carrier boards may
      not expose it, in which case power stays empty.
    - nvidia-smi (desktop/server GPUs): power comes from nvidia-smi's
      power.draw, which is genuinely GPU-only (the discrete card's own
      power rail).
    - powermetrics (macOS): the "GPU Power" line of its gpu_power sampler,
      GPU-only but estimated by macOS from the SoC's energy model rather
      than measured on a rail. powermetrics needs root, so it runs through
      `sudo -n` (non-interactive) unless the process is already root; allow
      it without a password via sudoers, e.g.
      `<user> ALL=(root) NOPASSWD: /usr/bin/powermetrics`. If sudo refuses,
      a warning is printed and power stays empty.

    If no backend is available, power samples stay empty.
    """

    _TEGRASTATS_POWER_RE = re.compile(r"VDD_CPU_GPU_CV (\d+)mW")
    _POWERMETRICS_POWER_RE = re.compile(r"GPU Power: (\d+(?:\.\d+)?) mW")

    def __init__(self, device: torch.device, interval: float = 0.2):
        self.device = device
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None
        self._start_time = None
        self.samples = []  # list of {"time_s", "memory_mb", "power_w"}
        self._tracked_peak_mb = None
        self._powermetrics = None
        self._powermetrics_fd = None
        self._powermetrics_thread = None
        self._powermetrics_power_w = None
        self._backend = self._detect_power_backend()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _detect_power_backend(self):
        if shutil.which("tegrastats"):
            return "tegrastats"
        if shutil.which("nvidia-smi"):
            return "nvidia-smi"
        if sys.platform == "darwin" and shutil.which("powermetrics"):
            if self._check_powermetrics():
                return "powermetrics"
        return None

    def _powermetrics_cmd(self, *args):
        cmd = ["powermetrics", "--samplers", "gpu_power", *args]
        return cmd if os.geteuid() == 0 else ["sudo", "-n", *cmd]

    def _check_powermetrics(self):
        """One short powermetrics sample, to fail early (and say why) when sudo
        needs a password or the output has no GPU Power line."""
        try:
            result = subprocess.run(
                self._powermetrics_cmd("-n", "1", "-i", "100"),
                capture_output=True, text=True, timeout=10,
            )
        except Exception as e:
            print(f"ResourceMonitor: powermetrics unavailable ({e}); power will be empty.")
            return False
        if result.returncode != 0 or not self._POWERMETRICS_POWER_RE.search(result.stdout):
            reason = result.stderr.strip() or "no 'GPU Power' line in its output"
            print(
                f"ResourceMonitor: powermetrics unavailable ({reason}); power will be empty. "
                "Allow it without a password, e.g. sudoers: "
                "<user> ALL=(root) NOPASSWD: /usr/bin/powermetrics"
            )
            return False
        return True

    def _start_powermetrics(self):
        import pty  # POSIX only; powermetrics is macOS only

        # powermetrics block-buffers when writing to a pipe, which would delay
        # readings by seconds; a pseudo-terminal makes it flush every line.
        master_fd, slave_fd = pty.openpty()
        try:
            self._powermetrics = subprocess.Popen(
                self._powermetrics_cmd("-i", str(max(int(self.interval * 1000), 100))),
                stdin=subprocess.DEVNULL, stdout=slave_fd, stderr=subprocess.DEVNULL,
            )
        finally:
            os.close(slave_fd)
        self._powermetrics_fd = master_fd
        self._powermetrics_thread = threading.Thread(target=self._read_powermetrics, daemon=True)
        self._powermetrics_thread.start()

    def _read_powermetrics(self):
        """Keep the most recent GPU Power reading for _run to pick up."""
        with open(self._powermetrics_fd, "r", errors="replace", closefd=False) as stream:
            try:
                for line in stream:
                    power_match = self._POWERMETRICS_POWER_RE.search(line)
                    if power_match:
                        self._powermetrics_power_w = float(power_match.group(1)) / 1000.0
            except OSError:  # the pty closes when powermetrics exits
                pass

    def _stop_powermetrics(self):
        if self._powermetrics is None:
            return
        self._powermetrics.terminate()
        try:
            self._powermetrics.wait(timeout=5)
        except Exception:
            self._powermetrics.kill()
        self._powermetrics_thread.join(timeout=5)
        os.close(self._powermetrics_fd)
        self._powermetrics = None

    def _sample_tegrastats_power(self):
        try:
            proc = subprocess.Popen(
                ("tegrastats", "--interval", "100"),
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
            try:
                out = proc.stdout.readline()
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
        except Exception:
            return None
        power_match = self._TEGRASTATS_POWER_RE.search(out)
        return int(power_match.group(1)) / 1000.0 if power_match else None

    def _sample_nvidia_smi_power(self):
        try:
            out = subprocess.check_output(
                ("nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"),
                text=True, stderr=subprocess.DEVNULL, timeout=2,
            )
            return float(out.strip().splitlines()[0])
        except Exception:
            return None

    def _sample_memory_mb(self):
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        if self.device.type == "mps":
            return torch.mps.current_allocated_memory() / (1024 ** 2)
        if psutil is not None:
            return psutil.Process().memory_info().rss / (1024 ** 2)
        return None

    def _run(self):
        while not self._stop_event.is_set():
            mem_mb = self._sample_memory_mb()
            if self._backend == "tegrastats":
                power_w = self._sample_tegrastats_power()
            elif self._backend == "nvidia-smi":
                power_w = self._sample_nvidia_smi_power()
            elif self._backend == "powermetrics":
                power_w = self._powermetrics_power_w
            else:
                power_w = None
            self.samples.append({
                "time_s": time.perf_counter() - self._start_time,
                "memory_mb": mem_mb,
                "power_w": power_w,
            })
            self._stop_event.wait(self.interval)

    def _record_peak(self, *_):
        mem_mb = self._sample_memory_mb()
        if mem_mb is not None and (self._tracked_peak_mb is None or mem_mb > self._tracked_peak_mb):
            self._tracked_peak_mb = mem_mb

    @contextlib.contextmanager
    def track_peak(self, model):
        """On MPS, read allocated memory after every submodule's forward while
        active, standing in for CUDA's exact peak counter. The hooks add Python
        overhead to every forward, so wrap untimed passes (warmup) only.
        No-op on other devices."""
        if self.device.type != "mps":
            yield
            return
        hooks = [module.register_forward_hook(self._record_peak) for module in model.modules()]
        try:
            yield
        finally:
            for hook in hooks:
                hook.remove()

    def start(self):
        self._start_time = time.perf_counter()
        if self._backend == "powermetrics":
            self._start_powermetrics()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._stop_powermetrics()

    def peak_memory_mb(self):
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        memory_samples = [s["memory_mb"] for s in self.samples if s["memory_mb"] is not None]
        if self._tracked_peak_mb is not None:
            memory_samples.append(self._tracked_peak_mb)
        return max(memory_samples) if memory_samples else None

    def peak_power_w(self):
        power_samples = [s["power_w"] for s in self.samples if s["power_w"] is not None]
        return max(power_samples) if power_samples else None
