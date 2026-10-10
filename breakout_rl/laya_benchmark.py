"""Reproducible reference/optimized HTTP comparison on one resident CUDA model."""

import ctypes
import hashlib
import json
import platform
import subprocess
import time
import uuid
from threading import Thread
from urllib.request import Request, urlopen
from urllib.parse import urlsplit

import numpy as np

from breakout_rl.laya_server import LayaDecisionService, create_laya_server
from breakout_rl.laya_vision_agent import load_laya_agent, LAYA_CODE_REVISION, LAYA_MODEL_REVISION


def process_memory():
    """Working set/private bytes on Windows; CUDA allocator is reported separately."""
    import torch
    result = {"cuda_allocated": torch.cuda.memory_allocated(),
              "cuda_reserved": torch.cuda.memory_reserved()}
    if platform.system() == "Windows":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in ("peak", "rss", "pool_peak", "pool", "nonpool_peak",
                                                    "nonpool", "pagefile", "pagefile_peak", "private")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32")
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        result.update(rss=counters.rss, private=counters.private)
    return result


def fixture_frames():
    """30 native RGB states from a deterministic raw-ALE action replay, not policy evaluation."""
    from breakout_env import make_breakout_raw_env
    frames, metadata, seen = [], [], set()
    for seed in (101, 202, 303):
        env = make_breakout_raw_env(render_mode="rgb_array")
        try:
            env.reset(seed=seed)
            rng = np.random.default_rng(seed)
            targets = (1, 60, 120, 180, 240, 360, 480, 720, 960, 1200)
            captured = 0
            for tick in range(1, 1201):
                action = 1 if tick % 120 == 1 else int(rng.integers(0, 4))
                _, _, terminated, truncated, _ = env.step(action)
                if captured < len(targets) and tick >= targets[captured]:
                    rgb = np.asarray(env.unwrapped.ale.getScreenRGB(), dtype=np.uint8).tobytes()
                    digest = hashlib.sha256(rgb).hexdigest()
                    if digest not in seen:
                        seen.add(digest)
                        captured += 1
                        frames.append(rgb)
                        metadata.append({"seed": seed, "replay_tick": tick,
                                         "ale_frame": env.unwrapped.ale.getFrameNumber(),
                                         "lives": env.unwrapped.ale.lives(), "sha256": digest})
                if terminated or truncated:
                    env.reset(seed=seed + tick)
        finally:
            env.close()
    if len(frames) != 30:
        raise RuntimeError("Fixture replay did not produce 30 distinct RGB states")
    return frames, metadata


def run_laya_benchmark(*, samples=100, rounds=3, stress=1000, url="http://127.0.0.1:8766"):
    import torch
    if samples < 100 or rounds < 3 or stress < 1000:
        raise ValueError("Acceptance benchmark requires >=100 samples, >=3 rounds and >=1000 stress requests")
    target = urlsplit(url)
    if target.scheme != "http" or target.hostname not in ("127.0.0.1", "localhost") or target.path not in ("", "/") or target.query or target.fragment:
        raise ValueError("Benchmark URL must be a local HTTP server/proxy origin")
    run_id = uuid.uuid4().hex
    frames, fixtures = fixture_frames()
    agent = load_laya_agent("cuda", preprocess="gpu")
    runtime = {"device": str(agent.device), "modelRevision": LAYA_MODEL_REVISION,
               "codeRevision": LAYA_CODE_REVISION, "benchmarkRunId": run_id}
    reference = LayaDecisionService(agent, runtime)
    optimized = LayaDecisionService(agent, runtime, optimized=True)
    optimized.policy.render_rgb = lambda: reference.frame
    server = create_laya_server(reference)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # Keep separate original handles because the server shares the reference service.
    original_policy, original_runtime = reference.policy, reference.runtime
    records = []

    def use(mode):
        reference.policy, reference.runtime = ((original_policy, original_runtime) if mode == "reference"
                                               else (optimized.policy, optimized.runtime))

    def request(frame):
        req = Request(url.rstrip("/") + "/api/laya/predict", data=frame, headers={
            "Content-Type": "application/octet-stream", "Origin": "http://127.0.0.1:5180"})
        started = time.perf_counter()
        with urlopen(req, timeout=30) as response:
            out = json.load(response)
        if out.get("benchmarkRunId") != run_id or out.get("inferenceMode") != reference.runtime["inferenceMode"]:
            raise RuntimeError("Benchmark URL reached another service or the wrong inference mode")
        return out, (time.perf_counter() - started) * 1000

    try:
        for mode in ("reference", "optimized"):
            use(mode)
            for i in range(20):
                request(frames[i % len(frames)])
        # Diagnostic CUDA events are measured separately from production A/B latency.
        use("optimized")
        optimized.policy.diagnostics = True
        profiles = [request(frame)[0]["timings"] for frame in frames]
        optimized.policy.diagnostics = False
        parity = []
        for frame in frames:
            use("reference")
            expected, _ = request(frame)
            use("optimized")
            actual, _ = request(frame)
            error = max(abs(a - b) for a, b in zip(expected["probabilities"], actual["probabilities"]))
            parity.append({"action_match": expected["actionIndex"] == actual["actionIndex"],
                           "probability_error": error})
        for round_index in range(rounds):
            for i in range(samples):
                for mode in (("reference", "optimized") if (i + round_index) % 2 == 0
                             else ("optimized", "reference")):
                    use(mode)
                    out, elapsed = request(frames[i % len(frames)])
                    records.append({"round": round_index, "mode": mode, "roundtrip_ms": elapsed,
                                    "inference_ms": out["inferenceMs"]})
            print(f"Finished A/B round {round_index + 1}/{rounds}", flush=True)
        use("optimized")
        torch.cuda.reset_peak_memory_stats()
        memory = [process_memory()]
        for i in range(stress):
            request(frames[i % len(frames)])
            if (i + 1) % 100 == 0:
                memory.append(process_memory())
                print(f"Memory stress {i + 1}/{stress}", flush=True)
        memory.append(process_memory())
        summary = []
        for r in range(rounds):
            for mode in ("reference", "optimized"):
                rows = [row for row in records if row["round"] == r and row["mode"] == mode]
                summary.append({"round": r, "mode": mode, **{
                    field: dict(zip(("p50", "p95", "p99"), np.percentile(
                        [row[field] for row in rows], (50, 95, 99)).tolist()))
                    for field in ("roundtrip_ms", "inference_ms")}})
        passes = all(p["action_match"] and p["probability_error"] <= 0.0002 for p in parity)
        for r in range(rounds):
            base, fast = summary[2 * r:2 * r + 2]
            passes &= (fast["roundtrip_ms"]["p50"] <= base["roundtrip_ms"]["p50"] * .8
                       and fast["roundtrip_ms"]["p95"] <= base["roundtrip_ms"]["p95"] * 1.05)
        return {"passed": bool(passes), "scope": "HTTP endpoint; excludes browser ALE/render/scheduling",
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(), "cpu_threads": torch.get_num_threads(),
                "platform": platform.platform(), "url": url, "samples_per_round": samples,
                "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
                "dtype": str(next(agent.model.parameters()).dtype), "preprocessing": agent.prep.to_config(),
                "model_revision": LAYA_MODEL_REVISION, "code_revision": LAYA_CODE_REVISION,
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
                "fixtures": fixtures, "parity": parity, "profiles": profiles,
                "summary": summary, "samples": records,
                "stress_requests": stress, "memory": memory,
                "cuda_peak_allocated": torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved": torch.cuda.max_memory_reserved()}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
