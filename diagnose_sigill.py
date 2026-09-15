#!/usr/bin/env python3
"""
SIGILL (Illegal instruction) diagnostic for the AICoverGen + rvc stack.

The "Illegal instruction (core dumped)" error almost always means a
binary was compiled with CPU instructions the host CPU doesn't have
(AVX2 / AVX-512 / FMA / etc.) OR an onnxruntime-gpu / torch binary
built for a CUDA version that doesn't match the host driver.

This script:
  1. Prints the CPU model + which instruction sets it supports
    (AVX, AVX2, AVX-512, FMA, F16C, SSE4_*).
  2. Reports the NVIDIA driver + CUDA version (if `nvidia-smi`
    is available).
  3. Runs `python -c "import <module>"` for each suspect module
    in a SEPARATE subprocess and reports the exit code.
    Exit code 0   → import OK
    Exit code 132 → SIGILL (128 + signal 4) — module crashed
                    with Illegal instruction.
    Exit code 139 → SIGSEGV — module crashed with segfault
                    (often the same root cause as SIGILL).
    Other         → import raised an exception with that exit
                    code (less common but still informative).

At the end it prints a summary and, if any module crashed, suggests
the most likely fix (downgrade that package to a version compiled
for older CPUs / a different CUDA version).

Usage:
    python diagnose_sigill.py

    # or from this script's path:
    /home/z/.venv/bin/python3.12 /home/z/my-project/scripts/diagnose_sigill.py
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


SUSPECT_MODULES = [
    # Order matters — test heavy ML modules from least-likely to
    # most-likely to crash.
    ("numpy",              "fundamental; almost never crashes"),
    ("scipy",              ">=1.15 requires AVX2 on x86_64 wheels"),
    ("torch",              "latest default build is CUDA 13; needs AVX2"),
    ("torchaudio",         "depends on torch; same AVX/CUDA concerns"),
    ("torchvision",        "depends on torch; same AVX/CUDA concerns"),
    ("triton",             "native code; can use AVX-512 / FMA"),
    ("onnxruntime",        "check CUDA version on .so files"),
    ("onnxruntime.capi.onnxruntime_pybind11_state",
                            "actual onnxruntime native binding — crashes here are SIGILL"),
    ("faiss",              "faiss-cpu wheels are AVX2-only"),
    ("librosa",            "uses numba (LLVM JIT) — can SIGILL on weird CPUs"),
    ("numba",              "librosa's JIT compiler backend"),
    ("transformers",       "uses tokenizers native extensions"),
    ("parselmouth",        "parselmouth Sound engine (pip name is praat-parselmouth)"),
    ("soundfile",          "libsndfile wrapper"),
    ("einops",             "pure-Python — should never crash"),
    ("omegaconf",          "pure-Python — should never crash"),
    ("rvc",                "the modern uziproj/rvc package"),
    ("rvc_adapter",        "AICoverGen's adapter (needs src/ on path)"),
]

SUSPECT_BINARIES = [
    # (path-glob relative to site-packages, what to look for)
    ("torch/lib/libtorch_cpu.so", "torch native lib (AVX-512 checks)"),
    ("torch/lib/libtorch_cuda.so", "torch CUDA lib"),
    ("onnxruntime/capi/onnxruntime_providers_cuda.so", "onnxruntime CUDA EP"),
    ("onnxruntime/capi/onnxruntime_pybind11_state.so",
        "onnxruntime pybind11 state — crashes here are SIGILL"),
    ("triton/_C/libtriton.so", "triton native compiler"),
    ("faiss/_swigfaiss.so", "faiss-cpu native binding (AVX2 only)"),
    ("scipy/_lib/_ccallback_c.so", "scipy native helper"),
]


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def section(title: str) -> None:
    print()
    print(f"--- {title} ---")


def get_cpu_info() -> dict:
    """Return {model_name, flags_set} from /proc/cpuinfo."""
    info = {"model_name": "(unknown)", "flags": set()}
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name") and info["model_name"] == "(unknown)":
                    info["model_name"] = line.split(":", 1)[1].strip()
                elif line.startswith("flags"):
                    info["flags"] = set(line.split(":", 1)[1].split())
                if info["model_name"] != "(unknown)" and info["flags"]:
                    break
    except Exception as e:
        info["model_name"] = f"(error: {e})"
    return info


def get_cuda_info() -> dict:
    """Return {driver_version, cuda_version} from nvidia-smi (if any)."""
    info = {"driver_version": "(no nvidia-smi)", "cuda_version": "(no nvidia-smi)"}
    smi = shutil.which("nvidia-smi")
    if not smi:
        return info
    try:
        out = subprocess.run([smi], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"Driver Version:\s*([\d.]+)", out)
        if m:
            info["driver_version"] = m.group(1)
        m = re.search(r"CUDA Version:\s*([\d.]+)", out)
        if m:
            info["cuda_version"] = m.group(1)
    except Exception as e:
        info["driver_version"] = f"(error: {e})"
    return info


def get_python_info() -> dict:
    """Return {executable, version, sys_path[:3]} for the current python."""
    return {
        "executable": sys.executable,
        "version": sys.version.split()[0],
        "platform": sys.platform,
    }


def diagnose_cpu(cpu: dict) -> list:
    """Return a list of warnings about missing instruction sets."""
    flags = cpu["flags"]
    warnings = []

    # Critical for modern ML wheels.
    if "avx" not in flags:
        warnings.append("CPU lacks AVX — almost no modern ML wheel will work.")
    if "avx2" not in flags:
        warnings.append(
            "CPU lacks AVX2 — scipy >= 1.15, torch 2.x, faiss-cpu, and "
            "numba/librosa will SIGILL on import."
        )
    if "fma" not in flags:
        warnings.append("CPU lacks FMA — some torch kernels may SIGILL.")
    if "sse4_2" not in flags:
        warnings.append("CPU lacks SSE4.2 — extremely old; almost nothing will work.")

    # AVX-512 — nice to have but not strictly required.
    avx512_variants = [f for f in flags if f.startswith("avx512")]
    if not avx512_variants:
        # AVX-512 is optional; just note it (not a warning).
        pass

    return warnings


def diagnose_cuda(cuda: dict) -> list:
    """Return warnings about CUDA / driver mismatches."""
    warnings = []
    if cuda["cuda_version"] == "(no nvidia-smi)":
        return warnings  # CPU-only machine — that's fine
    try:
        cv = float(cuda["cuda_version"])
    except ValueError:
        return warnings

    # The rvc package pulls torch 2.14+cu130 + onnxruntime-gpu 1.30
    # by default — both target CUDA 13.
    if cv < 13.0:
        warnings.append(
            f"Host CUDA is {cv} but the default rvc install pulls CUDA-13 "
            f"builds of torch and onnxruntime-gpu. This causes "
            f"'Illegal instruction (core dumped)' when onnxruntime "
            f"tries to load CUDA-13 libraries against a CUDA-{cv} driver. "
            f"FIX: pin torch==2.5.1 + onnxruntime-gpu==1.20.1 (CUDA 12.x)."
        )
    return warnings


def import_module_in_subprocess(module: str, hint: str,
                                  python_exec: str,
                                  extra_paths: list = None) -> dict:
    """Run `python -c "import <module>"` in a subprocess.

    Returns {module, exit_code, signal, stdout, stderr}.
    """
    env = dict(os.environ)
    if extra_paths:
        env["PYTHONPATH"] = os.pathsep.join(extra_paths) + os.pathsep + env.get("PYTHONPATH", "")

    code = "import " + module + "; print('OK')"
    try:
        result = subprocess.run(
            [python_exec, "-c", code],
            capture_output=True, text=True, timeout=60, env=env,
        )
        exit_code = result.returncode
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
    except subprocess.TimeoutExpired:
        return {
            "module": module,
            "exit_code": -1,
            "signal": None,
            "stdout": "",
            "stderr": "(timeout after 60s — likely stuck downloading model files)",
        }

    # Decode the signal for exit codes > 128.
    signal = None
    if exit_code > 128:
        signal = exit_code - 128
    return {
        "module": module,
        "exit_code": exit_code,
        "signal": signal,
        "stdout": stdout,
        "stderr": stderr,
    }


def main() -> int:
    banner("AICoverGen + rvc — SIGILL Diagnostic")
    print()
    print("This script tests each suspect ML module's import in a separate")
    print("subprocess and reports which one (if any) crashes with")
    print("'Illegal instruction (core dumped)' (exit code 132 = SIGILL).")

    # ----- Environment info --------------------------------------------
    section("Environment")
    py = get_python_info()
    print(f"Python executable : {py['executable']}")
    print(f"Python version    : {py['version']}")
    print(f"Platform          : {py['platform']}")

    cpu = get_cpu_info()
    print(f"CPU model         : {cpu['model_name']}")
    print(f"# CPU flags       : {len(cpu['flags'])}")

    # Which instruction sets are present.
    relevant_flags = sorted([
        f for f in cpu["flags"]
        if f.startswith(("sse", "avx", "fma", "f16c"))
    ])
    print(f"Instruction sets  : {', '.join(relevant_flags)}")

    cuda = get_cuda_info()
    print(f"NVIDIA driver     : {cuda['driver_version']}")
    print(f"CUDA version      : {cuda['cuda_version']}")

    # ----- Diagnose CPU -----------------------------------------------
    section("CPU analysis")
    warnings = diagnose_cpu(cpu)
    if not warnings:
        print("OK — CPU has all the instruction sets modern ML wheels need")
        print("     (AVX, AVX2, FMA, SSE4.2). SIGILL on this CPU is")
        print("     almost certainly NOT a CPU-instruction-set issue.")
    else:
        for w in warnings:
            print(f"  ⚠ {w}")

    # ----- Diagnose CUDA -----------------------------------------------
    section("CUDA analysis")
    warnings = diagnose_cuda(cuda)
    if not warnings:
        if cuda["cuda_version"] == "(no nvidia-smi)":
            print("OK — no NVIDIA GPU detected; running in CPU-only mode.")
            print("     (onnxruntime-gpu will fall back to CPU execution)")
        else:
            print(f"OK — CUDA {cuda['cuda_version']} is compatible with the pinned versions.")
    else:
        for w in warnings:
            print(f"  ⚠ {w}")

    # ----- Per-module import test -------------------------------------
    banner("Per-module import test (each in a fresh subprocess)")
    print(f"{'module':<48} {'exit':>5}  {'signal':<8}  result")
    print("-" * 90)

    crashed = []
    aico_vergen_src = str(Path("/home/z/my-project/AICoverGen/src").resolve())
    extra_paths = []
    if Path(aico_vergen_src).is_dir():
        extra_paths.append(aico_vergen_src)

    for module, hint in SUSPECT_MODULES:
        r = import_module_in_subprocess(
            module, hint, sys.executable, extra_paths=extra_paths,
        )
        ec = r["exit_code"]
        sig = r["signal"]
        if ec == 0:
            status = "OK"
        elif sig == 4:
            status = "SIGILL — " + hint
            crashed.append((module, "SIGILL", hint))
        elif sig == 11:
            status = "SIGSEGV — " + hint
            crashed.append((module, "SIGSEGV", hint))
        elif ec == -1:
            status = "TIMEOUT"
        else:
            # Import raised an exception with non-zero exit code.
            err = r["stderr"].split("\n")[-1][:80] if r["stderr"] else "?"
            status = f"exit {ec} — {err}"

        sig_str = f"SIG{sig}" if sig else "-"
        print(f"{module:<48} {ec:>5}  {sig_str:<8}  {status}")

    # ----- Summary -----------------------------------------------------
    banner("Summary & recommended fix")
    if not crashed:
        print("All suspect modules imported cleanly. If you still see")
        print("'Illegal instruction (core dumped)', it happens at RUNTIME")
        print("(not at import time). The most likely cause is one of the")
        print("native binaries in the SUSPECT_BINARIES list — try running")
        print("the actual AICoverGen command under strace to find which.")
        print()
        print("Also try:")
        print("  1. `pip install --upgrade --force-reinstall torch==2.5.1 \\")
        print("         torchvision==0.20.1 torchaudio==2.5.1 \\")
        print("         onnxruntime-gpu==1.20.1`")
        print("  2. Re-run this diagnostic.")
        return 0

    print(f"{len(crashed)} module(s) crashed:")
    for module, kind, hint in crashed:
        print(f"  • {module} — {kind} ({hint})")

    print()
    print("Recommended fix (in priority order):")
    print()
    print("1. PIN torch + onnxruntime to CUDA-12-compatible versions.")
    print("   The default rvc install pulls torch 2.14+cu130 and")
    print("   onnxruntime-gpu 1.30 — both target CUDA 13 and crash on")
    print("   CUDA 12.x drivers (the Colab / common-user default).")
    print()
    print("   Edit AICoverGen/requirements.txt and put these pins BEFORE")
    print("   the `git+https://github.com/uziproj/rvc.git` line:")
    print()
    print("       torch==2.5.1")
    print("       torchvision==0.20.1")
    print("       torchaudio==2.5.1")
    print("       onnxruntime-gpu==1.20.1")
    print()
    print("   Then `pip install -r requirements.txt` again.")
    print()
    print("2. If your CPU lacks AVX2 (very old / pre-2013), no modern")
    print("   ML wheel will work. Run AICoverGen on a newer machine")
    print("   or upgrade your CPU.")
    print()
    print("3. If you don't need GPU support at all, switch to CPU-only")
    print("   onnxruntime (frees you from CUDA driver issues):")
    print()
    print("       pip uninstall -y onnxruntime-gpu")
    print("       pip install onnxruntime  # CPU build")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
