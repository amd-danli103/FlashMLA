"""
3-way performance benchmark for sparse MLA prefill kernels:
  1. Triton  (triton_mla_kernels_prefill_optimized.py)
  2. Tilelang (sglang tilelang_kernel.py -> tilelang_sparse_fwd)
  3. PyTorch reference (ref.py -> ref_sparse_attn_fwd)

This script constructs test cases that ALL THREE implementations can run,
enabling meaningful performance comparison. Tilelang on AMD/HIP requires:
  - topk == 2048
  - tail_dim > 0 (d_qk > d_v, e.g. d_qk=576, d_v=512)
  - contiguous input tensors
  - no attn_sink / topk_length in prefill path

Test cases include:
  - Tilelang-compatible correctness cases (d_qk=576, topk=2048)
  - Tilelang-compatible performance cases (d_qk=576, topk=2048)
  - MODEL1 performance cases (d_qk=512, tilelang skipped)
  - V3.2 performance cases (d_qk=576, topk=2048)

Usage:
    # Run all tests (correctness + performance):
    python benchmark_sparse_mla_prefill_3way_with_tilelang.py

    # Correctness only:
    python benchmark_sparse_mla_prefill_3way_with_tilelang.py --correctness-only

    # Performance only:
    python benchmark_sparse_mla_prefill_3way_with_tilelang.py --perf-only

    # Specific config:
    python benchmark_sparse_mla_prefill_3way_with_tilelang.py --config tilelang_compat
    python benchmark_sparse_mla_prefill_3way_with_tilelang.py --config model1_config1
    python benchmark_sparse_mla_prefill_3way_with_tilelang.py --config v32

    # CSV output:
    python benchmark_sparse_mla_prefill_3way_with_tilelang.py --csv results.csv
"""

import argparse
import csv
import os
import platform
import sys
import time
from datetime import datetime
from typing import Optional

import torch
import kernelkit as kk

import lib
import ref
from lib import TestParam
from triton_mla_kernels_prefill_optimized import (
    triton_sparse_attn_fwd_optimized as triton_sparse_attn_fwd,
)

# ---------------------------------------------------------------------------
# Tilelang import (best-effort)
# ---------------------------------------------------------------------------
_tilelang_available = False
_tilelang_import_error = ""
try:
    sys.path.insert(0, "/home/danli103/sglang/python")
    from sglang.srt.layers.attention.nsa.tilelang_kernel import (
        tilelang_sparse_fwd,
    )
    _tilelang_available = True
except Exception as e:
    _tilelang_import_error = str(e)

_counter = kk.Counter()


# ---------------------------------------------------------------------------
# Environment info (following benchmark_sparse_mla_decode_report.py)
# ---------------------------------------------------------------------------

def print_env_info():
    """Print environment and version info for reproducibility."""
    print(f"{'=' * 80}")
    print("ENVIRONMENT INFO")
    print(f"{'=' * 80}")

    # Python
    print(f"  Python:        {platform.python_version()} ({sys.executable})")

    # PyTorch
    print(f"  PyTorch:       {torch.__version__}")

    # CUDA / HIP
    hip_ver = getattr(torch.version, "hip", None)
    cuda_ver = getattr(torch.version, "cuda", None)
    if hip_ver:
        print(f"  ROCm (HIP):    {hip_ver}")
    elif cuda_ver:
        print(f"  CUDA:          {cuda_ver}")

    # GPU
    gpu_name = torch.cuda.get_device_name(0)
    gpu_arch = torch.cuda.get_device_capability(0)
    print(f"  GPU:           {gpu_name}")
    print(f"  GPU Arch:      gfx{gpu_arch[0]}{gpu_arch[1]}0"
          f" (compute capability {gpu_arch[0]}.{gpu_arch[1]})")
    gpu_mem = torch.cuda.get_device_properties(0).total_memory
    print(f"  GPU Memory:    {gpu_mem / 1024**3:.1f} GB")

    # Triton
    try:
        import triton
        print(f"  Triton:        {triton.__version__}")
    except ImportError:
        print("  Triton:        not installed")

    # TileLang
    try:
        import tilelang
        print(f"  TileLang:      {tilelang.__version__}")
    except ImportError:
        print("  TileLang:      not installed")

    # Visible devices
    hip_dev = os.environ.get(
        "HIP_VISIBLE_DEVICES",
        os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"),
    )
    print(f"  VISIBLE_DEVS:  {hip_dev}")

    # Hostname & date
    print(f"  Hostname:      {platform.node()}")
    print(f"  Date:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"{'=' * 80}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _can_run_tilelang(p: TestParam) -> tuple:
    """Check whether tilelang can run for the given TestParam.

    Returns (can_run: bool, skip_reason: str).
    """
    if not _tilelang_available:
        return False, f"import failed: {_tilelang_import_error}"
    if p.topk != 2048:
        return False, f"topk={p.topk} != 2048"
    tail_dim = p.d_qk - p.d_v
    if tail_dim <= 0:
        return False, f"tail_dim={tail_dim} <= 0 (d_qk={p.d_qk}, d_v={p.d_v})"
    if p.have_attn_sink:
        return False, "have_attn_sink not supported"
    if p.have_topk_length:
        return False, "have_topk_length not supported"
    if p.is_all_indices_invalid:
        return False, "is_all_indices_invalid not supported"
    return True, ""


def _run_triton(t, p):
    return triton_sparse_attn_fwd(
        t.q, t.kv, t.indices, t.sm_scale, p.d_v,
        t.attn_sink, t.topk_length,
    )


def _run_tilelang(t, p):
    """Wrap tilelang_sparse_fwd; ensures contiguous inputs."""
    out = tilelang_sparse_fwd(
        q=t.q.contiguous(),
        kv=t.kv.contiguous(),
        indices=t.indices.contiguous(),
        sm_scale=t.sm_scale,
        d_v=p.d_v,
    )
    if out.dim() == 4:
        out = out.squeeze(0)
    return out


def _run_ref(p, t):
    return ref.ref_sparse_attn_fwd(p, t)


# ---------------------------------------------------------------------------
# Core test runner
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_test(
    p: TestParam,
    check_correctness: bool = True,
    run_perf: bool = True,
    csv_rows: Optional[list] = None,
) -> bool:
    if p.seed == -1:
        global _counter
        p.seed = _counter.next()

    tail_dim = p.d_qk - p.d_v
    print("=" * 72)
    print(f"Config: s_q={p.s_q}, s_kv={p.s_kv}, topk={p.topk}, "
          f"h_q={p.h_q}, d_qk={p.d_qk}, d_v={p.d_v}, tail_dim={tail_dim}")
    torch.cuda.empty_cache()

    t = lib.generate_testcase(p)
    torch.cuda.synchronize()

    can_tilelang, skip_reason = _can_run_tilelang(p)

    # ------------------------------------------------------------------
    # Performance benchmark
    # ------------------------------------------------------------------
    if run_perf and p.num_runs > 0:
        flops_and_mem = lib.count_flop_and_mem_vol(p, t)

        # --- Triton ---
        def bench_triton():
            return _run_triton(t, p)

        triton_time = kk.bench_by_cuda_events(
            bench_triton, num_warmups_each=5, num_runs_each=p.num_runs,
        )
        triton_tflops = flops_and_mem.fwd_flop / triton_time / 1e12
        triton_tbps = flops_and_mem.fwd_mem_vol / triton_time / 1e12
        print(f"  Triton:   {triton_time * 1e6:8.0f} us  "
              f"{triton_tflops:7.1f} TFlops  {triton_tbps:6.2f} TBps")

        # --- Tilelang ---
        tilelang_time = float("nan")
        tilelang_tflops = float("nan")
        tilelang_tbps = float("nan")
        if can_tilelang:
            def bench_tilelang():
                return _run_tilelang(t, p)

            tilelang_time = kk.bench_by_cuda_events(
                bench_tilelang, num_warmups_each=5,
                num_runs_each=p.num_runs,
            )
            tilelang_tflops = flops_and_mem.fwd_flop / tilelang_time / 1e12
            tilelang_tbps = flops_and_mem.fwd_mem_vol / tilelang_time / 1e12
            print(f"  Tilelang: {tilelang_time * 1e6:8.0f} us  "
                  f"{tilelang_tflops:7.1f} TFlops  "
                  f"{tilelang_tbps:6.2f} TBps")
        else:
            print(f"  Tilelang: SKIPPED ({skip_reason})")

        # --- Reference (PyTorch) ---
        def bench_ref():
            return _run_ref(p, t)

        ref_time = kk.bench_by_cuda_events(
            bench_ref, num_warmups_each=5, num_runs_each=p.num_runs,
        )
        ref_tflops = flops_and_mem.fwd_flop / ref_time / 1e12
        ref_tbps = flops_and_mem.fwd_mem_vol / ref_time / 1e12
        print(f"  Ref:      {ref_time * 1e6:8.0f} us  "
              f"{ref_tflops:7.1f} TFlops  {ref_tbps:6.2f} TBps")

        # --- Speedups ---
        print(f"  Speedup vs Ref:  Triton={ref_time / triton_time:.2f}x",
              end="")
        if can_tilelang:
            print(f"  Tilelang={ref_time / tilelang_time:.2f}x", end="")
        print()
        if can_tilelang and triton_time > 0 and tilelang_time > 0:
            ratio = tilelang_time / triton_time
            faster = "Triton" if ratio > 1 else "Tilelang"
            print(f"  Triton vs Tilelang: {ratio:.2f}x "
                  f"({faster} is faster)")

        # --- CSV row ---
        if csv_rows is not None:
            row = {
                "s_q": p.s_q,
                "s_kv": p.s_kv,
                "topk": p.topk,
                "h_q": p.h_q,
                "d_qk": p.d_qk,
                "d_v": p.d_v,
                "triton_us": f"{triton_time * 1e6:.0f}",
                "triton_tflops": f"{triton_tflops:.1f}",
                "ref_us": f"{ref_time * 1e6:.0f}",
                "ref_tflops": f"{ref_tflops:.1f}",
                "speedup_triton_vs_ref": f"{ref_time / triton_time:.2f}",
            }
            if can_tilelang:
                row["tilelang_us"] = f"{tilelang_time * 1e6:.0f}"
                row["tilelang_tflops"] = f"{tilelang_tflops:.1f}"
                row["speedup_tilelang_vs_ref"] = (
                    f"{ref_time / tilelang_time:.2f}"
                )
                row["speedup_triton_vs_tilelang"] = (
                    f"{tilelang_time / triton_time:.2f}"
                )
            else:
                row["tilelang_us"] = "N/A"
                row["tilelang_tflops"] = "N/A"
                row["speedup_tilelang_vs_ref"] = "N/A"
                row["speedup_triton_vs_tilelang"] = "N/A"
            csv_rows.append(row)

    # ------------------------------------------------------------------
    # Correctness check
    # ------------------------------------------------------------------
    is_correct = True
    if check_correctness:
        torch.cuda.synchronize()

        # Reference output
        ref_out_bf16, ref_out_fp32, ref_max_logits, ref_lse = _run_ref(p, t)
        ref_lse[ref_lse == float("-inf")] = float("+inf")
        torch.cuda.synchronize()

        # Triton output
        triton_out_bf16, triton_out_fp32, triton_max_logits, triton_lse = (
            _run_triton(t, p)
        )
        torch.cuda.synchronize()

        print("  [Correctness] Triton vs Ref:")
        is_correct &= kk.check_is_allclose(
            "    out", triton_out_bf16.float(), ref_out_fp32,
            abs_tol=8e-4, rel_tol=3.01 / 128, cos_diff_tol=7e-6,
        )
        is_correct &= kk.check_is_allclose(
            "    max_logits", triton_max_logits, ref_max_logits,
            abs_tol=1e-6, rel_tol=2.01 / 65536,
        )
        is_correct &= kk.check_is_allclose(
            "    lse", triton_lse, ref_lse,
            abs_tol=1e-6, rel_tol=2.01 / 65536,
        )

        # Tilelang output (only output tensor, no lse/max_logits)
        if can_tilelang:
            tilelang_out = _run_tilelang(t, p)
            torch.cuda.synchronize()

            print("  [Correctness] Tilelang vs Ref:")
            is_correct &= kk.check_is_allclose(
                "    out", tilelang_out.float(),
                ref_out_fp32[..., :p.d_v],
                abs_tol=8e-4, rel_tol=3.01 / 128, cos_diff_tol=7e-6,
            )
        else:
            print(f"  [Correctness] Tilelang: SKIPPED ({skip_reason})")

    return is_correct


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

def get_correctness_cases_model1():
    """MODEL1 correctness cases (d_qk=512, tail_dim=0).
    Tilelang cannot run these."""
    cases = []
    for h_q in [64, 128]:
        for s_kv, topk in [
            (256, 256),
            (512, 512),
            (1521, 512),
            (1840, 256),
        ]:
            for s_q in [1, 62, 213]:
                cases.append(TestParam(
                    s_q, s_kv, topk, h_q=h_q,
                    d_qk=512, d_v=512, num_runs=0,
                ))
    return cases


def get_correctness_cases_v32():
    """V3.2 correctness cases (d_qk=576, tail_dim=64).
    Tilelang skipped when topk != 2048."""
    cases = []
    for h_q in [64, 128]:
        for s_kv, topk in [
            (256, 256),
            (512, 512),
            (1521, 512),
        ]:
            for s_q in [1, 62, 213]:
                cases.append(TestParam(
                    s_q, s_kv, topk, h_q=h_q,
                    d_qk=576, d_v=512, num_runs=0,
                ))
    return cases


def get_correctness_cases_tilelang_compatible():
    """Cases where ALL THREE kernels can run:
    d_qk=576 (tail_dim=64) + topk=2048, no attn_sink/topk_length."""
    cases = []
    for h_q in [64, 128]:
        for s_kv in [4096, 8192]:
            for s_q in [1, 62, 213, 1024]:
                cases.append(TestParam(
                    s_q, s_kv, topk=2048, h_q=h_q,
                    d_qk=576, d_v=512, num_runs=0,
                ))
    return cases


def get_performance_cases_model1():
    """MODEL1 performance cases (tilelang skipped due to tail_dim=0)."""
    templates = [
        # MODEL1 CONFIG1: d_qk=512, h_q=64, topk=512
        (512, 64, 512, [8192, 32768, 49152, 65536]),
        # MODEL1 CONFIG2: d_qk=512, h_q=128, topk=1024
        (512, 128, 1024, [8192, 32768, 49152, 65536]),
    ]
    cases = []
    for d_qk, h_q, topk, s_kv_list in templates:
        for s_q in [4096]:
            for s_kv in s_kv_list:
                cases.append(TestParam(
                    s_q, s_kv, topk, h_q=h_q, d_qk=d_qk, d_v=512,
                    check_correctness=False, num_runs=10,
                ))
    return cases


def get_performance_cases_v32():
    """V3.2 performance cases (d_qk=576, topk=2048).
    Tilelang CAN run these."""
    templates = [
        (576, 128, 2048, [8192, 32768, 65536, 98304, 131072]),
    ]
    cases = []
    for d_qk, h_q, topk, s_kv_list in templates:
        for s_q in [4096]:
            for s_kv in s_kv_list:
                cases.append(TestParam(
                    s_q, s_kv, topk, h_q=h_q, d_qk=d_qk, d_v=512,
                    check_correctness=False, num_runs=10,
                ))
    return cases


def get_performance_cases_tilelang_compatible():
    """Performance cases where ALL THREE kernels can run:
    d_qk=576 + topk=2048, no attn_sink/topk_length."""
    templates = [
        (576, 64, 2048, [8192, 32768, 49152, 65536]),
        (576, 128, 2048, [8192, 32768, 49152, 65536]),
    ]
    cases = []
    for d_qk, h_q, topk, s_kv_list in templates:
        for s_q in [4096]:
            for s_kv in s_kv_list:
                cases.append(TestParam(
                    s_q, s_kv, topk, h_q=h_q, d_qk=d_qk, d_v=512,
                    check_correctness=False, num_runs=10,
                ))
    return cases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="3-way sparse MLA prefill benchmark "
                    "(Triton vs Tilelang vs PyTorch Ref)",
    )
    parser.add_argument(
        "--correctness-only", action="store_true",
        help="Run correctness checks only (no performance benchmark)",
    )
    parser.add_argument(
        "--perf-only", action="store_true",
        help="Run performance benchmark only (no correctness checks)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        choices=[
            "model1_config1", "model1_config2", "model1",
            "v32", "tilelang_compat", "all",
        ],
        help="Run only a specific config subset",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Write performance results to CSV file",
    )
    parser.add_argument(
        "--num-runs", type=int, default=10,
        help="Number of benchmark runs per case (default: 10)",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")

    # Print environment info
    print_env_info()

    print()
    print("=" * 72)
    print("3-Way Sparse MLA Prefill Kernel Benchmark")
    print("=" * 72)
    print(f"  Triton:   available")
    print(f"  Tilelang: {'available' if _tilelang_available else 'NOT available'}")
    if not _tilelang_available:
        print(f"            Error: {_tilelang_import_error}")
    print(f"  Ref:      available (PyTorch)")
    print()
    print("  Tilelang constraints on AMD/HIP:")
    print("    - topk must be 2048")
    print("    - tail_dim must be > 0 (d_qk > d_v, e.g. d_qk=576)")
    print("    - inputs must be contiguous")
    print("    - attn_sink / topk_length not supported in prefill path")
    print("=" * 72)

    cfg = args.config or "all"

    # Build test cases
    correctness_cases = []
    perf_cases = []

    if not args.perf_only:
        if cfg == "all":
            correctness_cases = (
                get_correctness_cases_model1()
                + get_correctness_cases_v32()
                + get_correctness_cases_tilelang_compatible()
            )
        elif cfg in ("model1", "model1_config1", "model1_config2"):
            correctness_cases = get_correctness_cases_model1()
        elif cfg == "v32":
            correctness_cases = get_correctness_cases_v32()
        elif cfg == "tilelang_compat":
            correctness_cases = get_correctness_cases_tilelang_compatible()

    if not args.correctness_only:
        if cfg == "all":
            perf_cases = (
                get_performance_cases_model1()
                + get_performance_cases_v32()
                + get_performance_cases_tilelang_compatible()
            )
        elif cfg == "model1_config1":
            perf_cases = [
                c for c in get_performance_cases_model1()
                if c.topk == 512 and c.h_q == 64
            ]
        elif cfg == "model1_config2":
            perf_cases = [
                c for c in get_performance_cases_model1()
                if c.topk == 1024 and c.h_q == 128
            ]
        elif cfg == "model1":
            perf_cases = get_performance_cases_model1()
        elif cfg == "v32":
            perf_cases = get_performance_cases_v32()
        elif cfg == "tilelang_compat":
            perf_cases = get_performance_cases_tilelang_compatible()

    # Override num_runs for performance cases
    for c in perf_cases:
        c.num_runs = args.num_runs

    all_cases = correctness_cases + perf_cases

    n_correctness = len(correctness_cases)
    n_perf = len(perf_cases)
    n_total = len(all_cases)

    print(f"\nTotal test cases: {n_total}")
    if n_correctness > 0:
        print(f"  Correctness cases: {n_correctness}")
    if n_perf > 0:
        print(f"  Performance cases: {n_perf}")
    print()

    if not all_cases:
        print("No test cases selected.")
        print("Check --config / --correctness-only / --perf-only flags.")
        return

    is_no_cooldown = lib.is_no_cooldown()
    csv_rows = [] if args.csv else None
    failed_cases = []
    tilelang_tested = 0
    tilelang_skipped = 0

    for i, test in enumerate(all_cases):
        if i > 0 and test.num_runs > 0 and not is_no_cooldown:
            time.sleep(0.3)

        do_correctness = test in correctness_cases
        do_perf = test in perf_cases

        can_tl, _ = _can_run_tilelang(test)
        if can_tl:
            tilelang_tested += 1
        else:
            tilelang_skipped += 1

        is_correct = run_test(
            test,
            check_correctness=do_correctness,
            run_perf=do_perf,
            csv_rows=csv_rows,
        )
        if not is_correct:
            failed_cases.append(test)

    # Write CSV
    if csv_rows and args.csv:
        fieldnames = [
            "s_q", "s_kv", "topk", "h_q", "d_qk", "d_v",
            "triton_us", "triton_tflops",
            "tilelang_us", "tilelang_tflops",
            "ref_us", "ref_tflops",
            "speedup_triton_vs_ref",
            "speedup_tilelang_vs_ref",
            "speedup_triton_vs_tilelang",
        ]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nPerformance results written to {args.csv}")

    # Summary
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Total cases:       {n_total}")
    print(f"  Tilelang tested:   {tilelang_tested}")
    print(f"  Tilelang skipped:  {tilelang_skipped}")
    print()
    if failed_cases:
        print(f"\033[31m\033[1m{len(failed_cases)} / {n_total} "
              f"cases FAILED:\033[0m")
        for case in failed_cases:
            print(f"    {case}")
        sys.exit(1)
    else:
        print(f"\033[32m\033[1mAll {n_total} cases passed!\033[0m")


if __name__ == "__main__":
    main()
