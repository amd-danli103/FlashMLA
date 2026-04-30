"""
Benchmark: Triton vs TileLang Native Sparse MLA Decode Kernels
==============================================================

Two implementations compared:
1. Triton-based (triton_mla_kernels_decode_optimized.py)
2. TileLang native (tilelang_kernel.py from sglang)

Correctness verified against PyTorch reference (ref.py).
26 MODEL1-related test configs from test_flash_mla_sparse_decoding_triton_v1.py.
"""

import sys
import os
import time
import dataclasses
from typing import Tuple, List, Optional
import csv

import rich.console
import rich.table
import numpy as np

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/danli103/sglang/python")

import kernelkit as kk
import lib
from lib import TestParam, RawTestParamForDecode as RawTestParam
import ref
from triton_mla_kernels_decode_optimized import triton_sparse_attn_decode
from sglang.srt.layers.attention.nsa.tilelang_kernel import (
    dpsk_v4_fp8_attention_fwd,
)


def gen_model1_testcases() -> List[RawTestParam]:
    """Generate the 26 MODEL1-related test configs."""
    base_and_bszs = [
        (RawTestParam(0, 64, 2, 1, 16384, True, topk=128, d_qk=512,
            extra_s_k=16384, extra_topk=512,
            block_size=256, extra_block_size=64),
         [2, 64, 74, 128, 148, 256]),
        (RawTestParam(0, 128, 2, 1, 16384, True, topk=128, d_qk=512,
            extra_s_k=16384, extra_topk=1024,
            block_size=256, extra_block_size=64),
         [2, 64, 74, 128, 148, 256]),
        (RawTestParam(0, 64, 2, 1, 16384, True, topk=128, d_qk=512,
            extra_s_k=16384, extra_topk=1024,
            block_size=256, extra_block_size=2,
            have_extra_topk_length=True),
         [2, 64, 74, 128, 148, 256]),
        (RawTestParam(0, 128, 2, 1, 16384, True, topk=128, d_qk=512,
            extra_s_k=16384, extra_topk=1024,
            block_size=256, extra_block_size=2,
            have_extra_topk_length=True),
         [2, 64, 74, 128, 148, 256]),
    ]
    production_cases = [
        dataclasses.replace(base, b=b)
        for base, bszs in base_and_bszs
        for b in bszs
    ]
    peak_perf_cases = [
        RawTestParam(148, h_q, 2, 1, 32768, True,
            topk=16384, d_qk=512)
        for h_q in [64, 128]
    ]
    all_cases = production_cases + peak_perf_cases
    for i, case in enumerate(production_cases):
        case.seed = i + 4
    peak_perf_cases[0].seed = 28
    peak_perf_cases[1].seed = 30
    return all_cases


def _sanitize_kvcache_for_tilelang(kv_cache_quantized):
    """Replace NaN values in quantized KV cache with zeros in-place.

    This is a test-only artifact: the test framework fills unused KV
    cache entries with NaN, but tilelang uses index 0 as fallback for
    invalid indices. In production this is not needed.
    """
    u8_view = kv_cache_quantized.view(torch.uint8)
    nan_mask = (u8_view == 127) | (u8_view == 255)
    u8_view[nan_mask] = 0
    return kv_cache_quantized


def run_triton(t, p):
    """Run Triton kernel (full end-to-end API call)."""
    return triton_sparse_attn_decode(
        t.q, t.kv_scope, t.extra_kv_scope,
        t.sm_scale, p.d_v, t.attn_sink)




@dataclasses.dataclass
class CaseResult:
    idx: int
    config_name: str
    p: TestParam
    triton_ok: bool
    triton_us: float
    triton_tflops: float
    tl_ok: bool
    tl_us: float
    tl_tflops: float
    ref_us: float
    ref_tflops: float
    compute_memory_ratio: float
    has_extra: bool


def get_config_name(p: TestParam) -> str:
    assert p.decode is not None
    if p.decode.extra_topk is None:
        return "PeakPerf"
    elif p.h_q == 64 and p.decode.extra_topk == 512:
        return "CFG1"
    elif (p.h_q == 128 and p.decode.extra_topk == 1024
          and p.decode.extra_block_size == 64):
        return "CFG2"
    elif (p.h_q == 64 and p.decode.extra_topk == 1024
          and p.decode.have_extra_topk_length):
        return "CFG3"
    elif (p.h_q == 128 and p.decode.extra_topk == 1024
          and p.decode.have_extra_topk_length):
        return "CFG4"
    return "?"


@torch.inference_mode()
def benchmark_case(idx, p, num_warmup=5, num_runs=20):
    assert p.decode is not None
    config_name = get_config_name(p)
    has_extra = p.decode.extra_topk is not None

    print(f"\n{'='*70}")
    print(f"[{idx+1}/26] {config_name}: b={p.decode.b}, h_q={p.h_q}, "
          f"topk={p.topk}", end="")
    if has_extra:
        print(f", extra_topk={p.decode.extra_topk}, "
              f"ebs={p.decode.extra_block_size}", end="")
    print()

    torch.cuda.empty_cache()
    t = lib.generate_testcase_for_decode(p)
    flops_mem = lib.count_flop_and_mem_vol_for_decode(p, t)
    cm_ratio = flops_mem.flop / flops_mem.mem_vol

    # Sanitize KV cache for tilelang (test artifact, not timed)
    _sanitize_kvcache_for_tilelang(t.kv_scope.get_kvcache_for_flash_mla())
    if t.extra_kv_scope is not None:
        _sanitize_kvcache_for_tilelang(
            t.extra_kv_scope.get_kvcache_for_flash_mla())

    # --- Reference (correctness baseline only) ---
    out_ref, lse_ref = ref.ref_sparse_attn_decode(p, t)
    ref_us = 0.0
    if num_runs > 0:
        ref_us = kk.bench_by_cuda_events(
            lambda: ref.ref_sparse_attn_decode(p, t),
            num_warmups_each=num_warmup,
            num_runs_each=num_runs) * 1e6
    ref_tflops = (flops_mem.flop / (ref_us / 1e6) / 1e12
                  if ref_us > 0 else 0.0)

    # --- Triton ---
    # Timing includes: dispatch logic, reshape, contiguous, buffer alloc,
    # gather kernel, attention kernel, output reshape.
    triton_ok = False
    triton_us = 0.0
    triton_tflops = 0.0
    try:
        out_tri, lse_tri = run_triton(t, p)
        torch.cuda.synchronize()
        tri_out_ok = kk.check_is_allclose(
            "tri_out", out_tri, out_ref,
            abs_tol=1e-3, rel_tol=2.01/128, cos_diff_tol=5e-6)
        tri_lse_ok = kk.check_is_allclose(
            "tri_lse", lse_tri, lse_ref,
            abs_tol=1e-6, rel_tol=8.01/65536)
        triton_ok = tri_out_ok and tri_lse_ok
        if num_runs > 0:
            triton_us = kk.bench_by_cuda_events(
                lambda: run_triton(t, p),
                num_warmups_each=num_warmup,
                num_runs_each=num_runs) * 1e6
            triton_tflops = flops_mem.flop / (triton_us / 1e6) / 1e12
    except Exception as e:
        print(f"  Triton ERROR: {e}")

    # --- TileLang native ---
    # Timing includes .contiguous() copies + _build_fp8_combined_view +
    # _pick_inner_iter + partial kernel + combine kernel + output alloc.
    # This matches Triton's end-to-end API call fairness: Triton receives
    # raw (possibly non-contiguous) tensors and handles everything internally.
    k1 = t.kv_scope.get_kvcache_for_flash_mla()
    tl1 = t.kv_scope.topk_length
    k2 = (t.extra_kv_scope.get_kvcache_for_flash_mla()
          if t.extra_kv_scope else None)
    tl2 = (t.extra_kv_scope.topk_length
           if t.extra_kv_scope else None)

    def run_tl_full():
        """Full TileLang call including .contiguous() copies."""
        q_c = t.q.contiguous()
        idx1 = t.kv_scope.indices_in_kvcache.contiguous()
        idx2 = (t.extra_kv_scope.indices_in_kvcache.contiguous()
                if t.extra_kv_scope else None)
        return dpsk_v4_fp8_attention_fwd(
            q=q_c, k_cache=k1, block_table=None,
            cache_seqlens=None, head_dim_v=p.d_v,
            tile_scheduler_metadata=None, num_splits=None,
            softmax_scale=t.sm_scale, causal=True,
            is_fp8_kvcache=True,
            indices=idx1,
            attn_sink=t.attn_sink,
            extra_k_cache=k2,
            extra_indices_in_kvcache=idx2,
            topk_length=tl1,
            extra_topk_length=tl2)

    tl_ok = False
    tl_us = 0.0
    tl_tflops = 0.0
    try:
        out_tl, lse_tl = run_tl_full()
        torch.cuda.synchronize()

        out_tl_cmp = out_tl[..., :p.d_v]
        lse_tl_cmp = lse_tl.transpose(1, 2)
        tl_out_ok = kk.check_is_allclose(
            "tl_out", out_tl_cmp, out_ref,
            abs_tol=1e-3, rel_tol=2.01/128, cos_diff_tol=5e-6)
        tl_lse_ok = kk.check_is_allclose(
            "tl_lse", lse_tl_cmp, lse_ref,
            abs_tol=1e-6, rel_tol=8.01/65536)
        tl_ok = tl_out_ok and tl_lse_ok
        if num_runs > 0:
            tl_us = kk.bench_by_cuda_events(
                run_tl_full,
                num_warmups_each=num_warmup,
                num_runs_each=num_runs) * 1e6
            tl_tflops = flops_mem.flop / (tl_us / 1e6) / 1e12
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  TileLang ERROR: {e}")

    status_tri = "PASS" if triton_ok else "FAIL"
    status_tl = "PASS" if tl_ok else "FAIL"
    print(f"  Triton:   {triton_us:>8.1f} us  {triton_tflops:>6.1f} TF  [{status_tri}]")
    print(f"  TileLang: {tl_us:>8.1f} us  {tl_tflops:>6.1f} TF  [{status_tl}]")
    print(f"  Ref:      {ref_us:>8.1f} us  {ref_tflops:>6.1f} TF")

    return CaseResult(
        idx=idx, config_name=config_name, p=p,
        triton_ok=triton_ok, triton_us=triton_us,
        triton_tflops=triton_tflops,
        tl_ok=tl_ok, tl_us=tl_us, tl_tflops=tl_tflops,
        ref_us=ref_us, ref_tflops=ref_tflops,
        compute_memory_ratio=cm_ratio, has_extra=has_extra,
    )


def geomean(vals):
    if not vals:
        return 0.0
    return float(np.exp(np.mean(np.log(vals))))


def print_report(results: List[CaseResult]):
    console = rich.console.Console(width=200)

    table = rich.table.Table(
        show_header=True, header_style="bold cyan",
        title="Sparse MLA Decode Benchmark: Triton vs TileLang Native",
        caption="Latency in microseconds. TFLOPS = Tera floating-point ops per second. Speedup = Ref_time / Kernel_time.")
    table.add_column("#", justify="right", width=3)
    table.add_column("Config", width=8)
    table.add_column("Batch\nSize", justify="right", width=5)
    table.add_column("Num\nHeads", justify="right", width=5)
    table.add_column("topk", justify="right", width=6)
    table.add_column("extra\ntopk", justify="right", width=6)
    table.add_column("Comp\n/Mem", justify="right", width=5)
    table.add_column("Triton\nLatency(us)", justify="right", width=11)
    table.add_column("Triton\nTFLOPS", justify="right", width=7)
    table.add_column("Triton\nPass?", justify="center", width=6)
    table.add_column("TileLang\nLatency(us)", justify="right", width=11)
    table.add_column("TileLang\nTFLOPS", justify="right", width=8)
    table.add_column("TileLang\nPass?", justify="center", width=8)
    table.add_column("Ref(PyTorch)\nLatency(us)", justify="right", width=13)
    table.add_column("Triton\nvs Ref", justify="right", width=7)
    table.add_column("TileLang\nvs Ref", justify="right", width=8)
    table.add_column("Triton vs\nTileLang", justify="right", width=10)

    for r in results:
        p = r.p
        assert p.decode is not None
        et = str(p.decode.extra_topk) if p.decode.extra_topk else "-"

        def f_us(v):
            return f"{v:.1f}" if v > 0 else "-"
        def f_tf(v):
            return f"{v:.1f}" if v > 0 else "-"
        def f_ok(v):
            return "[green]Y[/green]" if v else "[red]N[/red]"
        def f_sp(a, b):
            if a > 0 and b > 0:
                return f"{b/a:.2f}x"
            return "-"

        table.add_row(
            str(r.idx + 1), r.config_name,
            str(p.decode.b), str(p.h_q),
            str(p.topk), et,
            f"{r.compute_memory_ratio:.0f}",
            f_us(r.triton_us), f_tf(r.triton_tflops), f_ok(r.triton_ok),
            f_us(r.tl_us), f_tf(r.tl_tflops), f_ok(r.tl_ok),
            f_us(r.ref_us),
            f_sp(r.triton_us, r.ref_us),
            f_sp(r.tl_us, r.ref_us),
            f_sp(r.triton_us, r.tl_us),
        )
    console.print(table)

    # Correctness
    tri_pass = sum(1 for r in results if r.triton_ok)
    tl_pass = sum(1 for r in results if r.tl_ok)
    print(f"\n{'='*80}")
    print("CORRECTNESS")
    print(f"{'='*80}")
    print(f"  Triton:   {tri_pass}/{len(results)} passed")
    print(f"  TileLang: {tl_pass}/{len(results)} passed")

    # Performance
    print(f"\n{'='*80}")
    print("PERFORMANCE (geometric mean)")
    print(f"{'='*80}")

    tri_tf = [r.triton_tflops for r in results if r.triton_tflops > 0.1]
    tl_tf = [r.tl_tflops for r in results if r.tl_tflops > 0.1]
    ref_tf = [r.ref_tflops for r in results if r.ref_tflops > 0.1]

    if tri_tf:
        print(f"  Triton   TFlops geomean: {geomean(tri_tf):.1f}")
    if tl_tf:
        print(f"  TileLang TFlops geomean: {geomean(tl_tf):.1f}")
    if ref_tf:
        print(f"  Ref      TFlops geomean: {geomean(ref_tf):.1f}")

    tri_vs_ref = [r.ref_us / r.triton_us for r in results
                  if r.triton_us > 0 and r.ref_us > 0]
    tl_vs_ref = [r.ref_us / r.tl_us for r in results
                 if r.tl_us > 0 and r.ref_us > 0]
    tri_vs_tl = [r.tl_us / r.triton_us for r in results
                 if r.triton_us > 0 and r.tl_us > 0]

    print()
    if tri_vs_ref:
        print(f"  Triton   vs Ref speedup geomean: {geomean(tri_vs_ref):.2f}x")
    if tl_vs_ref:
        print(f"  TileLang vs Ref speedup geomean: {geomean(tl_vs_ref):.2f}x")
    if tri_vs_tl:
        print(f"  Triton vs TileLang speedup geomean: {geomean(tri_vs_tl):.2f}x")

    # Timing fairness note
    print(f"\n{'='*80}")
    print("TIMING METHODOLOGY")
    print(f"{'='*80}")
    print("""  Both kernels are timed using CUDA events (kk.bench_by_cuda_events).

  Triton timing includes:
    - Python dispatch logic (d_qk check, scope detection)
    - q.reshape() + .contiguous()
    - indices.reshape() + .contiguous()
    - Internal buffer allocation (gathered_kv, invalid_mask, output, lse)
    - Gather kernel launch (fused_gather_dequant_fp8)
    - Attention kernel launch (_unified_sparse_decode_kernel)
    - Output reshape (view)

  TileLang timing includes:
    - q.contiguous() (real copy if input is non-contiguous)
    - indices.contiguous() (real copy if input is non-contiguous)
    - _build_fp8_combined_view (reinterpret KV cache as uint32, no copy)
    - _pick_inner_iter (Python heuristic, ~1us)
    - Partial kernel launch (dpsk_v4_fp8_partial_kernel)
    - Combine kernel launch (dpsk_v4_combine_kernel)
    - Output tensor allocation

  NOT included in either timing:
    - NaN sanitization (test artifact, not needed in production)

  Fairness assessment:
    Both timings measure the full end-to-end API call including all
    data preparation. Triton receives raw (possibly non-contiguous)
    tensors and handles reshape/contiguous internally. TileLang also
    includes .contiguous() copies in its timed region. The .contiguous()
    overhead is ~1% of total time for production configs.
""")



def print_env_info():
    """Print environment and version info for reproducibility."""
    import platform
    import subprocess

    print(f"{'='*80}")
    print("ENVIRONMENT INFO")
    print(f"{'='*80}")

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

    # HIP_VISIBLE_DEVICES
    hip_dev = os.environ.get("HIP_VISIBLE_DEVICES",
              os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"))
    print(f"  VISIBLE_DEVS:  {hip_dev}")

    # Hostname
    print(f"  Hostname:      {platform.node()}")

    # Date
    from datetime import datetime
    print(f"  Date:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"{'='*80}")


def main():
    dtype = torch.bfloat16
    device = torch.device("cuda:0")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(32)

    import argparse
    parser = argparse.ArgumentParser(
        description="Benchmark Triton vs TileLang sparse MLA decode")
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--case", type=int, default=-1)
    args = parser.parse_args()

    num_runs = 0 if args.correctness_only else args.num_runs

    raw_cases = gen_model1_testcases()
    testcases = [t.to_test_param() for t in raw_cases]

    print_env_info()

    print(f"\n{'='*80}")
    print("Sparse MLA Decode: Triton vs TileLang Native")
    print(f"{'='*80}")
    print(f"Total test cases: {len(testcases)}")
    print(f"Benchmark runs: {num_runs}, Warmup: {args.num_warmup}")
    print(f"{'='*80}")

    if args.case >= 0:
        if args.case >= len(testcases):
            print(f"Error: case {args.case} out of range")
            sys.exit(1)
        indices = [args.case]
    else:
        indices = list(range(len(testcases)))

    results = []
    for i in indices:
        if len(results) > 0 and num_runs > 0:
            time.sleep(0.2)
        r = benchmark_case(i, testcases[i],
                           num_warmup=args.num_warmup,
                           num_runs=num_runs)
        results.append(r)

    print("\n\n")
    print_report(results)

    # Save CSV
    csv_path = os.path.join(os.path.dirname(__file__),
                            "benchmark_report_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case", "Config", "Bsz", "h_q", "topk", "extra_topk",
                     "C/M", "Triton_us", "Triton_TF", "Triton_OK",
                     "TileLang_us", "TileLang_TF", "TileLang_OK",
                     "Ref_us", "Ref_TF"])
        for r in results:
            p = r.p
            assert p.decode is not None
            w.writerow([
                r.idx + 1, r.config_name, p.decode.b, p.h_q,
                p.topk, p.decode.extra_topk or "-",
                f"{r.compute_memory_ratio:.1f}",
                f"{r.triton_us:.1f}", f"{r.triton_tflops:.1f}",
                "Y" if r.triton_ok else "N",
                f"{r.tl_us:.1f}", f"{r.tl_tflops:.1f}",
                "Y" if r.tl_ok else "N",
                f"{r.ref_us:.1f}", f"{r.ref_tflops:.1f}",
            ])
    print(f"\nResults saved to {csv_path}")

    all_ok = all(r.triton_ok and r.tl_ok for r in results)
    if not all_ok:
        print("\nWARNING: Some correctness checks failed!")
        sys.exit(1)
    print("\nAll correctness checks passed.")


if __name__ == "__main__":
    main()
