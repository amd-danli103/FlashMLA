import time
import sys

import torch
import kernelkit as kk

from lib import TestParam
import lib
import ref
# Use optimized Triton implementation instead of the original
from triton_mla_kernels_prefill_optimized import triton_sparse_attn_fwd_optimized as triton_sparse_attn_fwd

_counter = kk.Counter()

@torch.inference_mode()
def run_test(p: TestParam) -> bool:
    if p.seed == -1:
        global _counter
        p.seed = _counter.next()

    print("================")
    print(f"Running on {p}")
    torch.cuda.empty_cache()

    t = lib.generate_testcase(p)
    torch.cuda.synchronize()

    # Call Triton implementation
    def run_triton():
        return triton_sparse_attn_fwd(t.q, t.kv, t.indices, t.sm_scale, p.d_v, t.attn_sink, t.topk_length)

    prefill_ans_out, prefill_ans_out_fp32, prefill_ans_max_logits, prefill_ans_lse = run_triton()
    torch.cuda.synchronize()

    if p.num_runs > 0:
        flops_and_mem_vol = lib.count_flop_and_mem_vol(p, t)

        # Benchmark Triton implementation
        triton_time = kk.bench_by_cuda_events(run_triton, num_warmups_each=5, num_runs_each=p.num_runs)
        triton_flops = flops_and_mem_vol.fwd_flop/triton_time/1e12
        triton_mem_bw = flops_and_mem_vol.fwd_mem_vol/triton_time/1e12
        print(f"Triton:  {triton_time*1e6:4.0f} us, {triton_flops:6.1f} TFlops, {triton_mem_bw:4.2f} TBps")

        # Benchmark reference implementation for comparison
        def run_ref():
            return ref.ref_sparse_attn_fwd(p, t)
        ref_time = kk.bench_by_cuda_events(run_ref, num_warmups_each=5, num_runs_each=p.num_runs)
        ref_flops = flops_and_mem_vol.fwd_flop/ref_time/1e12
        ref_mem_bw = flops_and_mem_vol.fwd_mem_vol/ref_time/1e12
        print(f"Ref:     {ref_time*1e6:4.0f} us, {ref_flops:6.1f} TFlops, {ref_mem_bw:4.2f} TBps")
        print(f"Speedup: {ref_time/triton_time:.2f}x")

    if p.check_correctness:
        torch.cuda.synchronize()
        ref_out, ref_out_fp32, ref_max_logits, ref_lse = ref.ref_sparse_attn_fwd(p, t)
        ref_lse[ref_lse == float("-inf")] = float("+inf")
        torch.cuda.synchronize()

        is_correct = True
        is_correct &= kk.check_is_allclose("out", prefill_ans_out.float(), ref_out_fp32, abs_tol=8e-4, rel_tol=3.01/128, cos_diff_tol=7e-6)
        is_correct &= kk.check_is_allclose("max_logits", prefill_ans_max_logits, ref_max_logits, abs_tol=1e-6, rel_tol=2.01/65536)
        is_correct &= kk.check_is_allclose("lse", prefill_ans_lse, ref_lse, abs_tol=1e-6, rel_tol=2.01/65536)
        return is_correct
    else:
        return True


if __name__ == '__main__':
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')

    correctness_cases = [
        # Regular shapes
        TestParam(s_q, s_kv, topk, h_q=h_q, num_runs=0, d_qk=d_qk)
        for d_qk in [512, 576]
        for h_q in [
            128, 64
        ]
        for s_kv, topk in [
            # Regular shapes
            (128, 128),
            (256, 256),
            (512, 512),

            # Irregular shapes
            (592, 128),
            (1840, 256),
            (1592, 384),
            (1521, 512),

            # Irregular shapes with OOB TopK
            (95, 128),
            (153, 256),
            (114, 384),
        ]
        for s_q in [
            1, 62, 213
        ]
    ]

    correctness_cases_with_features = [
        TestParam(s_q, s_kv, topk, h_q=h_q, num_runs=0, have_attn_sink=have_attn_sink, have_topk_length=have_topk_length, d_qk=d_qk)
        for d_qk in [512, 576]
        for h_q in [
            128, 64
        ]
        for s_kv, topk in [
            (592, 128),
            (1840, 256),
            (1592, 384),
            (1521, 512),

            (95, 128),
            (153, 256),
            (114, 384),
        ]
        for s_q in [62, 213]
        for have_sink_lse in [False, True]
        for have_attn_sink in [False, True]
        for have_topk_length in [False, True]
    ]

    corner_cases = [
        TestParam(s_q, s_kv, topk, h_q=h_q, is_all_indices_invalid=True, num_runs=0, have_attn_sink=True, have_topk_length=True, d_qk=d_qk)
        for d_qk in [512, 576]
        for h_q in [
            128, 64
        ]
        for s_q, s_kv, topk in [
            (1, 128, 128),
            (1, 256, 256),
            (1234, 4321, 4096),
            (4096, 2048, 2048)
        ]
    ] + [
        # In these cases, some blocks may not have any valid topk indices
        TestParam(s_q, s_kv, topk, h_q=h_q, is_all_indices_invalid=False, num_runs=0, have_attn_sink=True, have_topk_length=True, d_qk=d_qk)
        for d_qk in [512, 576]
        for h_q in [
            128, 64
        ]
        for s_kv, topk in [
            (32, 2048),
            (64, 8192)
        ]
        for s_q in [1, 1024]
    ] + [
        # In this testcase, s_q is really large, so we cannot put it on the second dimension of grid shape
        # Note: Using s_q=8192 instead of 70000 for optimized kernel due to memory constraints
        # (70000 * 128 * 512 * 6 bytes for dual output = ~27GB which exceeds GPU memory)
        TestParam(8192, 256, 256, h_q=h_q, check_correctness=False, num_runs=0, have_attn_sink=True, have_topk_length=True, d_qk=d_qk)
        for d_qk in [512, 576]
        for h_q in [
            128, 64
        ]
    ]

    performance_case_templates = [
        # V3.2
        (576, 128, 2048, [8192, 32768, 65536, 98304, 131072]),
        # MODEL1 CONFIG1
        (512, 64, 512, [8192, 32768, 49152, 65536]),
        # MODEL1 CONFIG2
        (512, 128, 1024, [8192, 32768, 49152, 65536]),
    ]

    performance_cases = [
        TestParam(s_q, s_kv, topk, h_q=h_q, d_qk=d_qk, have_attn_sink=True)
        for (d_qk, h_q, topk, s_kv_list) in performance_case_templates
        for s_q in [4096]
        for s_kv in s_kv_list
    ]

    testcases = correctness_cases + correctness_cases_with_features + corner_cases + performance_cases

    is_no_cooldown = lib.is_no_cooldown()
    failed_cases = []
    for test in testcases:
        if test != testcases[0] and test.num_runs > 0 and not is_no_cooldown:
            time.sleep(0.3)
        is_correct = run_test(test)
        if not is_correct:
            failed_cases.append(test)
    
    if len(failed_cases) > 0:
        print(f"\033[31m\033[1m{len(failed_cases)} / {len(testcases)} cases failed:\033[0m")
        for case in failed_cases:
            print(f"    {case}")
        sys.exit(1)
    else:
        print(f"\033[32m\033[1mAll {len(testcases)} cases passed!\033[0m")
