# TileLang Dual-Scope Kernel Bug: Int32 Overflow in topk_length Sentinel

## Summary

The dpsk_v4_fp8_partial_kernel in tilelang_kernel.py produces incorrect
results (all zeros) for Phase 2 (extra scope) in dual-scope mode when
topk_length is None and certain batch size / block_size combinations
are used.

**Root Cause**: Signed int32 overflow in the validity check condition.

**Fix**: One-line change - reduce the sentinel value from 2^31-1 to 2^30.

**Patch**: tilelang_int32_overflow_fix.patch

---

## Failed Cases

Out of 24 dual-scope test cases, **11 failed** with the native dual-scope kernel:

| Config | Batch Sizes | extra_block_size | extra_topk_length | Result |
|--------|-------------|------------------|-------------------|--------|
| CFG1   | 2,64,74,128,148 | 64 | None | **FAIL** |
| CFG1   | 256 | 64 | None | PASS |
| CFG2   | 2,64,74,128,148,256 | 64 | None | **FAIL** |
| CFG3   | 2,64,74,128,148,256 | 2 | Set (small values) | PASS |
| CFG4   | 2,64,74,128,148,256 | 2 | Set (small values) | PASS |

**Pattern**: All failures occur when extra_topk_length=None (no topk length
limit for the extra scope). CFG3/CFG4 always pass because they have
have_extra_topk_length=True, providing small concrete values.

---

## Symptom

Phase 2 groups in the partial kernel output:
- **Output**: All zeros
- **LSE**: -2^30 (the "empty/no valid KV" sentinel)

This means the kernel treats ALL KV entries in the extra scope as invalid,
even though the indices and data are correct.

Phase 1 groups produce correct results (verified to match single-scope
kernel output exactly, diff = 0.0).

---

## Root Cause Analysis

### The Kernel Code (tilelang_kernel.py, line ~1750)

The Phase 2 validity check in the Python DSL:

    valid = (idx >= 0) & (pos < tk_len_2)

When topk_length is None, tk_len_2 is set to the sentinel value
_INT32_MAX = 2^31 - 1 = 2,147,483,647.

### The Generated C++ Code

The tilelang compiler optimizes the Phase 2 code. Instead of computing
pos = (group_i - n_groups_1) * inner_iter_2 * BI + ... (which requires
subtracting n_groups_1), it keeps pos as group_i * BI + ... and
compensates by adding n_groups_1 * BI to tk_len_2:

    // Phase 2 generated code (FAIL case: inner_iter_2=1, n_groups_1=2, BI=64)
    bool valid_1 = ((0 <= idx_1) &
        (pos < (tk_len_2 + 128)));  // 128 = n_groups_1 * BI = 2 * 64

### The Overflow

When tk_len_2 = INT32_MAX = 2,147,483,647:

    tk_len_2 + 128 = 2,147,483,647 + 128 = 2,147,483,775

This exceeds the int32 range (2^31 - 1 = 2,147,483,647), causing
**signed integer overflow**. In two's complement:

    2,147,483,775 - 2^32 = -2,147,483,521

So the condition becomes:

    pos < -2,147,483,521  // Always FALSE for any non-negative pos!

**Result**: valid is always false, all KV entries are masked out,
Phase 2 produces zeros.

### Why CFG3/CFG4 Pass

CFG3/CFG4 have have_extra_topk_length=True, so tk_len_2 is a small
concrete value (e.g., 214 or 515). Adding 128 to 515 gives 643, which
does NOT overflow int32.

### Why CFG1 b=256 Passes

For b=256, the compiler chooses inner_iter_2=8 (larger split-K).
With inner_iter_2=8, the compiler generates a proper loop variable
k_i_1 that starts from 0, so the condition is:

    bool valid_1 = ((0 <= idx_1) & (pos < tk_len_2));  // No addition!

No overflow occurs because tk_len_2 is not modified.

---

## Fix

**File**: python/sglang/srt/layers/attention/nsa/tilelang_kernel.py
**Line**: 1560

    -_INT32_MAX = 2**31 - 1
    +_INT32_MAX = 2**30  # Avoid int32 overflow in dual-scope Phase 2

**Rationale**: 2^30 = 1,073,741,824 is still far larger than any realistic
topk value (max ~16384 in our configs). Even with the largest possible
offset (n_groups_1 * BI <= topk <= 16384), 2^30 + 16384 = 1,073,758,208
is well within int32 range.

### Verification

After applying the fix, all 24 dual-scope cases pass correctness checks:
- CFG1: 6/6 PASS (was 1/6)
- CFG2: 6/6 PASS (was 0/6)
- CFG3: 6/6 PASS (was 6/6)
- CFG4: 6/6 PASS (was 6/6)

---

## How to Apply

    cd /path/to/sglang
    git apply tilelang_int32_overflow_fix.patch

Or manually edit line 1560 of
python/sglang/srt/layers/attention/nsa/tilelang_kernel.py.

---

## Performance Impact (After Fix)

With the bug fixed, Triton vs TileLang native comparison (26 cases, geomean):

| Kernel   | TFlops | vs Ref Speedup |
|----------|--------|----------------|
| Triton   | 62.7   | 4.02x          |
| TileLang | 20.5   | 1.31x          |
| Ref      | 15.6   | 1.0x           |

Triton vs TileLang speedup geomean: **3.06x**

Both kernels pass 26/26 correctness checks with the same tolerances:
- Output: abs_tol=1e-3, rel_tol=2.01/128, cos_diff_tol=5e-6
- LSE: abs_tol=1e-6, rel_tol=8.01/65536
