"""
test_swish.py
════════════════════════════════════════════════════════════════════
Complete test + benchmark suite for the custom Swish/Mish CUDA extension.

Tests:
  1.  Numerical correctness vs PyTorch reference
  2.  Gradient check (torch.autograd.gradcheck)
  3.  dtype coverage: float32, float16
  4.  Shape coverage: 1D, 2D, 3D, 4D tensors
  5.  Boundary values: large/small/zero inputs
  6.  Training loop — loss converges (backward works end-to-end)
  7.  FusedBiasSwish correctness + grad check
  8.  Drop-in replacement: same loss as nn.SiLU

Benchmarks:
  9.  Swish vs nn.SiLU: forward latency
  10. Swish vs nn.SiLU: backward latency
  11. Mish vs PyTorch Mish: forward latency
  12. FusedBiasSwish vs unfused (nn.Linear bias + SiLU): forward + backward
  13. Throughput (billion elements / second) across tensor sizes
  14. float32 vs float16 speed comparison
  15. Vec4 kernel vs standard kernel benchmark

Run:
    python test_swish.py
    python test_swish.py --quick    # skip long benchmarks
    python test_swish.py --no-plot  # skip matplotlib charts
════════════════════════════════════════════════════════════════════
"""

import sys
import time
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional

# ── import our module ─────────────────────────────────────────────────
from swish import (
    Swish, Mish, FusedBiasSwish,
    swish, mish, fused_bias_swish,
    SwishFunction, MishFunction,
    _swish_pytorch, _mish_pytorch,
    backend_info
)

# ── try import cuda extension for direct access ───────────────────────
try:
    import swish_cuda as _C
    CUDA_EXT = True
except ImportError:
    _C = None
    CUDA_EXT = False


# ════════════════════════════════════════════════════════════════════
# TEST UTILITIES
# ════════════════════════════════════════════════════════════════════

class _TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failures = []

    def ok(self, name: str):
        self.passed += 1
        print(f"  [PASS] {name}")

    def fail(self, name: str, reason: str):
        self.failed += 1
        self.failures.append((name, reason))
        print(f"  [FAIL] {name}  — {reason}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'═'*55}")
        print(f"  Tests: {self.passed}/{total} passed", end="")
        if self.failed:
            print(f"  ({self.failed} FAILED)")
            for name, reason in self.failures:
                print(f"    ✗ {name}: {reason}")
        else:
            print("  ✓ ALL PASS")
        print(f"{'═'*55}")
        return self.failed == 0


R = _TestResult()


def _gpu_time_ms(fn: Callable, warmup: int = 3, reps: int = 50) -> float:
    """Time a GPU function using CUDA events. Returns mean ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    ev_s.record()
    for _ in range(reps):
        fn()
    ev_e.record()
    torch.cuda.synchronize()
    return ev_s.elapsed_time(ev_e) / reps


def _check_close(a: torch.Tensor, b: torch.Tensor,
                 atol: float = 1e-4, rtol: float = 1e-4) -> tuple[bool, float]:
    max_err = (a - b).abs().max().item()
    ok = torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
    return ok, max_err


# ════════════════════════════════════════════════════════════════════
# SECTION 1 — CORRECTNESS TESTS
# ════════════════════════════════════════════════════════════════════

def test_swish_correctness():
    """Swish output matches PyTorch reference at float32 and float16."""
    print("\n── Test 1: Swish correctness ──────────────────────────────────")
    if not torch.cuda.is_available():
        print("  [SKIP] No CUDA")
        return

    # float32
    x32 = torch.randn(10000, device='cuda', dtype=torch.float32)
    ref = _swish_pytorch(x32)
    ours = swish(x32)
    ok, err = _check_close(ref, ours, atol=1e-5)
    if ok:
        R.ok(f"swish float32  max_err={err:.2e}")
    else:
        R.fail("swish float32", f"max_err={err:.2e}")

    # float16
    x16 = x32.half()
    ref16 = _swish_pytorch(x16).float()
    ours16 = swish(x16).float()
    ok16, err16 = _check_close(ref16, ours16, atol=5e-3)  # fp16 has less precision
    if ok16:
        R.ok(f"swish float16  max_err={err16:.2e}")
    else:
        R.fail("swish float16", f"max_err={err16:.2e}")

    # boundary values
    x_bound = torch.tensor([-100., -10., -1., 0., 1., 10., 100.],
                            device='cuda', dtype=torch.float32)
    ref_b = _swish_pytorch(x_bound)
    ours_b = swish(x_bound)
    ok_b, err_b = _check_close(ref_b, ours_b, atol=1e-4)
    if ok_b:
        R.ok(f"swish boundary values  max_err={err_b:.2e}")
    else:
        R.fail("swish boundary values", f"max_err={err_b:.2e}")


def test_mish_correctness():
    """Mish output matches PyTorch reference."""
    print("\n── Test 2: Mish correctness ───────────────────────────────────")
    if not torch.cuda.is_available():
        print("  [SKIP] No CUDA")
        return

    x = torch.randn(10000, device='cuda', dtype=torch.float32)
    ref = _mish_pytorch(x)
    ours = mish(x)
    ok, err = _check_close(ref, ours, atol=1e-4)
    if ok:
        R.ok(f"mish float32  max_err={err:.2e}")
    else:
        R.fail("mish float32", f"max_err={err:.2e}")

    # boundary
    x_b = torch.tensor([-20., -5., -1., 0., 1., 5., 20.],
                        device='cuda', dtype=torch.float32)
    ref_b = _mish_pytorch(x_b)
    ours_b = mish(x_b)
    ok_b, err_b = _check_close(ref_b, ours_b, atol=1e-4)
    if ok_b:
        R.ok(f"mish boundary values  max_err={err_b:.2e}")
    else:
        R.fail("mish boundary values", f"max_err={err_b:.2e}")


def test_fused_bias_swish_correctness():
    """FusedBiasSwish matches unfused (x+bias) then swish."""
    print("\n── Test 3: FusedBiasSwish correctness ─────────────────────────")
    if not torch.cuda.is_available():
        print("  [SKIP] No CUDA")
        return

    B, C = 64, 256
    x    = torch.randn(B, C, device='cuda')
    bias = torch.randn(C, device='cuda')

    ref  = _swish_pytorch(x + bias)
    ours = fused_bias_swish(x, bias)
    ok, err = _check_close(ref, ours, atol=1e-4)
    if ok:
        R.ok(f"fused_bias_swish [64,256]  max_err={err:.2e}")
    else:
        R.fail("fused_bias_swish [64,256]", f"max_err={err:.2e}")

    # 3D input [B, T, C]
    B, T, C = 4, 128, 512
    x3   = torch.randn(B, T, C, device='cuda')
    bias3 = torch.randn(C, device='cuda')
    ref3  = _swish_pytorch(x3 + bias3)
    ours3 = fused_bias_swish(x3, bias3)
    ok3, err3 = _check_close(ref3, ours3, atol=1e-4)
    if ok3:
        R.ok(f"fused_bias_swish [4,128,512]  max_err={err3:.2e}")
    else:
        R.fail("fused_bias_swish [4,128,512]", f"max_err={err3:.2e}")


def test_shape_invariance():
    """Kernels handle 1D, 2D, 3D, 4D tensors correctly."""
    print("\n── Test 4: Shape invariance ───────────────────────────────────")
    if not torch.cuda.is_available():
        print("  [SKIP] No CUDA")
        return

    shapes = [(1000,), (32, 256), (4, 16, 128), (2, 4, 8, 64)]
    for shape in shapes:
        x = torch.randn(*shape, device='cuda')
        ref  = _swish_pytorch(x)
        ours = swish(x)
        ok, err = _check_close(ref, ours)
        name = f"swish shape={list(shape)}"
        if ok:
            R.ok(f"{name}  err={err:.2e}")
        else:
            R.fail(name, f"err={err:.2e}")


# ════════════════════════════════════════════════════════════════════
# SECTION 2 — GRADIENT TESTS
# ════════════════════════════════════════════════════════════════════

def test_swish_gradcheck():
    """
    torch.autograd.gradcheck verifies analytical vs numerical gradient.
    Uses float64 inputs (required by gradcheck for numerical precision).
    Since our CUDA kernel only supports float32, gradcheck runs on
    the PyTorch fallback path — but this tests our backward formula.
    """
    print("\n── Test 5: Swish gradient check ───────────────────────────────")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # gradcheck needs float64, small input (for numerical stability)
    x = torch.randn(20, dtype=torch.float64, device=device, requires_grad=True)
    try:
        ok = torch.autograd.gradcheck(SwishFunction.apply, (x,),
                                       eps=1e-6, atol=1e-4, rtol=1e-4)
        if ok:
            R.ok("swish gradcheck (float64)")
        else:
            R.fail("swish gradcheck", "returned False")
    except Exception as e:
        R.fail("swish gradcheck", str(e))


def test_mish_gradcheck():
    """Mish gradient check using analytical backward."""
    print("\n── Test 6: Mish gradient check ────────────────────────────────")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(20, dtype=torch.float64, device=device, requires_grad=True)
    try:
        ok = torch.autograd.gradcheck(MishFunction.apply, (x,),
                                       eps=1e-6, atol=1e-4, rtol=1e-4)
        if ok:
            R.ok("mish gradcheck (float64)")
        else:
            R.fail("mish gradcheck", "returned False")
    except Exception as e:
        R.fail("mish gradcheck", str(e))


def test_fused_gradcheck():
    """FusedBiasSwish gradient check for both x and bias."""
    print("\n── Test 7: FusedBiasSwish gradient check ──────────────────────")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x    = torch.randn(8, 16, dtype=torch.float64, device=device, requires_grad=True)
    bias = torch.randn(16,   dtype=torch.float64, device=device, requires_grad=True)

    from swish import FusedBiasSwishFunction
    try:
        ok = torch.autograd.gradcheck(FusedBiasSwishFunction.apply, (x, bias),
                                       eps=1e-6, atol=1e-4, rtol=1e-4)
        if ok:
            R.ok("fused_bias_swish gradcheck (float64, grad_x + grad_bias)")
        else:
            R.fail("fused_bias_swish gradcheck", "returned False")
    except Exception as e:
        R.fail("fused_bias_swish gradcheck", str(e))


def test_swish_backward_values():
    """Verify backward gradient values match analytical formula."""
    print("\n── Test 8: Swish backward values ──────────────────────────────")
    if not torch.cuda.is_available():
        print("  [SKIP] No CUDA")
        return

    x = torch.randn(1000, device='cuda', requires_grad=True)
    grad_out = torch.ones_like(x)

    # reference: use PyTorch autograd on the functional form
    x_ref = x.clone().detach().requires_grad_(True)
    y_ref = _swish_pytorch(x_ref)
    y_ref.backward(grad_out)
    ref_grad = x_ref.grad.clone()

    # ours: use our custom backward
    x_ours = x.clone().detach().requires_grad_(True)
    y_ours = swish(x_ours)
    y_ours.backward(grad_out)
    our_grad = x_ours.grad.clone()

    ok, err = _check_close(ref_grad, our_grad, atol=1e-4)
    if ok:
        R.ok(f"swish backward values  max_err={err:.2e}")
    else:
        R.fail("swish backward values", f"max_err={err:.2e}")


# ════════════════════════════════════════════════════════════════════
# SECTION 3 — TRAINING TESTS
# ════════════════════════════════════════════════════════════════════

def test_training_loop_swish():
    """
    Train a small MLP for a few steps.
    Proves that:
    (a) backward computes valid gradients
    (b) parameters update correctly
    (c) loss decreases (model is learning)
    """
    print("\n── Test 9: Training loop (Swish MLP) ──────────────────────────")
    if not torch.cuda.is_available():
        print("  [SKIP] No CUDA")
        return

    torch.manual_seed(42)
    device = 'cuda'
    B, D_in, D_h, D_out = 32, 128, 256, 10

    # MLP with our Swish
    model = nn.Sequential(
        nn.Linear(D_in, D_h),
        Swish(),
        nn.Linear(D_h, D_h),
        Swish(),
        nn.Linear(D_h, D_out),
    ).to(device)

    # same model with nn.SiLU for comparison
    torch.manual_seed(42)
    model_ref = nn.Sequential(
        nn.Linear(D_in, D_h),
        nn.SiLU(),
        nn.Linear(D_h, D_h),
        nn.SiLU(),
        nn.Linear(D_h, D_out),
    ).to(device)

    # copy weights so both start identical
    model_ref.load_state_dict(model.state_dict())

    opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt_ref = torch.optim.Adam(model_ref.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    losses_ours, losses_ref = [], []
    for step in range(10):
        x = torch.randn(B, D_in, device=device)
        y = torch.randint(0, D_out, (B,), device=device)

        # ours
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
        losses_ours.append(loss.item())

        # reference
        opt_ref.zero_grad()
        loss_ref = criterion(model_ref(x), y)
        loss_ref.backward()
        opt_ref.step()
        losses_ref.append(loss_ref.item())

    # check: final losses are very close (same math, same weights)
    loss_diff = abs(losses_ours[-1] - losses_ref[-1])
    converging = losses_ours[-1] < losses_ours[0]  # loss decreased

    if loss_diff < 0.1:
        R.ok(f"training loss matches nn.SiLU  diff={loss_diff:.4f}")
    else:
        R.fail("training loss", f"diverged from nn.SiLU by {loss_diff:.4f}")

    if converging:
        R.ok(f"loss converging  {losses_ours[0]:.4f} → {losses_ours[-1]:.4f}")
    else:
        R.fail("loss converging", f"loss did not decrease: {losses_ours}")


def test_training_loop_fused():
    """Train a linear model that uses FusedBiasSwish."""
    print("\n── Test 10: Training loop (FusedBiasSwish) ────────────────────")
    if not torch.cuda.is_available():
        print("  [SKIP] No CUDA")
        return

    torch.manual_seed(0)
    device = 'cuda'
    B, D_in, D_out = 64, 256, 64

    class FusedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(D_in, D_out, bias=False)  # no bias in linear
            self.act    = FusedBiasSwish(D_out)                # fused bias+swish

        def forward(self, x):
            return self.act(self.linear(x))

    model = FusedModel().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    losses = []
    for step in range(5):
        x    = torch.randn(B, D_in, device=device)
        y    = torch.randn(B, D_out, device=device)
        loss = (model(x) - y).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # check grad flows to both weight and bias
    has_weight_grad = model.linear.weight.grad is not None
    has_bias_grad   = model.act.bias.grad is not None
    ok = has_weight_grad and has_bias_grad

    if ok:
        R.ok(f"FusedBiasSwish gradients flow to weight+bias ✓  losses={[f'{l:.3f}' for l in losses]}")
    else:
        R.fail("FusedBiasSwish gradients",
               f"weight_grad={has_weight_grad} bias_grad={has_bias_grad}")


# ════════════════════════════════════════════════════════════════════
# SECTION 4 — BENCHMARKS
# ════════════════════════════════════════════════════════════════════

def benchmark_swish_vs_silu(sizes=None, reps=100):
    """
    Latency: our Swish vs nn.SiLU (PyTorch built-in).

    PyTorch's SiLU is already highly optimised (uses cuDNN / ATen kernels).
    Our extension competes by removing dispatch overhead for the
    custom autograd path. For simple activations, the gap is small.
    The real win comes from the fused operations.
    """
    if sizes is None:
        sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

    print("\n── Benchmark 1: Swish forward vs nn.SiLU ──────────────────────")
    print(f"{'Elements':>12}  {'nn.SiLU (ms)':>14}  {'Our Swish (ms)':>16}  "
          f"{'Speedup':>9}  {'GELT/s (ours)':>14}")
    print("─" * 73)

    results = []
    silu = nn.SiLU()

    for N in sizes:
        x = torch.randn(N, device='cuda', dtype=torch.float32)

        t_silu  = _gpu_time_ms(lambda: silu(x), reps=reps)
        t_ours  = _gpu_time_ms(lambda: swish(x), reps=reps)
        speedup = t_silu / t_ours
        gelts   = N / (t_ours * 1e-3) / 1e9  # Giga-elements / second

        print(f"{N:>12,}  {t_silu:>14.4f}  {t_ours:>16.4f}  "
              f"{speedup:>9.3f}x  {gelts:>14.2f}")
        results.append((N, t_silu, t_ours, speedup))

    return results


def benchmark_swish_backward(sizes=None, reps=50):
    """Backward latency: our Swish vs nn.SiLU backward."""
    if sizes is None:
        sizes = [100_000, 1_000_000, 10_000_000]

    print("\n── Benchmark 2: Swish backward vs nn.SiLU backward ────────────")
    print(f"{'Elements':>12}  {'SiLU bwd (ms)':>15}  {'Our bwd (ms)':>14}  {'Speedup':>9}")
    print("─" * 55)

    silu = nn.SiLU()
    results = []

    for N in sizes:
        x_silu = torch.randn(N, device='cuda', requires_grad=True)
        x_ours = torch.randn(N, device='cuda', requires_grad=True)
        go = torch.ones(N, device='cuda')

        def silu_bwd():
            y = silu(x_silu)
            y.backward(go, retain_graph=True)

        def ours_bwd():
            y = swish(x_ours)
            y.backward(go, retain_graph=True)

        t_silu = _gpu_time_ms(silu_bwd, reps=reps)
        t_ours = _gpu_time_ms(ours_bwd, reps=reps)
        su = t_silu / t_ours

        print(f"{N:>12,}  {t_silu:>15.4f}  {t_ours:>14.4f}  {su:>9.3f}x")
        results.append((N, t_silu, t_ours, su))

    return results


def benchmark_mish(sizes=None, reps=100):
    """Mish vs PyTorch manual implementation."""
    if sizes is None:
        sizes = [100_000, 1_000_000, 10_000_000]

    print("\n── Benchmark 3: Mish vs PyTorch manual ────────────────────────")
    print(f"{'Elements':>12}  {'PT manual (ms)':>16}  {'Our Mish (ms)':>15}  {'Speedup':>9}")
    print("─" * 57)

    def pt_mish(x):
        return x * torch.tanh(F.softplus(x))

    results = []
    for N in sizes:
        x = torch.randn(N, device='cuda', dtype=torch.float32)

        t_pt   = _gpu_time_ms(lambda: pt_mish(x), reps=reps)
        t_ours = _gpu_time_ms(lambda: mish(x), reps=reps)
        su     = t_pt / t_ours

        print(f"{N:>12,}  {t_pt:>16.4f}  {t_ours:>15.4f}  {su:>9.3f}x")
        results.append((N, t_pt, t_ours, su))

    return results


def benchmark_fused_vs_unfused(sizes=None, reps=50):
    """
    The key benchmark: fused bias+swish vs two separate ops.

    Standard unfused sequence:
      z = x + bias   (kernel launch 1: reads x, bias; writes z)
      y = swish(z)   (kernel launch 2: reads z; writes y)
      = 2 reads + 2 writes = 4 HBM transactions

    Our fused kernel:
      y = swish(x + bias)   (1 kernel: reads x, bias; writes y)
      = 2 reads + 1 write = 3 HBM transactions  (25% fewer)

    Additionally: 1 fewer kernel launch = less scheduling overhead.
    """
    if sizes is None:
        sizes = [(64, 256), (64, 1024), (64, 4096), (256, 256), (256, 4096)]

    print("\n── Benchmark 4: FusedBiasSwish vs unfused ─────────────────────")
    print(f"{'Shape':>18}  {'Unfused fwd (ms)':>18}  {'Fused fwd (ms)':>16}  "
          f"{'Speedup':>9}  {'Mem saved':>10}")
    print("─" * 80)

    silu = nn.SiLU()
    results = []

    for (B, C) in sizes:
        x    = torch.randn(B, C, device='cuda')
        bias = torch.randn(C, device='cuda')

        def unfused():
            return silu(x + bias)

        def fused():
            return fused_bias_swish(x, bias)

        t_unf  = _gpu_time_ms(unfused, reps=reps)
        t_fus  = _gpu_time_ms(fused, reps=reps)
        su     = t_unf / t_fus
        # memory saved: intermediate (x+bias) tensor = B*C*4 bytes
        mem_mb = B * C * 4 / 1e6

        shape_str = f"[{B},{C}]"
        print(f"{shape_str:>18}  {t_unf:>18.4f}  {t_fus:>16.4f}  "
              f"{su:>9.3f}x  {mem_mb:>8.2f} MB")
        results.append((f"{B}×{C}", t_unf, t_fus, su, mem_mb))

    return results


def benchmark_fused_backward(reps=50):
    """
    Fused backward: computes grad_x and grad_bias in one kernel.
    Compare vs: separate backward for add and swish.
    """
    print("\n── Benchmark 5: FusedBiasSwish backward ───────────────────────")
    print(f"{'Shape':>14}  {'Unfused bwd (ms)':>18}  {'Fused bwd (ms)':>16}  {'Speedup':>9}")
    print("─" * 63)

    silu = nn.SiLU()
    shapes = [(64, 256), (64, 4096), (256, 4096)]

    for (B, C) in shapes:
        # unfused
        x1 = torch.randn(B, C, device='cuda', requires_grad=True)
        b1 = torch.randn(C, device='cuda', requires_grad=True)
        go = torch.ones(B, C, device='cuda')

        def unfused_bwd():
            y = silu(x1 + b1)
            y.backward(go, retain_graph=True)

        # fused
        x2 = torch.randn(B, C, device='cuda', requires_grad=True)
        b2 = torch.randn(C, device='cuda', requires_grad=True)

        def fused_bwd():
            y = fused_bias_swish(x2, b2)
            y.backward(go, retain_graph=True)

        t_unf = _gpu_time_ms(unfused_bwd, reps=reps)
        t_fus = _gpu_time_ms(fused_bwd, reps=reps)
        su = t_unf / t_fus
        print(f"[{B},{C}]{' ':>4}  {t_unf:>18.4f}  {t_fus:>16.4f}  {su:>9.3f}x")


def benchmark_dtype(reps=100):
    """float32 vs float16 throughput."""
    print("\n── Benchmark 6: float32 vs float16 ────────────────────────────")
    print(f"{'dtype':>10}  {'N=10M (ms)':>14}  {'GELT/s':>10}")
    print("─" * 38)

    N = 10_000_000
    for dtype, name in [(torch.float32, 'float32'), (torch.float16, 'float16')]:
        x = torch.randn(N, device='cuda', dtype=dtype)
        t = _gpu_time_ms(lambda: swish(x), reps=reps)
        gelts = N / (t * 1e-3) / 1e9
        print(f"{name:>10}  {t:>14.4f}  {gelts:>10.2f}")


def benchmark_vec4(reps=100):
    """Vectorised (float4) vs standard swish kernel."""
    if not CUDA_EXT:
        print("\n  [SKIP] CUDA extension not built — skipping vec4 benchmark")
        return

    print("\n── Benchmark 7: Standard vs float4 vectorised swish ───────────")
    print(f"{'Elements':>12}  {'Standard (ms)':>15}  {'Vec4 (ms)':>11}  {'Speedup':>9}")
    print("─" * 52)

    for N in [1_000_000, 10_000_000, 100_000_000]:
        x = torch.randn(N, device='cuda', dtype=torch.float32)
        t_std  = _gpu_time_ms(lambda: _C.swish_forward(x), reps=reps)
        t_vec4 = _gpu_time_ms(lambda: _C.swish_vec4_forward(x), reps=reps)
        su = t_std / t_vec4
        print(f"{N:>12,}  {t_std:>15.4f}  {t_vec4:>11.4f}  {su:>9.3f}x")


# ════════════════════════════════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════════════════════════════════

def plot_results(swish_results, fused_results):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('white')

    # ── Plot 1: Swish throughput ────────────────────────────────────
    ax = axes[0]
    Ns     = [r[0] for r in swish_results]
    t_silu = [r[1] for r in swish_results]
    t_ours = [r[2] for r in swish_results]
    x = np.arange(len(Ns))
    w = 0.35
    ax.bar(x - w/2, t_silu, w, label='nn.SiLU', color='#888888', alpha=0.85)
    ax.bar(x + w/2, t_ours, w, label='Our Swish', color='#2a7abf', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n//1000}K' if n < 1e6 else f'{n//1000000}M' for n in Ns],
                       fontsize=9)
    ax.set_yscale('log'); ax.set_xlabel('Tensor size')
    ax.set_ylabel('Latency (ms, log scale)')
    ax.set_title('Swish vs nn.SiLU — forward', fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    # ── Plot 2: Fused vs unfused speedup ────────────────────────────
    ax2 = axes[1]
    shapes   = [r[0] for r in fused_results]
    t_unf    = [r[1] for r in fused_results]
    t_fus    = [r[2] for r in fused_results]
    speedups = [r[3] for r in fused_results]
    x2 = np.arange(len(shapes))
    ax2.bar(x2 - w/2, t_unf, w, label='Unfused (add + SiLU)', color='#e07b39', alpha=0.85)
    ax2.bar(x2 + w/2, t_fus, w, label='Fused (our kernel)', color='#1a9e5c', alpha=0.85)
    ax2.set_xticks(x2); ax2.set_xticklabels(shapes, fontsize=8)
    ax2.set_xlabel('Shape [B, C]'); ax2.set_ylabel('Latency (ms)')
    ax2.set_title('FusedBiasSwish vs unfused', fontweight='bold')
    ax2.legend(); ax2.grid(axis='y', alpha=0.3)

    # ── Plot 3: Memory savings from fusion ───────────────────────────
    ax3 = axes[2]
    mem_saved = [r[4] for r in fused_results]
    bars = ax3.bar(x2, mem_saved, 0.5, color='#7f77dd', alpha=0.85)
    for bar, sp in zip(bars, speedups):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.005,
                 f'{sp:.2f}×', ha='center', fontsize=9, color='#3C3489', fontweight='bold')
    ax3.set_xticks(x2); ax3.set_xticklabels(shapes, fontsize=8)
    ax3.set_xlabel('Shape [B, C]')
    ax3.set_ylabel('HBM saved (MB) — intermediate tensor avoided')
    ax3.set_title('Memory savings from operator fusion\n(speedup labels)', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    plt.suptitle('Custom CUDA Activation Extension — Benchmark Results',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('swish_benchmark.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved swish_benchmark.png — put this in your README!")


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick',   action='store_true', help='Skip large tensor benchmarks')
    parser.add_argument('--no-plot', action='store_true', help='Skip matplotlib charts')
    parser.add_argument('--tests-only', action='store_true', help='Skip all benchmarks')
    args = parser.parse_args()

    print("═" * 55)
    print("  Custom CUDA Swish/Mish Extension — Test + Benchmark")
    print("═" * 55)
    info = backend_info()
    print(f"  CUDA extension: {'loaded ✓' if info['cuda_extension_loaded'] else 'NOT LOADED (fallback)'}")
    print(f"  GPU: {info['device']}")
    print(f"  PyTorch: {info['torch_version']}")
    print()

    # ── Run all correctness tests ─────────────────────────────────────
    test_swish_correctness()
    test_mish_correctness()
    test_fused_bias_swish_correctness()
    test_shape_invariance()
    test_swish_gradcheck()
    test_mish_gradcheck()
    test_fused_gradcheck()
    test_swish_backward_values()
    test_training_loop_swish()
    test_training_loop_fused()

    all_ok = R.summary()

    if args.tests_only or not torch.cuda.is_available():
        return 0 if all_ok else 1

    # ── Run benchmarks ────────────────────────────────────────────────
    print("\n" + "═"*55)
    print("  BENCHMARKS")
    print("═"*55)

    sizes_fwd = [1_000, 100_000, 1_000_000, 10_000_000] if not args.quick \
                else [100_000, 1_000_000]

    swish_res  = benchmark_swish_vs_silu(sizes=sizes_fwd)
    benchmark_swish_backward(sizes=[100_000, 1_000_000] if not args.quick
                             else [1_000_000])
    benchmark_mish(sizes=[100_000, 1_000_000] if not args.quick else [1_000_000])
    fused_res  = benchmark_fused_vs_unfused()
    benchmark_fused_backward()
    benchmark_dtype()
    benchmark_vec4()

    if not args.no_plot:
        plot_results(swish_res, fused_res)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
