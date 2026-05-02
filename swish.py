"""
swish.py
════════════════════════════════════════════════════════════════════
Python interface to the custom CUDA activation extension.

Provides three layers of abstraction:

  Layer 1 — torch.autograd.Function  (raw CUDA dispatch + custom backward)
    SwishFunction          forward/backward for Swish
    MishFunction           forward/backward for Mish
    FusedBiasSwishFunction forward/backward for fused bias+activation

  Layer 2 — nn.Module  (drop-in replacements)
    Swish                  replaces nn.SiLU
    Mish                   replaces custom Mish implementations
    FusedBiasSwish         replaces nn.Linear (bias) + nn.SiLU

  Layer 3 — Functional API  (like torch.nn.functional)
    swish(x)
    mish(x)
    fused_bias_swish(x, bias)

Usage:
    from swish import Swish, Mish, FusedBiasSwish

    # Drop-in for nn.SiLU:
    model = nn.Sequential(nn.Linear(512, 512), Swish())

    # Fused linear + activation:
    layer = FusedBiasSwish(in_features=512, out_features=256)
    out = layer(x)

Install:
    pip install -e .  (from the directory containing setup.py)
════════════════════════════════════════════════════════════════════
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

# ── Try to import the compiled CUDA extension ─────────────────────────
try:
    import swish_cuda as _C
    _CUDA_AVAILABLE = True
except ImportError:
    _C = None
    _CUDA_AVAILABLE = False
    import warnings
    warnings.warn(
        "[swish.py] CUDA extension not built. Using PyTorch fallback. "
        "Run `pip install -e .` to build the CUDA extension.",
        stacklevel=2
    )


# ── Pure-Python / PyTorch fallbacks ───────────────────────────────────
# These are used when the extension is not built, and also for
# float64 gradient checking (gradcheck requires float64, but our
# CUDA kernels only support float32/float16).

def _swish_pytorch(x: Tensor) -> Tensor:
    """Swish in pure PyTorch. Equivalent to F.silu(x)."""
    return x * torch.sigmoid(x)

def _mish_pytorch(x: Tensor) -> Tensor:
    """Mish in pure PyTorch."""
    return x * torch.tanh(F.softplus(x))

def _swish_grad_pytorch(x: Tensor) -> Tensor:
    """Analytical gradient of Swish w.r.t. x."""
    s  = torch.sigmoid(x)
    sw = x * s
    return s + sw * (1.0 - s)

def _mish_grad_pytorch(x: Tensor) -> Tensor:
    """Analytical gradient of Mish w.r.t. x."""
    sp    = F.softplus(x)
    t     = torch.tanh(sp)
    sech2 = 1.0 - t**2
    sig   = torch.sigmoid(x)
    return t + x * sech2 * sig


# ════════════════════════════════════════════════════════════════════
# LAYER 1 — torch.autograd.Function
# ════════════════════════════════════════════════════════════════════

class SwishFunction(torch.autograd.Function):
    """
    Custom autograd Function for Swish activation.

    Forward:  y = x * sigmoid(x)
    Backward: grad_x = grad_out * swish'(x)
               where swish'(x) = sigmoid(x) * (1 + x*(1-sigmoid(x)))

    Saves only x (not y) — computes grad from x directly.
    This avoids one HBM round-trip vs saving y.

    Memory footprint:
      Standard torch.sigmoid + multiply: saves x, sigmoid(x), and y
      Our function: saves only x
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        # save_for_backward registers tensors to be saved for the backward pass
        # these are accessible in backward() via ctx.saved_tensors
        ctx.save_for_backward(x)

        if _CUDA_AVAILABLE and x.is_cuda and x.dtype in (torch.float32, torch.float16):
            return _C.swish_forward(x.contiguous())
        else:
            return _swish_pytorch(x)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        # ctx.saved_tensors returns a tuple of all saved tensors
        x, = ctx.saved_tensors

        if _CUDA_AVAILABLE and x.is_cuda and x.dtype == torch.float32:
            grad_in = _C.swish_backward(x.contiguous(), grad_out.contiguous())
        else:
            # fallback: analytical grad in PyTorch
            # also used by gradcheck (which uses float64)
            grad_in = grad_out * _swish_grad_pytorch(x)

        # return one gradient per input argument to forward()
        # SwishFunction.forward(ctx, x) has 1 non-ctx arg → return 1 gradient
        return grad_in


class MishFunction(torch.autograd.Function):
    """
    Custom autograd Function for Mish activation.

    Forward:  y = x * tanh(softplus(x))
    Backward: grad_x = grad_out * mish'(x)
               mish'(x) = tanh(sp) + x * sech²(sp) * sigmoid(x)
               where sp = softplus(x) = ln(1 + e^x)

    The backward kernel recomputes sp and t from x — no need to
    save the intermediate tanh/softplus values.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        if _CUDA_AVAILABLE and x.is_cuda and x.dtype in (torch.float32, torch.float16):
            return _C.mish_forward(x.contiguous())
        else:
            return _mish_pytorch(x)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x, = ctx.saved_tensors
        if _CUDA_AVAILABLE and x.is_cuda and x.dtype == torch.float32:
            grad_in = _C.mish_backward(x.contiguous(), grad_out.contiguous())
        else:
            grad_in = grad_out * _mish_grad_pytorch(x)
        return grad_in


class FusedBiasSwishFunction(torch.autograd.Function):
    """
    Custom autograd Function for fused bias addition + Swish.

    Forward:  y = swish(x + bias)
    Backward: grad_x    = grad_out * swish'(x + bias)
              grad_bias = sum(grad_out * swish'(x + bias), over batch)

    Fusion benefit: x + bias and swish() happen in the same kernel.
    Standard approach would be two separate kernel launches:
      kernel 1: z = x + bias   (write z to HBM)
      kernel 2: y = swish(z)   (read z from HBM)
    Our fused kernel: reads x + bias, writes y — one fewer HBM write.

    Saves: x and bias (not the intermediate x+bias) → lower memory.
    """

    @staticmethod
    def forward(ctx, x: Tensor, bias: Tensor) -> Tensor:
        ctx.save_for_backward(x, bias)
        if _CUDA_AVAILABLE and x.is_cuda:
            return _C.fused_bias_swish_forward(
                x.contiguous(), bias.contiguous())
        else:
            return _swish_pytorch(x + bias)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x, bias = ctx.saved_tensors
        if _CUDA_AVAILABLE and x.is_cuda and x.dtype == torch.float32:
            grad_x, grad_bias = _C.fused_bias_swish_backward(
                x.contiguous(), bias.contiguous(), grad_out.contiguous())
        else:
            # PyTorch fallback backward
            z      = x + bias
            dact   = _swish_grad_pytorch(z)
            grad_x = grad_out * dact
            # grad_bias: sum over all dims except the last (channel) dim
            grad_bias = (grad_out * dact).reshape(-1, bias.shape[0]).sum(0)
        return grad_x, grad_bias


# ════════════════════════════════════════════════════════════════════
# LAYER 2 — nn.Module
# ════════════════════════════════════════════════════════════════════

class Swish(nn.Module):
    """
    Swish activation module — drop-in replacement for nn.SiLU.

    f(x) = x * sigmoid(x)

    Used in: EfficientNet, GPT-Neo, PaLM, LLaMA, MobileNetV3

    Example:
        model = nn.Sequential(
            nn.Linear(512, 512),
            Swish(),    # instead of nn.ReLU() or nn.SiLU()
            nn.Linear(512, 10)
        )
    """

    def forward(self, x: Tensor) -> Tensor:
        return SwishFunction.apply(x)

    def extra_repr(self) -> str:
        return "custom_cuda=True" if _CUDA_AVAILABLE else "cuda_ext=False (fallback)"


class Mish(nn.Module):
    """
    Mish activation module.

    f(x) = x * tanh(ln(1 + e^x))

    Used in: YOLOv4, YOLOv5, many vision detection models.
    Generally outperforms ReLU on image tasks; competitive with Swish.

    Example:
        self.act = Mish()
    """

    def forward(self, x: Tensor) -> Tensor:
        return MishFunction.apply(x)

    def extra_repr(self) -> str:
        return "custom_cuda=True" if _CUDA_AVAILABLE else "cuda_ext=False (fallback)"


class FusedBiasSwish(nn.Module):
    """
    Fused linear layer (without weight) + Swish activation.

    Equivalent to: nn.Linear(in_features, out_features, bias=True) + Swish()
    BUT: the bias addition and Swish are fused into a single CUDA kernel.

    Typically used AFTER a weight matrix multiplication:
        y = W @ x + bias   (standard nn.Linear with bias=False)
        y = FusedBiasSwish(features)(y)

    Or as a complete fused linear:
        layer = FusedLinear(512, 256)   # weight only, no bias
        act   = FusedBiasSwish(256)
        out   = act(layer(x))

    Example showing parameter savings from single kernel:
        # Standard approach (3 kernels: matmul, add, swish):
        out = F.silu(F.linear(x, weight, bias))

        # Fused approach (2 kernels: matmul + fused_bias_swish):
        out = fused_bias_swish(F.linear(x, weight), bias)
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.bias = nn.Parameter(torch.zeros(num_features))
        self._init_bias()

    def _init_bias(self):
        # initialise with small random values (same as Linear default)
        bound = 1 / math.sqrt(self.num_features)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        # x: [..., num_features] — any number of leading dims
        return FusedBiasSwishFunction.apply(x, self.bias)

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}"


# ════════════════════════════════════════════════════════════════════
# LAYER 3 — Functional API
# ════════════════════════════════════════════════════════════════════

def swish(x: Tensor) -> Tensor:
    """Swish activation: f(x) = x * sigmoid(x). Differentiable."""
    return SwishFunction.apply(x)

def mish(x: Tensor) -> Tensor:
    """Mish activation: f(x) = x * tanh(softplus(x)). Differentiable."""
    return MishFunction.apply(x)

def fused_bias_swish(x: Tensor, bias: Tensor) -> Tensor:
    """Fused bias + Swish: f(x, b) = swish(x + b). Differentiable."""
    return FusedBiasSwishFunction.apply(x, bias)


# ════════════════════════════════════════════════════════════════════
# UTILITY — show backend info
# ════════════════════════════════════════════════════════════════════

def backend_info() -> dict:
    """Return info about the active backend."""
    return {
        "cuda_extension_loaded": _CUDA_AVAILABLE,
        "extension_name": "swish_cuda" if _CUDA_AVAILABLE else None,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
    }
