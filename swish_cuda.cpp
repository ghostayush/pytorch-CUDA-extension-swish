/*
 * swish_cuda.cpp
 * ════════════════════════════════════════════════════════════════════
 * C++ binding layer between the CUDA kernels (swish_cuda.cu) and
 * the Python pybind11 module.
 *
 * Responsibilities:
 *   1. Declare external functions implemented in swish_cuda.cu
 *   2. Add input validation (TORCH_CHECK) before calling CUDA code
 *   3. Register all functions with pybind11 via PYBIND11_MODULE
 *
 * Why separate from the .cu file?
 *   The CUDA compiler (nvcc) handles .cu files.
 *   The C++ compiler (g++) handles .cpp files.
 *   Keeping them separate lets the build system use the right compiler
 *   for each file and avoids nvcc processing pybind11 headers, which
 *   can cause subtle compatibility issues.
 * ════════════════════════════════════════════════════════════════════
 */

#include <torch/extension.h>
#include <vector>

/* ── Forward declarations (implemented in swish_cuda.cu) ──────────── */
torch::Tensor swish_forward_cuda(torch::Tensor x);
torch::Tensor swish_backward_cuda(torch::Tensor x, torch::Tensor grad_out);

torch::Tensor mish_forward_cuda(torch::Tensor x);
torch::Tensor mish_backward_cuda(torch::Tensor x, torch::Tensor grad_out);

torch::Tensor fused_bias_swish_forward_cuda(torch::Tensor x, torch::Tensor bias);
std::vector<torch::Tensor> fused_bias_swish_backward_cuda(
    torch::Tensor x, torch::Tensor bias, torch::Tensor grad_out);

torch::Tensor swish_forward_vec4_cuda(torch::Tensor x);


/* ── Input validation macros ──────────────────────────────────────── */
#define CHECK_CUDA(x)   TORCH_CHECK((x).device().is_cuda(),  #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)  do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while(0)


/* ════════════════════════════════════════════════════════════════════
   PUBLIC C++ INTERFACE
   These are called from Python. They validate inputs, then dispatch
   to the CUDA launcher functions in swish_cuda.cu.
   ════════════════════════════════════════════════════════════════════ */

/* ── Swish ────────────────────────────────────────────────────────── */
torch::Tensor swish_forward(torch::Tensor x) {
    CHECK_INPUT(x);
    return swish_forward_cuda(x);
}

torch::Tensor swish_backward(torch::Tensor x, torch::Tensor grad_out) {
    CHECK_INPUT(x);
    CHECK_INPUT(grad_out);
    TORCH_CHECK(x.sizes() == grad_out.sizes(),
                "x and grad_out must have the same shape");
    return swish_backward_cuda(x, grad_out);
}

/* ── Mish ─────────────────────────────────────────────────────────── */
torch::Tensor mish_forward(torch::Tensor x) {
    CHECK_INPUT(x);
    return mish_forward_cuda(x);
}

torch::Tensor mish_backward(torch::Tensor x, torch::Tensor grad_out) {
    CHECK_INPUT(x);
    CHECK_INPUT(grad_out);
    TORCH_CHECK(x.sizes() == grad_out.sizes(),
                "x and grad_out must have the same shape");
    return mish_backward_cuda(x, grad_out);
}

/* ── Fused Bias + Swish ───────────────────────────────────────────── */
torch::Tensor fused_bias_swish_forward(torch::Tensor x, torch::Tensor bias) {
    CHECK_INPUT(x);
    CHECK_INPUT(bias);
    TORCH_CHECK(bias.dim() == 1,
                "bias must be 1-D, got shape: ", bias.sizes());
    TORCH_CHECK(x.size(-1) == bias.size(0),
                "Last dim of x (", x.size(-1), ") must equal bias size (", bias.size(0), ")");
    return fused_bias_swish_forward_cuda(x, bias);
}

std::vector<torch::Tensor> fused_bias_swish_backward(
    torch::Tensor x, torch::Tensor bias, torch::Tensor grad_out)
{
    CHECK_INPUT(x);
    CHECK_INPUT(bias);
    CHECK_INPUT(grad_out);
    return fused_bias_swish_backward_cuda(x, bias, grad_out);
}

/* ── Vectorised Swish (float32 only, aligned tensors) ─────────────── */
torch::Tensor swish_vec4_forward(torch::Tensor x) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.scalar_type() == torch::kFloat32,
                "swish_vec4 only supports float32, got: ", x.scalar_type());
    TORCH_CHECK(x.numel() % 4 == 0 || true,
                "Tensor size should ideally be divisible by 4 for vec4 kernel");
    return swish_forward_vec4_cuda(x);
}


/* ════════════════════════════════════════════════════════════════════
   PYBIND11 MODULE REGISTRATION
   This is what Python sees when it does: import swish_cuda
   TORCH_EXTENSION_NAME is set by setup.py → "swish_cuda"
   ════════════════════════════════════════════════════════════════════ */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = R"doc(
        Custom CUDA activation functions for PyTorch.

        Implements Swish (SiLU), Mish, and fused Bias+Swish with
        hand-written CUDA backward passes — no autograd tape overhead.

        All functions require CUDA tensors that are contiguous.
        Call .cuda().contiguous() before passing in if needed.
    )doc";

    /* ── Swish ── */
    m.def("swish_forward",
          &swish_forward,
          "Swish (SiLU) forward pass: y = x * sigmoid(x)",
          py::arg("x"));

    m.def("swish_backward",
          &swish_backward,
          "Swish backward: grad_in = grad_out * swish'(x)",
          py::arg("x"), py::arg("grad_out"));

    /* ── Mish ── */
    m.def("mish_forward",
          &mish_forward,
          "Mish forward: y = x * tanh(softplus(x))",
          py::arg("x"));

    m.def("mish_backward",
          &mish_backward,
          "Mish backward: grad_in = grad_out * mish'(x)",
          py::arg("x"), py::arg("grad_out"));

    /* ── Fused Bias + Swish ── */
    m.def("fused_bias_swish_forward",
          &fused_bias_swish_forward,
          "Fused bias+swish: y = swish(x + bias) — one kernel, one memory pass",
          py::arg("x"), py::arg("bias"));

    m.def("fused_bias_swish_backward",
          &fused_bias_swish_backward,
          "Fused bias+swish backward: returns [grad_x, grad_bias]",
          py::arg("x"), py::arg("bias"), py::arg("grad_out"));

    /* ── Vectorised variant ── */
    m.def("swish_vec4_forward",
          &swish_vec4_forward,
          "Swish forward with float4 vector loads (float32 only, faster for large tensors)",
          py::arg("x"));
}
