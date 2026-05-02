/*
 * swish_cuda.cu
 * ════════════════════════════════════════════════════════════════════
 * Project 4: Custom PyTorch CUDA Extension
 *
 * Implements fused activation kernels with hand-written backward passes.
 * Two activations + variants:
 *
 *   1. Swish (SiLU)         forward + backward
 *      f(x) = x * sigmoid(x) = x / (1 + e^-x)
 *      Used in: EfficientNet, GPT-Neo, PaLM, LLaMA
 *
 *   2. Mish                 forward + backward
 *      f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
 *      Used in: YOLOv4, many vision models
 *
 *   3. Fused Bias + Swish   forward + backward
 *      f(x, b) = swish(x + b)
 *      Key optimisation: fuses two ops → one memory round-trip
 *
 * Why "fused" matters:
 *   Standard PyTorch: y = swish(x + bias)
 *     → kernel 1: reads x, reads bias, writes (x+bias) to HBM
 *     → kernel 2: reads (x+bias), writes swish(x+bias) to HBM
 *     = 2 reads + 2 writes = 4 HBM round-trips
 *
 *   Our fused kernel:
 *     → reads x, reads bias, writes swish(x+bias)
 *     = 2 reads + 1 write = 3 HBM round-trips  (25% less memory traffic)
 *
 * Compile (used by setup.py — not called directly):
 *   nvcc -O2 -arch=sm_75 --use_fast_math swish_cuda.cu ...
 * ════════════════════════════════════════════════════════════════════
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>   /* for AT_DISPATCH_FLOATING_TYPES */

/* ── constants ──────────────────────────────────────────────────── */
#define BLOCK_SIZE   256       /* threads per block for 1-D kernels */
#define WARP_SIZE     32

/* ── error check ────────────────────────────────────────────────── */
#define CUDA_CHECK(x) do { \
    cudaError_t e=(x); \
    if(e!=cudaSuccess){ \
        printf("CUDA error %s:%d  %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
    } \
} while(0)


/* ════════════════════════════════════════════════════════════════════
   DEVICE HELPERS
   Inline device functions shared across all kernels.
   Using __forceinline__ and __device__ lets the compiler inline them
   into each kernel without function-call overhead.
   ════════════════════════════════════════════════════════════════════ */

/* sigmoid: σ(x) = 1 / (1 + e^-x) */
template <typename T>
__device__ __forceinline__ T sigmoid(T x) {
    return (T)1.0 / ((T)1.0 + exp(-x));
}

/* swish: f(x) = x * σ(x) */
template <typename T>
__device__ __forceinline__ T swish_fwd(T x) {
    T s = sigmoid(x);
    return x * s;
}

/*
 * swish derivative: d/dx [x * σ(x)]
 *   = σ(x) + x * σ(x) * (1 - σ(x))
 *   = σ(x) * (1 + x * (1 - σ(x)))
 *   = σ(x) * (1 + x - x*σ(x))
 *   = σ(x) + swish(x) * (1 - σ(x))
 *
 * We compute this from x directly — no need to re-read swish output.
 * This is why the backward kernel only needs to save x, not swish(x).
 */
template <typename T>
__device__ __forceinline__ T swish_bwd(T x) {
    T s    = sigmoid(x);
    T sw   = x * s;           /* swish(x) */
    return s + sw * ((T)1.0 - s);
}

/*
 * Mish: f(x) = x * tanh(softplus(x))
 *            = x * tanh(ln(1 + e^x))
 *
 * Numerically stable implementation:
 *   softplus(x) = ln(1 + e^x)
 *   For large x: softplus(x) ≈ x (avoid overflow in e^x)
 *   We use: ln(1 + e^x) = x + ln(1 + e^-x) for x > 20
 */
template <typename T>
__device__ __forceinline__ T softplus(T x) {
    /* stable log(1 + exp(x)) */
    return (x > (T)20.0) ? x : log((T)1.0 + exp(x));
}

template <typename T>
__device__ __forceinline__ T mish_fwd(T x) {
    return x * tanh(softplus(x));
}

/*
 * Mish derivative: d/dx [x * tanh(softplus(x))]
 *   Let sp = softplus(x) = ln(1 + e^x)
 *   Let t  = tanh(sp)
 *   Let σ  = sigmoid(x) = e^x / (1 + e^x)
 *
 *   d/dx [x * tanh(sp)] = tanh(sp) + x * (1 - tanh²(sp)) * σ(x)
 *                       = t + x * sech²(sp) * σ(x)
 *
 * We compute this from x only — key for memory efficiency.
 */
template <typename T>
__device__ __forceinline__ T mish_bwd(T x) {
    T sp  = softplus(x);
    T t   = tanh(sp);
    T sech2 = (T)1.0 - t * t;    /* sech²(sp) = 1 - tanh²(sp) */
    T sig = sigmoid(x);
    return t + x * sech2 * sig;
}


/* ════════════════════════════════════════════════════════════════════
   KERNEL 1 — Swish forward
   ════════════════════════════════════════════════════════════════════ */
template <typename scalar_t>
__global__ void swish_forward_kernel(
    const scalar_t* __restrict__ x,
    scalar_t*       __restrict__ y,
    int64_t N)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        y[i] = swish_fwd(x[i]);
}

/* ════════════════════════════════════════════════════════════════════
   KERNEL 2 — Swish backward
   Inputs : x (saved from forward), grad_output (from next layer)
   Outputs: grad_input (passed to previous layer)
   Chain rule: grad_input = grad_output * d(swish)/dx
   ════════════════════════════════════════════════════════════════════ */
template <typename scalar_t>
__global__ void swish_backward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ grad_out,
    scalar_t*       __restrict__ grad_in,
    int64_t N)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        grad_in[i] = grad_out[i] * swish_bwd(x[i]);
}

/* ════════════════════════════════════════════════════════════════════
   KERNEL 3 — Mish forward
   ════════════════════════════════════════════════════════════════════ */
template <typename scalar_t>
__global__ void mish_forward_kernel(
    const scalar_t* __restrict__ x,
    scalar_t*       __restrict__ y,
    int64_t N)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        y[i] = mish_fwd(x[i]);
}

/* ════════════════════════════════════════════════════════════════════
   KERNEL 4 — Mish backward
   ════════════════════════════════════════════════════════════════════ */
template <typename scalar_t>
__global__ void mish_backward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ grad_out,
    scalar_t*       __restrict__ grad_in,
    int64_t N)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        grad_in[i] = grad_out[i] * mish_bwd(x[i]);
}


/* ════════════════════════════════════════════════════════════════════
   KERNEL 5 — Fused Bias + Swish forward
   Computes: y[i] = swish(x[i] + bias[i % C])
   where C is the number of channels (bias is broadcast across batch).

   Shape convention: x is [N, C] (or any shape where last dim = C).
   bias is [C].

   Memory saving vs two kernels:
     Standard: (x+bias) written to HBM, then swish reads it
     Fused:    addition and swish happen in registers — never hits HBM
   ════════════════════════════════════════════════════════════════════ */
template <typename scalar_t>
__global__ void fused_bias_swish_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ bias,
    scalar_t*       __restrict__ y,
    int64_t N,     /* total elements */
    int64_t C)     /* number of channels (bias length) */
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        scalar_t val = x[i] + bias[i % C];   /* fused add */
        y[i] = swish_fwd(val);                /* fused swish */
    }
}

/* ════════════════════════════════════════════════════════════════════
   KERNEL 6 — Fused Bias + Swish backward
   Produces: grad_x[i]    = grad_out[i] * swish'(x[i] + bias[i%C])
             grad_bias[c] = sum over all i where i%C==c of
                            grad_out[i] * swish'(x[i] + bias[i%C])

   grad_bias reduction:
     Each thread computes its local grad_bias contribution.
     We use atomic add to accumulate into the shared bias gradient.
     For large N, a reduction kernel would be faster, but atomicAdd
     is correct and simpler for the extension interface.
   ════════════════════════════════════════════════════════════════════ */
template <typename scalar_t>
__global__ void fused_bias_swish_backward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ grad_out,
    scalar_t*       __restrict__ grad_x,
    scalar_t*       __restrict__ grad_bias,   /* [C], accumulated with atomicAdd */
    int64_t N,
    int64_t C)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int64_t c     = i % C;
        scalar_t val  = x[i] + bias[c];
        scalar_t dact = swish_bwd(val);          /* d(swish)/d(val) */
        scalar_t g    = grad_out[i] * dact;

        grad_x[i] = g;                           /* dx: straightforward */
        atomicAdd(&grad_bias[c], g);             /* db: accumulate across batch */
    }
}


/* ════════════════════════════════════════════════════════════════════
   KERNEL 7 — Vectorised Swish forward (float4)
   Processes 4 floats per thread using float4 vector loads.
   float4 load = one 128-bit memory transaction instead of four 32-bit.
   Improves memory bandwidth utilisation by ~4× for large tensors.
   Only works for float32 (not templated).
   ════════════════════════════════════════════════════════════════════ */
__global__ void swish_forward_vec4_kernel(
    const float* __restrict__ x,
    float*       __restrict__ y,
    int64_t N)
{
    int64_t i4 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t i  = i4 * 4;
    if (i + 3 < N) {
        /* vector load: reads 4 floats in one 128-bit transaction */
        float4 vx = reinterpret_cast<const float4*>(x)[i4];
        float4 vy;
        vy.x = swish_fwd(vx.x);
        vy.y = swish_fwd(vx.y);
        vy.z = swish_fwd(vx.z);
        vy.w = swish_fwd(vx.w);
        reinterpret_cast<float4*>(y)[i4] = vy;
    } else {
        /* tail: handle remaining elements individually */
        for (int64_t j = i; j < N && j < i + 4; ++j)
            y[j] = swish_fwd(x[j]);
    }
}


/* ════════════════════════════════════════════════════════════════════
   C++ LAUNCHER FUNCTIONS
   Called from swish_cuda.cpp (the pybind11 binding).
   These bridge the AT_DISPATCH_FLOATING_TYPES macro (which selects
   the right template instantiation) with the kernel launch syntax.
   ════════════════════════════════════════════════════════════════════ */

/* ── Swish forward ─────────────────────────────────────────────── */
torch::Tensor swish_forward_cuda(torch::Tensor x) {
    auto y  = torch::empty_like(x);
    int64_t N = x.numel();
    int blocks = (int)((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "swish_forward_cuda", ([&] {
        swish_forward_kernel<scalar_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            N);
    }));
    return y;
}

/* ── Swish backward ────────────────────────────────────────────── */
torch::Tensor swish_backward_cuda(torch::Tensor x, torch::Tensor grad_out) {
    auto grad_in = torch::empty_like(x);
    int64_t N = x.numel();
    int blocks = (int)((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "swish_backward_cuda", ([&] {
        swish_backward_kernel<scalar_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            grad_out.data_ptr<scalar_t>(),
            grad_in.data_ptr<scalar_t>(),
            N);
    }));
    return grad_in;
}

/* ── Mish forward ──────────────────────────────────────────────── */
torch::Tensor mish_forward_cuda(torch::Tensor x) {
    auto y  = torch::empty_like(x);
    int64_t N = x.numel();
    int blocks = (int)((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "mish_forward_cuda", ([&] {
        mish_forward_kernel<scalar_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            N);
    }));
    return y;
}

/* ── Mish backward ─────────────────────────────────────────────── */
torch::Tensor mish_backward_cuda(torch::Tensor x, torch::Tensor grad_out) {
    auto grad_in = torch::empty_like(x);
    int64_t N = x.numel();
    int blocks = (int)((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "mish_backward_cuda", ([&] {
        mish_backward_kernel<scalar_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            grad_out.data_ptr<scalar_t>(),
            grad_in.data_ptr<scalar_t>(),
            N);
    }));
    return grad_in;
}

/* ── Fused bias + swish forward ────────────────────────────────── */
torch::Tensor fused_bias_swish_forward_cuda(torch::Tensor x, torch::Tensor bias) {
    auto y  = torch::empty_like(x);
    int64_t N = x.numel();
    int64_t C = bias.numel();
    int blocks = (int)((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_bias_swish_forward_cuda", ([&] {
        fused_bias_swish_forward_kernel<scalar_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            N, C);
    }));
    return y;
}

/* ── Fused bias + swish backward ───────────────────────────────── */
std::vector<torch::Tensor> fused_bias_swish_backward_cuda(
    torch::Tensor x, torch::Tensor bias, torch::Tensor grad_out)
{
    auto grad_x    = torch::empty_like(x);
    auto grad_bias = torch::zeros_like(bias);   /* zeros: atomicAdd accumulates into this */
    int64_t N = x.numel();
    int64_t C = bias.numel();
    int blocks = (int)((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_bias_swish_backward_cuda", ([&] {
        fused_bias_swish_backward_kernel<scalar_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            grad_out.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            grad_bias.data_ptr<scalar_t>(),
            N, C);
    }));
    return {grad_x, grad_bias};
}

/* ── Vectorised swish forward (float32 only) ───────────────────── */
torch::Tensor swish_forward_vec4_cuda(torch::Tensor x) {
    TORCH_CHECK(x.scalar_type() == torch::kFloat32,
                "swish_vec4 only supports float32");
    auto y = torch::empty_like(x);
    int64_t N = x.numel();
    int64_t N4 = (N + 3) / 4;
    int blocks = (int)((N4 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    auto stream = at::cuda::getCurrentCUDAStream();

    swish_forward_vec4_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N);
    return y;
}
