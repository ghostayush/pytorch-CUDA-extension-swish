"""
setup.py
════════════════════════════════════════════════════════════════════
Build configuration for the custom Swish/Mish CUDA extension.

Usage:
    pip install -e .                        # install in editable (dev) mode
    pip install -e . --no-build-isolation   # faster, skips venv
    python setup.py build_ext --inplace     # build .so in current directory

After installation:
    import swish_cuda                       # compiled CUDA module
    from swish import Swish, Mish, FusedBiasSwish  # Python wrappers

GPU architecture flags:
    The script auto-detects your GPU's compute capability.
    If building without a GPU (for CI/packaging), it defaults to sm_75.
    Override with: TORCH_CUDA_ARCH_LIST="8.0" pip install -e .

Files compiled:
    swish_cuda.cpp  — C++ pybind11 binding (compiled by g++)
    swish_cuda.cu   — CUDA kernels (compiled by nvcc)
════════════════════════════════════════════════════════════════════
"""

import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_gpu_arch_flags():
    """
    Detect GPU compute capability and return appropriate -arch flags.
    Falls back to sm_75 (T4) if no GPU is available.
    """
    if not torch.cuda.is_available():
        print("[setup.py] No GPU detected — defaulting to sm_75 (T4)")
        return ["-arch=sm_75"]

    # respect TORCH_CUDA_ARCH_LIST env var if set
    arch_list_env = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    if arch_list_env:
        arches = arch_list_env.replace(".", "").split()
        flags = [f"-arch=sm_{a}" for a in arches]
        print(f"[setup.py] Using TORCH_CUDA_ARCH_LIST: {flags}")
        return flags

    # auto-detect from current GPU
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    device_name = torch.cuda.get_device_name(0)
    print(f"[setup.py] Detected GPU: {device_name} ({arch})")
    return [f"-arch={arch}"]


arch_flags = get_gpu_arch_flags()

# ── Compiler flags ────────────────────────────────────────────────────
nvcc_flags = [
    "-O2",
    "--use_fast_math",          # __expf, __sinf etc — ~2× faster, ULP-level precision loss
    "--maxrregcount=64",        # cap register use → higher occupancy
    "-std=c++17",
    "--expt-relaxed-constexpr", # allow constexpr in device code
    "--threads=4",              # parallel compilation within nvcc
] + arch_flags

cxx_flags = [
    "-O2",
    "-std=c++17",
]

# On Linux, add frame pointers for better profiling with Nsight
if os.name != 'nt':
    cxx_flags.append("-fno-omit-frame-pointer")


setup(
    name="swish_cuda",
    version="0.2.0",
    description="Custom CUDA activation functions: Swish, Mish, FusedBiasSwish",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    author="Your Name",
    python_requires=">=3.8",
    packages=find_packages(exclude=["tests*"]),
    ext_modules=[
        CUDAExtension(
            name="swish_cuda",               # import name in Python
            sources=[
                "swish_cuda.cpp",            # C++ binding (g++)
                "swish_cuda.cu",             # CUDA kernels (nvcc)
            ],
            extra_compile_args={
                "cxx":  cxx_flags,
                "nvcc": nvcc_flags,
            },
            # Link against CUDA math library (for expf, tanhf etc in device code)
            libraries=["cudart"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(
            use_ninja=True,      # use Ninja build system (faster than make)
            no_python_abi_suffix=False,
        )
    },
    install_requires=[
        "torch>=1.13",
    ],
    extras_require={
        "test": ["pytest", "matplotlib", "numpy"],
    },
    zip_safe=False,
)
