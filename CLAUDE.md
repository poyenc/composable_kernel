# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Composable Kernel (CK) is AMD's high-performance GPU kernel library for machine learning workloads. It uses HIP C++ with a tile-based programming model and tensor coordinate transformation techniques.

**Two implementations exist:**
- **CK Tile** (`include/ck_tile/`) - Modern tile-programming API, preferred for new development
- **Legacy CK** (`include/ck/`) - Older implementation, still supported

## Build Commands

```bash
# Development build (from build directory)
mkdir build && cd build
../script/cmake-ck-dev.sh .. "gfx908;gfx90a;gfx942"
make -j32  # Use ~2GB RAM per thread

# Build specific targets
make tile_example_fmha_fwd    # FMHA forward example
make tile_example_fmha_bwd    # FMHA backward example
make ckProfiler               # Performance profiler

# Tests
make smoke      # Quick tests (<30s each)
make regression # Long tests (>=30s each)
make check      # All tests

# Single test
ctest -R "fmha" -V
```

**CMake options:**
- `GPU_TARGETS="gfx908;gfx90a;gfx942"` - Target GPU architectures
- `DTYPES="fp32;fp16;fp8;bf16;int8"` - Data types to build
- `BUILD_DEV=ON` - Development mode

## Code Formatting

CK uses clang-format. Install pre-commit hooks:
```bash
sudo script/install_precommit.sh
```

Disable temporarily with `git commit --no-verify`.

## Architecture

### Four-Layer Structure
1. **Templated Tile Operators** - Low-level tile operations
2. **Templated Kernel/Invoker** - Kernel templates with tile operators
3. **Instantiated Kernel/Invoker** - Concrete kernel instances
4. **Client API** - User-facing API

### Key Directories
- `include/ck_tile/core/` - Core utilities (containers, data types, coordinate transforms)
- `include/ck_tile/ops/` - Operator implementations (fmha, gemm, softmax, etc.)
- `include/ck_tile/ops/fmha/pipeline/` - FMHA pipeline implementations (performance-critical)
- `example/ck_tile/` - Working examples with build recipes
- `codegen/` - Python-based kernel code generation
- `profiler/` - Performance profiling tools

### FMHA (Flash Attention) Architecture

#### Directory Structure
```
include/ck_tile/ops/fmha/
├── kernel/           # Kernel entry points (fmha_fwd_kernel.hpp, fmha_fwd_v3_kernel.hpp)
├── pipeline/         # Pipeline implementations (performance-critical)
├── block/            # Block-level components (masking, dropout, position encoding)
└── api/              # High-level API wrappers
```

#### Kernel Template Structure

The kernel (`FmhaFwdKernel`, `FmhaFwdV3Kernel`) has two key template parameters:
- `FmhaPipeline` - Block tile pipeline handling Q*K and P*V computations
- `EpiloguePipeline` - Post-processing and output storage

**Key data types extracted from pipeline:**
- `QDataType`, `KDataType`, `VDataType` - Input types (fp8, fp16, bf16)
- `PDataType` - Attention probability type after softmax
- `SaccDataType` - Scratch accumulator (typically float)
- `ODataType` - Output type

**Configuration flags:**
- `kIsGroupMode` - Variable-length sequences via seqstart pointers
- `kPadSeqLenQ/K`, `kPadHeadDimQ/V` - Padding control
- `kHasLogitsSoftCap` - Gemma-style logits softcap
- `kStoreLSE` - Store log-sum-exp for backward pass
- `QScaleEnum` - FP8 quantization (PERTENSOR, NONE)

#### Pipeline Implementations

| Pipeline | Name | Description |
|----------|------|-------------|
| `BlockFmhaPipelineQRKSVS` | "qr" | LDS-based, all QKV in LDS. For medium sequences. |
| `BlockFmhaPipelineQRKSVSAsync` | "qr_async" | Q in registers, async K/V loading. For longer sequences. |
| `BlockFmhaFwdV3Pipeline` | "v3" | Next-gen with warp group coordination and instruction scheduling. |
| `BlockFmhaPipelineSplitKV` | - | Multi-pass with reduction for very long sequences. |
| `BlockFmhaPipelinePagedKV` | - | KV-cache paging for inference. |

#### Attention Computation Flow (Online Softmax)

```
Phase 1: GEMM0 (Q × K^T → S)
├── Load Q tile (M0 × K0) into registers
├── Loop over K tiles (N0 × K0):
│   ├── Async load K tile to LDS
│   ├── Sync barrier
│   └── Block GEMM with MFMA → S accumulator
└── Apply scale: S *= 1/sqrt(hdim)

Phase 2: Online Softmax
├── Row-wise max: m_j = max(S_j)
├── Optional: logits softcap (tanh transform)
├── Exponential: P = exp(S - m_j)
├── Row-wise sum: l_j = sum(P_j)
└── Rescale accumulator: O *= exp(m_old - m_new)

Phase 3: GEMM1 (P × V → O)
├── Convert P to compute type
├── Load V tiles (K1 × N1)
├── Block GEMM with MFMA → O accumulator
└── Finalize: O /= l_j

Phase 4: Epilogue
├── Convert O to output type
├── Optional: store LSE = m/log(2) + log(l)
└── Write O tile to DRAM
```

#### Memory Management

**LDS Layout:**
- K tiles: N0 × K0, double-buffered for async prefetch
- V tiles: K1 × N1, bank-conflict-aware padding
- Size computed via `Policy::GetSmemSize<Problem>()`

**Async Copy Pattern:**
```cpp
async_load_tile_raw(k_lds_window, k_dram_window);  // Non-blocking
move_tile_window(k_dram_window, {kN0, 0});
// ... GEMM computation overlaps with load ...
s_waitcnt_vmcnt<0>();  // Wait before use
```

**Prefetching Strategy:** Load K[i+1] while computing with K[i]

#### Block-Level Components

**Masking (`block_masking.hpp`):**
- `MASK_FROM_TOP_LEFT` - Causal (lower triangular)
- `MASK_FROM_BOTTOM_RIGHT` - Future tokens
- Local attention via `window_size_left/right`
- `GenericAttentionMask::GetTileRangeAlongX()` - Skip fully masked tiles

**Quantization (`block_attention_quant_scale_enum.hpp`):**
- `NONE` - Standard float operations
- `PERTENSOR` - Single scale per Q/K/V tensor (FP8)
- Flow: `Q_fp8 * scale → float → compute → saturate → O_fp8`

#### Policy/Trait Configuration

**TileFmhaTraits** - Core configuration:
```cpp
template <
    bool kPadSeqLenQ, kPadSeqLenK,
    bool kPadHeadDimQ, kPadHeadDimV,
    bool kHasLogitsSoftCap,
    BlockAttentionBiasEnum BiasEnum,
    bool kStoreLSE,
    bool kHasDropout,
    BlockAttentionQuantScaleEnum QScaleEnum
>
struct TileFmhaTraits;
```

**Default Policy** provides:
- Alignment hints for DRAM loads
- GEMM configurations (MFMA instruction selection)
- LDS store/load descriptors
- Register tile distributions

#### Grid/Block Organization

```cpp
dim3 GridSize(batch_size, nhead, ceil(max_seqlen_q / kM0) * ceil(hdim_v / kN1));
dim3 BlockSize(kBlockSize);  // Typically 256-512 threads
```

#### V3 Pipeline Optimizations

- **Warp Group Specialization** - 2 warp groups (4 waves each) with different roles
- **Phase Scheduling** - Explicit barriers for MFMA/VALU/TRANS timing
- **Packed FP32** - `v_pk_mul_f32` for two operations per instruction
- **Fast Exp2** - Bit manipulation approximation

## Key Concepts

- **Tile** - Fixed-size data chunk processed by a thread block
- **Block Tile** - Tile owned by entire thread block
- **Wave Tile** - Tile owned by a wavefront (64 threads on AMD)
- **LDS** - Local Data Share (AMD's shared memory)
- **MFMA** - Matrix Fused Multiply-Add (AMD's matrix core instruction)
- **XDL** - Crosslane Data Layout instructions

See `TERMINOLOGY.md` and `ACRONYMS.md` for complete references.

## Common Variable Naming

| Symbol | Meaning |
|--------|---------|
| M, N, K | GEMM dimensions: A[M,K] × B[K,N] = C[M,N] |
| Q, K, V | Query, Key, Value (attention) |
| S | Sequence length |
| D | Head dimension |
| B | Batch size |
| H | Number of attention heads |

## Running FMHA Examples

```bash
# Basic FMHA forward
./bin/tile_example_fmha_fwd -b=1 -h=16 -s=16384 -d=128

# With FP8
./bin/tile_example_fmha_fwd -b=1 -h=8 -s=4096 -d=128 -prec=fp8

# Group mode (variable length)
./bin/tile_example_fmha_fwd -mode=1 -b=2 -h=8 -s=1024,2048 -d=128

# With causal mask
./bin/tile_example_fmha_fwd -b=1 -h=8 -s=4096 -d=128 -mask=t
```

Use `-?` flag to see all options.

## Codegen System

Kernels are instantiated into separate files via Python scripts to enable parallel compilation. This section explains how the FMHA codegen works using `example/ck_tile/01_fmha/` as reference.

### Why Codegen?

CK kernels are heavily templated. A single FMHA kernel can have dozens of template parameters (data type, tile sizes, pipeline variant, padding flags, mask type, etc.). Compiling all variants in a single translation unit would be extremely slow. The codegen system:

1. **Generates one .cpp file per kernel variant** - Enables parallel compilation
2. **Creates a dispatch API file** - Runtime selection of the best kernel for given parameters
3. **Filters variants by "receipt"** - Product-specific builds (Flash Attention, PyTorch, Aiter, etc.)

### Directory Structure

```
example/ck_tile/01_fmha/
├── generate.py              # Main entry point
├── codegen/
│   ├── __init__.py
│   ├── cmake_config.py      # Output directory config
│   ├── arch.py              # GPU architecture traits (gfx9, gfx950, gfx12)
│   ├── cpp_symbol_map.py    # Python-to-C++ constant mappings
│   ├── utils.py             # File writing, dedup checking
│   └── ops/
│       ├── __init__.py
│       ├── fmha_fwd.py      # Forward pass kernel generation
│       ├── fmha_bwd.py      # Backward pass kernel generation
│       ├── fmha_fwd_splitkv.py
│       ├── fmha_fwd_appendkv.py
│       └── fmha_pagedkv_prefill.py
├── CMakeLists.txt           # CMake integration
└── fmha_fwd.hpp             # Shared header included by generated files
```

### CMake Integration (CMakeLists.txt)

The build system uses a two-phase approach:

#### Phase 1: List Generation (CMake Configure Time)
```cmake
# At configure time, run generate.py --list_blobs to get file list
execute_process(
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/generate.py
    --targets ${FMHA_TARGETS_ARG}
    --api ${FMHA_FWD_APIS}
    --optdim 32,64,80,128,256
    --list_blobs ${CMAKE_CURRENT_BINARY_DIR}/fwd_blob_list.txt
)

# Read the file list
file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/fwd_blob_list.txt FMHA_FWD_GEN_BLOBS)
```

#### Phase 2: Code Generation (Build Time)
```cmake
# Custom command to generate actual source files
add_custom_command(
  OUTPUT ${FMHA_FWD_GEN_BLOBS}
  COMMAND ${Python3_EXECUTABLE} ${FMHA_FWD_CODE_GEN_COMMON_ARGS}
    --output_dir ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CODE_GEN_SCRIPTS}
  COMMENT "Generate CK Tile FMHA FWD kernels"
)

# Create object library from generated sources
add_library(tile_fmha_fwd_instances OBJECT EXCLUDE_FROM_ALL)
target_sources(tile_fmha_fwd_instances PRIVATE ${FMHA_FWD_GEN_BLOBS})
```

This separation ensures CMake knows the output files before they exist, enabling proper dependency tracking.

### Python Codegen Flow (generate.py)

```
generate.py --api fwd --targets gfx9,gfx950 --receipt 100 --output_dir ./build
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Handler Discovery                                                        │
│    - pkgutil.iter_modules() finds all modules under codegen.ops/           │
│    - Each module exports (list_blobs, write_blobs) functions               │
│    - Handlers keyed by API name: {"fwd": (list_blobs, write_blobs), ...}   │
└─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. Kernel Enumeration (get_fwd_blobs in fmha_fwd.py)                       │
│    For each (factory, dtype, hdim, mode, tile, pipeline):                  │
│      - Check compatibility rules (arch-specific)                           │
│      - Apply product filter (receipt)                                       │
│      - Apply optional fnmatch filter                                        │
│      - Create FmhaFwdKernel + register FmhaFwdApiTrait                     │
└─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. Code Generation                                                          │
│    - write_single_fwd_kernel(): Render each kernel to .cpp file            │
│    - write_fwd_api(): Render dispatch logic to fmha_fwd_api.cpp            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Classes (fmha_fwd.py)

#### FmhaFwdTileSize
Defines block/warp tile dimensions for GEMM0 (Q×K) and GEMM1 (P×V):
```python
@dataclass
class FmhaFwdTileSize:
    F_bm0: int   # Block tile M for GEMM0 (along Q seqlen)
    F_bn0: int   # Block tile N for GEMM0 (along K seqlen)
    F_bk0: int   # Block tile K for GEMM0 (head dim unroll)
    F_bn1: int   # Block tile N for GEMM1 (V head dim)
    F_bk1: int   # Block tile K for GEMM1 (along K seqlen)
    F_bk0max: int  # Max K0 tile for pipelines loading full Q tile
    F_rm0, F_rn0, F_rk0: int  # Warp distribution for GEMM0
    F_rm1, F_rn1, F_rk1: int  # Warp distribution for GEMM1
    F_wm0, F_wn0, F_wk0: int  # Wave tile sizes for GEMM0
    F_wm1, F_wn1, F_wk1: int  # Wave tile sizes for GEMM1
    F_occupancy: int  # Target occupancy (-1 = auto)
```

#### FmhaFwdPipeline
Defines feature flags for a pipeline variant:
```python
@dataclass
class FmhaFwdPipeline:
    tag: str          # "qr", "qr_async", "qr_async_trload_v3", etc.
    F_vlayout: str    # "row" or "col" (V memory layout)
    F_spad: str       # "t"/"f" - Q seqlen padding support
    F_skpad: str      # "t"/"f" - K seqlen padding support
    F_dpad: str       # "t"/"f" - Q head dim padding support
    F_dvpad: str      # "t"/"f" - V head dim padding support
    F_logits: str     # "t"/"f" - logits softcap (Gemma)
    F_bias: str       # "no", "bias", "alibi"
    F_lse: str        # "t"/"f" - store log-sum-exp
    F_dropout: str    # "t"/"f" - dropout support
    F_qscale: str     # "no", "pertensor" (FP8 quantization)
    F_mask: str       # "no", "causal", "s_mask", etc.
    F_skip: str       # "t"/"f" - skip small seqlen optimization
    F_trload: str     # "t"/"f" - transpose load optimization
    F_sink: str       # "t"/"f" - attention sink support
```

#### FmhaFwdKernel
Combines architecture, problem config, tile, and pipeline into a complete kernel:
```python
@dataclass
class FmhaFwdKernel:
    F_arch: ArchTrait      # gfx9, gfx950, gfx12
    F_hdim: int            # Head dimension (32, 64, 128, 256)
    F_dtype: str           # "fp16", "bf16", "fp8bf16", etc.
    F_mode: str            # "batch" or "group"
    F_tile: FmhaFwdTileSize
    F_pipeline: FmhaFwdPipeline
```

The `render()` method generates the complete .cpp file using string templates.

### Kernel Filename Convention

Generated files follow this pattern:
```
fmha_fwd_d{hdim}_{dtype}_{mode}_{tile_params}_{pipeline_params}_{arch}.cpp
```

Example:
```
fmha_fwd_d128_fp8bf16_batch_b256x64x128x128x64x128_r8x1x1_r8x1x1_w32x32x32_w32x32x32_qr_async_trload_v3_vrow_psskddv_nlogits_nbias_nmask_nlse_ndropout_nskip_pertensor_trload_nsink_gfx950.cpp
```

### Generated Kernel Structure

Each generated .cpp file contains:

```cpp
// Header with arch guard
#if !defined(__HIP_DEVICE_COMPILE__) || (defined(__gfx950__))

// Type aliases from template parameters
using fmha_dtype = FmhaFwdFp8Bf16;
using fmha_block_tile = ck_tile::sequence<256, 64, 128, 128, 64, 128>;
using fmha_shape = ck_tile::TileFmhaShape<...>;
using fmha_traits = ck_tile::TileFmhaTraits<true, true, false, false, ...>;
using fmha_mask = ck_tile::SimplifiedGenericAttentionMask<false>;
using fmha_pipeline_problem = ck_tile::BlockFmhaPipelineProblem<...>;
using fmha_pipeline = ck_tile::BlockFmhaFwdV3Pipeline<fmha_pipeline_problem>;
using fmha_kernel = ck_tile::FmhaFwdV3Kernel<fmha_pipeline, fmha_epilogue>;

// Explicit template instantiation
using trait = fmha_fwd_traits_<128, FmhaFwdFp8Bf16, false, 256, 64, ...>;

template<>
float fmha_fwd_<trait, ck_tile::gfx950_t>(const ck_tile::stream_config& s, fmha_fwd_args a) {
    using k_ = fmha_kernel;
    auto [kargs, grids] = fmha_fwd_v3_create_kargs_and_grids<k_>(a);
    return ck_tile::launch_kernel(...);
}

#endif
```

### Generated API Dispatcher (fmha_fwd_api.cpp)

The API file contains runtime dispatch logic:

```cpp
float fmha_fwd(fmha_fwd_traits traits, fmha_fwd_args args, const ck_tile::stream_config& config) {
    // Check if V3 pipeline can be used
    const bool can_dispatch_v3 = (device_name == "gfx950") &&
                                  (traits.data_type == "fp8bf16") && ...;

    if (can_dispatch_v3) {
        return fmha_fwd_v3(traits, args, config);
    } else {
        return fmha_fwd_v2(traits, args, config);
    }
}

// Nested dispatch by: arch → dtype → hdim → (mode, vlayout, mask, bias, ...)
float fmha_fwd_v2(fmha_fwd_traits t, fmha_fwd_args a, const ck_tile::stream_config& s) {
    if(device_name.compare(0, 6, "gfx950") == 0) {
        if(t.data_type.compare("fp8bf16") == 0) {
            if(t.hdim_q <= 128 && t.hdim_v <= 128) {
                if((t.is_group_mode == false) && (t.mask_type == mask_enum::no_mask) && ...) {
                    using trait_ = fmha_fwd_traits_<128, FmhaFwdFp8Bf16, false, ...>;
                    return fmha_fwd_<trait_, ck_tile::gfx950_t>(s, a);
                }
                // ... more variants
            }
        }
    }
}
```

### Compatibility Rules

Each architecture factory defines rules that filter valid (tile, pipeline) combinations:

```python
class CompatibilityRuleFactoryGfx950(CompatibilityRuleFactoryGfx9):
    @classmethod
    def get_rules(cls) -> List[CompatibilityRule]:
        rules = super().get_rules()

        def check_tile_pipeline(problem_ctx, kernel_ctx):
            # Only v3 pipeline uses 256x8warp tiles
            is_v3_tile = (kernel_ctx.tile.F_bm0 == 256 and num_warps == 8)
            is_v3_pipeline = (kernel_ctx.pipeline.tag == "qr_async_trload_v3")
            return is_v3_tile == is_v3_pipeline

        rules.append(check_tile_pipeline)
        return rules
```

### Product Receipts

The `--receipt` argument selects kernel subsets for different products:

| Receipt | Product | Description |
|---------|---------|-------------|
| 0 | Default | Standard coverage, no fp32 |
| 2-3 | Flash Attention | fp16/bf16, row-major, no skip |
| 4 | PyTorch | fp16/bf16, batch mode only |
| 100 | Aiter mha_fwd | fp16/bf16/fp8bf16, batch mode |
| 200 | Aiter mha_varlen_fwd | Group mode variant |
| 600 | Aiter C++ API | Both batch and group modes |
| 800 | fp32 only | All variations for fp32 |

Example usage:
```bash
# Generate kernels for Aiter integration
python generate.py --targets gfx9,gfx950 --api fwd --receipt 600 --output_dir ./build
```

### Symbol Maps (cpp_symbol_map.py)

Maps Python string values to C++ types/enums:

```python
PIPELINE_MAP = {
    "qr": "ck_tile::BlockFmhaPipelineQRKSVS",
    "qr_async": "ck_tile::BlockFmhaPipelineQRKSVSAsync",
    "qr_async_trload_v3": "ck_tile::BlockFmhaFwdV3Pipeline",
}

FWD_DTYPE_MAP = {
    "fp16": "FmhaFwdFp16",
    "bf16": "FmhaFwdBf16",
    "fp8bf16": "FmhaFwdFp8Bf16",
}

MASK_MAP = {
    "no": "FmhaMasks::NoMask",
    "causal": "FmhaMasks::CausalMask",
}
```

### Debugging: Find Which Kernel Variant Is Used

When `fmha_fwd()` returns incorrect results, use these steps to identify the dispatched kernel:

**1. Enable kernel name logging:**

```cpp
ck_tile::stream_config config;
config.stream_id_ = stream;
config.log_level_ = 1;  // Enable logging

float time = fmha_fwd(traits, args, config);
// Prints: ", fmha_fwd_d128_fp8bf16_batch_b256x64x128..."
```

**2. Decode the kernel name:**

```
fmha_fwd_d128_fp8bf16_batch_b256x64x128x128x64x128_r8x1x1_r8x1x1_w32x32x32_w32x32x32_qr_async_trload_v3_vrow_psskddv_nlogits_nbias_nmask_nlse_ndropout_nskip_pertensor_trload_nsink
        │    │       │     │                      │             │                  │                   │    │       │
        │    │       │     │                      │             │                  │                   │    │       └── feature flags
        │    │       │     │                      │             │                  │                   │    └── padding flags
        │    │       │     │                      │             │                  │                   └── V layout
        │    │       │     │                      │             │                  └── pipeline tag
        │    │       │     │                      │             └── wave tile sizes (gemm0, gemm1)
        │    │       │     │                      └── warp distribution (gemm0, gemm1)
        │    │       │     └── block tile sizes (bm0, bn0, bk0, bn1, bk1, bk0max)
        │    │       └── batch/group mode
        │    └── data type
        └── head dimension
```

**3. Find the generated source file:**

```bash
# Kernel name maps to filename with _{arch}.cpp suffix
# For CK standalone build:
find build/ -name "fmha_fwd_d128_fp8bf16*v3*.cpp"

# For Aiter JIT build:
find aiter/jit/build/ -name "fmha_fwd_d128_fp8bf16*v3*.cpp"
```

**4. Find the pipeline/kernel implementation:**

The generated .cpp file uses these headers based on pipeline tag:

| Pipeline Tag | Pipeline Header | Kernel Header |
|--------------|-----------------|---------------|
| `qr` | `block_fmha_pipeline_qr_ks_vs.hpp` | `fmha_fwd_kernel.hpp` |
| `qr_async` | `block_fmha_pipeline_qr_ks_vs_async.hpp` | `fmha_fwd_kernel.hpp` |
| `qr_async_trload_v3` | `block_fmha_fwd_v3_pipeline.hpp` | `fmha_fwd_v3_kernel.hpp` |

All located in `include/ck_tile/ops/fmha/pipeline/` and `include/ck_tile/ops/fmha/kernel/`.

### Adding a New Kernel Variant

1. **Define tile sizes** in `get_hdim_tile_size_dict()`:
   ```python
   (128, 128): [FmhaFwdTileSize(256, 64, 128, 128, 64, 128, 8, 1, 1, ...)]
   ```

2. **Add pipeline configuration** in `get_pipelines()`:
   ```python
   pipelines.append(FmhaFwdPipeline("qr_async_trload_v3", "row", "t", "t", ...))
   ```

3. **Add compatibility rules** if needed in the factory class

4. **Update symbol maps** for any new C++ types

5. **Regenerate**: `python generate.py --list_blobs ... && python generate.py --output_dir ...`
