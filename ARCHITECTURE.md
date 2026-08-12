# FlashInfer 代码架构与流程分析

> 分析时间：基于当前仓库 HEAD（`d57bfb19`，约 2025-08 状态）
> 代码规模：`include/` + `flashinfer/` + `csrc/` 合计约 **18 万行**（C++/CUDA + Python）

---

## 1. 项目概览

**FlashInfer** 是一个面向 LLM 推理/服务的 **GPU 内核库**，核心设计决策是：

- **JIT（Just-In-Time）编译为默认路径**：内核代码随参数（dtype、head_dim 等）动态特化编译，改 `.cuh` 源码无需重装包即可生效，天然适合开发迭代。
- **框架无关的内核层**：`include/` 下的 CUDA 内核只接受裸指针（raw pointer），不依赖 PyTorch；框架绑定由 `csrc/` 通过 **TVM-FFI** 统一 ABI 完成，理论上可对接 Python/C++/Rust 等多种语言。
- **面向现代 GPU 架构**：支持 SM75/80/86/89/90/100/103/107/110/120/121，每个架构有专属的 kernel 变体与编译 flag（如 sm90a、sm100a、sm120f）。
- **覆盖完整推理原语**：Prefill/Decode 注意力（含 MLA、POD、KDA、NVFP4 等变体）、GEMM（FP4/FP8/BF16/MXFP4）、MoE（CUTLASS / TRT-LLM / CuTe-DSL 多后端）、采样/TopK、量化、Norm、RoPE、通信（AllReduce/AllToAll）等。

---

## 2. 顶层目录结构与职责

```
flashinfer/
├── include/flashinfer/       # ① CUDA 内核模板层（header-only，框架无关，裸指针）
│   ├── attention/            #    Attention 内核（decode/prefill/MLA/POD/hopper/sm120/sparse_mla...）
│   ├── gemm/                 #    GEMM 内核与 CUTLASS 封装
│   ├── fused_moe/            #    MoE 内核
│   ├── comm/                 #    通信内核（TRT-LLM allreduce/alltoall 等）
│   ├── mamba/                #    Mamba SSM 内核
│   ├── norm/ topk.cuh sampling.cuh pos_enc.cuh mma.cuh utils.cuh ...
│
├── csrc/                     # ② C++ 启动层（TVM-FFI 绑定 + 参数分发 + 内核 launch）
│   ├── *_jit_binding.cu      #    TVM-FFI 导出（DLL_EXPORT_TYPED_FUNC）
│   ├── *.cu                  #    启动器 launcher（TensorView → 参数结构 → launch kernel）
│   ├── *_customize_config.jinja / *_kernel_inst.jinja   # 类型特化模板
│   └── tvm_ffi_utils.h       #    DLPack/TVM-FFI 工具
│
├── flashinfer/               # ③ Python 包（高层 API + JIT 编排 + 分发 + 工具）
│   ├── jit/                  #    JIT 系统核心
│   │   ├── core.py           #      JitSpec 抽象基类 / JitSpecNvcc / build_and_load 生命周期
│   │   ├── env.py            #      路径环境（workspace/cache/AOT/cubin 目录解析）
│   │   ├── cpp_ext.py        #      ninja build.ninja 生成 + nvcc/cxx 编译执行
│   │   ├── attention/gemm/fused_moe/...  # 各操作的 gen_*_module()（代码生成器）
│   │   ├── cute_dsl_core.py  #      CuTe-DSL 内核的 JIT 子类（JitSpecCuteDsl）
│   │   └── cubin_loader.py   #      预编译 cubin 下载/校验
│   ├── attention/ gemm/ fused_moe/ mla/ ...  # 各领域的高层 Python API
│   ├── decode.py prefill.py  #    核心 Wrapper（plan/run 模式，工作量缓存）
│   ├── aot.py                #    AOT 预编译入口（生成 flashinfer-jit-cache 包）
│   ├── api_logging.py        #    @flashinfer_api 装饰器（日志/诊断/数据 dump）
│   ├── fi_trace.py / trace/  #    调用追踪（基准定义 JSON）
│   ├── trace_apply/          #    Trace Apply（运行时内核替换，可选）
│   ├── autotuner/            #    自动调优框架
│   └── __main__.py           #    CLI（collect-env、artifacts、tactics-blocklist 等）
│
├── build_backend.py          # ④ 自定义 PEP 517 构建后端（editable/wheel/sdist + 子模块 patching）
├── 3rdparty/                 #    第三方依赖（cutlass、spdlog、cccl 子模块）
├── tests/ benchmarks/ docs/ ci/ docker/  # 测试、基准、文档、CI、容器
└── flashinfer-jit-cache/ flashinfer-cubin/  # 可选预编译发行包
```

**铁律（Framework Separation）**：`include/` 严禁包含 Torch 头文件；内核只碰裸指针，所有框架张量处理都发生在 `csrc/`。

---

## 3. 核心架构：三层 + 一个 JIT 引擎

```
┌─────────────────────────────────────────────────────────────────────┐
│  Python 层  (flashinfer/)                                            │
│  ┌───────────────────────┐   ┌────────────────────────────────────┐ │
│  │ 高层 API / Wrapper     │   │ JIT 编排层 (flashinfer/jit/)        │ │
│  │ - BatchDecodeWithPage │   │ - gen_*_module() 生成源码与 JitSpec  │ │
│  │   dKVCacheWrapper     │──▶│ - JitSpec.build_and_load()          │ │
│  │ - @functools.cache    │   │ - Jinja 渲染 → .cu 写入 gen 目录     │ │
│  │ - @flashinfer_api     │   │ - ninja 编译 → .so → tvm_ffi 加载    │ │
│  └───────────────────────┘   └────────────────────────────────────┘ │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ TVM-FFI module 句柄（.run/.plan/.workspace_size）
┌───────────────────────────────▼─────────────────────────────────────┐
│  C++ 启动层  (csrc/)                                                │
│  - *_jit_binding.cu：TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, ...)         │
│  - *.cu launcher：TensorView → 参数结构体 → DISPATCH_* 宏 → launch   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ 内核 launch（编译期特化模板参数）
┌───────────────────────────────▼─────────────────────────────────────┐
│  CUDA 内核层  (include/flashinfer/)                                 │
│  - template <HEAD_DIM, POS_ENCODING_MODE, AttentionVariant,...>     │
│  - __global__ SingleDecodeWithKVCacheKernel / BatchDecode...Kernel   │
│  - 纯 CUDA + CuTe/CUTLASS，无框架依赖，裸指针 Params 结构体          │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.1 三层职责划分

| 层 | 目录 | 职责 | 依赖 |
|---|---|---|---|
| Python API 层 | `flashinfer/` | 张量形状/参数校验、后端选择、wrapper 状态管理、JIT 模块编排、日志/trace | torch, tvm_ffi |
| C++ 启动层 | `csrc/` | 接收 TensorView、做编译期分发（DISPATCH_* 宏）、填充 Params、launch kernel | TVM-FFI, DLPack |
| 内核层 | `include/flashinfer/` | 纯计算内核模板、共享内存调度、MMA 指令、Tile 循环 | CUDA, CuTe, CUTLASS |

---

## 4. JIT 编译系统（本仓库最核心的架构）

JIT 系统分三层实现，见 `flashinfer/jit/core.py` 与 CLAUDE.md 的对应描述：

### 4.1 JitSpec 生命周期（`core.py`）

`JitSpec`（抽象基类）定义了一个内核模块的完整生命周期，每种编译工具链一个子类：

- `JitSpecNvcc`（nvcc + ninja，`gen_jit_spec()` 返回）
- `JitSpecCuteDsl`（CuTe-DSL 内核，`cute_dsl_core.py`，产物为 `.o` 而非 `.so`）

```
用户调用 gen_*_module()
      │
      ▼
JitSpec.build_and_load()          ← 模板方法，封装缓存/锁/JIT 开关策略
      │
      ├─ ① try_load()             快速路径：AOT 产物存在则直接加载
      │
      ├─ ② FileLock(lock_path)    跨进程锁（防多进程同时编译同一模块）
      │     └─ try_load() 二次检查（等待期间别的进程可能已编译好）
      │
      ├─ ③ FLASHINFER_DISABLE_JIT 检查：设置了则抛 MissingJITCacheError
      │
      ├─ ④ build()                 write_ninja() → run_ninja()
      │                              - build.ninja：.cu/.cpp → .cuda.o/.o → .so
      │                              - ninja 依赖扫描自动实现增量编译
      │
      └─ ⑤ load()                  tvm_ffi.load_module(<name>.so)
```

**关键设计点**：
- **AOT 优先**：`try_load()` 只信任 AOT 产物；JIT 路径的 `.so` 新鲜度完全交给 ninja 的依赖扫描，miss 就进入 `build()`，此时 ninja 若无变化会 no-op，开销极小。
- **写目录规则**：永远只写 `FLASHINFER_GEN_SRC_DIR`（生成源码）和 `FLASHINFER_JIT_DIR`（编译产物）；`FLASHINFER_CSRC_DIR`（模板）与 `FLASHINFER_AOT_DIR`（预编译）只读。
- **全局注册表** `jit_spec_registry`：所有创建的 JitSpec 被登记，供 CLI（`flashinfer jit-status` 等）查询编译状态。

### 4.2 代码生成模式（`flashinfer/jit/**`）

所有 `gen_*_module()` 遵循统一五步模式：

```python
def gen_single_decode_module(dtype_q, dtype_kv, ..., head_dim_qk, ...) -> JitSpec:
    # 1. 由参数计算唯一 URI（缓存键）
    uri = get_single_decode_uri(dtype_q, dtype_kv, ..., head_dim_qk, ...)
    # 2. 生成目录 = FLASHINFER_GEN_SRC_DIR / uri
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    # 3. 渲染 Jinja 配置模板 → single_decode_config.inc（类型特化）
    config_templ.render(dtype_q=..., head_dim_qk=..., variant_name=..., ...)
    # 4. 渲染/拷贝源码：single_decode_kernel.cu（内核实例化）+ launcher + binding
    #    write_if_different() 保证磁盘内容不变时不动 mtime（配合 ninja 增量）
    # 5. 返回 gen_jit_spec(uri, sources, extra_cuda_cflags=[...])
```

**Jinja 模板的作用**：把 Python 侧的参数（dtype、head_dim、pos_encoding_mode、attention variant 等）**编译期烙进 C++**，例如：

```
渲染前: using DTypeIn = {{ dtype_q }}; constexpr int HEAD_DIM = {{ head_dim_qk }};
渲染后: using DTypeIn = half;       constexpr int HEAD_DIM = 128;
```

这样同一套模板 C++ 代码可为不同具体类型各编译一份，配合 `DISPATCH_*` 宏在启动层做运行期参数→编译期特化的展开。

### 4.3 编译执行（`cpp_ext.py`）

- `generate_ninja_build_for_op()` 手工生成 `build.ninja`（**不引入 CMake**），两条 rule：
  - `cuda_compile`：`nvcc -MMD -MF $out.d ...`（depfile 依赖追踪）
  - `link` / `nvcc_link`：`cxx -shared ...` 或 `nvcc -shared`（device linking）
- 编译 flags 分层：全局 flags（`CompilationContext.get_nvcc_flags_list()`，按检测到的 GPU 架构生成 `-gencode` 列表）+ 模块专属 flags（如 large-head 模块只编 SM100+）+ 环境变量注入（`FLASHINFER_EXTRA_*FLAGS`）。
- nvcc 并行度：`FLASHINFER_NVCC_THREADS`（单进程内线程数，`--threads=N`）与 `MAX_JOBS`（并行 ninja 任务数）两个维度。
- 多模块批量编译：`build_jit_specs()` 把多个子 ninja 拼成一个聚合 ninja 一次跑完（AOT 用）。

### 4.4 编译上下文与架构分发（`compilation_context.py`）

`CompilationContext` 负责确定目标架构集合：

1. 读 `FLASHINFER_CUDA_ARCH_LIST`（环境变量）→ 否则自动探测本机 GPU（`torch.cuda.get_device_capability`）
2. 架构后缀规范化：SM9x→`a`，SM10+→`a`，SM12→`f`/`a`（需 CUDA≥12.9），SM<9 无后缀
3. `get_nvcc_flags_list(supported_major_versions=[...])` 让模块声明自己支持的 SM 主版本，不在列表内的架构自动过滤，从而**同一份源码在不同架构上编译出不同的 `-gencode` fatbin**
4. 特例：SM107（Rubin）在 CUTLASS 原生支持前映射到 `sm_100f` 目标（`map_sm107_to_100f`）

### 4.5 两级缓存与失效

| 级别 | 载体 | 说明 |
|---|---|---|
| Python 内存级 | `@functools.cache`（`get_*_module(*args)`） | 同一进程内相同参数只 `build_and_load` 一次 |
| 磁盘级 | `~/.cache/flashinfer/<version>/<arch>/cached_ops/<uri>/` | `.so` + `build.ninja`；ninja depfile 负责源码变更检测 |

缓存目录按版本 + 排序后的架构列表组织（`env.py::_get_workspace_dir_name`），保证确定性。失效条件 = 源码变化 / 编译 flag 变化 / 架构变化 / 版本变化。清缓存：`rm -rf ~/.cache/flashinfer/`。

### 4.6 CuTe-DSL 内核缓存（`cute_dsl_core.py`）

`nvidia-cutlass-dsl` 的 `cute.compile()` 无持久缓存，因此 `JitSpecCuteDsl` 把编译产物 `export_to_c()` 成 `.o` 文件存到 `cached_ops/<module>_<arch>_cute_dsl/`，每目录一个 `meta.json` + 每个特化一个 `.o`；失效按模块粒度（DSL 版本或内核源码 SHA256 变化即整模块重建）。参考实现：`flashinfer/quantization/kernels/nvfp4_quantize.py`。

---

## 5. 一次 API 调用的完整数据流（以 single decode 为例）

```
用户调用 single_decode_with_kv_cache(q, k, v, ...)
      │
      ▼
① flashinfer/decode.py::single_decode_with_kv_cache   (@flashinfer_api + @functools.cache)
      │   参数解析（dtype、head_dim、pos_encoding 等）
      ▼
② get_single_decode_module(*args)                     (functools.cache 命中则跳过)
      │   gen_single_decode_module(...).build_and_load()
      ▼
③ flashinfer/jit/attention/modules.py::gen_single_decode_module
      │   uri = get_single_decode_uri(...)             ← 缓存键
      │   gen_customize_single_decode_module(...)
      ▼
④ 代码生成（写 ~/.cache/flashinfer/.../generated/<uri>/）
      │   single_decode_config.inc  （Jinja 渲染：dtype/head_dim/variant 烙进 C++）
      │   single_decode_kernel.cu   （内核模板实例化）
      │   single_decode.cu          （拷贝自 csrc/）
      │   single_decode_jit_binding.cu（拷贝自 csrc/）
      ▼
⑤ JitSpecNvcc.build_and_load()
      │   write_ninja() → ninja 编译（增量）→ <uri>.so
      │   tvm_ffi.load_module(<uri>.so)   ← 得到 module 句柄
      ▼
⑥ 注册 torch custom op
      │   register_custom_op("flashinfer::<uri>_run", ...)   ← torch.compile / CUDA graph 兼容
      │   register_fake_op(...)                              ← FakeTensor（shape 推断）
      ▼
⑦ module.run(q, k, v, tmp, o, ...)     ← TVM-FFI 调用进入 C++
      │
      ▼
⑧ csrc/single_decode.cu::single_decode_with_kv_cache
      │   TensorView 解包 → DISPATCH_context/DISPATCH_DTYPE 宏（运行期→编译期）
      │   → 填充 Params 结构体 → launch SingleDecodeWithKVCacheKernel<HEAD_DIM, POS_ENC,...>
      ▼
⑨ include/flashinfer/attention/decode.cuh
      │   __global__ 内核：paged KV 访问、Tile 循环、SMEM staging、MMA、softmax 在线归约
      ▼
⑩ 输出写回 o / lse 张量（torch 张量，CUDA 流内异步）
```

---

## 6. 关键架构模式

### 6.1 Plan / Run 模式（批量 wrapper 的默认形态）

`BatchDecodeWithPagedKVCacheWrapper` / `BatchPrefillWithPagedKVCacheWrapper` 等把一次调用拆成两阶段：

```
plan(indptr, indices, last_page_len, num_heads, head_dim, page_size, ...)
  ├─ 调用模块的 plan（host 侧：计算分区/切分/工作量）
  ├─ 调用 workspace_size → 分配 float/int workspace buffer（复用，可 reset）
  ├─ 把与形状相关的元数据固化在 wrapper 上
  └─ 可被 CUDA Graph 捕获（plan 阶段用 int 张量传信息，避免 host 依赖）

run(q, paged_k_cache, paged_v_cache, o, ...)
  ├─ 仅依赖已固化的 plan 与 workspace，无 host 分支
  ├─ 适合 torch.compile / CUDA graph replay
  └─ 输出可能含 lse（forward_return_lse 变体）
```

workspace 用**全局缓冲区 + reset_workspace_buffer()** 复用，避免每步分配；`begin_forward/end_forward` 用于 batch 大小动态变化时的工作量重算。

### 6.2 TVM-FFI 绑定

- **跨语言统一 ABI**：`csrc/*_jit_binding.cu` 里 `TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, cpp_func)` 导出一个模块级函数集合（`run`/`plan`/`workspace_size` 等），Python 侧 `tvm_ffi.load_module()` 后以 `module.run(...)` 调用。
- **张量传参**：TensorView（DLPack）零拷贝；`tvm_ffi_utils.h` 封装 dtype 编解码。
- **Torch 集成**：`register_custom_op("flashinfer::<uri>_run")` + `register_fake_op` 让内核参与 `torch.compile`（inductor 可识别）与 CUDA Graph；`mutates_args` 声明原地修改的张量。

### 6.3 分发宏（Dispatch Macros）

组合参数空间用宏嵌套展开，在 `.jinja` 渲染后生成：

```cpp
DISPATCH_DTYPE(input_dtype, DTypeIn, {
  DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
    LaunchKernel<DTypeIn, BLOCK_SIZE>(...);
  });
});
```

### 6.4 `@flashinfer_api` 装饰器（`api_logging.py`）

- `FLASHINFER_LOGLEVEL`：0 关闭（零开销，直接返回原函数）/ 1 函数名 / 3 输入输出 / 5 + 张量统计 / 10 + 张量 dump 到磁盘（崩溃可复现）
- **crash-safe**：输入在执行**前**记录，CUDA illegal memory access 时信息不丢
- `trace=` 参数挂接 `TraceTemplate`，驱动 `fi_trace()` 自动 dump 基准定义 JSON

### 6.5 后端分发（Backend Dispatch）

同一操作常有多个后端实现，例如 decode/prefill：`fa2`（自家 CUDA）、`fa3`（Hopper FlashAttention-3）、`cudnn`、`cute-dsl`、`trtllm`、`cutlass` 等。`backend` 参数 + `@backend_requirement`（计算能力/后端能力检查）+ `FLASHINFER_*` 环境变量开关（如 `FLASHINFER_DISABLE_TINYGEMM2_SM100`、`FLASHINFER_FORCE_...`）构成回退链。

### 6.6 自动调优（`autotuner/`）

- 对可配置的 kernel（如 MLA sm120、FMHA-v2）在候选 tactics 上计时选优
- 计时器自动选择：普通环境用 `cudaEvent`，机密计算（Confidential Computing）下 `cudaEventElapsedTime` 不可靠，改用 GPU `%globaltimer` 寄存器（`FLASHINFER_AUTOTUNE_TIMER` 可强制）
- 结果可序列化落盘（`FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=1` 复用），`FLASHINFER_TACTICS_BLOCKLIST` 支持跳过已知会导致 hang/crash 的 tactics

---

## 7. 功能模块地图

| 模块 | Python API（`flashinfer/`） | JIT 生成器（`flashinfer/jit/`） | 内核（`include/` + `csrc/`） |
|---|---|---|---|
| Prefill 注意力 | `prefill.py`、`attention/_core.py` | `jit/attention/modules.py` | `attention/prefill.cuh`、`csrc/batch_prefill*.cu`（含 sm90/FP8 变体） |
| Decode 注意力 | `decode.py` | `jit/attention/modules.py` | `attention/decode.cuh`、`csrc/batch_decode*.cu`（含 MLA sm80/sm90） |
| MLA | `mla/`、`decode.py` | `jit/mla.py` | `attention/mla*.cuh`、`csrc/batch_mla*`、`sparse_mla_sm120*` |
| POD（Point of Decode） | `pod.py` | `jit/attention/modules.py` | `attention/batch_pod.cuh`、`csrc/batch_pod*.cu` |
| Cascade（共享前缀） | `cascade.py` | `jit/cascade.py` | `attention/cascade.cuh` |
| KDA（Delta rule） | `kda*.py`、`gdn_*.py` | `jit/flash_kda*.py` | `attention/blackwell/`、`csrc/kda/`、`csrc/gdn*` |
| GEMM | `gemm/`、`grouped_mm/` | `jit/gemm/`、`jit/tinygemm2.py` 等 | `gemm/*.cuh`、`csrc/*_gemm*.cu`（FP4/FP8/MXFP4/BF16，CUTLASS/CuTe-DSL/cuBLASLt/TRT-LLM） |
| MoE | `fused_moe/` | `jit/fused_moe.py`、`jit/bgmv_moe.py` 等 | `fused_moe/*.cuh`、`csrc/fused_moe/`（CUTLASS/TRT-LLM/CuTe-DSL 后端） |
| 量化 | `quantization/`、`fp4/fp8_quantization.py` | `jit/fp4/fp8_quantization.py` 等 | `quantization.cuh`、`csrc/quantization.cu` |
| 采样/TopK | `sampling.py`、`topk*.py` | `jit/sampling.py`、`jit/topk.py` | `sampling.cuh`、`topk.cuh`、`csrc/topk.cu` |
| Norm | `norm/` | `jit/norm.py`、`jit/rmsnorm_silu.py` | `norm/*.cuh`、`csrc/norm.cu`、`rmsnorm_silu.cu` |
| RoPE | `rope.py` | `jit/rope.py` | `pos_enc.cuh`、`csrc/rope.cu` |
| Page/KV 管理 | `page.py` | `jit/page.py` | `page.cuh`、`csrc/page.cu` |
| 通信 | `comm/` | `jit/comm.py` | `comm/*.cuh`、`csrc/trtllm_*`、`mixed_comm.cu` |
| Mamba/SSM | `mamba/`、`msa_ops/` | `jit/mamba.py` | `mamba/*.cuh`、`csrc/selective_state_update*.cu` |
| NVFP4 Attention | `nvfp4_attention_sm120.py` | `jit/nvfp4_attention_sm120.py` | `csrc/nvfp4_attention_sm120/` |

---

## 8. 构建与分发体系

### 8.1 构建后端（`build_backend.py`，PEP 517）

自定义构建后端完成：

1. **子模块 patching**：对 `3rdparty/cutlass` 等应用 `3rdparty_patches/` 中的补丁
2. **可选 NVIDIA EP 构建**：探测并编译 NIXL、NCCL 等扩展包（`_build_nixl_ep` / `_build_nccl_ep`），探测失败自动跳过（`_gate_backend`）
3. **数据目录**：editable 安装用符号链接把 `include/`、`csrc/`、`3rdparty/cutlass` 暴露为 `flashinfer/data/{include,csrc,cutlass}`，供 JIT 编译引用（`env.py` 读取）
4. **版本元数据**：`_build_meta.py` 从 `version.txt` + git 生成

### 8.2 AOT 预编译（`aot.py`）

预编译发行包（`flashinfer-jit-cache`）流程：`register_default_modules()` 注册所有 `gen_*_module()` → 枚举 dtype/head_dim/arch 组合 → `build_jit_specs()` 批量编译 → 产物打进 wheel。运行时 `env.py::_get_aot_dir()` 优先读 `flashinfer_jit_cache` 包的目录（带版本一致性校验，可用 `FLASHINFER_DISABLE_VERSION_CHECK` 绕过）。

### 8.3 Cubin 分发（`flashinfer-cubin` + `cubin_loader.py`）

部分内核（TRT-LLM、CUDA-Tile 等）以**预编译 cubin** 分发而非源码 JIT：`cubin_loader.py` 按 `FLASHINFER_CUBINS_REPOSITORY`（可配镜像/离线）下载、SHA256 校验、文件锁防并发，缓存于 `FLASHINFER_CUBIN_DIR`；`FLASHINFER_NO_DOWNLOAD=1` 强制本地缺失即报错（CI/内网）。

### 8.4 CLI（`__main__.py`）

`python -m flashinfer <cmd>` 提供：`collect-env`（环境报告）、`artifacts download`（下载 cubin）、`jit-status`/`jit-clear`（JIT 状态查询/清理）、`show-config`、`tactics-blocklist generate` 等。

---

## 9. 质量基础设施

| 环节 | 工具/命令 | 说明 |
|---|---|---|
| 测试 | `pytest tests/` | 按模块分目录（attention/gemm/moe/comm/jit/trace/autotuner...）；`conftest.py` 自动跳过 OOM 测试；GPU 架构守卫 `is_sm90a_supported()` 等 |
| 基准 | `benchmarks/flashinfer_benchmark.py` | 统一框架，多后端对比（fa2/fa3/cudnn/cutlass/trtllm/cublas），CUPTI 计时（无则回落 CUDA Event），CSV 输出 |
| 代码检查 | `pre-commit run -a` | ruff/clang-format 等钩子 |
| CI | `Jenkinsfile`、`.github/workflows/`、`ci/`、`scripts/task_run_unit_tests.sh` | 多架构矩阵、sharding 支持 |
| 文档同步 | `docs/`、`.claude/skills/` | 架构说明与技能教程随代码同步更新 |

---

## 10. 架构演进要点（近期提交反映的趋势）

1. **内核层大幅向 CuTe-DSL / CUTLASS 迁移**：新内核（MoE B12x、FP4 GEMM、MLA sm120、NVFP4）越来越多走 Python DSL 或 CUTLASS 生成，而非手写 CUDA（`cute_dsl/`、`jit/cute_dsl_core.py`）。
2. **架构专门化加剧**：SM100/SM120/SM12x 专属内核与 `-gencode` 特化越来越多，`CompilationContext` 的 supported_major_versions 机制保证老架构不受影响。
3. **观测性成为一等公民**：`@flashinfer_api` 日志/数据 dump、`fi_trace`/`trace_apply`、API 日志统计模块（`jit/api_log_stats.py`）持续增强。
4. **向后兼容与 AOT 并存**：JIT 是开发默认，AOT/cubin 是部署优化，二者由 `JitSpec.build_and_load` 统一调度，保证同一套代码路径。
5. **与 vLLM/TRT-LLM 生态深度耦合**：大量 `trtllm_*` 内核、routing replay、低延迟 GEMM 表明其作为推理框架底层内核库的定位。

---

## 11. 快速导航（新人上手建议）

- **理解 JIT 流程**：读 `flashinfer/jit/core.py`（JitSpec 生命周期）+ `flashinfer/jit/attention/modules.py`（一个 gen 函数即可）
- **理解一次调用**：从 `flashinfer/decode.py::single_decode_with_kv_cache` 跟到 `csrc/single_decode.cu` 再到 `include/flashinfer/attention/decode.cuh`
- **新增内核**：按 CLAUDE.md「Adding a New Operation」十一步流程（kernel → launcher → binding → Jinja → gen → API → 测试 → AOT → 导出 → trace → 示例）
- **调试**：`FLASHINFER_JIT_VERBOSE=1` / `FLASHINFER_LOGLEVEL=3` / `FLASHINFER_JIT_DEBUG=1`，产物在 `~/.cache/flashinfer/<ver>/<arch>/`
