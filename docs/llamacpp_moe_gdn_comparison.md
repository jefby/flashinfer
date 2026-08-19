# FlashInfer vs llama.cpp：Qwen3.5/3.6 MoE + GDN 支持对比

> 分析日期：2026-08-19（2026-08-24 随上游 main 更新刷新）
> 背景：Qwen3.6-35B-A3B（35B 总参 / 3B 激活 MoE，Gated Delta Attention 线性注意力 + MoE）在 NVIDIA DRIVE Thor U（SM110，CUDA 13.2）等带宽受限机器上的解码优化选型。
> 结论来源：直接对比两仓库源码（本仓库 `flashinfer/` 与 `llama.cpp` 的 `src/models/qwen35moe.cpp`、`ggml/src/ggml-cuda/gated_delta_net.cu` 等）。

---

## 1. 定位差异（根本不同）

| | **FlashInfer** | **llama.cpp** |
|---|---|---|
| 定位 | 数据中心**内核库**（vLLM/SGLang/TRT-LLM 底层算子） | 独立**推理引擎**（桌面/嵌入/边缘/本地） |
| 并发 | 高并发服务端批处理、CUDA Graph、plan/run | 单/少量请求，server 模式有限并发 |
| 调用方式 | Python API，嵌入框架 | 独立可执行 + GGUF 模型格式 |
| 模型格式 | 不解析文件格式，只收内存张量（DLPACK/TVM-FFI） | 原生解析 GGUF |

## 2. GDN 实现方式对比（核心结论）

### llama.cpp（`qwen35moe.cpp` → `delta-net-base.cpp` → `gated_delta_net.cu`）

**既有手写 CUDA 内核，也有纯算子组合路径，融合内核默认开启**：

```
build_delta_net (T=1 判断)
  ├─ 自回归 T=1:
  │    ├─ fused_gdn_ar=true（默认）→ build_delta_net_fused
  │    │        └─ ggml_gated_delta_net 算子 → gated_delta_net.cu
  │    │           【手写 CUDA 内核，327 行】
  │    │           warp 级优化：一列一 warp、fastmodulo/fastdiv、warp reduce
  │    └─ fused 关闭 → build_delta_net_autoregressive（~12 个通用算子组合）
  └─ Prefill K>1:
       ├─ fused_gdn_ch=true（默认）→ 同一内核 chunked 模式
       └─ fused 关闭 → build_delta_net_chunking（~31 个算子：cumsum/solve_tri/逐 chunk 循环）
```

- 多后端：CUDA / Vulkan / OpenCL / SYCL / Hexagon / OpenVINO
- CUDA 架构：默认编译列表不含 SM110，需 `GGML_NATIVE` 在 Thor U 现场编译

### FlashInfer（`flashinfer/gdn_kernels/`，9 个专用内核 + 实验性融合算子）

| 变体 | 用途 |
|---|---|
| `gdn_decode_bf16_state.py` | 自回归 T=1，基础 state |
| `gdn_decode_bf16_wy_output_only.py` / `wy_ucache.py` / `wy_ucache_flush.py` | WY 输出布局 + unified-cache 带宽优化（ucache_flush 2820/3751 行） |
| `gdn_decode_pretranspose.py` / `nontranspose.py` | V-major / K-major state 布局 |
| `gdn_decode_mtp.py` | 投机解码（T>1 多 token） |
| `experimental/gdn_fused_decode*.py` | **实验性**：`gdn_fused_decode_step` 将 b/a projection GEMV、causal conv1d 状态更新、q/k/v split 与 gated delta-rule decode 融合为单一 op（SM120 特化，registry 注册层几何） |
| `blackwell/`（CuTe-DSL） | chunked prefill + CP + tile scheduler |
| `delta_rule_dsl/` | sm90 / sm120 prefill |

- 架构特化：sm90a / sm100a / **sm110a** / sm120f（`compute_110a` 显式支持，CUDA ≥ 13.0）

## 3. MoE 与量化对比

| 维度 | **FlashInfer** | **llama.cpp** |
|---|---|---|
| MoE 内核 | CUTLASS fused / TRT-LLM / CuTe-DSL / BGMV 多后端，路由+激活+finalize 融合 | topk-moe.cu（warp softmax、sigmoid、delayed softmax）+ cuBLASLt GEMM |
| 权重量化 | W4A16（NVFP4/MXFP4）、FP8、FP4 KV cache | IQ2/3/4 系列、Q4_K/Q5_K/Q6_K/Q8_0、MXFP4、F8 |
| Expert offload | 无（假设全 GPU 驻留） | ✅ GPU→CPU→磁盘分级卸载 |
| SM110 支持 | ✅ 显式 `compute_110a` | ⚠️ 默认不含，需 native 编译 |

### SM110 上 MoE 权重 4-bit 化的已知缺口（FlashInfer）

- W4A16 两个后端均不覆盖 SM110：CUTLASS 版仅 SM90（`_CUTLASS_W4A16_ARCHS = (90,)`）、B12x 版仅 SM120/121
- TRT-LLM routed MoE 现注册 `_TRTLLM_ROUTED_ARCHS = (100, 103, 107)`；NVFP4/MXFP4 支持 SM100/103/107，W4A16 仅支持 SM100/SM107 且在 SM103 上禁用
- CUTLASS BF16 dense 已加入 SM110（`_CUTLASS_BF16_ARCHS` 含 110），但 MoE W4A16 两个后端仍不覆盖
- SM110 上 MoE 权重带宽优化目前走 CUTLASS fused MoE + FP8 路径

## 4. 性能差距预期（Thor U / SM110 带宽受限场景）

| 场景 | 预期 |
|---|---|
| 自回归 T=1 解码 | 两者均有融合单内核 → **数量级相当**（均 O(1) state，不读长 KV） |
| FlashInfer 领先点 | state 布局按需选择（K-major/V-major 匹配 h_k=16/h_v=32 非对称头）、ucache 减少 state 读写带宽、sm110a 特化 gencode |
| llama.cpp fused 关闭时 | 退化到 12-31 算子组合路径，多 launch + 中间张量 → **预估慢 2-5×** |
| Prefill | llama.cpp 单态 fused 内核；FlashInfer CuTe-DSL tile scheduler/CP，大 batch 领先明显 |
| 架构匹配 | llama.cpp 无 sm110 特化；FlashInfer 直接 compute_110a，带宽敏感内核再拉开数个百分点 |

## 5. 选型结论

| 需求 | 选谁 |
|---|---|
| 服务端多用户吞吐、GDN 架构级带宽优化、SM110 原生支持 | **FlashInfer**（+ SGLang） |
| 单流低延迟、内存受限、2-3bit 激进量化/offload、全平台 | **llama.cpp** |
| 无 CUDA 环境（CPU/手机/AMD） | llama.cpp |

**一句话**：FlashInfer 胜在「SM110 特化 + GDN 布局/带宽深度优化（ucache、K/V-major、MTP）+ 服务端吞吐」；llama.cpp 胜在「IQ 低比特量化 + offload 灵活性 + 全平台覆盖」。llama.cpp 的 GDN 有默认开启的手写融合内核，数量级与 FlashInfer 持平，但特化深度不及。

## 6. 实测验证建议（Thor U）

```bash
# FlashInfer 侧
export FLASHINFER_CUDA_ARCH_LIST="11.0"
python benchmarks/bench_gdn_decode.py --preset qwen3-next --batch-size 1 32 128
python benchmarks/bench_gdn_prefill.py
python benchmarks/bench_gdn_fused_decode.py   # 实验性融合 decode 算子

# llama.cpp 侧（需 GGML_NATIVE 现场编译以支持 SM110）
cmake -B build -DGGML_NATIVE=ON -DCMAKE_CUDA_ARCHITECTURES=native
./build/bin/llama-cli -m qwen3.5-35b-a3b.gguf -t 1 --no-mmap
```

对比同一 batch/seq 下的 tok/s 与带宽利用率（ncu 的 dram__bytes.sum.per_second）。
