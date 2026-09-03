# 更新日志

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [1.2.0] - 2026-09-03

围绕"显存装不下模型"这一场景的性能优化。在 RTX 5060 Ti 16GB + Qwen3.8-27B
（`qwen35` 混合注意力架构）上，吐字速度 **6.39 → 16.34 token/s（2.56×）**。

完整测量数据、方法论与无效路径记录见 [docs/performance-tuning.md](docs/performance-tuning.md)。

### 新增

- **显存预算诊断**。权重超出可用显存时，llama.cpp 会静默把层卸载到 CPU，
  速度掉数倍却毫无提示。现在启动时直接指出瓶颈，并给出可执行的建议：
  > 权重 15.3GB 超出可用显存 14.6GB，必然有一部分层留在 CPU 上，这是吐字慢的主因；
  > 当前 ctx_size=16384 还要额外占用 288MiB KV cache…若要让权重完整驻留显存，
  > 需换用体积 ≤13.8GB 的量化档位

  权重装得下、但上下文过长时，则给出当前显存下建议的 `ctx_size` 上限。

- `run.kv_cache_mb()` —— 按模型元数据估算 KV cache 显存占用。
  正确处理**混合注意力架构**：`full_attention_interval=4` 的模型每 4 层才有 1 层
  带 KV，其余是常数大小的 SSM 循环状态。按全层计算会把 KV **高估 4 倍**，
  进而误判上下文预算。

- `run.max_ctx_for_vram()` —— 反解当前显存下权重仍能完整驻留的最大上下文长度。

### 变更

- `gguf_meta()` 新增读取注意力相关字段（`head_count_kv` / `key_length` /
  `value_length` / `full_attention_interval` / `nextn_predict_layers`），
  改为按后缀表统一提取。此前只读 `block_count` 与 `context_length`，
  无法估算 KV cache。

- 27B / 16GB 高压档的 `--fit-target` 由 256 MiB 收紧到 64 MiB（`maximum` 档）。
  权重比显存大时，留给驱动的余量每多一点就少一点权重驻留 GPU；
  实测提速约 15%。`auto` 档保持 256 MiB 的保守值。

### 文档

- 新增 [docs/performance-tuning.md](docs/performance-tuning.md)：完整调优日志，
  含各档位实测数据、量化档位质量对比（困惑度实测）、复现方法，
  以及**已验证无效的优化路径**（见下）。

### 已验证无效（不要重复尝试）

这些手段看起来合理，实测均为负收益，细节见性能文档第 4 节：

| 手段 | 结果 |
|---|---|
| `-ot token_embd.weight=CPU` | 12.91 → **8.41** t/s，会干扰 `--fit` 的张量级自动放置 |
| `--spec-type draft-mtp` | 15.98 → **11.90** t/s，多占约 350MiB 显存把权重挤到 CPU |
| `--spec-type ngram-*` | +2%~6%，在噪声内 |
| 手工指定 `-ngl` | 均劣于 `-ngl auto --fit on` |

> 调优过程中曾单次测到 21.27 t/s（+45%），中位数复测后未能复现，属异常值。
> 本机单次测量波动可达 ±30%，所有结论均改用 3 次取中位数。

### 测试

新增 4 个用例，覆盖 KV 估算（含混合注意力与元数据缺失）与两类显存预警分支。

---

## [1.1.1] 及更早

见 git 提交历史。
