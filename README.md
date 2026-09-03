# 🦙 llama-deploy

**一键部署 llama.cpp + GGUF 大语言模型的智能管理工具**

支持 Windows / Linux / macOS / 树莓派，对中文用户友好。

---

## ✨ 功能特性

| 功能 | 说明 |
|------|------|
| 🔍 **模型市场** | 搜索 HuggingFace / ModelScope / 热门 GGUF 发布者的模型 |
| 📚 **模型库** | 管理本地已下载的模型，一键切换激活 |
| 🎮 **GPU 加速** | 自动检测单卡/多卡，支持 CUDA / Metal / Vulkan / CPU |
| 👁️ **视觉能力** | 自动配对并规范命名主模型与 mmproj，复制到其他设备仍能识别 |
| 🌐 **Web 管理器** | 浏览器操作，可视化配置编辑 |
| 🤖 **OpenAI 兼容 API** | 兼容 OpenAI 格式，方便接入各种应用 |
| 📦 **一键部署** | 自动下载引擎 + 模型 + 编译，全程无需手动操作 |
| 🪞 **国内镜像** | 内置 HuggingFace / GitHub 镜像加速 |

---

## 📁 项目结构

```
llama-deploy/
├── config.example.jsonc # 可公开的示例配置
├── config.jsonc      # [本地生成，不提交] 真实配置
├── deploy.py         # 一键部署脚本
├── run.py            # 启动/管理脚本
├── manager.py        # Web 管理界面
├── uninstall.py      # 卸载工具
├── README.md         # 本文档
├── llama.cpp/        # [自动下载] llama.cpp 引擎
├── models/           # [自动下载] 模型文件
│   ├── *.gguf        # 模型本体
│   └── vision/       # 视觉模块
└── .hf-venv/         # [自动创建] 下载工具虚拟环境
```

---

## 🚀 快速开始

### 环境要求

- **Python 3.8+**
- **操作系统**：Windows 10+ / Ubuntu 20.04+ / macOS 12+ / Raspberry Pi OS Bookworm
- **可选**：NVIDIA GPU + 驱动（启用 CUDA 加速）

### 三步上手

```bash
# 1️⃣ 获取项目并启动管理器
git clone https://github.com/d05628/llama-deploy.git
cd llama-deploy
python manager.py

# 2️⃣ 浏览器打开（首次启动会生成本地 config.jsonc）
#    http://localhost:9090

# 3️⃣ 在网页上操作：搜索模型 → 选择下载 → 一键部署 → 启动服务
```

管理器默认只监听本机 `127.0.0.1`。确实需要从可信局域网管理时，可显式执行 `python manager.py --host 0.0.0.0`。

### 或者使用命令行

```bash
# 首次使用命令行部署时，先从示例创建本地配置
# Windows PowerShell: Copy-Item config.example.jsonc config.jsonc
# Linux/macOS:       cp config.example.jsonc config.jsonc

# 一键部署（根据 config.jsonc 自动下载引擎和模型）
python deploy.py

# 交互式聊天
python run.py chat

# 启动 API 服务器
python run.py server

# 带视觉能力启动
python run.py server --vision

# 后台启动
python run.py server --background

# 停止服务器
python run.py stop

# 查看状态
python run.py status
```

### 更新 llama.cpp

先停止 `llama-server`、`llama-cli` 等所有 llama.cpp 进程，然后在 Web 管理器中点击「升级 llama.cpp」，或执行：

```bash
python deploy.py --upgrade-llama
```

更新器会扫描官方最新的完整 nightly Release，精确匹配操作系统、CPU 架构和 CPU/CUDA/Vulkan 后端。CUDA 主包与对应版本的运行时会成对下载；镜像失效时自动回退 GitHub 官方直链。旧引擎只有在下载、解压和两个程序的版本验证全部成功后才会删除，失败会自动回滚。

匿名 GitHub API 被限流时会自动改用官方 Release feed。也可以通过 `GITHUB_TOKEN` 或 `GH_TOKEN` 环境变量提供只读 Token 以提高 API 配额；Token 不应写入 `config.jsonc`。

---

## 🌐 Web 管理器

启动管理器后，浏览器打开 `http://localhost:9090`：

### 🔍 模型市场
- 搜索 HuggingFace、HF 镜像（国内加速）、ModelScope、热门 GGUF 发布者
- 根据设备内存/显存自动评估模型兼容性
- 选择主模型后自动挑选同仓库、同模型家族的视觉模块
- 本地使用规范化文件名，远端原始路径单独保存，因此子目录和非 HuggingFace 直链也能正确下载

### 👁️ 主模型与视觉模型自动配对

下载时，视觉模块会自动使用主模型名作为前缀。例如：

```text
Qwen3-VL-8B-Q4_K_M.gguf
Qwen3-VL-8B-Q4_K_M.mmproj-f16.gguf
```

如果主模型是分片文件，命名时会自动去掉 `-00001-of-xxxxx`。配置中仍保留视觉模块在远端仓库里的原始名称，因此改名不会影响重新下载。将主模型和对应的 `.mmproj-*.gguf` 一起复制到另一台设备的 `models/` 与 `models/vision/` 后，模型库可按名称重新匹配，不依赖原机器的绝对路径。

### 📚 模型库
- 查看所有已下载的 GGUF 模型
- 一键切换激活模型（需重启服务器生效）
- 删除不需要的模型释放空间

### ⚙️ 配置编辑
- 可视化编辑所有参数
- GPU 后端选择（auto / cuda / vulkan / cpu）
- 采样参数调整
- 镜像地址配置

### 🚀 部署管理
- 一键部署，实时彩色日志
- 启动/停止服务器
- 实时状态监控

### 📊 系统信息
- 硬件检测（CPU / 内存 / GPU / 磁盘）
- 基于硬件的 AI 智能建议

---

## ⚙️ 配置说明

`config.jsonc` 支持 `//` 注释，由 Web 管理器首次运行时自动生成。它可能包含本机模型选择、内网地址或 API 密钥，已被 `.gitignore` 排除；请勿提交。公开配置示例位于 `config.example.jsonc`。

### 模型配置

| 参数 | 说明 | 示例 |
|------|------|------|
| `model.repo_id` | HuggingFace 仓库 ID | `unsloth/Qwen3.5-0.8B-GGUF` |
| `model.model_file` | 模型 GGUF 文件名 | `Qwen3.5-0.8B-Q4_K_M.gguf` |
| `model.mmproj_file` | 自动生成的配对视觉模块文件名（留空禁用） | `Qwen3-VL-8B-Q4_K_M.mmproj-f16.gguf` |

### GPU 配置

| 参数 | 说明 | 默认 |
|------|------|------|
| `gpu.backend` | 计算后端 | `auto` |
| `gpu.gpu_layers` | GPU 卸载层数（`-1`=按实时显存自动拟合，`0`=CPU） | `-1` |
| `gpu.flash_attention` | Flash Attention | `true` |

**backend 选项：**

| 值 | 说明 |
|----|------|
| `auto` | 自动检测：CUDA > Vulkan > CPU |
| `cuda` | NVIDIA CUDA（需要 NVIDIA 显卡 + 驱动） |
| `vulkan` | 通用 GPU 加速（AMD/NVIDIA/Intel） |
| `cpu` | 仅 CPU |

### 服务器配置

| 参数 | 说明 | 默认 |
|------|------|------|
| `server.host` | 监听地址 | `0.0.0.0` |
| `server.port` | 监听端口 | `8080` |
| `server.threads` | CPU 线程数（0=自动） | `0` |
| `server.ctx_size` | 上下文长度 | `8192` |
| `server.enable_thinking` | 思考/推理模式 | `false` |
| `server.reasoning_budget` | 思考 token 预算（`-1` 为不限制） | `512` |

### 采样参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `sampling.temperature` | 温度（越高越随机） | `0.7` |
| `sampling.top_k` | Top-K 采样 | `20` |
| `sampling.top_p` | Top-P 核采样 | `0.8` |
| `sampling.presence_penalty` | 重复惩罚（防止重复） | `1.5` |
| `sampling.max_tokens` | 单次最大生成数 | `2048` |
| `performance.spec_type` | speculative decoding（可手动设 `draft-mtp` 测 MTP/nextn 模型） | `off` |
| `performance.spec_draft_n_max` | MTP 单步草稿 token 数 | `3` |
| `performance.profile` | `auto` 自适应 / `maximum` 极限显存利用 / `compatible` 兼容优先 | `auto` |
| `performance.cache_type_k/v` | KV cache 类型；`auto` 在显存高压或长上下文时自动采用 `q8_0` | `auto` |
| `performance.batch_size` / `ubatch_size` | `0` 交给新版 llama.cpp 按设备自动拟合；非零为手动固定值 | `0` |
| `performance.fit_target_mb` | 自动拟合后每张 GPU 预留显存；`0` 会按 GPU、模型和视觉模块动态计算 | `0` |

### 性能自适应

- 生成线程使用实际物理核心，提示词批处理使用逻辑核心，兼顾单 token 延迟与提示词吞吐。
- CUDA 会汇总多张 NVIDIA GPU 的可用显存；Apple Silicon 自动选 Metal；Windows 同时识别 Vulkan loader 与开发工具。
- 新版 llama.cpp 支持时使用 `-ngl auto` 与 `--fit`，按启动时的实时空闲显存决定卸载，而不是绑定某台机器的层数。
- Flash Attention、连续批处理和 KV cache 复用会按能力启用；MTP 仅在用户显式请求且模型、当前二进制均支持时启用。
- MoE GPU 层估算按完整常驻权重计算，避免把“每 token 激活参数少”误当成“显存只需相同比例”而导致 OOM。
- 能读取 NVIDIA compute capability；RTX 50 / Blackwell 会采用更激进但仍由 `--fit` 保护的显存策略。源码构建检测到 CUDA Toolkit 12.8+ 时会生成 `sm_120` 原生架构。

若追求单路输出速度，优先选择能完整放入显存的合适量化；`batch_size` 主要影响长提示词和并发吞吐，不会等比例提高逐 token 生成速度。

### RTX 5060 Ti 16GB 与 Qwen3.8

Qwen3.8-27B Q4_K_M 文件本身已非常接近 16GB 显存容量，Windows 桌面占用、CUDA 工作区和 KV cache 会使它通常无法 100% 常驻显存。本项目的 `auto` 策略会自动采用：

- 单路 `parallel=1`、Flash Attention、`-ngl auto` 和实时显存拟合；
- Q8 KV cache，把更多显存留给权重；
- 约 384MB 的目标余量；Web 管理器选择“极限性能”后会尝试 256MB；
- 物理核心负责逐 token 解码、逻辑核心负责提示词批处理。

15+ token/s 是合理的调优目标，但并非固定保证：Q4_K_M 仍可能有少数层留在 CPU，最终速度很依赖 CPU、DDR5 带宽、当时空闲显存和上下文长度。若以吐字速度优先，约 14.6GB 的 IQ4_XS 或更小量化更容易全量放入 16GB 显存，往往比“量化稍高但跨 CPU/GPU”的组合更快。

在目标电脑上先关闭占显存的软件，再运行：

```powershell
python run.py benchmark --sweep
```

结果中的 `tg128` 是单路生成速度。五组结果里选择能稳定完成且 `tg128` 最高的 `fit-target`，填入 Web 管理器的“GPU 预留显存”；如果 256/384MB OOM，就使用下一档。普通复测可运行 `python run.py benchmark`。

Qwen3.8-Flash-Next 是总参数规模远大于 27B、每 token 只激活一部分参数的实验模型；“激活参数少”并不代表整个 GGUF 不需要装入内存。当前公开量化通常已超过这台机器 64GB RAM 的舒适容量，而且 llama.cpp 的 Flash-Next MTP 加速仍在实验阶段。因此本项目支持识别并启动兼容量化，但默认不启用其实验 MTP；在 5060 Ti 16GB + 64GB RAM 上不把它列为稳定高速目标。强行依赖页面文件可以启动某些超低量化，却通常会严重拖慢吐字。

RTX 50 要获得原生 Blackwell 内核，优先安装最新 NVIDIA 驱动并使用更新器选出的最新兼容 CUDA 包：

```powershell
python deploy.py --upgrade-llama
python run.py status
```

若更新器提示所选预编译包低于 CUDA 12.8，它仍可兼容运行，但没有 `sm_120` 原生构建优化；高级用户可安装 CUDA Toolkit 12.8+、CMake 和 Visual Studio C++ 工具链后运行 `python deploy.py --build-from-source`。该操作同样会先备份旧引擎，并在构建或验证失败时自动回滚。

### 下载配置

| 参数 | 说明 | 默认 |
|------|------|------|
| `download.hf_mirror` | HuggingFace 镜像 | `https://hf-mirror.com` |
| `download.github_mirror` | 可选 GitHub 镜像（失败自动回退官方直链） | 空 |
| `download.timeout` | 下载超时（秒） | `300` |
| `download.retries` | 重试次数 | `3` |

---

## 📱 设备推荐

### 树莓派 5

| 内存 | 推荐模型 | 量化 | ctx_size |
|------|----------|------|----------|
| 4GB | Qwen3.5-0.8B | Q4_K_M | 1024-2048 |
| 8GB | Qwen3.5-2B | Q4_K_M | 2048-4096 |

**树莓派额外准备：**

```bash
# 增加 Swap（必须）
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup && sudo dphys-swapfile swapon

# 安装编译依赖
sudo apt install -y build-essential cmake git libcurl4-openssl-dev libopenblas-dev

# 主动散热（必须，否则降速 20-30%）
```

### Windows / Linux 桌面

| 显卡显存 | 推荐模型 | 量化 |
|----------|----------|------|
| 无独显 | 0.8B-2B | Q4_K_M |
| 4GB | 2B-3B | Q4_K_M |
| 6GB | 3B-7B | Q4_K_M |
| 8GB+ | 7B-14B | Q4_K_M |
| 16GB+ | 14B-32B | Q4_K_M / Q8_0 |

---

## 🔌 API 使用

启动服务器后，提供 OpenAI 兼容 API：

### 文本对话

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "你好"}],
    "temperature": 0.7
  }'
```

### 图片理解（需启动视觉模式）

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "描述这张图片"},
        {"type": "image_url", "image_url": {"url": "图片URL"}}
      ]
    }]
  }'
```

### Python 调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "你好"}],
)
print(response.choices[0].message.content)
```

---

## 🔧 开机自启（Linux / 树莓派）

```bash
sudo tee /etc/systemd/system/llama.service > /dev/null << 'EOF'
[Unit]
Description=llama-deploy Server
After=network.target

[Service]
Type=simple
User=你的用户名
WorkingDirectory=/home/你的用户名/llama-deploy
ExecStart=/usr/bin/python3 run.py server --background
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable llama.service
sudo systemctl start llama.service
```

---

## 🗑️ 卸载

```bash
python uninstall.py
```

支持选择性卸载：

- **完全卸载** — 删除引擎 + 模型 + 配置
- **仅删除引擎** — 保留模型
- **仅删除模型** — 保留引擎
- **仅清理临时文件**

完全删除整个项目：

```bash
# Windows
rmdir /s /q llama-deploy

# Linux / macOS
rm -rf llama-deploy
```

---

## ❓ 常见问题

### 安装部署

| 问题 | 解决方案 |
|------|----------|
| git clone HTTP/2 报错 | 脚本已自动设置 HTTP/1.1 |
| pip 报 externally-managed | 脚本使用独立虚拟环境 |
| HuggingFace 连不上 | 配置 `hf_mirror` 为 `https://hf-mirror.com` |
| 下载的文件只有几百字节 | Xet 指针文件，脚本会自动用 huggingface_hub 重新下载 |
| Windows 编译失败 | 脚本自动下载预编译版本，无需编译 |
| llama.cpp 无法更新 | 先停止所有 llama.cpp 进程，再运行 `python deploy.py --upgrade-llama`；更新器会自动包含 nightly 预发布 |
| GitHub API 提示 rate limit | 会自动改用官方 Release feed；也可临时设置 `GITHUB_TOKEN` |
| GitHub 镜像返回网页/无效 ZIP | 会校验 ZIP、大小与可用的 SHA-256，并自动回退官方直链 |
| Windows 终端 Emoji 乱码 | 脚本已内置 UTF-8 修复 |

### 运行问题

| 问题 | 解决方案 |
|------|----------|
| 模型回复很慢 | 关闭思考模式 `enable_thinking: false` |
| 内存不足被 kill | 降低 `ctx_size`，或换更小的模型/量化 |
| GPU 没有被使用 | 确认 `gpu.backend` 为 `auto` 或 `cuda`，需要 CUDA 版引擎 |
| `--color` 报错 | 新版 llama.cpp 需要 `--color auto` |
| `-fa` 报错 | 新版需要 `-fa on` |
| 服务器启动后看不到状态 | 旧进程未关闭，执行 `taskkill /F /IM llama-server.exe` |
| ModelScope 搜不到模型 | API 不稳定，会自动回退到 HuggingFace 搜索 |

### 局域网与外部对接

Web 管理器的「部署管理」页提供「发布到局域网」和「兼容网关」功能：

| 用途 | 地址示例 |
|------|----------|
| llama.cpp 原生 Web/API | `http://192.168.x.x:8080` |
| OpenAI 兼容 API | `http://192.168.x.x:8080/v1` |
| 兼容网关（Ollama / Claude Code / OpenAI） | `http://192.168.x.x:11434` |

Claude Code / Anthropic 兼容：

```powershell
$env:ANTHROPIC_BASE_URL="http://192.168.x.x:11434"
$env:ANTHROPIC_API_KEY="local-no-key-needed"
$env:ANTHROPIC_MODEL="llama-deploy-local"
$env:ANTHROPIC_SMALL_FAST_MODEL="llama-deploy-local"
$env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1"
claude
```

Ollama 兼容：

```bash
OLLAMA_HOST=http://192.168.x.x:11434
```

Codex / OpenAI 兼容：

```powershell
$env:OPENAI_BASE_URL="http://192.168.x.x:11434/v1"
$env:OPENAI_API_KEY="local-no-key-needed"
codex
```

Gemini CLI 兼容：

```powershell
$env:GOOGLE_GEMINI_BASE_URL="http://192.168.x.x:11434"
$env:GEMINI_BASE_URL="http://192.168.x.x:11434"
$env:GEMINI_API_KEY="local-no-key-needed"
gemini
```

VSCode / JetBrains 插件建议优先选择 OpenAI-compatible 或 Ollama provider：

```text
OpenAI base URL: http://192.168.x.x:11434/v1
Ollama base URL: http://192.168.x.x:11434
Model: llama-deploy-local
API key: local-no-key-needed
```

如果其他设备无法访问，优先检查 Windows 防火墙是否放行 `8080` 和 `11434` 端口。

> 安全提示：管理器没有登录认证，默认因此只监听 `127.0.0.1`。推理服务与兼容网关的示例配置会监听局域网；`local-no-key-needed` 只是兼容占位值，不是访问控制。请只在可信内网使用，或在反向代理上配置 TLS、身份认证和防火墙白名单，切勿直接暴露到公网。

### 树莓派专项

| 问题 | 解决方案 |
|------|----------|
| 编译 llama.cpp 很慢 | 正常，约 10-20 分钟 |
| Swap 不够 | 扩展到 2GB：修改 `/etc/dphys-swapfile` |
| CPU 温度过高降频 | 必须安装主动散热 |
| SD 卡加载模型慢 | 建议使用 NVMe SSD |

---

## 🧪 测试

```bash
# 快速回归测试（不下载模型或引擎）
python -m unittest discover -s tests -v

# 可选：在临时目录下载并运行官方最新 Windows CPU 包
# PowerShell
$env:RUN_LLAMA_UPDATE_INTEGRATION="1"
python -m unittest discover -s tests -p test_live_update.py -v
```

真实下载测试只使用系统临时目录，结束后自动清理，不会覆盖当前 `llama.cpp/`。

---

## 📋 项目文件清单

| 文件 | 用途 | 状态 |
|------|------|------|
| `config.example.jsonc` | 脱敏示例配置 | ✅ |
| `config.jsonc` | 本地私有配置 | 自动生成且不提交 |
| `deploy.py` | 一键部署 | ✅ |
| `run.py` | 启动管理 | ✅ |
| `manager.py` | Web 管理界面 | ✅ |
| `compat.py` | Ollama / Claude Code / OpenAI 兼容网关 | ✅ |
| `uninstall.py` | 卸载工具 | 🆕 |
| `tests/` | 更新器回归与真实下载测试 | ✅ |
| `README.md` | 完整文档 | 🆕 |

---

## 📜 许可证

本工具为开源项目，使用 [MIT 许可证](LICENSE)。

llama.cpp 遵循 MIT 许可证。模型文件遵循各自的许可协议（如 Apache 2.0）。

---

## 🙏 致谢

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — 高性能 LLM 推理引擎
- [Qwen](https://github.com/QwenLM/Qwen) — 通义千问大模型
- [HuggingFace](https://huggingface.co) — 模型托管平台
- [unsloth](https://github.com/unslothai/unsloth) — 高质量 GGUF 量化
