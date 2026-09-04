# 代码审查清单（2026-09-04）

对 llama-deploy v1.2.1 全量代码（8503 行）的问题统计。

> **状态：本清单中的 16 项已在 v1.3.0 全部修复**，本文档保留为问题记录与回归依据。
> 修复说明见 [CHANGELOG.md](../CHANGELOG.md)。

严重度：🔴 需尽快修 · 🟡 应该修 · 🟢 可以缓

| # | 严重度 | 位置 | 问题 |
|---|---|---|---|
| A1 | 🔴 | compat.py | `api_key` 从未被校验，网关默认对整个局域网开放 |
| A2 | 🔴 | manager.py:1998 | `Allow-Origin: *` + 零鉴权，任意网页可控制管理器 |
| A3 | 🟡 | manager.py:1749 | 删除接口未过滤 glob 通配符，`*.gguf` 可删模型 |
| A4 | 🟡 | deploy.py:1444 | `extractall()` 无路径校验（zip slip） |
| A5 | 🟡 | deploy.py:1188 | 下载校验 fail-open：无 digest 即放行 |
| A6 | 🟡 | manager.py:176 | `pid_running()` 不校验进程身份，可能强杀无关进程 |
| A7 | 🟡 | manager.py 前端 | `innerHTML` 生拼外部数据，全项目无 HTML 转义 |
| B1 | 🟡 | 三处 | 配置默认值三份副本，无同步机制 |
| B2 | 🟡 | manager.py:2473 | `api()` 从不检查 HTTP 状态码 |
| B3 | 🟢 | 全项目 | 38 处 `except: pass` 静默吞错 |
| B4 | 🟢 | manager.py | `activate()` 不校验模型文件是否存在 |
| C1 | 🟡 | manager.py:2131 | 几乎无响应式；窄屏把设备信息整个隐藏 |
| C2 | 🟢 | manager.py | 零 `aria-*`，25 个 `<label>` 全无 `for=` |
| C3 | 🟢 | manager.py | 21 处 `alert()`，与已有 `showToast()` 并存 |
| C4 | 🟢 | manager.py:3327 | 启动横幅 ASCII 边框随 host 长度错位 |
| C5 | 🟢 | manager.py:3302 | 轮询定时器永不清除，页面隐藏仍在跑 |

---

## A. 安全

### 🔴 A1 — compat 网关的 `api_key` 完全无效

`api_key` 在 compat.py 里**只出现一次**，就是从配置读出来那行（compat.py:72）。
`do_POST` / `do_GET` 里没有任何鉴权分支。

而网关默认 `host: "0.0.0.0"`、`port: 11434`。也就是说**局域网内任何设备都能
直接调用你的模型**，无需任何凭证。

比"没有鉴权"更糟的是：配置里明晃晃写着 `"api_key": "local-no-key-needed"`，
会让人以为改成真 key 就安全了 —— 改了也没用。

修法二选一：真正实现校验（比较 `Authorization: Bearer` / `X-Api-Key`），
或者删掉这个配置项并把默认监听地址改成 `127.0.0.1`。

### 🔴 A2 — 管理器对任意网页开放

管理器默认绑 `127.0.0.1`（这点是对的），但对所有响应都发：

```python
self.send_header("Access-Control-Allow-Origin", "*")
```

且**没有任何鉴权**。浏览器允许网页向 localhost 发跨域请求，而 `Allow-Origin: *`
明确授权对方**读取响应**。后果是：你浏览任意网站时，那个站点的 JS 可以

- 读 `/api/config` —— 里面有网关 API key、内网地址
- 写 `/api/config` —— 改模型、改监听地址
- 调 `/api/models/delete` —— 删模型
- 调 `/api/server/start|stop`、`/api/deploy`、`/api/lan/publish`

修法：去掉通配 CORS（管理界面是同源的，本就不需要），
或加一个启动时生成、写进页面的一次性 token。

### 🟡 A3 — 删除接口可用通配符

```python
if ".." in filename or "/" in filename or "\\" in filename:
    return {"status": "error", "message": "非法文件名"}
...
for f in MODELS_DIR.rglob(filename):
```

挡了路径穿越，但没挡 glob 元字符。`rglob("*.gguf")` 会匹配到文件并删除。
单次只删一个，但配合 A2 就是"访问恶意网页 → 模型被删"。

顺带：应当校验解析后的路径确实位于 `MODELS_DIR` 内（防符号链接）。

### 🟡 A4 — 解压未校验成员路径

deploy.py:1444 `archive.extractall(BASE_DIR)`。压缩包正常来自 GitHub release，
但 `download.github_mirror` 是**用户可配置的第三方镜像**。含 `../` 条目的恶意包
可以写到项目目录之外。应逐条校验成员路径。

### 🟡 A5 — 下载校验 fail-open

```python
if not digest.lower().startswith("sha256:"):
    return True
```

拿不到 digest 就直接放行。逻辑上"没校验"和"校验通过"返回同一个值，
调用方无法区分。至少应把两种情况分开，并在无 digest 时告警。

### 🟡 A6 — PID 校验不认身份

`pid_running()` 只确认"存在这个 PID 的进程"，不确认它是不是 llama-server。
停止流程用 `taskkill /F /PID`。若 PID 文件过期且该 PID 被系统复用，
会**强杀一个无关进程**。应同时核对进程名或命令行。

### 🟡 A7 — 前端无 HTML 转义

全项目没有 escapeHtml 函数，20 处 `innerHTML` 直接拼接外部数据：

```js
'<span>'+id+'</span>'+sizeTag+          // id 来自模型源 API
```

`escapedId` 只转义了反斜杠和单引号（为了 `onclick` 的 JS 字符串上下文），
**没有做 HTML 转义**。

HuggingFace 官方 repo id 字符集受限，直接注入不易。但管理器同时支持
ModelScope / Ollama / Gitee AI / **用户可配置的 hf_mirror**。
恶意或被劫持的源返回带标签的 `id`、`author`、`filename` 即可注入 →
存储型 XSS → 配合 A2 等于完全控制管理器。

## B. 正确性与健壮性

### 🟡 B1 — 配置默认值三份副本

| 位置 | 形式 |
|---|---|
| manager.py `default_config()` | Python dict |
| manager.py `defaultCfg()` (JS) | JS 对象 |
| `config.example.jsonc` | JSONC |

目前三份值一致，但没有任何机制保证。新增配置项漏改一处就会静默漂移 ——
这和 v1.2.1 修掉的"量化识别四份副本"是同一类问题，
建议同样收敛到单一来源（Python 为准，JS 通过接口取，示例文件由脚本生成或加测试比对）。

### 🟡 B2 — `api()` 不检查 HTTP 状态码

```js
async function api(url,method,body){
  try{
    ...
    var r=await fetch(url,opts);
    return await r.json();
  }catch(e){return{error:e.message}}
}
```

`r.ok` 从未判断。服务端返回 404 / 500，只要 body 是合法 JSON，
就会被当成正常数据继续往下渲染。应先判 `r.ok` 再解析。

### 🟢 B3 — 38 处静默吞错

`except: pass` / `except Exception: pass` 分布：manager 15、deploy 11、run 9、compat 3；
另有 3 处裸 `except:`（均在 manager.py），会连 `KeyboardInterrupt` 一起吞掉。

不必全改，但至少：裸 `except:` 应改为 `except Exception:`；
涉及用户可见行为的分支应记日志而非静默。
这类静默正是显存/量化那批 bug 能长期潜伏的原因之一。

### 🟢 B4 — `activate()` 不校验文件存在

可以把 `model_file` 设成任意字符串并写入配置，直到启动服务才报错。
应在切换时就确认文件确实在 `models/` 下。

## C. UI / 交互

### 🟡 C1 — 几乎没有响应式

3300 行的界面里只有一个断点：

```css
@media(max-width:768px){
  .sidebar{width:60px}.sidebar .nav-text,.sidebar .logo span{display:none}
  .main{margin-left:60px;padding:16px}.form-row{grid-template-columns:1fr}.sys-info{display:none}
}
```

只处理了侧边栏收窄。更糟的是 `.sys-info{display:none}` —— 窄屏下
**把设备信息整块藏掉**，而那里正是显存预算这类最该看的内容。
模型卡片、表格在手机宽度（375px）下会溢出。

### 🟢 C2 — 无障碍缺失

零 `aria-*` 属性；25 个 `<label>` **全部没有 `for=`**，
点击标签不会聚焦对应输入框，读屏软件也无法建立关联。

### 🟢 C3 — 两套反馈机制并存

21 处 `alert()` / `confirm()` 阻塞式弹窗，而项目里已经实现了 `showToast()`。
风格不统一，且 `alert()` 会打断操作流。

### 🟢 C4 — 启动横幅边框错位

```
║  📡 监听地址:  http://{host}:{port}             ║
```
`host` 为 `0.0.0.0`(7 字符) 或 `127.0.0.1`(9 字符) 时右边框对不齐；
emoji 在多数终端占双宽，也会破坏对齐。建议改用不依赖等宽假设的排版。

### 🟢 C5 — 轮询定时器不清理

`setInterval(pollStatus,10000)` 从不 `clearInterval`，
页面切到后台仍持续请求。可用 `document.visibilityState` 暂停。

---

## 修复建议顺序

1. **A1 + A2** —— 安全影响面最大，且改动都很小（加鉴权 / 去掉通配 CORS）
2. **A3 + A7** —— 与 A2 组合放大危害
3. **B1 + B2** —— 防止后续再出现同类静默错误
4. **C1** —— 直接影响日常使用体验
5. 其余按需
