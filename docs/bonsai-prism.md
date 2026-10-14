# Bonsai 2 PQ2_0 / PTQ1_0

`Ternary-Bonsai-2-27B-PQ2_0.gguf` and `PTQ1_0.gguf` require the
[author's Prism llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp).
Their GGML tensor types are 142 and 143. Stock llama.cpp support for the architecture
or older Bonsai Q1_0/Q2_0 models does not establish support for these packings.

The installer uses the complete release pinned by
[Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo),
[`prism-b10743-adfffbe`](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b10743-adfffbe).
For Windows NVIDIA GPUs, install the author's CUDA 12.4 package and matching runtime:

```powershell
python deploy.py --install-prism --prism-cuda 12.4
```

This downloads and verifies the release's SHA256 values, checks the executable/DLL
set, then publishes a complete directory under `engines/prism/<tag>/<asset-stem>/`.
It does not load a model, start/stop services, change `config.jsonc`, or replace
`llama.cpp/`. Installation integrity is separate from successful inference: run a
bounded actual inference check before adopting either packing. PTQ1_0's smaller
file is not evidence that it is faster; the model card describes hardware-specific
unpacking costs. This project's Prism installer, registration and runtime routing
currently support the flat Windows release layout only. Other platforms must use
the author's build/setup instructions independently; `run.py` reports `unsupported`
instead of treating a native non-Windows Prism engine as a registered Windows one.

The engine's `.llama-deploy.json` records `provider`/`engine`, the author repository,
release `tag`, short `commit`, `backend`, supported `quant_types`, `asset_digest`
and `runtime_digest` (both `sha256:<hex>`), and `binary_sha256` (the bare hex SHA256
of `llama-server.exe`). Integrations can reuse this manifest without a second
installation record. The executable hash identifies the executable; the verified
archive hashes cover the corresponding bundled DLLs.

`runtime_files_sha256` maps every DLL filename in the binary directory to its
complete SHA256. `executable_files_sha256` likewise records every `.exe`, including
`llama-server.exe`, `llama-cli.exe`, and `llama-bench.exe` when shipped. Engine
selection and installation reuse verify both exact inventories before selecting
any executable; changed, missing, or added EXE/DLL files fail before inference.
The server's original `binary_sha256` is also retained and checked.

An older manifest containing only the server hash, or server plus DLL hashes,
does not silently acquire trust for other executable files. Selection refuses it
until explicit migration. For a just-completed trusted archive-verified installation:

```powershell
python deploy.py --register-prism-manifest "engines/prism/prism-b10743-adfffbe/llama-prism-b10743-adfffbe-bin-win-cuda-12.4-x64"
```

This only completes the manifest after matching its archive hashes to the author's
release, preserves its existing data, and does not download or execute binaries.
Use it only immediately after a trusted archive-verified installation: this adds
the first baseline for files that the old manifest did not record, and does not
independently prove those previously unrecorded files came from the archive.
Existing server, EXE and DLL hashes must all pass before any missing field can be
added; migration cannot accept changed, missing or added files in a registered
inventory. A complete manifest is verified without rewriting it. For an older
installation whose unrecorded files are no longer trusted, retain it and make a
separate verified installation instead of using registration to repair it.

With the default `engine.provider=auto`, `run.py` reads GGUF tensor descriptors
and selects a registered Prism engine for types 142/143. Filename recognition also
works before download. Normal quantizations retain their existing engine selection.
The local model library labels these formats and displays their Prism requirement.
Missing/incompatible Prism engines raise `unsupported` before starting inference;
they do not fall back to stock binaries or a different compute backend.

Optional configuration (merge into your existing file; do not replace it):

```json
"engine": {
  "provider": "auto",
  "directory": "",
  "cuda_version": "12.4"
}
```

`provider` accepts `auto`, `upstream`, and `prism`. Choosing `upstream` with either
special packing fails explicitly. `directory` can pin a registered version folder,
relative to this project or absolute; point it at a retained older Prism directory
to roll back. The stock engine and its existing rollback mechanism are independent.
The install command always writes a separate pinned release directory and refuses
to overwrite an existing incomplete directory. `--prism-cuda` overrides the install
CUDA version; the default is 12.4. CUDA main/runtime packages must match exactly.

The [model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) states that
reasoning defaults to `xhigh`, recommends `medium` for shorter output, and says
`low` does not work as a low-effort mode. The downloaded GGUF templates explicitly
test `enable_thinking`; `run.py` supplies
`--chat-template-kwargs '{"enable_thinking":false}'` when
`server.enable_thinking=false`. With thinking enabled, optional
`server.reasoning_effort` accepts `medium` or `xhigh` (default). This is distinct
from `--reasoning off`, which controls reasoning extraction. API clients can use
`chat_template_kwargs: {"enable_thinking": false}`; request overrides must agree
with the intended mode. No global sampling or context settings are changed.

The `--install-prism` entrypoint is the installation-only path. Ordinary deployment
also installs Prism when the selected model needs it, then continues the existing
model download/verification flow.
