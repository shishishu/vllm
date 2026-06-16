# vLLM Build Experience

Date: 2026-06-16 to 2026-06-17  
Workspace: `/home/shishishu/Documents/vllm`

## Result

Build and validation succeeded.

- vLLM import: passed
- `torch.cuda.is_available()`: `True`
- Real vLLM inference: passed with `Qwen/Qwen2.5-0.5B-Instruct`
- Interactive multi-turn chat: passed
- GPU monitoring run: passed

## Source Tree

The directory looked like a valid vLLM source root because it contained:

- `pyproject.toml`
- `setup.py`
- `CMakeLists.txt`
- `vllm/__init__.py`

However, `.git` was an empty directory, so Git metadata was unavailable.

Commit:

```text
unavailable: git rev-parse failed because .git is empty
```

This mattered because vLLM uses `setuptools-scm` to derive the package version.

## Environment

Observed system environment:

```text
OS: Ubuntu 26.04 LTS
Kernel: Linux 7.0.0-22-generic x86_64
GPU: NVIDIA GeForce RTX 2080
GPU memory: 8192 MiB
Driver: 595.71.05
CUDA runtime shown by driver: 13.2
GPU compute capability: 7.5
System Python: Python 3.14.4
Build Python: Python 3.12.13 in .venv
System gcc: 15.2.0
System nvcc: not found
System cmake: not found
System ninja: not found
```

Resource snapshot:

```text
Disk: about 175 GiB free after install
Memory: 30 GiB total, about 26 GiB available
Swap: 8 GiB
```

Installed environment size:

```text
.venv: about 8.4 GiB
.hf_cache: about 954 MiB
```

## Build Strategy

Because system `nvcc` was missing, the successful build used vLLM's
precompiled extension path while still installing the current source tree as an
editable install.

Important: this was not `pip install vllm` from a published wheel. The final
installed package came from the local source tree's editable wheel:

```text
vllm==0.0.0+local.precompiled
torch==2.11.0+cu130
```

## Commands Used

Install `uv` into a temporary user location:

```bash
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh
UV_UNMANAGED_INSTALL=/tmp/uv-bin sh /tmp/uv-install.sh
```

Create a Python 3.12 virtual environment:

```bash
/tmp/uv-bin/uv venv --python 3.12 .venv
```

First editable install attempt:

```bash
MAX_JOBS=2 VLLM_USE_PRECOMPILED=1 \
/tmp/uv-bin/uv pip install -e . --torch-backend=auto --prerelease=allow
```

This failed because `.git` was empty and `setuptools-scm` could not determine a
version.

Successful build retry with an explicit local version:

```bash
MAX_JOBS=2 VLLM_USE_PRECOMPILED=1 \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+local \
/tmp/uv-bin/uv pip install -e . --torch-backend=auto --prerelease=allow
```

The build phase succeeded, but one dependency download failed. The missing
dependency was retried separately:

```bash
/tmp/uv-bin/uv pip install llvmlite==0.47.0
```

Then the already built local editable wheel was installed directly to avoid
re-fetching vLLM nightly precompiled metadata:

```bash
/tmp/uv-bin/uv pip install \
/home/shishishu/.cache/uv/sdists-v9/editable/02b45f9b7419fd37/3esVCNxQJOOwtNvb/vllm-0.0.0+local.precompiled-0.editable-cp312-cp312-linux_x86_64.whl \
--torch-backend=auto --prerelease=allow
```

## Validation

Import and CUDA validation:

```bash
.venv/bin/python -c 'import torch, vllm; print("vllm", getattr(vllm, "__version__", "unknown")); print("torch", torch.__version__); print("torch_cuda", torch.version.cuda); print("cuda_available", torch.cuda.is_available()); print("device_count", torch.cuda.device_count()); print("device_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
```

Observed result:

```text
vllm 0.0.0+local
torch 2.11.0+cu130
torch_cuda 13.0
cuda_available True
device_count 1
device_name NVIDIA GeForce RTX 2080
```

Small model validation used:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

The key runtime settings for RTX 2080 were:

```bash
ALL_PROXY=
all_proxy=
VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_HOME=/home/shishishu/Documents/vllm/.venv/lib/python3.12/site-packages/nvidia/cu13
PATH=/home/shishishu/Documents/vllm/.venv/bin:/home/shishishu/Documents/vllm/.venv/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH
```

The Python `LLM(...)` call also needed:

```python
attention_backend="TRITON_ATTN"
```

This avoided FlashInfer JIT problems on RTX 2080 / compute capability 7.5.

## Problems and Fixes

### 1. Empty `.git` broke version detection

Symptom:

```text
LookupError: setuptools-scm was unable to detect version
```

Fix:

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+local
```

### 2. Dependency download failed

Symptom:

```text
Failed to download llvmlite==0.47.0
tls handshake eof
```

Fix:

```bash
/tmp/uv-bin/uv pip install llvmlite==0.47.0
```

Then resume install.

### 3. Hugging Face download failed due to proxy scheme

Symptom:

```text
ValueError: Unknown scheme for proxy URL URL('socks://127.0.0.1:7890')
```

Cause:

```text
ALL_PROXY=socks://127.0.0.1:7890
```

Fix:

```bash
ALL_PROXY=
all_proxy=
```

The existing `HTTP_PROXY` and `HTTPS_PROXY` values used `http://127.0.0.1:7890`
and worked.

### 4. FlashInfer JIT could not find nvcc

Symptom:

```text
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

Fix:

```bash
CUDA_HOME=/home/shishishu/Documents/vllm/.venv/lib/python3.12/site-packages/nvidia/cu13
PATH=/home/shishishu/Documents/vllm/.venv/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH
```

### 5. FlashInfer JIT could not find ninja

Symptom:

```text
FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
```

Fix:

```bash
PATH=/home/shishishu/Documents/vllm/.venv/bin:$PATH
```

### 6. FlashInfer JIT header/compiler mismatch

Symptom:

```text
#error "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
```

This happened while FlashInfer tried to JIT compile sampling and attention
kernels.

Fix:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0
```

and in the vLLM Python code:

```python
attention_backend="TRITON_ATTN"
```

After this, Qwen inference succeeded.

## Working Inference Command

Interactive local chat:

```bash
env HF_HOME=/home/shishishu/Documents/vllm/.hf_cache \
CUDA_VISIBLE_DEVICES=0 \
VLLM_LOGGING_LEVEL=INFO \
ALL_PROXY= \
all_proxy= \
VLLM_USE_FLASHINFER_SAMPLER=0 \
CUDA_HOME=/home/shishishu/Documents/vllm/.venv/lib/python3.12/site-packages/nvidia/cu13 \
PATH=/home/shishishu/Documents/vllm/.venv/bin:/home/shishishu/Documents/vllm/.venv/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH \
.venv/bin/python dev_start/interactive_qwen_chat.py
```

Commands inside chat:

```text
/clear  reset context
/exit   quit
/quit   quit
```

## GPU Monitoring Result

The monitored multi-turn run sampled GPU stats 87 times across about 46.3
seconds.

```text
GPU: NVIDIA GeForce RTX 2080, 8192 MiB

Memory used:
  min 561 MiB
  avg 1116 MiB
  max 5925 MiB

GPU utilization:
  min 3%
  avg 28.84%
  max 56%

Memory controller utilization:
  min 0%
  avg 3.47%
  max 20%

Power:
  min 2.68 W
  avg 40.32 W
  max 172.68 W

Temperature:
  min 51 C
  avg 55.62 C
  max 64 C
```

## Log Files

Main logs:

- `dev_start/uv-venv.log`
- `dev_start/editable-install-precompiled.log`
- `dev_start/editable-install-precompiled-retry-version.log`
- `dev_start/install-llvmlite-retry.log`
- `dev_start/install-local-editable-wheel.log`
- `dev_start/validate-import-cuda.log`
- `dev_start/validate-vllm-qwen-0.5b-retry-triton-attn.log`
- `dev_start/validate-qwen-multiturn.log`
- `dev_start/validate-qwen-multiturn-rerun.log`
- `dev_start/validate-qwen-multiturn-gpu.csv`
- `dev_start/validate-qwen-multiturn-gpu-summary.json`

## Reusing This on a Fresh Clone

Use a fresh `.venv` in the clone. Do not reuse this directory's `.venv`, because
editable install points at the source tree it was installed from.

For a fresh clone at `/home/shishishu/Documents/vllm`:

```bash
cd /home/shishishu/Documents/vllm
/tmp/uv-bin/uv venv --python 3.12 .venv
MAX_JOBS=2 VLLM_USE_PRECOMPILED=1 \
/tmp/uv-bin/uv pip install -e . --torch-backend=auto --prerelease=allow
mkdir -p dev_start
```

Then run:

```bash
env HF_HOME=/home/shishishu/Documents/vllm/.hf_cache \
CUDA_VISIBLE_DEVICES=0 \
VLLM_LOGGING_LEVEL=INFO \
ALL_PROXY= \
all_proxy= \
VLLM_USE_FLASHINFER_SAMPLER=0 \
CUDA_HOME=/home/shishishu/Documents/vllm/.venv/lib/python3.12/site-packages/nvidia/cu13 \
PATH=/home/shishishu/Documents/vllm/.venv/bin:/home/shishishu/Documents/vllm/.venv/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH \
.venv/bin/python dev_start/interactive_qwen_chat.py
```
