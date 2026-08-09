# 单请求 KV Cache 生命周期 FAQ

本文整理围绕 Qwen3-1.7B 单请求实验产生的连续提问与结论。重点是建立从
Prefill、首个输出 token、Decode 到 Prefix Cache 复用的动态直觉，并区分逻辑
抽象、物理显存和具体 Attention backend 的实现。

## 1. 第一个 output token 已经生成，它有自己的 KV Cache 吗？

没有。

Prefill 使用最后一个 Prompt token 的 hidden state 产生 logits，再从 logits 中采样
得到 `output token #1`。此时它只是刚被选出的 token ID，还没有作为模型输入执行
forward，因此还没有属于自己的 Q/K/V。

```text
Prompt Prefill
  -> 最后一个 Prompt token 的 logits
  -> sampling
  -> output token #1（尚无自己的 KV）

下一轮 Decode
  -> output token #1 作为输入
  -> 生成并保存它的 K/V
  -> 产生 output token #2
```

## 2. 首个输出 token 产生时，是谁的 Query 在做 Attention？

是最后一个 Prompt token 的 Query。它与 Prompt 范围内的 K/V 做 causal
Attention，并由最终结果产生首个输出 token 的 logits。

最后一个 Prompt token 会和包括自己在内的 K/V 做 Attention：

```text
Q_last attends to K/V[0 ... last]
```

Causal mask 禁止看未来 token，但不禁止看自己。到下一轮 Decode，才会生成
`output token #1` 的 Q/K/V；它的 Query 同样可以关注历史 Prompt 和自己当前的
K/V。

## 3. KV Cache 的物理池什么时候创建？是否占显存？

物理 KV Cache 池通常在 Engine 初始化和显存规划阶段创建，在请求到来前已经占用
GPU 显存。请求到来时主要是在池中分配 block ID，而不是临时向 CUDA 创建一整套
新的 KV Cache。

应区分三个动作：

```text
Engine 初始化：创建固定容量的物理 KV Cache 池，占用显存
请求运行：从池中取得 block，ref_cnt 增加
请求结束：解除持有，block 回到 free pool；显存池仍然存在
Engine 销毁：物理池及其 GPU 显存才真正释放
```

## 4. 物理 block 总数由什么决定？会按需求从 100 增加到 200 吗？

通常不会在运行中按请求需求动态扩容。Engine 初始化时根据可用显存预算和每个
block 的字节数确定池容量，核心关系为：

```text
num_gpu_blocks ~= 可用于 KV Cache 的显存预算 / 每个物理 block 的字节数
```

预算受到模型权重、运行时峰值、CUDA Graph、临时 workspace、
`gpu_memory_utilization`、KV dtype、层数、KV heads、head dimension、block size
和并行配置等因素影响。请求超过当前 free blocks 时通常会等待、抢占、重计算或
驱逐可复用缓存，而不是现场扩大物理池。

并发度为 1 表示同一时刻只运行一个请求（或压测只发一个请求），并不表示每个请求
都重新初始化 Engine。

## 5. 一个 token 的 KV 为 112 KiB，算大吗？

对单个 token 看不算大，但 KV 占用随 token 数、并发请求数线性增长，因此累计后很
可观。

Qwen3-1.7B 实验配置下：

```text
每 token KV
= 28 layers * 8 KV heads * 128 head_dim * (K + V) * 2 bytes
= 114,688 bytes
= 112 KiB

block_size = 16 时：
每个完整物理 block = 16 * 112 KiB = 1.75 MiB
```

例如单请求持有 2048 tokens，理论 KV 约为 224 MiB；多个长请求并发后会迅速达到
数 GiB。

## 6. Request 什么时候申请和释放 KV block？

请求进入调度后，调度器会在 Prefill 执行前为本轮计划计算的 token 分配所需物理
blocks。KV 数值随后在模型逐层 forward 时生成并写入。

Decode 时，当前 block 未满就继续写入；跨越 block 边界时再取得新 block。请求结束
后解除其对 blocks 的引用：

- 未启用 Prefix Cache：blocks 可直接作为普通空闲块复用。
- 启用 Prefix Cache：`ref_cnt` 可以降到 0，但 hash 和有效 KV 内容仍可保留，block
  同时进入可驱逐的 free queue。

因此 Request release 不等于释放 Engine 的 GPU 显存。

## 7. Prefix Cache 下 `ref_cnt: 1 -> 0` 后如何复用？

`ref_cnt=0` 只表示没有活跃请求持有该 block，不表示缓存内容立即清除。该 block
可以同时满足：

```text
在 free queue 中：必要时可被重新分配或驱逐
仍在 hash map 中：相同 prefix 到来时可以命中
```

新请求命中后，`touch()` 将 block 从 free queue 移除，并令 `ref_cnt: 0 -> 1`：

```python
if block.ref_cnt == 0 and not block.is_null:
    self.free_block_queue.remove(block)
block.ref_cnt += 1
```

代码位置：
[block_pool.py](vllm/v1/core/block_pool.py#L702)。

可以把 `ref_cnt=0` 到重新命中或被驱逐之间理解为 cached-but-inactive 时间，但当前
GPU Prefix Cache 没有基于时间的固定 active-time/TTL 语义。

## 8. Prefix 如何通过 hash 匹配？

vLLM 对完整 token block 计算链式 hash：

```text
h0 = H(NONE_HASH, block_tokens_0, extra_keys)
h1 = H(h0,        block_tokens_1, extra_keys)
h2 = H(h1,        block_tokens_2, extra_keys)
```

`parent_hash` 把当前 block 和它之前的完整上下文绑定。因此，即使两个请求中间出现
相同 token block，只要前面的上下文不同，最终 hash 也不同。

假设 `block_size=4`：

```text
Request A:
A0=[10,11,12,13] -> h0
A1=[20,21,22,23] -> h1=H(h0,A1,...)
A2=[30,31,32,33] -> h2=H(h1,A2,...)

Request C:
C0=[10,11,12,13] -> h0，命中 A0
C1=[20,21,22,23] -> h1，命中 A1
C2=[40,41,42,43] -> hX，未命中
```

查找从第一个 block 开始，遇到第一个 miss 就停止，所以得到的是“从请求开头开始
连续命中的最长 block prefix”，不会跳过中间 miss 去匹配后面的 block。

关键代码：

- hash 公式：[kv_cache_utils.py](vllm/v1/core/kv_cache_utils.py#L596)
- 链式生成请求 hashes：
  [kv_cache_utils.py](vllm/v1/core/kv_cache_utils.py#L691)
- Request 保存并更新 hashes：[request.py](vllm/v1/request.py#L262)
- 最长连续 prefix 查找：
  [single_type_kv_cache_manager.py](vllm/v1/core/single_type_kv_cache_manager.py#L682)
- hash 到物理 block 的查询：[block_pool.py](vllm/v1/core/block_pool.py#L198)
- 完整 block 写入 hash map：[block_pool.py](vllm/v1/core/block_pool.py#L607)

即使整个 Prompt 都命中，vLLM 通常仍限制最大命中长度为
`request.num_tokens - 1`，以便重新计算最后一个 token 并取得 logits，见
[kv_cache_manager.py](vllm/v1/core/kv_cache_manager.py#L253)。

## 9. A、B、C 顺序发送，A 与 C 前缀相同，C 一定能复用 A 吗？

不一定。C 能复用 A 需要同时满足：

- Prefix Cache 已启用。
- A 对应的完整 blocks 已被注册到 hash map。
- C 从开头开始具有相同的 token block、parent hash 和 extra keys。
- C 到来前，这些 `ref_cnt=0` 的 cached blocks 尚未被驱逐或覆盖。

B 与 A 没有共同前缀并不直接使 A 失效。如果 free blocks 足够，A 的 cached blocks
通常仍可保留，C 可以命中；如果 B 或其他请求带来分配压力，A 的空闲 cached blocks
可能被选中驱逐，C 就需要重新计算。

## 10. 什么会触发 Prefix Cache 驱逐？有 cache timeout 吗？

常见触发条件是物理 block 分配压力：free pool 需要提供 block，而某个 cached block
的 `ref_cnt=0`，它就属于可驱逐候选。其他失效来源还包括 cache reset、显式 evict、
block 被重新分配，以及 Engine 关闭。

当前 GPU Prefix Cache 通常没有“缓存 N 秒后自动删除”的 TTL/cache-timeout 配置。
缓存寿命主要由容量压力和淘汰顺序决定，而不是墙钟时间。`ref_cnt>0` 的 block 被活跃
请求持有，不应被驱逐；`ref_cnt=0` 的 block 可以继续命中，但没有保留承诺。

## 11. 命中多个 blocks 后，KV 是串行读取还是并行读取？

需要区分 CPU 元数据阶段和 GPU Attention 阶段：

| 阶段 | 行为 |
| --- | --- |
| Prefix hash 查询 | CPU 按 prefix 顺序逐 block 检查，首个 miss 后停止 |
| `touch`/`ref_cnt` 更新 | CPU 逐 block 更新元数据 |
| 实际 K/V 数据访问 | GPU 按 head、query、token/block tile 等维度并行访问 |

Prefix 命中阶段只得到 physical block IDs，不读取 K/V 张量。Attention kernel 通过
block table 间接访问离散物理 blocks；它不是先完整读完 block 0，再完整读 block 1
的简单串行过程。具体并行粒度和归并方式由 Attention backend 与 kernel 决定。

## 12. 一个 block 是否包含其中每个 token 的所有层 K/V？

逻辑上是。一个 `block_id` 代表最多 `block_size` 个 token 在所有相关 Transformer
层中的 K/V 槽位：

```text
block_id = 37
  -> Layer 0  cache slot 37：这些 token 的 K/V
  -> Layer 1  cache slot 37：这些 token 的 K/V
  ...
  -> Layer 27 cache slot 37：这些 token 的 K/V
```

模型执行到第 L 层时，为当前 token 计算 `K_L/V_L`，并写入该层对应的
`block_id + token_offset`。所有层执行结束后，该 token 才具有完整的跨层 KV。

CPU 上的 `KVCacheBlock` 主要保存 `block_id`、`ref_cnt`、`block_hash` 等元数据；真实
K/V 数值位于 GPU 上各层的 KV Cache tensor 中。

## 13. 一个 block 的所有 K/V 是否位于连续显存？

不保证。如果“一个 block”指同一 `block_id` 跨所有层的完整 K/V，那么它通常不是
一个连续显存对象，而是各层 KV Cache tensor 中相同编号的切片。

```text
Layer 0 的 block 37 -> 地址区域 A
Layer 1 的 block 37 -> 地址区域 B
...
Layer 27 的 block 37 -> 地址区域 Z
```

同一层内的 block slice 通常属于规则分配的大 tensor，但 K/V 是否相邻、维度顺序、
stride、padding 和 packing 均不能笼统保证。具体布局取决于：

- 推理引擎及其版本。
- Attention backend，如 FlashAttention、FlashInfer、Triton。
- MHA、GQA、MLA 等模型结构。
- KV dtype、并行切分及 kernel 对齐要求。

因此 `block_id` 应理解为跨层一致的寻址索引，而不是一段包含所有层数据的连续显存
指针。前述“一个 block 为 1.75 MiB”指同一 block ID 跨所有层占用的总和。

## 14. Prefill 写 KV 与产生第一个 output token 是并行的吗？

第一个 output token 依赖 Prefill 的最终 logits，因此不是一个可以脱离 Prefill 独立
运行的任务：

```text
逐层 Q/K/V 和 Attention
  -> 最后一层 hidden state
  -> LM Head logits
  -> sampling
  -> output token #1
```

但这里需要区分两个命题：

1. Attention 必须使用 Prompt 的 K/V **数值**，这是数学数据依赖。
2. Attention 是否必须先等这些数值写入持久 **KV Cache**，属于 backend 实现选择。

理论上 backend 可以直接用 fresh K/V 计算 Prefill Attention，同时或随后保存 Cache；
也可以先写 Cache，再从 Cache 读取。

当前 vLLM FlashAttention decoder 路径采用显式的“先更新 Cache，再 Attention”顺序。
Attention wrapper 先调用 `unified_kv_cache_update()`，并通过 dummy dependency 防止
`torch.compile` 重排，见
[attention.py](vllm/model_executor/layers/attention/attention.py#L543) 和
[attention.py](vllm/model_executor/layers/attention/attention.py#L775)。

FlashAttention backend 使用 `reshape_and_cache_flash()` 写入，见
[flash_attn.py](vllm/v1/attention/backends/flash_attn.py#L1098)；随后 Attention kernel
以 `key_cache`、`value_cache` 和 `block_table` 为输入，见
[flash_attn.py](vllm/v1/attention/backends/flash_attn.py#L1041)。

CUDA kernel launch 对 CPU 可以是异步的，但 GPU stream/data dependency 会保证
Attention 不会在必要的 Cache 更新完成前读取错误数据。

## 15. 为什么采用“先写 KV Cache，再读取”的设计？是否有额外负担？

有额外负担，但这是有意的工程权衡。

KV Cache 写入本身无法省略，因为未来 Decode 必须使用 Prompt KV。额外成本主要是：

- 独立 `reshape_and_cache` kernel launch。
- 将 K/V scatter 到 paged physical blocks。
- Attention 再次读取刚写入的当前 K/V。
- 相应显存带宽和执行依赖。

这样设计的主要收益是把所有 K/V 统一成一个数据源：

```text
Prefix Cache 中的历史 KV
+ 本轮新计算的 Prompt KV
+ Decode token KV
        ↓
统一通过 block_table 访问 Paged KV Cache
```

否则 kernel 需要同时处理 cached KV 和临时 fresh KV，或者进行昂贵的拼接，并处理
两者边界、causal mask、混合 Prefill/Decode、chunked prefill 等情况。统一 Cache
路径显著简化了调度和 backend 接口。

性能上：

- Prefill 每个 token 都要写 KV，长 Prompt 下有可见成本，但 Attention 和矩阵乘法
  通常也很重。
- Decode 每轮只写一个新 token，却要读取整个历史 KV，因此历史读取往往比新增写入
  更占主导。
- 刚写数据可能部分命中 GPU cache 层级，但不能依赖这一点消除成本。

其他 backend 可以选择把 KV update 融入 Attention，或同时读取 cached KV 与 fresh
KV。vLLM 通过 `forward_includes_kv_cache_update` 表达这种差异；当前
FlashAttention backend 将其设为 `False`，见
[flash_attn.py](vllm/v1/attention/backends/flash_attn.py#L86)。

最终应记住：

```text
K/V 数值是产生 logits 的必要数据依赖；
是否先写入 KV Cache 再计算 Attention，是 backend 的实现策略；
当前 FlashAttention 路径选择统一 Paged KV Cache，付出一定带宽成本以换取统一调度。
```
