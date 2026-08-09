# Topic 2：Block 与 Paged KV 管理 FAQ

本文回答主任务讨论过程中反复出现的机制问题。实验数字和完整映射图见
[`TOPIC_2_SUMMARY.md`](TOPIC_2_SUMMARY.md)，研究延伸见
[`TOPIC_2_THINKING.md`](TOPIC_2_THINKING.md)。

## 1. 为什么 KV Cache 要切成固定大小的 Block？

因为请求长度、到达时间和结束时间都不固定。若每个请求必须申请一段连续、可继续
扩展的显存，随着并发请求反复进入和退出，会出现外部碎片、扩容搬迁和难以复用的
空洞。

固定 Block 带来：

- 按需增页，不必预留最大上下文；
- 请求增长时不搬迁旧 KV；
- 请求结束时按页回收；
- 不同请求可以复用任意空闲页；
- Prefix Cache 可以按完整页共享；
- 外部碎片转化为最多 `block_size - 1` slots 的尾页内部碎片。

代价是需要 Block Table，并且最后一页可能没有填满。

## 2. Token、Logical Block 和 Physical Block 怎么对应？

以 `block_size=16` 为例：

```text
token 0…15   → Logical Block 0
token 16…31  → Logical Block 1
token 32…47  → Logical Block 2
```

对 token 位置 `i`：

```text
logical_block = i // 16
offset        = i % 16
```

Logical Block 只描述请求中的逻辑页号；Block Table 再决定它映射到哪个 Physical
Block。例如：

```text
L0 → P417
L1 → P423
L2 → P419
```

## 3. Prompt 从 16 增加到 17 个 token 时发生什么？

Scheduler 在执行 GPU forward 前发现 1 个 Block 不够，于是分配第二个 Physical
Block，并更新 Block Table：

```text
16 tokens:
L0 → P0，16/16 valid

17 tokens:
L0 → P0，16/16 valid
L1 → P1， 1/16 valid
```

旧 Block 不扩容、不搬迁。GPU forward 时，每一层一次处理本轮 scheduled token
范围，根据 slot mapping 把前 16 个 token 的 K/V 写入 `P0`，第 17 个写入 `P1`。

## 4. 最后一个未填满 Block 怎么处理？

它仍占用完整 Physical Block。以 17 tokens 为例，第二页只有 1 个有效 slot，另外
15 个是内部碎片。

关闭 Prefix Cache 时，这些空 slot 仍属于当前请求的尾页，后续 Decode 可以继续
追加；不能同时借给另一个普通请求。请求结束后整页一起回收。

## 5. 一个请求可以使用不连续 Physical Block 吗？

可以，而且并发和反复回收后很常见。Block Table 保证：

```text
连续 Logical Blocks
        ↓
任意 Physical Block IDs
```

Attention 按 Block Table 收集地址，不要求物理连续，也不要求先做一次 KV 拼接。
物理离散可能影响局部性和地址处理开销，但 PagedAttention 的设计目标就是在这种
布局下直接工作；不能仅凭 ID 不连续推导出显著性能下降，需要单独 profiling。

## 6. 多个活动请求可以共享同一个 Physical Block 吗？

本轮 `prefix_caching=False`，因此不同活动请求不共享普通 Physical Block。

开启 Prefix Cache 后，多个请求可以引用同一组已经提交并可复用的完整 Prefix
Blocks，此时 `ref_cnt` 可以大于 1。一个请求释放引用不代表立刻覆盖数据；只有没有
活动引用且缓存策略允许淘汰时，这个页才可作为新的分配对象。

## 7. 为什么基础实验保持 Prefix Cache 关闭？

为了先隔离最基本的分配、增长和释放语义：

- `ref_cnt=1` 可直接解释为当前请求持有；
- Finish 后 `1→0` 可直接观察回收；
- 不会把 Prefix 命中、共享引用、缓存保留和 LRU 淘汰混入结果。

Prefix Cache 不是不重要，而是下一层问题；它适合在基础映射正确后单独研究。

## 8. 什么时候 `ref_cnt` 从 0 变为 1？

在 BlockPool 分配操作内部：

```text
从 Free Queue 取出 Block
→ 确认 ref_cnt == 0
→ ref_cnt += 1
→ 返回给请求并加入 Block Table
```

因此从调用者视角，成功得到的新 Block 已经是 `ref_cnt=1`。不应把“取出”和“增加
引用”理解成两个可以被其他请求插入的独立事务。

## 9. 请求结束时是先 `ref_cnt=0`，还是先释放？

“释放”是一次 Block Manager 操作，不是先做一个无归属的 `ref_cnt=0` 状态再调用
另一个释放函数：

```text
free_blocks(block)
  → ref_cnt -= 1
  → 如果结果为 0，将 Block 放回 Free Queue/缓存淘汰队列
```

关闭 Prefix Cache且没有共享引用时，通常就是 `1→0→进入 Free Queue`。有共享引用
时可能只是 `2→1`，此时 Block 仍被其他请求使用。

## 10. 请求什么时候被判断为结束？

当引擎确认满足终止条件，例如达到 `max_tokens`、生成 EOS、命中 stop 条件或取消，
请求进入 finished 状态。Scheduler 随后从 running 集合移除请求，并调用 KV Cache
Manager 释放该请求的 Block 引用。

采样出最后一个 output token 和释放 KV 属于相邻但不同的运行时事件；Tracer 应以
Scheduler 的 finish/free 事件判断，而不是仅看 output 数量猜测。

## 11. Prompt=17、Output=3 时 KV 怎么读写？

简化时间线：

```text
Prefill:
  allocate P0/P1
  每层写 prompt token 0…16 的 K/V
  Attention 读取 prompt KV
  computed=17，采样 y0

Decode 1:
  输入 y0
  每层写 y0 的 K/V 到 P1 offset=1
  Attention 读取 prompt + y0 的 KV
  computed=18，采样 y1

Decode 2:
  输入 y1
  每层写 y1 的 K/V 到 P1 offset=2
  Attention 读取 prompt + y0 + y1 的 KV
  computed=19，采样 y2

Finish:
  y2 不再进入下一次 forward，所以没有 y2 的 KV
  P0/P1 的 ref_cnt 1→0，返回 Free Queue
```

## 12. Prefill 是批量写，还是 Block by Block 写？

从模型执行角度，是按 layer 处理本轮的一批 tokens；从内存目标看，这批 token 可以
通过 slot mapping scatter 到多个 Physical Block。

```text
Layer 0: 生成本轮全部 token 的 K/V → scatter 到对应 slots
Layer 1: 生成本轮全部 token 的 K/V → scatter 到对应 slots
...
```

它不是 CPU 外层循环：

```text
先完整写完 P0 的所有 layers
再完整写 P1 的所有 layers
```

不同 Attention backend 可能融合或拆分 cache update，但 Block Table 的逻辑不变。

## 13. 17-token Prefill 中，P0 写 16 个、P1 写 1 个，会明显速度不一致吗？

两个目标页写入量不同，但不能把它理解成两个串行、各自有独立墙钟时间的 Block
任务。它们通常属于同一个 layer 的 batched cache-update/scatter kernel。

边界可能带来少量地址计算、分支和不完全利用，但 Prefill 总耗时主要还包括 QKV
projection、Attention、MLP 等。要判断 16+1 是否比 17 个连续 slots 慢，需要 GPU
kernel profiling；本轮 CPU Block Trace 不能给出这个结论。

## 14. Decode 写 KV 与 Prefill 有什么不同？

数据格式和寻址机制相同，粒度不同：

| 阶段 | 每个请求每轮典型写入 | 主要特征 |
| --- | ---: | --- |
| Prefill | 一段 Prompt/chunk | 大批量、计算密集，可跨多页 |
| Decode | 1 token | 小写入、反复读取越来越长的历史 KV |
| MTP/Spec Decode | 多个候选 tokens | 多写入，但包含可能被拒绝的 provisional KV |

Decode 不需要另一种 Block 管理器；它只是在每轮继续使用尾页，页满时再申请新页。

## 15. 为什么最终 KV 数是 `prompt + output - 1`？

KV 对应“已经作为模型输入执行过 forward 的 token”。Prefill 计算全部 Prompt，随后
采样第一个 output；之后每个 output token 要到下一步作为输入时才生成自己的 KV。

请求在采样出最后一个 output 后立即结束，所以最后一个 output 没有后续 forward，
也没有对应 KV。若它随后作为新一轮会话输入，才会在下一轮生成 KV。

## 16. 基础任务需要考虑 Output 变化吗？

需要，但不必覆盖很大的组合矩阵。Prompt 边界验证 Prefill 分页；少量 Output=2、
17、18 的组合验证 Decode 如何填满尾页和跨入新页。这能回答 Prompt 与 Decode 两种
跨页模式是否使用同一套机制。

## 17. 是否需要比较不同 `block_size` 的性能？

本轮不需要。固定 16 是为了先验证抽象和边界。改变 Block Size 会同时影响：

- 尾页内部碎片；
- Block Table 长度；
- Prefix Cache 命中粒度；
- 分配/回收次数；
- Attention kernel 的页处理方式。

这是独立性能 Topic，需扩大 Prompt、并发和 Prefix workload，并做 GPU profiling；
不能用本轮固定 Block Size 数据回答。

## 18. 为什么 Prompt 范围从 15/16/17 扩展到 2047/2048/2049？

扩展是为了验证边界规律不会只在很短输入成立，并让并发实验更容易产生离散 Physical
IDs。它增强的是正确性和内存布局覆盖，不是 Block Size 性能对比。

## 19. Chunked Prefill 的粒度等于 Block Size 吗？

不等于。Block Size 是 KV 内存分页粒度；Prefill chunk 是 Scheduler 每轮允许计算的
token 数，通常受 `max_num_batched_tokens`、本轮其他请求、模型限制和剩余 Prompt
长度共同决定。

实验中 token budget 为 512：

```text
第 1 轮：512 - Decode 1 - Short Prefill 16 = Long Prefill 495
后续轮：512 - Decode 1 = Long Prefill 511
最后：21
```

因此得到 `[495, 511, 511, 511, 21]`，明显不是 16 的 Block 粒度。

## 20. `chunked_size` 是哪个参数？

在本实验配置中没有独立固定的 `chunked_size`。控制上限的是
`max_num_batched_tokens=512`，实际 chunk 是每轮分给该请求的剩余 token budget。

谈论 chunk 时应区分：

```text
Block Size             = KV 内存页大小
max_num_batched_tokens = 一轮调度总 token budget
actual prefill chunk   = 本轮真正分给某请求的 tokens
```

## 21. 并发大于 1 时，Prefill 主要有什么变化？

- 多个新请求可能在同一 Scheduler step 做 Prefill；
- Scheduler 先按策略和 token budget 决定各自份额；
- 各请求独立分配 Block、维护 Block Table；
- GPU forward 将它们组成混合 batch；
- 长 Prefill 会拉长同一 iteration，影响其他请求 TTFT/ITL；
- 开启 Chunked Prefill 后，长 Prompt 可跨多个 iteration，被重新调度。

## 22. 并发大于 1 时，Decode 主要有什么变化？

每个活跃 Decode 请求通常贡献 1 个 token，形成 batched decode。每个请求仍读自己的
历史 KV、写自己的新 slot；请求页满或结束的时刻不同，因此 Block 分配和回收交错，
Physical IDs 更容易离散。

吞吐因 batch 增大而提高，但单请求可能承担更长 iteration 和排队时间。

## 23. 并发分配是否需要处理竞争？

需要保证一致性，但当前中心 Scheduler/Block Manager 把核心 allocate/free 决策串行
组织，并不是每个请求线程无锁修改 Free Queue。因此这里主要是调度与容量竞争：

```text
多个请求争用有限 token budget
+
多个请求争用有限 KV Blocks
```

而不是本实验中观测到的 Python 层数据竞争。

## 24. 不连续 Physical Blocks 会影响读写性能吗？

理论上可能影响：地址加载、访存合并、TLB/缓存局部性和传输 descriptor 数量；但
PagedAttention 正是为非连续页设计，Block 仍是固定大小的连续小页。

所以正确结论是：

> 物理离散是可支持的；性能影响需用控制 Physical Layout 的 kernel/microbenchmark
> 验证，不能从 Trace 中 Physical ID 的视觉顺序直接下结论。

## 25. `chunked_prefill=False` 时如何触发 Continuous Batching？

关键是请求到达时间重叠，而不是必须开启 Chunked Prefill：

1. 先让一个请求进入 Decode；
2. 在它尚未结束时加入新请求；
3. 下一轮 Scheduler 同时选择已有 Decode 和新 Prefill；
4. Trace 中一个 engine step 出现多个请求/phase。

Chunked Prefill 决定长 Prefill 是否可以跨 iteration 切分；Continuous Batching 决定
每轮是否动态组合不同请求。

## 26. 哪些日志能证明发生了 Continuous Batching？

最直接证据是同一个 `engine_step` 中存在多个请求：

```text
engine_step=54
  ongoing   phase=DECODE  scheduled_tokens=1
  newcomer  phase=PREFILL scheduled_tokens=16
  long      phase=PREFILL scheduled_tokens=2049
```

辅助证据包括：

- 相同 step/start/end 时间；
- step request count 大于 1；
- 同一轮同时出现 Prefill 和 Decode；
- 各请求在该轮分别发生 allocate 或复用尾页；
- forward 完成后多个请求的 `num_computed_tokens` 同时推进。

## 27. 未开启 Chunked Prefill 时，长 Prompt 会影响其他请求 TTFT 吗？

会。实验中一个 2049-token Prefill 与 16-token newcomer、1-token Decode 进入同一
iteration：

```text
baseline newcomer TTFT = 24.41 ms
with long prefill       = 817.11 ms
```

Continuous Batching 让它们同轮执行，但 iteration 的完成仍是共同同步边界，所以短
请求也要等长 Prefill forward 完成。

## 28. Chunked Prefill 的核心结论是什么？

它主要改善调度公平性和尾延迟，而不是让长 Prompt 的计算消失：

- newcomer TTFT：817.11 → 79.56 ms；
- ongoing 加入轮 ITL：817.27 → 79.60 ms；
- ongoing 最大 ITL：817.27 → 270.14 ms；
- long 自身 TTFT：只改善约 7.15%。

它把一个长的不可重新调度区间拆成多个较短区间，让其他请求在 chunk 边界获得进展。

## 29. 为什么同样是 511-token chunk，后面几轮耗时更长？

Prefill Attention 的成本不仅取决于本轮 Query token 数，还取决于它们需要关注的历史
上下文长度。随着长请求已经 computed 的 prefix 增长，后续 chunk 面对更长 K/V，
所以即使 scheduled tokens 相同，墙钟时间也可能增加。

## 30. Block 分配和回收是不是主要性能瓶颈？

本实验测得 CPU 路径：

```text
allocate_slots mean = 0.0107 ms
free mean           = 0.0185 ms
```

它们很小，但不等于 GPU KV update 很小。Block 管理耗时测的是元数据和 Free Queue；
GPU 生成、layout transform、scatter、读取 KV 的时间需要单独用 CUDA/Nsight 分析。

## 31. 什么情况下会发生 Preemption？

当 Scheduler 想推进请求，但可用 KV Blocks 不足，并且无法仅通过等待/其他策略满足
本轮分配时，可能抢占已有请求，释放其 KV Blocks。

本轮基础实验容量足够，没有观测到 preemption；因此这里只是机制说明。

## 32. Preemption 后为什么需要 Recompute？

如果抢占策略直接释放请求 KV，而没有把它 swap/offload 到其他层级，恢复请求时只剩
Prompt 和已经生成的 token 序列，没有 K/V 状态。引擎必须重新 forward 这些 token
以重建 KV。

Recompute 会再次消耗 Scheduler token budget，因为 budget 衡量“本轮需要执行多少
token 的 Transformer 计算”，而不是“这些 token 历史上是否算过”。Logical token
序列不变，但重新分配后的 Physical Block IDs 可以完全不同。

## 33. 为什么不用 CPU/SSD 保存所有被抢占 KV？

可以，这属于 swap/offload，但要比较：

```text
恢复成本 = 传输 KV 的时间
重算成本 = 重新执行 Transformer 的时间
```

结果取决于模型、上下文、PCIe/RDMA/SSD 带宽、缓存命中和系统负载。不能笼统断言
swap 或 recompute 永远更好；它们是 KV 分层管理中的策略选择。
