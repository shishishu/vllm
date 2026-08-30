# Topic 3：Automatic Prefix Caching FAQ

实验配置、数据与关系图见 [`TOPIC_3_SUMMARY.md`](TOPIC_3_SUMMARY.md)。

## 1. 两个请求在什么条件下可以复用 KV？

后一个请求必须在模型计算条件一致的前提下，从 token 0 开始拥有相同的完整 token
Blocks。实际 hash 除 token IDs 和前一 Block hash 外，还会纳入会影响 KV 正确性的
额外输入，例如 LoRA、multimodal 等相关 key。

文本看起来相同不够；Tokenizer 产生的 token IDs 和相关计算上下文必须匹配。采样
温度、`max_tokens` 等生成参数不改变已有 Prompt KV，本身不阻止 Prefix 命中。

## 2. Prefix Cache 的匹配粒度是 token 还是 Block？

Prompt 内容最终由 token IDs 定义，但缓存查找、命中计数和 Physical KV 复用以完整
Block 为粒度。本实验 `block_size=16`，所以命中长度为 16 的倍数。

若前 2108 tokens 相同，最后 4 tokens 不同，只能命中 2096；同一尾部 Block 中未改
的 12 tokens 也要和修改的 4 tokens 一起重算。

## 3. 为什么只能复用最长连续前缀？

因为 token 位置的 KV 依赖它之前的全部上下文。vLLM 使用链式 Block hash：后一个
Block hash 包含前一个 Block hash。Block 0 变化会使所有后续 hash 都变化，查找只能
返回从 Block 0 开始连续匹配的最长前缀。

Prompt 中间或末尾恰好出现相同 token 片段，不能脱离它前面的历史上下文单独复用。

## 4. 修改开头和修改结尾，命中有什么不同？

本实验 2112-token Prompt：

| 修改 | 最长公共前缀 | 实际命中 |
| --- | ---: | ---: |
| 第一个 token | 0 | 0 |
| 64-token 后缀 | 2048 | 2048 |
| 最后 4 tokens | 2108 | 2096 |

改开头破坏整条 hash 链；改结尾只影响包含修改位置的 Block 及其后续 Block。

## 5. 完全相同的 2112-token Prompt，为什么只命中 2096？

vLLM 为当前请求保留至少一个 token 的实际计算，最大命中不能覆盖整个 Prompt。再按
16-token Block 向下对齐：

```text
floor((2112 - 1) / 16) × 16 = 2096
```

所以 Probe 仍调度 16-token Prefill，并新分配一个尾部 Block。

## 6. 请求结束后，为什么 KV 还能被下一个请求命中？

结束时释放的是请求对 Block 的活动引用，不一定立即抹除数据：

```text
ref_cnt: 1 → 0
cached:  false → true（已提交的完整 Block）
```

`ref_cnt=0` 表示没有活动请求持有、允许淘汰；`cached=true` 表示 hash 映射和 KV 内容
仍保留。下一个请求命中时，同一个 Physical Block 的 `ref_cnt` 再从 0 增至 1。

## 7. Prefix Cache 什么时候真正被释放或覆盖？

主要有三种情况：

1. 新请求需要空间时，缓存策略从 `ref_cnt=0` 的可淘汰 Blocks 中选择页，移除旧 hash
   映射并覆盖内容；
2. 显式调用 `reset_prefix_cache`；
3. Engine 退出，整个 KV Block Pool 销毁。

不是必须等显存完全耗尽后才存在“可释放”资格；Block 在 `ref_cnt=0` 后已可被复用，
但只要没有分配压力，保留数据能提高未来命中率。`ref_cnt>0` 的页不能被驱逐。

## 8. APC 主要影响 TTFT、TPOT，还是吞吐？

首先影响 TTFT，因为命中的 Prefix 不再执行 Prefill。本实验 A/D 的 TTFT 下降约 96%，
B 下降约 91%。

TPOT 基本不变：每个新 token 的 Decode 仍要读取当前完整上下文并做一次 forward。
吞吐在共享 Prefix workload 下也会提高，因为 GPU 少做大量 Prefill；并发 8 时，本实验
完全相同 Prompt 的 requests/s 提高 2.73 倍，共享 1024-token Prefix 提高 2.35 倍。

固定 2112-token Prompt 的补充曲线中，命中从 0 增至 2096 后，TTFT 中位数从
777.49 ms 降至 40.00 ms，降幅 94.86%。但命中长度与 TTFT 不是简单线性关系，因为
剩余 query 除 QKV/MLP 外，还要对缓存 Prefix 执行 causal Attention。

## 9. Prefix Cache 开启后，每个请求是否复制一份命中 KV？

不会。多个请求的 Block Table 可以指向相同 Physical Blocks，并通过 `ref_cnt` 保护
共享页。本实验 8 个并发相同 Probe 共享 67 个命中 Blocks；同一 Block 在各请求 lookup
前的引用计数依次为 0、1、2、3、4、5、6、7。

只有各请求不共享的后缀、尾部和后续 Decode KV 才需要分别分配 Physical Blocks。
`ref_cnt>1` 能证明活动持有区间重叠，但要证明请求处于同一个 GPU batch，还要检查它们
是否有相同 `engine_step`。

## 10. Prefix Cache 与单请求 Decode KV Cache 是一回事吗？

底层都使用同一个 KV Block Pool，但语义不同：

- 单请求 Decode KV：活动请求必须持有的历史状态，不能关闭；
- Prefix Cache：已完成完整 Blocks 的跨请求 hash 索引和保留策略，可开关、可淘汰。

因此 `enable_prefix_caching=False` 只关闭跨请求复用，不会让 Decode 丢失历史 KV。

## 11. 为什么单请求实验保持并发为 1？

为了只改变 Prompt 内容和 APC 开关，让 hit、Prefill 和 TTFT 能直接归因于 Prefix
复用。并发会同时引入 batch 竞争、排队、共享引用和显存压力。

完成基础验证后，本 Topic 再单独扩展并发 1/4/8，并使用能在 APC-off 下全部驻留的
1088-token Prompt，避免把容量拆批误认为 Prefix Cache 性能差异。

## 12. 日志怎样证明“缓存存在但请求已释放”？

需要把三个事件连起来，而不是只看一个字段：

1. Seed 的 `FINISH_AFTER_RELEASE`：完整 Blocks 为 `ref_cnt=0, cached=true`；
2. Probe 的 `PREFIX_LOOKUP`：相同 Physical Block IDs 被命中，touch 前仍为
   `ref_cnt=0`；
3. Probe 的首个 `SCHEDULE_STEP`：这些 IDs 被列为 `reused_block_ids`，当前
   `ref_cnt=1`。

这条链同时证明了“解除请求引用”“缓存未立即删除”和“下一个请求复用同一物理页”。

## 13. 关闭 APC 后为什么仍能看到 KV Block 分配？

因为模型仍需要当前请求自己的 KV 历史完成 Prefill 和连续 Decode。关闭组中
`prefix_hit_tokens=0`，Probe 的 2112-token Prompt 全部重算；但请求执行期间仍会按需
分配和持有普通 KV Blocks，请求结束后再释放。

## 14. 这些延迟能直接代表生产性能吗？

不能。实验使用 eager 模式、RTX 2080，并开启逐调度 step 的详细 JSONL Trace；绝对
延迟包含日志和实验控制开销。结论适合验证机制及同机 APC-on/off 相对差异。生产评估
应关闭详细 Tracer，并使用目标硬件、真实 Prompt 分布和并发模型重新 benchmark。

## 15. 多并发复用是并行还是串行？

两者都有，发生在不同层次：

- Scheduler 按请求串行执行 Prefix lookup、Block touch 和 `ref_cnt` 更新；
- 多个请求的 Block Table 可以同时指向相同 Physical Blocks；
- 同一 `engine_step` 选中的请求进入一个 batched GPU forward。

本实验并发 8 的真实 Trace 中，前 7 个 Probe 在 step 2282 做 Prefill；第 8 个在
step 2283 加入，当时前 7 个做 Decode，形成“7 Decode + 1 Prefill”的 mixed batch。
此时共享 Block `P544` 的 `ref_cnt=8`。

## 16. `ref_cnt` 怎样反映并发共享过程？

`P544` 从 Seed 结束后的 `ref_cnt=0, cached=true` 开始。8 个 Probe 依次 lookup/touch：

```text
lookup before touch: 0,1,2,3,4,5,6,7
touch 后活动引用:   1,2,3,4,5,6,7,8
请求依次结束后:     7,6,5,4,3,2,1,0
cached:             始终 true
```

因此 `ref_cnt` 证明同一个 Physical Block 被多少活动 Block Table 持有，以及何时可以
驱逐。它不记录 GPU 算术过程；判断同批执行还需联合 Physical Block ID、`engine_step`、
phase 和 scheduled tokens。

## 17. Batch size=8 时，是否只计算一次 Attention？

不是。8 个请求共享同一份 Prefix K/V 物理存储，但各自仍有 Query，并分别计算：

```text
scores_i  = Q_i × K_context_iᵀ
weights_i = softmax(scores_i + causal_mask)
output_i  = weights_i × V_context_i
```

GPU 用 batched/ragged kernel 并行处理这些逻辑 Attention，不会计算一次 Attention
output 后复制给 8 个请求。请求生成分叉后，Query 和私有历史不同，更不可能共享结果。

## 18. 复用多数 KV 后，剩余 tokens 如何计算 KV？

设命中 `P` tokens、剩余 `M` tokens。对每一层，只为 `M` 个 suffix tokens 计算新的
Q/K/V，把新 K/V 写入该请求的私有 Blocks；Prefix Blocks 保持只读共享。

第 `j` 个 suffix query 的上下文为：

```text
全部 Prefix K/V + suffix[0:j+1] 的 K/V
```

所有 `M` 个 Q/K/V projection 可批量计算，Attention 也可并行处理多个 query rows，
用 causal mask 保证 suffix token 不能看到未来位置。

## 19. 命中 Prefix 后，还需要计算“完整 Attention”吗？

需要完整历史语义，但不需要重新计算全部 query rows。完整矩阵可拆为：

```text
                         Keys
                   Prefix P    Suffix M
Queries  Prefix P      A_pp       0
         Suffix M      A_sp       A_ss
```

APC 跳过已完成的 `A_pp`，只计算 `A_sp + A_ss`。对每个 suffix query，Prefix 和可见
suffix scores 必须参加同一个 Softmax，不能分别归一化后直接相加。

计算量为：

```text
M × P + M × (M + 1) / 2
```

例如 `P=2048, M=64`，每个 head 计算 133152 个 query-key pairs，而完整 2112-token
Prefill 为 2231328 个，剩余约 5.97%。

## 20. Attention 时是否要搬运或复制 reused KV？

不会把 Prefix KV 持久化复制成每请求一份，也不会为了 Attention 先拼成连续 K/V。
Kernel 根据 Block Table 直接访问共享 Prefix Blocks 和私有 suffix Blocks。

但 GPU 计算仍需正常读取数据：

```text
HBM → L2 → shared memory/registers → 计算单元
```

所以“无物理复制”不等于“无数据读取”。多个请求访问相同地址可能利用 L2，但不能仅凭
APC 假设 Batch size=8 时 Prefix KV 只从 HBM 读取一次；这需要 GPU profiling 验证。

## 21. 同一 batch 中只有部分请求命中，会怎样？

每个请求仍只计算自己的 miss 部分。例如 scheduled tokens 为 `16/64/2112` 的三个
请求可以组成 ragged mixed batch；前两个不会因第三个 miss 而重算 Prefix。

但一次 GPU forward 是共同的同步边界，长 miss 会增加整个 step 的工作量，可能拖高
短 hit 请求的 TTFT。整体吞吐通常仍优于全部 miss，具体收益取决于总新 query tokens、
上下文长度、Attention 工作量、命中率和 GPU batch 效率。

当前并发实验中同一 Suite 的 Probe 命中长度一致，没有直接测量 mixed hit/miss 同批；
因此这部分是基于调度与计算路径的机制结论，仍应增加专门 workload 做定量验证。

## 22. 为什么 Prefix hit—TTFT 曲线不是严格线性？

命中增加会同时减少两类工作：

1. 剩余 tokens 的 QKV/MLP 等 token-wise 计算；
2. 剩余 query 对 Prefix 和 suffix 的 causal Attention pairs。

仅用 `prefill_tokens` 线性拟合，本实验 `R²=0.9648`；加入 causal attention pairs 后
`R²=0.9960`。另外，GPU kernel shape、固定开销、详细 Trace 和系统抖动会造成局部
非单调；18 个点中只有 0→128 命中出现一次中位数上升，不能据此否定总体趋势。
