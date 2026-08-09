# Topic 2：从 Paged KV 到 Transient KV Runtime 的思考

## 1. 文档定位

本文记录从 Topic 2 主实验和两个 side chat 延伸出的研究判断。它不把设想冒充实验
结论：凡是涉及 KV Async Commit、Prefix split attention、P–D direct commit、MTP
或 SSD 分层的内容，都需要后续 prototype 和 profiling 证实。

已验证的基础事实见 [`TOPIC_2_SUMMARY.md`](TOPIC_2_SUMMARY.md)，现有机制的逐项解释
见 [`TOPIC_2_FAQ.md`](TOPIC_2_FAQ.md)。

## 2. 从 Block 管理得到的更大抽象

Topic 2 表面上研究的是固定 Block，实际揭示了三层状态：

```text
Logical sequence
  token 顺序、请求语义

Mapping state
  Logical Block → Physical Block

Physical KV data
  每一层、每个 token 的 K/V bytes
```

三层不必同时存在或同步变化：

- token 序列仍在，但 KV 可以因 preemption 被释放；
- Logical Block 不变，recompute 后 Physical Block 可以变化；
- Block 已分配，但某些 layer/token slots 的数据可能尚未写完；
- 数据已经写完，但完整 Block 可能尚未发布给 Prefix Cache；
- P 节点已经生成 KV，D 节点可能仍在接收。

因此更完整的 KV 生命周期不是简单的 allocated/free，而是：

```text
Allocated
   ↓
Transient / Produced
   ↓
Committed
   ↓
Published / Shared
   ↓
Cached / Remote / Offloaded
   ↓
Evicted or Recomputed
```

这为 KV Async Commit 提供了问题背景。

## 3. 什么是 Transient KV

Transient KV 指当前 forward 已经由 QKV projection 产生、能被当前 Attention 消费，
但尚未保证写入 persistent paged KV 的 K/V。

它不是：

- 不需要缓存的 KV；
- 历史 KV 的替代品；
- 普通异步网络传输的另一个名字。

历史 token 的 KV 仍来自 persistent paged cache。Transient 只描述本轮新增 token 在
“产生”和“持久可见”之间的窗口：

```text
History: persistent paged K/V
Current: transient contiguous K/V
Future:  current K/V commit 后成为 persistent
```

## 4. Cache-first 与 KV Async Commit

当前常见 cache-first 依赖可以抽象为：

```text
QKV
 ↓
quantize / layout transform / paged scatter
 ↓
persistent KV ready
 ↓
Attention(old + new)
```

KV Async Commit 希望改成：

```text
                         ┌─→ Attention(old persistent + new transient)
QKV → transient K/V ─────┤
                         └─→ async local/remote commit
```

它的核心不是少读一次 KV，而是改变依赖：当前 Attention 不再必须等待本轮新 KV 的
persistent materialization；Commit 只需在未来消费者读取该 KV 前完成。

可以把每个 `layer × token range` 的截止条件写成：

```text
produce(l, t) < current_attention(l, t)
commit(l, t)  < next_consumer_of_persistent_KV(l, t)
```

中间的 slack 可以覆盖 Attention、MLP、后续层、LM Head、sampling，甚至下一 token
到达该层之前的一段执行时间。

## 5. 为什么最小同步单位不应该是整个 Block

Block 是内存分配单位，但一次 forward 按 layer 产生 KV。一个 16-token Block 可能：

- 在某一层已经写完；
- 在下一层尚未产生；
- 作为 Prefix Block 尚未达到完整可发布条件。

因此至少要区分：

| 状态 | 合理粒度 |
| --- | --- |
| 内存已预留 | Physical Block |
| 当前计算可消费 | layer × token range |
| persistent data ready | layer × token range / slot range |
| Prefix Cache 可共享 | 完整、已验证的逻辑 Block |

若把“整个请求或整个 Block 全部完成”作为唯一 ready bit，会丢掉按层流水的重叠机会；
若粒度过细，则 completion bitmap、event 和 queue 开销可能抵消收益。

## 6. Prefill：最值得先验证的路径

### 6.1 为什么可能有收益

Prefill 一轮产生一段 token 的 KV，写入量大，容易摊薄 kernel/event 开销，并且有
较长计算窗口隐藏 commit：

```text
QKV(chunk)
 ├─→ current Attention
 └─→ paged scatter / quantization / remote transfer
```

潜在收益来自：

- cache-update kernel 退出当前 Attention 的强依赖链；
- layout conversion、quantization、scatter 与计算重叠；
- 专用 Prefill 节点可能不再长期保存本地 paged KV；
- P→D 场景可能避免 `P paged write → P reread → network send`。

### 6.2 Chunked Prefill 的关系

Chunked Prefill 解决调度粒度，Async Commit 解决单个 forward 内的数据依赖，两者
正交但会相互影响：

```text
Chunked Prefill:
  长 Prompt 切成多个可重新调度的 iteration

Async Commit:
  每个 iteration 内，让新 KV persistence 与计算重叠
```

Chunk 太小会增加 commit 次数和同步开销；Chunk 太大又会恶化 TTFT/ITL。未来调度
可能同时考虑 token budget 和 commit queue/backpressure，而不是只看 scheduled
tokens。

## 7. Decode：单 token 收益为什么更难

普通 Decode 每个请求每轮通常只新增 1 个 KV token。每层写入很小：

```text
new KV bytes 小
kernel/event/queue 固定开销相对大
下一 token 很快就要读取该 KV
可隐藏窗口短
```

因此直接为每个 layer、每个 token 启动独立 Async Commit，可能比同步写更慢。更可行
的方向包括：

- persistent GPU work queue，避免频繁 launch；
- 多层 grouped commit；
- cache update 与其他 epilogue 融合，但保留 write-behind 语义；
- 只对高并发 batch 使用异步路径；
- 根据 HBM 带宽和 deadline 动态退回 cache-first。

### 7.1 MTP/Spec Decode 是否更有空间

MTP 一轮产生多个候选 token，确实扩大了写入粒度和可摊薄空间。但它增加
provisional 状态：候选 token 可能被拒绝。

理想状态机：

```text
Produced candidates
      ↓
Provisional KV
      ↓ verify
Accepted prefix ─→ Committed
Rejected suffix ─→ Discard / reclaim
```

创新机会不仅是把多 token KV 批量写入，还包括只提交 accepted prefix，减少无效
HBM 写入和 P→D 网络流量。难点是 rollback、compact/scatter、Block 边界和 accepted
length 的同步。

## 8. Prefix Cache：收益与复杂度同时增加

Prefix 命中后的 Prefill 不是“全部 KV 都是新的”，而是：

```text
read old persistent prefix KV
+
generate new suffix KV
+
append/publish new suffix KV
```

Async Commit 下，Attention 需要同时消费：

```text
old prefix: persistent paged KV
new suffix: transient contiguous KV
```

### 8.1 不能简单把两次 Attention 输出相加

正确结果是：

```text
softmax(Q [K_prefix, K_suffix]^T) [V_prefix, V_suffix]
```

不能分别计算两个 softmax 后直接相加，因为归一化分母不同。实现需要：

- 单 kernel 同时支持 paged old KV 与 contiguous new KV；或
- 两个分支分别计算 online-softmax/LSE 中间状态，再精确 merge。

若两段结果为 `(m1, l1, o1)` 和 `(m2, l2, o2)`：

```text
m = max(m1, m2)
l = exp(m1-m) * l1 + exp(m2-m) * l2
o = exp(m1-m) * o1 + exp(m2-m) * o2
result = o / l
```

### 8.2 Prefix 越长，不代表 Async Commit 收益越大

如果 20K prefix 已命中、只新增 500-token suffix，真正需要 commit 的只有 500 个
token。Prefix 越长、命中越高，新写入量反而可能越小；与此同时 split-source
Attention 和发布一致性更复杂。

所以 Prefix 的论文价值可能更高，但纯性能收益不是单调的。应分别测：

- Prefix length；
- new suffix length；
- cache-update hidden ratio；
- split Attention 额外开销；
- 完整 Block publish 延迟。

## 9. P–D 分离：最强的系统使用场景

### 9.1 D 为什么需要完整 Prompt KV

D 生成第一个 output token 时，Query 必须关注整个 Prompt，因此逻辑上必须能够访问
每一层、全部有效 Prompt token 的 KV。物理上可以：

- 全部进入 D GPU；
- 先进入 D CPU DRAM，再逐层异步加载；
- Prefix 已在 D，只传缺失 suffix；
- 位于共享 KV Store，由 D 按需读取。

handoff 后：

```text
D: 持有/可访问 Prompt KV，并继续追加 Decode KV
P: 正常 Decode 不再依赖；ACK 后可释放或保留缓存副本
```

完整 handoff 降低的是 D 在 Decode 期间对 P 的持续依赖；handoff 本身仍依赖 P 计算、
网络传输和 ready/ACK 协议。

### 9.2 Direct Remote Commit

最激进路径是让 D 提前分配 Physical Blocks，把 destination descriptors 发给 P：

```text
D allocate blocks
      ↓
P produce transient K/V
      ├─→ P current Attention
      └─→ transform/RDMA scatter → D blocks
                                      ↓
                               layer/chunk ready
                                      ↓
                                  D Decode
```

相对传统路径，它可能跳过：

```text
P local paged write
→ connector reread P paged blocks
→ transfer
```

P 可成为纯计算节点，D 成为该请求 KV 的长期 owner。

### 9.3 新增复杂度

- D 必须提前分配并传递可靠的目标地址/Block Table；
- transient source 在 DMA 完成前不能覆盖；
- 需要 layer/chunk ready bitmap、memory fence 和 deadline；
- 取消请求时要协调 P、网络和 D 回收；
- 传输失败时若 P 没有完整 persistent 副本，重试和 recompute 更复杂；
- commit queue 拥堵会把平均隐藏的成本转化为 p99 TPOT stall；
- Prefix Cache 需要区分 data-ready、committed 和 published。

## 10. 与 Mooncake/Kimi 公开架构的边界

[Mooncake](https://arxiv.org/abs/2407.00079) 已公开：P/D 分离、Chunked Pipeline
Parallelism、按层 Prefix KV load、新 KV 的异步 store/stream、RDMA 传输、D 侧异步
load 以及 cache-aware scheduling。因此以下表述不能作为新的核心贡献：

> “Prefill 生成 KV 后异步传给 D，并与后续计算重叠。”

这已经是 Mooncake 的主要设计之一。

相对 Mooncake 仍可能有价值的差异是：

| 维度 | Mooncake 公开方案 | Async Commit 潜在新增 |
| --- | --- | --- |
| 异步边界 | 外部 store/transfer | 从 QKV 产生后分叉 |
| 当前 Attention | 依赖已物化 KV 的常见路径 | 直接消费 transient new KV |
| P local paged KV | 通常仍是传输来源或中间状态 | 可延迟或跳过 |
| P→D | 传输 persistent KV | transient 直接 commit D pages |
| Ownership | 调度/缓存中心化管理 | D-owned allocation + commit protocol |

当前仓库的 MooncakeConnector 也体现了这个边界：
`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`
中的 `save_kv_layer()` 是空实现，Connector 注册已存在的 GPU paged KV buffers，并
按请求/Block 生成传输 descriptors。也就是说，当前开源路径并未展示 transient
K/V 直接写入 D pages。

基于 Mooncake，最稳妥的创新表述应是：

> 解耦当前 Attention 对新增 KV 的消费与本地/远端 KV persistence，使专用 P 节点
> 可以避免同步本地 paged materialization，并通过 destination-owned protocol
> 直接提交到 D。

这只说明相对 Mooncake 有差异；宣称全局学术 novelty 前仍需系统检查 DistServe、
Splitwise、NIXL/LMCache、HMA-Serve、FlowKV、FlashAttention append-attention 和
TensorRT-LLM generation attention 等 prior art。

## 11. 与 Tutti/SSD KV Cache 思路的关系

[Tutti](https://arxiv.org/abs/2605.03375) 研究的是 SSD-backed KV Cache，不等同于
当前层的 Async Commit，但提供了三个重要启发：

1. 数据路径直接并不代表控制路径已经优化。GDS 可让 SSD→GPU 数据绕过 CPU，CPU
   逐次提交碎片 I/O 仍可能成为瓶颈。
2. 读取历史 KV 位于当前请求关键路径，而写回新 KV 更多影响未来复用，二者应有不同
   优先级；读优先、写延迟更符合 serving 目标。
3. 只有让存储调度理解 Transformer 的 layer execution 和 slack，才能真正隐藏 I/O。

这与 Async Commit 的共同原则是：

```text
不要只优化搬运带宽；还要重新设计谁提交、何时可见、何时必须完成。
```

若把 commit target 扩展为统一层次：

```text
L0 transient buffer
L1 local HBM paged KV
L2 D-node HBM / CPU DRAM
L3 remote KV Store
L4 SSD
```

那么问题会从单个 kernel 优化扩展为 compute–memory co-scheduling。不过目标越统一，
一致性、容错和 backpressure 也越复杂，不能在第一个 prototype 中一次解决。

## 12. Agentic Workload 为什么重要

Coding Agent 运行很久，不代表 KV 会永久留在 GPU：会话可能因为请求结束、资源压力、
路由迁移或缓存淘汰而释放 KV。再次执行时：

- 若 Prefix KV 仍在 HBM/DRAM/SSD/远端 Store，可恢复或复用；
- 若只保留文本，必须对历史 Prompt 重新 Prefill；
- 历史越长，重建的 Prefill tokens 越大。

Async Commit 不能消除“没有缓存时的重算”，但可以改善新 KV 从计算到各层级持久化的
路径。对 Agent 更完整的问题是：

```text
哪些 KV 立即 commit？
哪些只保持 transient？
哪些发布为共享 Prefix？
哪些下沉到 DRAM/SSD？
哪些直接淘汰并接受未来 recompute？
```

因此 Agent 场景真正连接的是 Async Commit、Prefix Cache、offload 和 session
lifecycle，而不是单独一个 cache-update kernel。

## 13. 风险与可能证伪方向

### 13.1 性能前提可能不成立

- cache update 占 Prefill critical path 的比例可能很小；
- Attention 已接近 HBM 带宽上限，异步 scatter 只会争抢带宽；
- 现有 fused append-attention 已经覆盖大部分收益；
- transient buffer 延长生命周期后占用额外 HBM；
- Decode 的 event/queue 开销大于被隐藏写入时间。

### 13.2 正确性风险

- Attention 通过 persistent Block Table 读到仍在写的 slot；
- 当前 step 使用 BF16 transient、未来使用 FP8 persistent，数值语义不一致；
- Spec Decode rejected suffix 被错误发布；
- P 释放 source 早于 RDMA completion；
- D 收到部分 layer 后错误地把整个请求标记 ready。

### 13.3 系统风险

- commit queue 堵塞导致下一 token missed deadline；
- fallback/recompute 使 P/D ownership 模糊；
- Prefix publication、共享 ref count 和远端副本一致性难以组合；
- 多 backend、sliding window、multimodal mask、量化 KV 增加接口矩阵。

## 14. 建议的验证路线

### 阶段 0：先测瓶颈

用 Nsight Systems/CUDA events 拆分每层：

```text
QKV projection
cache update / quantize / scatter
Attention
MLP
connector read/transform/transfer
```

先回答 cache update 是否真的在关键路径、占多少、是否与 Attention 争用相同资源。

### 阶段 1：纯 Prefill 最小原型

限制条件：无 Prefix、BF16、固定 shape、无 P/D。

对比：

```text
Baseline: QKV → paged scatter → paged Attention
Prototype: QKV → transient Attention
                   ∥ async paged scatter
```

验证输出一致性、hidden ratio、HBM 流量、峰值显存和吞吐。

### 阶段 2：Prefix hit + suffix

实现 persistent prefix + transient suffix 的精确 Attention merge，测试不同 Prefix/
Suffix 比例。只有这个阶段才能判断 side chat 中“Prefix 更像论文方向”的判断是否成立。

### 阶段 3：P–D direct commit

让 D 预分配 blocks，比较：

1. P paged staging 后发送；
2. P contiguous staging 后发送；
3. transient 直接 scatter/commit 到 D。

同时测 handoff latency、P HBM 占用、NIC 利用率、D ready stall、失败恢复和 p99。

### 阶段 4：Decode 与 MTP

比较 token-by-token、grouped commit、persistent work queue；MTP 额外测 accepted ratio、
无效写入和无效网络流量。

## 15. 必须记录的指标

| 类别 | 指标 |
| --- | --- |
| Correctness | logits/output 一致性、FP8 误差、spec acceptance |
| Critical path | cache-update 时间、Attention 时间、可重叠窗口 |
| Commit | hidden ratio、queue depth、deadline miss、wait time |
| Memory | transient buffer、P/D paged KV、峰值 HBM |
| Network | bytes、descriptors、RDMA overlap、重传 |
| Serving | TTFT、TPOT、ITL p50/p95/p99、吞吐 |
| Lifecycle | allocated/produced/committed/published timestamps |

只报告平均 throughput 不足以证明价值，因为 Async Commit 最危险的失败模式是平均
写入被隐藏、但高负载 queue 堵塞导致 p99 Decode stall。

## 16. 当前最可信的研究判断

1. 固定 Block 和 Block Table 已解决“放在哪里”，但没有完整表达“数据是否 ready”。
2. 单纯把 KV 写 kernel 放到另一条 stream，创新性和收益都有限。
3. Prefill，特别是专用 P 节点，是最可能先获得收益的场景。
4. 普通单 token Decode 的收益较弱；MTP 提供更大粒度，但需要 provisional commit。
5. Prefix 场景的系统价值高，但新 KV 写量可能更小，必须实测而不能凭直觉判断。
6. Mooncake 已覆盖 layer-wise 异步传输；剩余差异应聚焦 transient consumption、
   P local materialization bypass 和 D-owned direct commit。
7. Tutti 说明控制路径、执行感知调度和读写优先级，与带宽本身同样重要。
8. 最终有价值的抽象可能不是一个 kernel，而是带 deadline、visibility、ownership、
   rollback 和多 commit target 的 Transient/Persistent KV Runtime。

一句话概括：

> Topic 2 从“token 如何映射到物理页”出发，进一步暴露了一个尚未被 Block Table
> 表达的问题：新 KV 何时从当前计算可见，转变为未来消费者、远端节点和 Prefix
> Cache 都可安全使用的持久状态。
