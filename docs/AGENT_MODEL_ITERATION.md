# Structured Agent 模型迭代记录

本文合并记录 entity-v2 Typed Set Transformer agent 的重要设计演进、实验结果
和当前状态。实现过程中已经修复、且不影响后续设计判断的局部代码问题不再逐项
罗列；协议和基础网络的字段级说明见
[TYPED_SET_TRANSFORMER_AGENT.md](TYPED_SET_TRANSFORMER_AGENT.md)。

## 1. 当前状态

当前 Bridge 协议仍是 `sts2-entity-v2` / `candidate-v2`，模型输入为
`typed-set-tensor-v4`，动作语义为
`entity-actions-v2-monotonic-choice`。训练使用 MaskablePPO、无位置编码的
Typed Set Transformer 和 candidate-aware actor，不使用 LSTM/GRU，也不考虑
多人游戏。

当前推荐的本机 checkpoint 是：

```text
output/typed_set_v4_moe_monotonic_100k_20260805/
  best_model/best_model.zip
```

最新 100 个固定种子评估（seed `100109..100208`）：

| Policy | Wins | Mean floor | Median | Max | Truncated |
| --- | ---: | ---: | ---: | ---: | ---: |
| v4 75k best | 0/100 | **8.78** | 7.0 | 20 | 0 |
| Masked random | 0/100 | 3.41 | - | 10 | 0 |

它已经稳定优于随机策略，但仍然没有采样到胜局，因此还不是通关 agent。

## 2. 基础架构

### 2.1 Observation

`STS2EntityRunEnv` 将完整 run snapshot 转换为固定形状的 Dict observation：

- global categorical/numeric：阶段、角色、房间、层数、HP、金币等；
- entity set：牌、玩家、敌人、能力、遗物、药水、地图节点和事件实体；
- candidate set：157 个冻结 action slot 对应的语义候选及合法 mask。

每个实体 token 由 content ID、entity type、zone、rarity、owner、upgrade、
affliction、enchantment、status、count bucket 等 embedding 与 numeric projection
相加得到。永久牌组中的完全相同牌会聚合为一个 token，并保留精确 count 数值和
count embedding，避免克隆类效果令 sequence 长度失控。

### 2.2 Typed Set Transformer

实体顺序没有游戏语义，因此 encoder 不使用 sequence positional encoding：

```text
typed entity embeddings
    -> ISAB layers
    -> PMA memory tokens
    -> global state

candidate embeddings
    -> cross-attention to memory
    -> candidate states
```

地图当前作为带 `row/col/type/visited/reachable` 的普通 set tokens 注入，尚未
显式编码边。

### 2.3 Actor 与 critic

critic 从 permutation-invariant global state 估计 `V(s)`。当前 actor 使用
global-state-gated、4-expert candidate scorer：

```text
gate = softmax(MLP(global_state))
logit_i = sum_e gate_e * scorer_e(global_state, candidate_i)
```

专家共享 entity encoder，可以自行专门化到战斗、地图、奖励、商店或事件，
但没有按 phase 硬编码拆网。当前模型为 4,743,945 参数。

## 3. 重要历史演进

### 3.1 v2：第一个结构化神经消费者

最初版本证明了以下完整链路可行：entity-v2 JSON snapshot、padded Dict tensor、
Typed Set Transformer、自定义 SB3 policy、MaskablePPO、SubprocVecEnv、保存/加载、
评估和实机 Bridge runner。

20,480-step smoke checkpoint 在 20 个固定种子上平均 5.60 层、最高 10 层；
random 为平均 3.45、最高 7。这个结果只证明网络学到了部分有效行为，不代表
模型已经充分训练。

### 3.2 v3：无碰撞词表与 reward shaping

早期 categorical encoding 使用稳定 hash 后对固定桶数取模，不同实体可能
碰撞。v3 改为 tokenizer 式、版本化的显式词表：

- `0=PAD`、`1=UNKNOWN`；
- 已知 token 排序去重后获得唯一 ID；
- 当前 v0.110 词表总 embedding size 为 2685；
- vocabulary hash 和 tensor layout hash 随 checkpoint 保存并在加载时验证。

v3 同时将 card modifiers 拆成独立类别字段，扩展 relic/modifier numeric 状态，
并加入可选的 decomposed reward shaping：

```text
r = terminal reward
  + 0.02 * new floors
  + 0.02 * combats won
  - 0.20 * normalized HP lost
  - 0.001 per step
```

原始任务 reward 没有改变：胜利 `+1`、死亡 `-1`、中间步骤 `0`。每个 shaping
分量独立记录，便于检查 reward hacking 和做消融。

v3 100,352-step CUDA 实验达到 91.36 step/s。最终模型在 20 个种子上平均
6.85 层、最高 13 层，但不同 checkpoint 的固定种子表现没有随训练步数单调
上升。这说明 on-policy training return 不能替代独立评估，也促成了后续的
best-checkpoint 机制。

### 3.3 v4：候选语义、MoE 与能力评估

v4 保持 wire protocol 不变，主要升级训练侧：

- candidate 明确保留 content/option/zone，并和相应语义实体对齐；
- 多选动作采用单调 selected set，达到 `max_select` 后只允许 confirm；
- actor 升级为全局门控的 4-expert scorer；
- 学习率默认从 `3e-4` 线性衰减到 `3e-5`；
- 单场战斗训练保护上限收紧到 50 回合，并单独记录触发次数；
- UNKNOWN 按 global/entity/candidate 字段槽位统计；
- 支持从兼容 checkpoint 恢复训练。

周期评估使用固定种子并按下式选择 checkpoint：

```text
capability = 100 * win_rate + mean_floor - 10 * truncation_rate
```

恢复训练会在 timestep 0 先评估源模型，防止退化的微调结果仅因“在本次 run
中最好”而覆盖基线。

## 4. v4 实验结果

主实验配置为 seed 109、4 environments、CUDA、decomposed shaping、100k
requested steps；实际因 rollout 对齐完成 100,352 steps，用时 1,244.1 秒，
包含周期评估后为 80.38 step/s。

10-seed 周期评估为：

| Checkpoint | Mean floor | Median | Max | Truncated | Capability |
| --- | ---: | ---: | ---: | ---: | ---: |
| 25k | 6.4 | - | 10 | 1/10 | 5.4 |
| 50k | 7.6 | - | 16 | 0/10 | 7.6 |
| 75k | 9.2 | 9.5 | 14 | 1/10 | 8.2 |
| 100k | 7.6 | - | 14 | 0/10 | 7.6 |

用当前环境重新评估后的 20-seed 对比：

| Policy | Wins | Mean floor | Max | Truncated |
| --- | ---: | ---: | ---: | ---: |
| Random | 0/20 | 3.45 | 7 | 0 |
| v3 final 100k（历史环境） | 0/20 | 6.85 | 13 | 2 |
| v4 final 100k | 0/20 | 7.00 | 14 | 0 |
| v4 75k best | 0/20 | **8.65** | **20** | 0 |
| v4 fine-tune best | 0/20 | 8.25 | 16 | 0 |

从 75k best 以 `1e-4 -> 1e-5` 微调 51,200 steps 后，没有稳定超过源模型。
因此“训练更久”和“最终 checkpoint”都不能直接等同于能力更强。

100-seed 结果进一步确认 75k best 的平均层数为 8.78，而 random 为 3.41；
结论比原先 20-seed 更稳定，但依然是 0 胜。

## 5. 评估并行与耗时

原评估逐 episode 顺序执行。当前实现支持 `--n-envs`，使用 SubprocVecEnv 并行
模拟环境，并把 observation 组成 batch 后一次送入 policy。相同 20 个 seed 的
结果在顺序、4-env 和 8-env 下完全一致。

本机 RTX 4070 SUPER / 16 logical CPU 的实测：

| Evaluation | 20 seeds wall time | Speedup |
| --- | ---: | ---: |
| Sequential | 81.4 s | 1.00x |
| 4 envs | 63.1 s | 1.29x |
| 8 envs | 47.4 s | 1.72x |

100-seed policy + random 完整评估使用 8 envs 共 256.8 秒。环境并行主要增加
CPU 的 simulator、snapshot/tensorizer 和进程通信负载；同时 policy inference
batch 变大，使 GPU 利用率略有改善。加速不会线性增长，因为 episode 长度不同、
固定批次会等待最慢环境，Windows 进程通信也有成本。

新的默认设置是：训练中周期评估 20 seeds，最终评估 100 seeds，评估并行度
8。千级 seed 评估应改成动态任务调度，避免固定 VecEnv 的尾部等待。

## 6. 训练环境并行

训练也可以增加并行环境，并且当前瓶颈下确实有效。不过 SB3 的每次 rollout
大小是：

```text
rollout transitions = n_envs * n_steps
```

只增加 `n_envs` 会同时增大 PPO rollout buffer、降低更新频率并改变学习行为。
公平的吞吐对比应同步减小 `n_steps`。保持每次 rollout 为 2048 transitions 的
CUDA 实测为：

| n_envs × n_steps | Throughput | Rollout | PPO update |
| --- | ---: | ---: | ---: |
| 4 × 512 | 113.5 step/s | 30.0 s | 6.1 s |
| 8 × 256 | 153.0 step/s | 20.4 s | 6.4 s |
| 16 × 128 | **181.4 step/s** | 16.1 s | 6.4 s |

16 envs 比 4 envs 快约 60%。新的训练默认值改成 `16 × 256`，和旧默认
`4 × 1024` 一样保持 4096-transition rollout；显存和 RAM 仍应在长训时监控。
若机器同时运行游戏或其他重负载，可显式退回 8 envs。

## 7. 当前限制与后续方向

- 100-seed 仍为 0 胜，完整 run 胜率仍是最终指标。
- v4 长训中 UNKNOWN rate 很低但非零；下一次长训应利用按字段统计定位并扩词表。
- 地图尚无 edge-aware encoder，不能直接表达路线汇合和远期路径价值。
- reward shaping 仍是启发式，应做系数消融并考虑 potential-based shaping。
- 下一阶段优先考虑共享 encoder 上的辅助 value targets：下一战胜负、结束 HP、
  floor/act progress；其次是轻量地图 graph encoder。
- 在没有胜局样本时，scripted/heuristic policy 的 behavior cloning warm start 或
  curriculum 很可能比继续堆大 actor 更有样本效率。
- GPU update 已不是主要瓶颈；性能工程应继续优化 snapshot/tensorizer 的重复
  构造和异步 rollout。
