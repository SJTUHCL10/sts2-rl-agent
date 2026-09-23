# Structured Agent 模型迭代记录

本文合并记录 entity-v2 Typed Set Transformer agent 的重要设计演进、实验结果
和当前状态。实现过程中已经修复、且不影响后续设计判断的局部代码问题不再逐项
罗列；协议和基础网络的字段级说明见
[TYPED_SET_TRANSFORMER_AGENT.md](TYPED_SET_TRANSFORMER_AGENT.md)。

## 1. 当前状态

当前 Bridge 协议仍是 `sts2-entity-v2` / `candidate-v2`。代码中的新接口为
`typed-set-tensor-v5` / `entity-actions-v3-extended-choice`。两个药水 ID
规范化别名修复后，已完成兼容当前代码的 500k 长训；旧的 v4 500k 与修复前的
v5 100k 模型因接口或词表 hash 不同，不能直接用于当前代码。
此 500k checkpoint 训练时的模拟器尚有副怪阻止战斗结束的 parity 错误；
规则现已修复，模型可加载用于复评，但尚未在修复后的环境中重训。
训练使用 MaskablePPO、无位置编码的
Typed Set Transformer 和 candidate-aware actor，不使用 LSTM/GRU，也不考虑
多人游戏。

历史 v4 能力基线 checkpoint 是：

```text
output/typed_set_v4_retrain_500k_4env_20260922/
  best_model/best_model.zip
```

这是设备迁移后在 `0.111.0` 词表上重新训练的 4-env checkpoint；完整 run
为 503,808 steps，固定种子能力评估在约 425k 选择了 best。最新 100 个固定
种子评估（seed `100109..100208`）：

| Policy | Wins | Mean floor | Median | Max | Truncated |
| --- | ---: | ---: | ---: | ---: | ---: |
| v4 425k best | 0/100 | **8.64** | 7.0 | 16 | 0 |
| v4 500k final | 0/100 | 8.51 | 7.0 | 31 | 0 |
| Masked random | 0/100 | 3.39 | 3.0 | 10 | 0 |

它已经稳定优于随机策略，但仍然没有采样到胜局，因此还不是通关 agent。
同一设备和种子下的 100k 实验中，16-env / 256-step 吞吐配置达到 158
step/s，但 100-seed mean floor 只有 6.08；4-env / 1024-step 配置为 96.4
step/s、mean floor 7.12。500k 长训继续采用 4-env / 1024-step，包含周期评估
的整体吞吐为 184.4 step/s。

### 1.1 v5 UNKNOWN 诊断试训（2026-09-23）

`output/typed_set_v5_unknown_diag_100k_4env_20260923/` 使用 4 env × 1024
step、CUDA、默认 reward shaping；请求 100k，rollout 对齐后的实际步数为
102,400。训练耗时 706 秒（约 145 step/s）。固定 20 seed 的周期均层：
0k 5.75、25k 6.80、50k 7.75、75k 8.20、100k 8.20；best checkpoint
在 75k 选出。final 在固定 100 seed 上为 0 胜、均层 8.01、中位 6.5、
最高 16；75k best 在同组 100 seed 上为 0 胜、均层 8.11、中位 7、
最高 18；同组随机策略均层 3.38。试训当时选择该 run 的
`best_model/best_model.zip`；别名修复后其 layout hash 已不兼容当前代码，
需重新训练。试训不能与 v4 500k 的模型能力直接等同。

训练 observation 的 UNKNOWN rate 为 `5.78e-6`（749 次编码），没有 entity
overflow。按 worker 记录的原始 token 日志包含 reset 等额外 observation，
因此计数与训练 callback 略有差异。已定位的漏词仅为：

- `FairyInABottle` → `FAIRY_IN_ABOTTLE`，词表中为 `FAIRY_IN_A_BOTTLE`；
- `ShipInABottle` → `SHIP_IN_ABOTTLE`，词表中为 `SHIP_IN_A_BOTTLE`。

两者发生在 `entity.content`，后者还出现在药水动作的
`candidate.source_content`。这表示字段已进入 observation，只是没有对应的
词表 ID。诊断日志在 `unknown_diagnostics/worker_*.jsonl`，可运行：

```powershell
python scripts/summarize_unknown_diagnostics.py `
  output/typed_set_v5_unknown_diag_100k_4env_20260923/unknown_diagnostics
```

每条日志按 worker、字段和规范化 token 聚合次数，并保存首次出现的 phase、
entity/candidate ID、动作槽及原始值。两个别名已在
`categorical_vocabulary.py` 规范化为词表已有 token，并将别名表纳入词表 hash；
针对性回归测试已确认不再产生 UNKNOWN。修复前的 checkpoint 会因 hash
不匹配而明确拒绝加载；修复后的长训结果见下一节。

### 1.2 v5 别名修复后 500k 长训（2026-09-23）

`output/typed_set_v5_alias_fixed_500k_4env_20260923/` 使用 4 env × 1024
step、CUDA 与默认 reward shaping。请求 500k，按 rollout 对齐实际完成
503,808 steps，训练耗时 3,394 秒（148.4 step/s）。训练 observation 的
UNKNOWN rate 与 entity overflow 均为 0。训练完成并确认产物后，已删除本次
5 个定期恢复 checkpoint；保留 `best_model/best_model.zip`、`final_model.zip`、
评估文件与元数据。

20 个固定种子的周期评估在 50k 达到最高均层 7.00，因此预设规则推荐该
best checkpoint。另以同一组 100 个种子（`100109..100208`）复评：

| Policy | Wins | Mean floor | Median | Max | Truncated |
| --- | ---: | ---: | ---: | ---: | ---: |
| v5 50k best | 0/100 | 7.32 | 6 | 16 | 0 |
| v5 500k final | 0/100 | 7.72 | 7 | 16 | 0 |

final 名义高 0.40 层，但逐种子差异较大；不能仅凭这 100 局断言其稳定优于
best。模型元数据仍遵循训练前设定的 20-seed best 选择规则，并保留 final
供进一步比较。脚本内置的另 100-seed final 评估同样为均层 7.72、0 胜；
其随机策略基线均层 3.38。相同 100-seed 集合上的历史 v4 best 为 8.64 层；
本次 v5 未提升这个能力指标，但架构和词表同时改变，不能将差距归因于别名修复。
当前仍未采样到通关局。

### 1.3 第一幕瓶颈诊断（2026-09-23）

上述 final 的 100 局（seed `100109..100208`）最高仅 16 层，21 局因单场
战斗超过训练设定的 50 回合而被模拟器强制判负；其中 **20 局**在终止时只剩
带 `MINION` 的 `EYE_WITH_TEETH` 存活，没有存活的主敌人。seed `100110`
可稳定复现：第 3 层主怪 `FOGMOG` 已死亡，副怪仍为 6 HP，战斗不结束，
一直进行到 50 回合上限。放宽到 200 回合仍不结束。这不是单纯的 agent
空过：50 回合内有 97 个非结束回合动作。

这是已定位的模拟器 parity 缺口，而非观察信息不足。修复前的
`CombatState._check_combat_end()` 以 `alive_enemies` 是否为空判胜，
其中仍包括副怪；游戏反编译源码 `CombatManager.IsCombatEnding()` 则在没有
存活**主敌人**时结束，`CreatureCmd.Kill()` 还会在最后一个主敌人死亡时
杀死剩余副怪。该行为已修正并增加 Fogmog 场景测试；不能
把旧模型在这个错误环境中的失败率解释为纯粹的网络容量或地图选路问题。

全程训练结束 4,884 局、0 胜、仅 5 局进入第二幕，503 局触发战斗回合
上限；其中最后 100k steps 共结束约 916 局，仅 1 局进入第二幕。地图边存在于
v2 snapshot 的 `map_edges` 中，却没有进入当前 tensor；这妨碍路线汇合和
远期路径价值判断。为隔离地图决策，另用 final 模型控制非地图动作，只在
`MAP_CHOICE` 阶段均匀随机选合法节点，对前 30 个固定种子
（`100109..100138`）做探索性评估：原策略均层 7.47、随机地图 7.27，
战斗回合上限分别 7/30、6/30。样本小且仍受模拟器错误影响，不足以证明
地图无用，但不支持“缺地图边是当前第一幕瓶颈”的优先判断。此前的完全随机
基线随机的是**全部动作**，不能用它推断地图单独的影响。

### 1.4 副怪战斗结束规则修复与旧模型复评（2026-09-23）

`CombatState.kill_creature()` 现在在最后一个主敌人死亡时依次处理存活
`MINION` 副怪的死亡钩子，再结束战斗；`_check_combat_end()` 仅以存活主敌人
判定能否胜利，并保留其他死亡效果对结束战斗的阻断。
`IllusionPower` 不再自行阻止战斗结束。Fogmog/Eye with Teeth、
多个主敌人及主敌人存活时 Eye 复活均有回归测试；全套为
4,742 passed、1 skipped。

修复没有改变 observation/action/vocabulary，因此旧 500k final 模型仍能
加载，但它的权重**仍由错误环境训练得到**。同一模型、同一 100 个固定种子
在修复前后的完整 run 评估：

| 环境规则 | Mean floor | Max floor | 50 回合强制失败 | Wins |
| --- | ---: | ---: | ---: | ---: |
| 修复前 | 7.72 | 16 | 21/100 | 0/100 |
| 修复后 | 8.75 | 16 | 0/100 | 0/100 |

逐种子比较有 22 局到达更高楼层、0 局下降、78 局不变。seed `100110`
不再在第 3 层 Fogmog 战斗卡死，而是继续至第 5 层。修复后的逐局结果在
`final_evaluation_post_minion_fix_100seed_100109.json`。这量化了 parity
修复的直接收益，但仍未进入第二幕/采样到胜局；不能当作重训后的能力。

## 2. 基础架构

### 2.1 Observation

`STS2EntityRunEnv` 将完整 run snapshot 转换为固定形状的 Dict observation。
本节以下的 157-slot/PMA 描述是 v4 长训时的结构；v5 的变化见下一节。

v4 observation：

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

### 2.2.1 v5 实现（已完成 500k 训练）

- 每只敌人的当前 intent 按至多三个条目编码在对应 creature entity：类型、
  伤害、次数、总预计伤害及战斗 index。Bridge 同时保留原第一条 intent 字段。
- candidate 直接 cross-attend 全体 contextual entity；PMA 仍服务 global state、
  critic 和 actor expert gate。candidate 还通过 `source_row`/`target_row` 显式
  获取对应 entity token，因此相同牌或怪物实例不会仅凭 content ID 混淆。
- 冻结的前 157 个 v1 slot 不变；entity policy 新增 128 个共享扩展选择槽，
  总共 285 个动作，容纳更多卡牌奖励、火堆选项和水晶球格子。固定槽仍是过渡
  方案，并非最终动态 action set。

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

训练脚本默认把 TensorBoard event 写到本次输出目录的 `tb_logs/`，并每 25k
timesteps 刷新 `training_progress.png`、`capability_progress.png` 和
`optimization_progress.png`。前两者展示 episode rolling curve 和固定种子
能力评估；优化图展示 learning rate、approx KL、clip fraction、entropy、
explained variance、policy/value/total loss 和累计吞吐。对应的耐久日志是
`training_curve.jsonl`、`capability_evaluations.jsonl` 和
`optimization_metrics.jsonl`。已有 JSONL 日志可以独立重绘：

```powershell
python scripts/plot_agent_v2_training.py output/<run-name> --window 100
tensorboard --logdir output/<run-name>/tb_logs
```

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

### 6.1 Tensorizer 热路径优化

2026-09-21 的本机 profile 显示，entity-v2 的主要 CPU 开销不是 snapshot 本身，
而是 tensorizer 对每个标量调用 NumPy、重复编码相同 categorical token、重复计算
schema manifest，以及每步扫描完整 padded observation 做 Gym space 校验。训练热
路径现已改为标量 Python clamp、有界 categorical ID cache、进程级 manifest
template，并将完整 space 校验改为显式 `validate=True` 的测试/诊断选项。

同一 RTX 4070 SUPER 上的结果：

| Benchmark | Before | After | Change |
| --- | ---: | ---: | ---: |
| entity-v2 environment, 5000 steps | 83.5 step/s | 276.7 step/s | 3.31x |
| entity observation overhead | 11.58 ms/step | 3.06 ms/step | -73.6% |
| 4-env CUDA PPO, 4096 steps | 142.3 step/s | 177.3 step/s | +24.6% |
| PPO rollout time | 22.81 s | 16.33 s | -28.4% |
| PPO total time | 28.78 s | 23.10 s | -19.7% |

优化没有修改 tensor shape、vocabulary 或 feature-layout hash，已有 v4 checkpoint
保持兼容。

### 6.2 Snapshot 与 action-mask 去重

2026-09-22 继续处理 wrapper/snapshot 重复工作：entity wrapper 不再构造随后丢弃
的 151-element v1 observation；`info["action_mask"]` 同时供当前 structured
observation 和下一次 policy 取 mask 使用；完整 run decision 不再先构造 combat
candidates 后立即覆盖；act map 的不可变节点拓扑和边只序列化一次，visited/
reachable 状态仍逐步刷新。直接修改底层私有模拟器状态的测试/诊断工具需要调用
`invalidate_action_mask_cache()`，正常的 Gym `reset/step` 流程无需处理缓存。

同一基准相对 6.1 优化后的版本继续改善：

| Benchmark | 6.1 后 | 6.2 后 | Change |
| --- | ---: | ---: | ---: |
| entity-v2 environment, 5000 steps | 276.7 step/s | **353.4 step/s** | +27.7% |
| entity observation overhead | 3.06 ms/step | **2.42 ms/step** | -21.0% |
| 4-env CUDA PPO, 4096 steps | 177.3 step/s | **183.6 step/s** | +3.6% |
| PPO rollout time | 16.33 s | **15.57 s** | -4.7% |
| PPO total time | 23.10 s | **22.31 s** | -3.4% |

没有缓存完整 persistent run snapshot：HP、deck card instance、relic counter 和
potion state 都可能在相邻 combat action 间变化，缺少统一 revision 时缓存整块数据
有产生陈旧观测的风险。当前只缓存生命周期和失效边界明确的 map topology。下一步
若继续优化同步 CPU 路径，应先为 RunState 增加正式 revision，再按 inventory/map/
player 分片缓存；否则应评估异步 rollout collector 的复杂度和收益。

## 7. 当前限制与后续方向

1. **已修关键 parity，下一步是干净重训。** 最后主敌人死亡时副怪仍使战斗
   无法结束的问题已有 Fogmog/Eye with Teeth 场景测试，旧模型固定种子
   均层从 7.72 升至 8.75。仍应按死亡怪物、战斗回合、卡牌行动和地图节点
   分解其余失败原因；现有 checkpoint 权重受旧环境影响。
2. **验证信息瓶颈。** 当前每只怪的当前 intent、伤害/次数、战斗 index，
   手牌实例、source/target entity 指针已进入模型，UNKNOWN/overflow 为 0；
   不能再称这些信息“缺失”。但 map edges 没进 tensor；`counters` 只保留
   数值总和，丢失各计数器名称/结构；最多两个 affliction、两个 enchantment
   的截断仍在。卡牌/遗物/能力的具体效果规则没有显式关系表示，主要靠
   content embedding 从交互中学习。应优先做观测扰动与分阶段评估，确认
   哪项确实影响决策，而不是笼统加宽网络。
3. **做对照实验。** 修复 parity 后，扩大上述固定其余策略、只随机/正常
   地图的实验样本并覆盖多个训练 seed；记录第一幕 boss/第二幕到达率、
   真实死亡与战斗回合上限。reward shaping 用同种子、多个训练 seed 做当前系数、
   去掉各分量、potential-based shaping 的消融。现有 floor/combat-win/
   HP-loss/step 奖励是启发式，不能保证不改变最优策略。当前
   `gamma * gae_lambda = 0.999 * 0.95 ≈ 0.949`，100 步外 TD 残差的直接
   GAE 权重约 0.005；4,884 局平均约 103 步，终局奖励虽通常落在同一
   1024-step rollout 内，长程信用分配依然困难。可考虑辅助战斗结果/结束
   HP 目标，或符合 `gamma*Phi(s')-Phi(s)`、终局 `Phi=0` 的 shaping。
   目前 `step_penalty=0.001` 与 `gamma=0.999` 已近似抵消单纯拖延
   `-1` 终局损失的折扣收益；不能把 Fogmog 的长战斗直接归咎于步惩罚过小。
4. **扩训顺序。** 当前约 550 万参数，修复前的 500k 仍 0 胜且权重受
   parity 错误污染；不建议直接跑 5M 或增大模型。下一轮可从头以相同
   4×1024 配置跑 1M（最好多 seed），观察第二幕率和后半程能力曲线，再决定
   是否到 5M。若仍难获得胜局，优先考虑 scripted/heuristic 示范预训练、
   分阶段 curriculum 或下一战胜负/结束 HP 等辅助目标。更大网络只有在
   排除环境与训练信号问题后、出现明确欠拟合证据时再试。
5. GPU update 仍不是主要吞吐瓶颈；后续性能工程可评估安全的 RunState
   revision 分片缓存或异步 rollout，但不能把吞吐改进当作能力改进。
