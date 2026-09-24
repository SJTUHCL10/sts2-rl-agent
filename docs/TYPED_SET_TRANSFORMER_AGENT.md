# Typed Set Transformer Agent v2

> 本文第 1–8 节记录 v4 长训时的设计和 157-slot 基线。最近的 v5 1M
> checkpoint 在修复副怪结束规则后训练，同种子 100 局均层约 12.2、0 胜。
> 当前代码已升级到 `typed-set-tensor-v6`，修正卡牌 affliction 与附魔强度
> 观测，尚无兼容的 v6 checkpoint。旧 v5 模型只能通过
> [轨迹查看器](TRAJECTORY_VIEWER.md)的显式 legacy 投影用于诊断，不能
> 静默作为 v6 模型加载或继续训练。实验详见
> [模型迭代记录](AGENT_MODEL_ITERATION.md)。

> 轨迹审计还修复了燃烧之血重复治疗与训练进程事件未注册。v6 额外接收
> 事件 ID/选项和战后药水/遗物候选指针；历史 v5 训练数据不能与修复后的
> 环境直接比较，必须从头训练。

## v5 接口增量

| Tensor | v5 shape | 新含义 |
| --- | ---: | --- |
| `entity_categorical` | `[N, 16]` | 原 13 字段 + 最多三个当前 intent 类型 |
| `entity_numeric` | `[N, 43]` | 原 34 字段 + 战斗 index、三组 intent 伤害/次数、intent 数量和总伤害 |
| `candidate_categorical` | `[285, 7]` | 前 157 个旧槽 + 128 个复用的额外选择槽 |
| `candidate_numeric` | `[285, 15]` | 与候选槽一一对应 |
| `candidate_source_row` / `candidate_target_row` | `[285]` | 指向 entity tensor 的行号；无指向时为 `-1` |

每只怪物的 **当前** move 中所有 intent（至多三个）写入该怪物 entity，
不把未来招式表当作当前意图。Bridge 的 `enemies` 与模拟器的 `creatures`
统一进入 creature entity；`combat_index` 与打牌目标 slot 顺序对应。
手牌中的重复卡牌各保留实例 entity，source 指针区分实例；药水、奖励选项、
地图/事件选项以及水晶球格子同样可作 source，敌人目标可作 target。
永久牌组中的重复副本仍按原规则聚合。

candidate cross-attention 的 key/value 改为全部 contextual entity，而非 PMA
摘要；PMA 仍为 global state、critic 和 actor expert gate 提供全局信息。
保留前 157 个 v1 slot 的解释，额外 128 槽按当前 screen 的选项顺序复用，
可覆盖水晶球的 121 个格子、超过五个的火堆选项及更多奖励选项。该上限仍是
固定空间过渡设计，未来可替换为真正的动态候选集合。

水晶球事件初次选项后显示隐藏格子，每个可选格子都是有坐标、可点状态的
`CRYSTAL_CELL` entity。候选动作的 source 指向该格子；模型只看到当前
公开信息，不会看到未翻开格子的内容。前四个格子沿用旧事件槽，第五个起
使用扩展槽，结束时的 proceed 保持事件选项语义。

## v6 卡牌修饰观测

游戏的卡牌同时最多有一个 affliction、一个 enchantment。v5 的模拟器
`CardInstance.affliction` 已保存当前负面附着，但 snapshot 错读不存在的
`afflictions` 属性；附魔字典的数值也被 tensorizer 丢弃。v6 snapshot 将
当前 affliction 序列化为单元素映射，entity categorical 的每种修饰只用
第一个类型槽；`entity_numeric` 从 `[N, 43]` 变为 `[N, 45]`，末尾新增
affliction/enchantment 强度的 signed-log 数值。旧的第二类型槽保留为空，
避免改动其余字段偏移；layout hash 随版本改变，必须重训。

## 1. 目标与边界

本版本把 agent 从“固定向量打平后经过 MLP”升级为面向实体集合的
actor-critic，但不同时更换游戏逻辑、动作语义和 PPO 算法。设计目标是：

- 直接消费 `sts2-entity-v2` / `candidate-v2` 中的牌、角色、怪物、能力、
  遗物、药水、地图节点和当前选项；
- 不依赖实体在数组中的任意顺序；
- 同一网络覆盖战斗内和战斗外决策，同时允许以后拆成多个 encoder/head；
- 保留 157 个 v1 action slot 及其 mask，使现有模拟器和 Bridge 命令解码继续
  可用；
- 把 environment、tensorizer、encoder、policy head 和 RL algorithm 分层，
  便于替换网络或训练算法；
- 当前只以单人游戏为训练和验证目标；
- 当前状态视为 Markov state，不使用 LSTM/GRU，也不编码动作历史。

实现入口：

- `sts2_env/agent_v2/tensorizer.py`：JSON snapshot 到固定形状 tensor；
- `sts2_env/gym_env/entity_run_env.py`：结构化 Gymnasium 环境；
- `sts2_env/models/typed_set_transformer.py`：encoder、actor head、critic；
- `scripts/train_agent_v2.py`：训练/评估入口。

## 2. 为什么继续使用 SB3

当前继续使用 `sb3-contrib` 的 `MaskablePPO`，没有更换 RL 框架。

原因是本阶段真正缺失的是 observation encoder 和 candidate-aware policy，
不是 PPO rollout/GAE/minibatch 实现。MaskablePPO 已支持 `Dict` observation、
离散动作、并行环境和 invalid-action masking；SB3 也允许通过
`BaseFeaturesExtractor` 与自定义 actor-critic policy 替换网络。因此保留它
可以减少训练框架改动，并让新模型与旧 MLP smoke model 使用相同的算法基线。

需要注意：

- 多进程 `SubprocVecEnv` 下，`action_masks()` 必须由环境本身实现；不能依赖
  父进程中的 `ActionMasker`；
- 普通 `EvalCallback` 不理解 action mask，正式加入周期评估时必须使用
  mask-aware 的自定义评估 callback 或显式传 mask；
- SB3 的 rollout buffer 仍然保存 padded `Dict` observation。若以后实体上限、
  batch 或模型显著增大导致内存成为瓶颈，可以保留本模型接口并把算法后端换成
  CleanRL、RLlib 或自研 PPO，而不改模拟器状态协议。

参考：

- [Set Transformer, ICML 2019](https://proceedings.mlr.press/v97/lee19d.html)
- [sb3-contrib MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)
- [Stable-Baselines3 custom policy](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)
- [Ng et al., policy-invariant potential-based reward shaping, ICML 1999](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)

## 3. Observation tensor

`STS2EntityRunEnv` 包装现有 `STS2RunEnv`。底层 157 个动作的执行、mask 和奖励
不变，只有 observation 从 151 维 v1 向量替换为 `spaces.Dict`：

| Tensor | Shape | 含义 |
| --- | ---: | --- |
| `global_categorical` | `[4]` | phase、character、room、screen type |
| `global_numeric` | `[16]` | act/floor、HP、gold、deck/relic/potion 数量、round、终局等 |
| `entity_categorical` | `[N, 13]` | type、content ID、zone、subtype、rarity、owner role、upgrade、两个 affliction、两个 enchantment、status、count bucket |
| `entity_numeric` | `[N, 34]` | HP/block/energy/cost/damage/amount/count/map 坐标、modifier 数量及实体状态 |
| `entity_mask` | `[N]` | padding mask |
| `candidate_categorical` | `[157, 7]` | action type、slot、source/target content、phase、option ID、source zone |
| `candidate_numeric` | `[157, 15]` | enabled、price/cost、坐标、索引、selected、selection bounds、can-confirm 等 |

当前默认 `N=384`。`typed-set-tensor-v4` 不再使用哈希桶：类别值由版本化词表
显式枚举，`0=PAD`、`1=UNKNOWN`、其余 token 各有唯一 ID。词表 hash 和
tensorizer layout hash 写入模型 metadata；新增词表或改变字段布局后旧 checkpoint
会明确拒绝加载，而不会把新实体静默映射到已有实体的 embedding。

### 3.1 Entity embedding

每个实体 token 不是只有一个 entity ID embedding，而是以下各字段 embedding
之和，再加 numeric MLP：

```text
token =
    entity_type_embedding
  + content_id_embedding
  + zone_embedding
  + subtype_embedding
  + rarity_embedding
  + owner_embedding
  + upgrade_embedding
  + affliction_embedding
  + enchantment_embedding
  + count_embedding
  + numeric_projection
```

之后通过 entity-type-conditioned FiLM residual：

```text
x <- x + FFN(x) * (1 + tanh(type_scale[type])) + type_shift[type]
```

这使牌、能力、遗物、怪物等可以共享注意力空间，同时仍有不同的类型路径。以后
可以把这个模块替换为完全独立的 type experts，而不改变 observation contract。

`entity_id` 中的运行时 instance number 不进入 content embedding。候选动作会先
通过 snapshot 的实体表把 `source_id`/`target_id` 解析为稳定的 card ID、
relic ID 或 monster ID，否则同一张牌在不同局中会错误地得到不同类别。

多选界面的 `selected`、`selected_count`、`min_select`、`max_select` 和
`can_confirm` 必须属于当前 observation。缺少这些字段会让“选择”和“取消同一
项”在模型看来完全相同，确定性策略可能无限切换同一项。该问题不能用 RNN
合理解决，因为它本质上是当前状态缺失。

### 3.2 重复牌与超大牌组

战斗中的 hand/draw/discard/exhaust 仍按实例逐 token 编码，因为费用修改、
enchantment、affliction、临时变量和可打出状态可能不同。

永久牌组以及当前战斗中不可直接选择的 `draw/discard/exhaust` pile 使用语义
签名分区聚合；`zone` 本身属于签名，所以不同 pile 不会合并。签名还包含 card
ID、升级、附魔、诅咒/affliction、combat vars 及其他可见字段，但排除 instance
ID、zone index 和 playable。
同签名的 `n` 张牌变为一个 token：

```text
(card semantic token, exact numeric count=n, logarithmic count embedding)
```

因此 `克隆` 把同一张牌复制到极大数量时，sequence length 不随副本数增长。
若存在超过 `N` 种不同语义实体，tensorizer 会保留高优先级的当前屏幕实体，
并用一个 `OVERFLOW` token 记录被截断数量。这个保护是异常兜底，不应作为普通
大牌组的主要压缩方式。

### 3.3 地图

第一版没有单独的 graph neural network。每个地图节点是一个 set token，包含：

- `node_type` embedding；
- `row`、`col` 数值；
- `visited`、`reachable` 标记。

Transformer 不使用 sequence positional encoding；`row/col` 是游戏语义特征，
不是数组位置。当前版本没有显式 message passing over edges。后续可增加 Graph
Transformer/GNN map encoder，再把输出作为一种 entity memory 注入全局集合。
注意 v2 snapshot 已携带 `map_edges`，但 v5 tensorizer 只取 `map_nodes`；
这是**传输层有、模型输入层没有**的信息。候选 map node 的 source 指针和
`reachable` 只能告诉模型当前合法节点，不能恢复任意两节点之间的边或路线汇合。
不过一次 30-seed 探索性对照中，保持 final 模型负责其他阶段、仅在地图
阶段随机选合法节点，均层为 7.27；原策略为 7.47。这个小样本不排除
地图边的长期价值；该对照使用的是修复前的模拟器规则，应在新环境中重做，
再决定 graph encoder 的优先级。

## 4. Typed Set Transformer

实体编码器不加位置编码，结构如下：

```text
typed entity embeddings [B, N, d]
        |
        +--> ISAB x L
                |
                +--> PMA with K memory seeds [B, K, d]
                            |
global fields embedding ----+--> global state [B, d]
                            |
candidate embeddings [B,157,d]
        +------------------- cross attention to K memories
                            |
                     candidate states [B,157,d]
```

默认配置：

- `d_model=64`
- `num_heads=4`
- `num_inducing_points=16`
- `num_isab_layers=2`
- `num_memory_tokens=8`
- FFN expansion `2x`
- dropout `0`

ISAB（Induced Set Attention Block）把实体 self-attention 从约 `O(N²)` 降为
`O(NM)`。PMA（Pooling by Multihead Attention）把任意数量实体汇聚为少量 memory
token。候选动作只 cross-attend 到这 `K` 个 memory，而不是对 384 个实体做完整
cross-attention，因此 157 个动作的开销可控。

只要同步置换 entity tensor 和 `entity_mask`，global/candidate 输出保持不变；
回归测试对此做了数值验证。

## 5. Actor 与 critic

### 5.1 Candidate-aware actor

actor 不使用一个 `Linear(d, 157)` 固定分类头。每个 action slot 都有自己的
candidate token。v4 使用由 global state 门控的四个共享 scorer experts：

```text
gate = softmax(MLP(global_state))
logit_i = sum_e gate_e * MLP_e([global_state, candidate_state_i])
```

优点：

- 同一套参数可比较 card、potion、map、event、shop 等不同候选；
- option 的 card/relic/price/source/target 信息可以直接影响自己的 logit；
- action slot embedding 保留当前 157 动作的兼容性；
- 以后切换为真正动态长度 action space 时，scorer 本身无需改变。

MaskablePPO 在 categorical distribution 前把无效 slot 置为不可选。模型不会
学习“用极小 logit 猜测无效动作”。

### 5.2 Critic

critic 只消费 permutation-invariant global state：

```text
V(s) = MLP(global_state)
```

actor 与 critic 共享 Typed Set Transformer，减少计算量。后续若发现 value
learning 干扰 candidate representation，可通过配置拆成独立 encoder。

## 6. 战斗内外是否拆网络

第一版不拆两个完整网络，原因是 deck/relic/power 等跨阶段语义应共享，而且目前
训练数据量不足以支撑两个大型 encoder。phase embedding、entity type embedding
和 candidate action type 已允许网络学习阶段差异。

保留的实验扩展点：

1. 共享 entity encoder，使用 combat/run 两个 candidate scorer；
2. combat encoder 与 run encoder 完全分离；
3. 共享低层 embedding，不同阶段使用独立 ISAB；
4. 加入 mixture-of-experts router。

这些变化都只涉及 `sts2_env/models/`，不需要改 `STS2EntityRunEnv`。

## 7. 为什么暂时不使用时序网络

当前假设完整可见 snapshot 对单人 STS2 决策近似满足 Markov 性：

- 当前牌堆、能力、怪物 intent、地图节点和候选选项已经编码；遗物/能力
  `counters` 仍被压成总和，地图边仍未进入 tensor，这是待验证的信息损失；
- 需要记忆的“历史”原则上应成为游戏状态字段，例如遗物计数器，而不是交给
  LSTM 猜测；
- Bridge 丢字段时增加状态协议和 parity test，比隐藏状态补偿更可审计。

因此本版不使用 LSTM/GRU。若以后发现隐藏随机状态、事件历史或部分可观测信息
确实影响最优决策，再引入 recurrent policy，并把它作为独立实验。

## 8. 训练

安装：

```powershell
conda activate C:\Users\A\Codes\sts\.conda\sts
cd C:\Users\A\Codes\sts\sts2-rl-agent
python -m pip install -e ".[train,dev]"
```

以下是 v6 输入的从头训练示例；v5 1M 结果不能通过 `--resume-from`
直接续训到 v6：

```powershell
python scripts/train_agent_v2.py `
  --total-timesteps 1000000 `
  --n-envs 4 `
  --n-steps 1024 `
  --batch-size 256 `
  --device cuda `
  --checkpoint-freq 100000 `
  --eval-freq 25000 `
  --output-dir output/typed_set_v6_modifier_fixed_1m_4env
```

周期评估使用 mask-aware `CapabilityEvalCallback`，按固定种子的胜率、平均层数
和截断率选择 `best_model`；默认每 25,000 environment steps 评估一次。
`--eval-freq 0` 可关闭周期评估。checkpoint 频率按总 environment steps 指定，
脚本会根据 `n_envs` 换算 callback 的调用频率。

本阶段工程 smoke run：

```powershell
python scripts/train_agent_v2.py `
  --total-timesteps 20480 `
  --n-envs 4 `
  --n-steps 512 `
  --batch-size 256 `
  --n-epochs 4 `
  --seed 109 `
  --output-dir output/typed_set_v2_smoke_20260727
```

输出包括：

- `final_model.zip`
- `model_metadata.json`

metadata 保存网络、tensorizer、PPO 超参数、协议/vocab/layout hash、运行速度和
完整 run evaluation。模型输出仍由 `.gitignore` 排除，不进入 Git。

`sts2_env.bridge.agent_runner` 会自动识别 entity-v2 `Dict` observation
checkpoint。实机运行时，它先验证 Bridge 的 v2 schema/hash，再用同一个
tensorizer 编码 live snapshot，并复用 157-slot action mask/decoder：

```powershell
$modelPath = "output/typed_set_v2_smoke_final_20260727_v2/final_model.zip"
if (-not (Test-Path $modelPath)) {
  throw "Missing local entity-v2 checkpoint: $modelPath"
}

python -m sts2_env.bridge.agent_runner `
  --model-path $modelPath `
  --record-replay artifacts/typed_set_v2_live.json
```

这里使用的是本机现有的 20,480-step entity-v2 工程 smoke checkpoint；
`output/typed_set_v2/final_model.zip` 只有在训练时明确指定该 output directory
后才会存在。这条加载路径已经有 Python 自动化覆盖；模型质量足够之前，不应
把工程 smoke checkpoint 当作可靠的自动通关 agent。

Bridge 的单项 card-select 已能使用该路径。需要一次提交多个 index 的实机
multi-select 仍受现有 `RlCardSelector` 命令语义限制；在修改 Bridge 为增量
toggle/confirm 或增加 `choose_many` policy action 之前，不应声称 v2 模型已经
覆盖所有实机多选界面。模拟器训练侧已完整编码 toggle/confirm 状态。

20,480 steps 只用于验证 rollout、mask、反向传播、保存、加载和完整 run evaluation
闭环。稀疏终局奖励下不能据此评价网络好坏。正式实验至少需要：

- 多个 seed；
- 与 v1 MLP、random 和 heuristic baseline 使用相同 evaluation seeds；
- 充分训练步数；
- 报告 win rate、平均/最大楼层、每阶段 action entropy、invalid mask 数量；
- 对 tensorizer overflow、unknown vocabulary token 和 candidate alignment 做监控。

2026-07-27 的冻结 `typed-set-tensor-v2` smoke checkpoint 位于
`output/typed_set_v2_smoke_final_20260727_v2/`。CPU 训练 20,480 steps
耗时 486.2 秒（42.1 steps/s），policy 有 6,279,554 个参数。相同 20 个
evaluation seeds 上：

| Policy | Win | Mean floor | Max floor | Truncated |
| --- | ---: | ---: | ---: | ---: |
| Typed Set smoke | 0/20 | 5.60 | 10 | 2 |
| Masked random | 0/20 | 3.45 | 7 | 0 |

两个截断 episode 均为短训练策略在合法 multi-select 界面反复 toggle 同一项。
逐步诊断确认 `selected`、`selected_count` 和 `can_confirm` 正确交替，故不是
observation/mask 卡死；保留该结果可避免把未收敛策略误报成接口故障。

## 9. 已知限制与下一步

- 修复前 500k v5 final 的 100 局评估中有 21 局触发 50 回合战斗上限，
  其中 20 局只剩 `MINION` 副怪。规则修复后，同一模型的 100 局均层从
  7.72 升至 8.75，战斗回合强制失败从 21 降至 0；但 checkpoint 权重仍
  来自错误环境，应从头重训，不能把旧模型复评当成新训练结果。
- 当前网络已有约 550 万参数，critic explained variance 在 500k 末约
  0.73；这不能证明容量充分，但没有证据表明“加大参数量”比修环境、改训练
  信号更优先。缺少完整通关样本、战斗超时和 20-seed 评估波动更值得先处理。
- v3 训练默认启用可配置的 floor/combat-win/HP-loss/step shaping，并继续单独保留
  原始终局 `+1/-1`。各分量写进每个 episode 的 JSONL，便于做消融而不把 shaping
  写死在网络中。当前奖励并非 policy-invariant 的 potential-based 形式。
  在修正后的模拟器中，可比较默认系数、逐项移除与
  `r' = r + gamma * Phi(s') - Phi(s)`（终局 `Phi=0`）的方案，并按多个
  seed 比较真实胜率、第二幕率及战斗超时，而不仅是训练回报。
- v3 牌、能力、遗物及结构字段使用显式、版本化 vocabulary。未知值仍有专用
  `UNKNOWN` ID 以保证运行安全，但训练前应通过覆盖审计使其计数为零。
- v6 按游戏的单张卡牌上限编码一个 affliction 和一个 enchantment，含强度；
  对违反单值约束的 snapshot 直接报错，以免悄悄截断。v5 的两槽历史设计
  保留在前述表格中。
- 地图暂未编码 edges；协议中有边，tensorizer 丢弃。应在 parity 修复后用
  固定战斗策略的 map-only 消融确认实际影响，再设计 edge-aware encoder。
- v5 已将选择项的 content/option/zone 和 source/target entity 指针与候选
  对齐，但输出仍建立在 285 个固定 padded 槽上（前 157 个兼容旧槽）。
  协议已有 semantic `candidate_id`，未来可实现动态候选分布并消除固定槽。
- 不保证多人 ownership/targeting parity；多人游戏明确不在本阶段范围内。
- 当前训练机器已安装 `torch 2.12.1+cu130`，并在 RTX 4070 SUPER 上验证
  CUDA 训练。CPU/GPU 分阶段性能数据、复现命令和优化建议见
  [AGENT_V2_PERFORMANCE.md](AGENT_V2_PERFORMANCE.md)。
