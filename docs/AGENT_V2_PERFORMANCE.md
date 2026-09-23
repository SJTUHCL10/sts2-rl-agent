# Entity-v2 训练性能基准

本文记录 Typed Set Transformer + MaskablePPO 的 CPU/GPU 对照方法和
2026-07-28 的本机测量结果。它用于定位训练瓶颈，不代表策略质量。

## 基准配置

- GPU：NVIDIA GeForce RTX 4070 SUPER（12 GiB）
- PyTorch：`2.12.1+cu130`
- CUDA runtime：13.0
- policy 参数量：6,279,554
- observation：`typed-set-tensor-v2`，最多 384 个 entity
- PPO：4 个 `SubprocVecEnv`，`n_steps=256`，`batch_size=256`，
  `n_epochs=4`
- 每组 PPO 对照：4,096 environment steps，seed 109

CPU 和 GPU 使用同一个 CUDA 版 PyTorch、相同模型结构和超参数。这样对照的
唯一主要变量是 SB3 的 `device`，而不是两个不同的 PyTorch build。

可复现命令：

```powershell
python scripts/benchmark_agent_v2.py `
  --device cpu `
  --environment-steps 2000 `
  --forward-iterations 30 `
  --backward-iterations 10 `
  --ppo-timesteps 4096 `
  --n-envs 4 `
  --n-steps 256 `
  --batch-size 256 `
  --n-epochs 4 `
  --output output/benchmark_agent_v2_cpu.json

python scripts/benchmark_agent_v2.py `
  --device cuda `
  --skip-environment `
  --forward-iterations 30 `
  --backward-iterations 10 `
  --ppo-timesteps 4096 `
  --n-envs 4 `
  --n-steps 256 `
  --batch-size 256 `
  --n-epochs 4 `
  --output output/benchmark_agent_v2_gpu.json
```

输出 JSON 被 `output/` 的 gitignore 规则排除；需要长期比较时，应把关键结果
抄入实验记录，而不是提交每次运行产生的原始文件。

## 测量结果

### 网络计算

| 工作负载 | CPU | GPU | GPU 相对速度 |
| --- | ---: | ---: | ---: |
| forward，batch 1 | 3.70 ms/batch | 5.15 ms/batch | 0.72x |
| forward，batch 4 | 5.78 ms/batch | 5.96 ms/batch | 0.97x |
| forward，batch 256 | 247.09 ms/batch | 24.62 ms/batch | 10.04x |
| forward + backward，batch 256 | 751.97 ms/batch | 77.15 ms/batch | 9.75x |

GPU 的 PyTorch peak allocated memory 是约 1,506 MiB。训练期间一次
`nvidia-smi` 采样为 89% utilization、3,618 MiB device memory 和 117 W；
后者包含 CUDA context、allocator reserve 等，不等同于 PyTorch 的 active
allocation。

结论很明确：4070 SUPER 对 PPO 的大 batch 参数更新约有 9.7 倍优势，但
rollout 时一次只推理 4 个 observation，kernel launch、host/device 传输等固定
开销抵消了并行收益。小 batch 推理在这组配置上没有从 GPU 获益。

### 完整 PPO

| 指标 | CPU | GPU | 变化 |
| --- | ---: | ---: | ---: |
| 总耗时 | 81.93 s | 32.16 s | 2.55x faster |
| 总吞吐 | 50.0 step/s | 127.4 step/s | 2.55x |
| rollout | 36.16 s（44.1%） | 26.21 s（81.5%） | 1.38x |
| PPO update | 45.74 s（55.8%） | 5.93 s（18.4%） | 7.72x |

因此：

- CPU 训练时，最大单项成本是 PPO 网络更新，而不是环境模拟。
- 换到 GPU 后，网络更新已不再是瓶颈；采样阶段占总时间约 82%。
- rollout 同时包含环境步进、v2 observation 构造、进程通信和 batch=4
  policy inference。结合上面的纯 forward 测量，小 batch 推理约
  6 ms/vector-step，并没有被 GPU 加速；剩余主要是环境输入流水线。

### 环境与 observation 流水线

单进程、2,000 步随机合法动作测量：

| 路径 | 吞吐 | 每步耗时 |
| --- | ---: | ---: |
| 原始 `STS2RunEnv` | 1,561 step/s | 0.64 ms |
| `STS2EntityRunEnv` | 75.8 step/s | 13.19 ms |
| v2 snapshot + tensorization 等额外工作 | — | 12.55 ms |

这里的差值包含 run snapshot 序列化、entity/candidate 构造、padding 和
tensorization，不能全部归因于 Transformer，也不能全部称为纯游戏模拟。
原始游戏规则步进本身只占 entity-v2 单进程路径约 5%；当前最值得优化的是
observation 构造，而不是卡牌/怪物规则执行。

## 当前 v5 长训实测（2026-09-23）

上文是 2026-07 的 v2 控制基准，不应用其参数量或吞吐直接推断 v5。
修复药水别名后的 v5 使用约 550 万参数、4 env × 1024 steps、CUDA、
25k 周期评估，实际 503,808 environment steps 耗时 3,394 秒，端到端
148.4 step/s。按这一**单次运行**粗略外推，1M 约 1.9 小时、5M 约
9.4 小时；评估频率、局长和环境/模型修改会改变实际耗时。

吞吐不是当前第一优先级：固定 100 局中有 20 局因模拟器未在只剩
`MINION` 副怪时结束战斗而打满 50 回合，这会污染训练回报和后续幕的
采样。该 parity 缺口现已修复，但此处的 148.4 step/s 是修复前的吞吐，
不能直接用来预测新环境中更长轨迹的速度。下一轮应在修复后的环境中
重新测量 1M/5M 的成本；不要用扩大参数量或更长训练掩盖其他环境错误。
能力诊断详见
[AGENT_MODEL_ITERATION.md](AGENT_MODEL_ITERATION.md)。

## 训练建议

正式训练应使用 `--device cuda`（训练脚本当前支持把 device 传给 SB3），并
保留较大的 update batch。当前 4-env 配置已经得到约 2.55 倍端到端加速。

下一轮性能工程的优先级：

1. profile `RunStateSerializer` 与 `EntityTensorizer`，避免每步重复创建不变的
   deck、relic、地图 ID 和 padding 数组；
2. 增加并行环境数，直到 GPU update 吞吐与 CPU rollout 吞吐取得平衡；
3. 再比较 `n_envs=8/16`、不同 `n_steps` 和 batch size，同时监控 RAM、
   GPU memory 与 episode throughput；
4. 若继续优化 rollout 推理，可评估异步采样或批量 inference；仅把 batch=4
   forward 搬到 GPU 不会带来收益。

基准脚本把环境、纯 policy 计算和完整 PPO 分开计时，后续更换网络或算法时
可以沿用同一套测量口径。
