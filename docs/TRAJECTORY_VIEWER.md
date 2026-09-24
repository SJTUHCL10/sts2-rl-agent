# Entity-v2 测试轨迹查看器

`scripts/trace_agent_v2.py` 对模拟器运行完整的单人测试局，逐个策略决策
记录行动前后状态，并输出可检索、按楼层展开的离线 HTML 和机器可读 JSON。
无需启动 Web 服务，也不依赖外部脚本或字体。界面可切换中文和英文；卡牌、
怪物、能力等 content ID 保留原始标识，避免使用不完整的翻译表。

运行最近的 v5 1M best checkpoint（只读 legacy v5 输入投影）：

```powershell
cd C:\Users\Administrator\codes\sts\sts2-rl-agent
C:\Users\Administrator\codes\sts\.conda\sts\python.exe scripts/trace_agent_v2.py `
  --model-path output/typed_set_v5_minion_fixed_1m_4env_20260924/best_model/best_model.zip `
  --seed 100109 `
  --episodes 1 `
  --device cuda `
  --output-dir output/trajectory_v5_diagnostics
```

同一命令也适用于当前带 `model_metadata.json` 的 v6 checkpoint，例如将
`--model-path` 改为 `output/typed_set_v6_reward_fixed_500k_4env_20260924/final_model.zip`。
脚本按 checkpoint 内记录的 layout hash 选择**严格匹配**的投影，不允许静默混用。
v5 投影仅用于重现旧策略决策，不能用于 v6 续训或实机推理。无需模型时可用
`--random` 跑诊断。`--episodes N` 从给定种子开始，每个种子生成独立的
`seed_<seed>.json` 和 `seed_<seed>.html`。直接在浏览器打开 HTML 即可查看。

每层按决策步列出阶段、房间、回合、动作类型与来源/目标。展开后可看到
行动前后 HP、格挡、能量、金币、手牌及其修饰、逐行排列的敌人 HP/意图、
卡牌/药水奖励候选、事件 ID 与选项、能力、药水、遗物、牌组和 reward 分量。
层摘要会列出金币变化与新获得的遗物。普通宝箱在打开时先获得 42–52 金币
（高进阶“贫穷”按 0.75 倍向下取整），随后提供遗物；模拟器在进入宝箱
决策前自动打开宝箱结算金币，`COLLECT` 仅领取遗物，与 Bridge 的决策时点一致。
JSON 保存相同的结构化记录，便于进一步
筛选第一幕 Boss、低血量入场或特定卡牌的行为。
如果动作槽合法却缺少对应的语义 candidate，查看器会用环境实际执行的动作
补充标签并显示警告；JSON 的 `missing_candidate_steps` 汇总此类情况。
目前药水/遗物奖励已补语义候选及物品来源指针，旧 v5 模型仍以原有空候选
投影执行；重放旧模型时另以 `legacy_model_missing_candidate_steps` 和专门
警告标示历史缺口，不把新 snapshot 的候选误称为旧模型已感知。独立运行的
训练进程现在会注册全部内置事件；过去的 v5 训练/评估
没有加载事件模块，事件层实际只有默认的 Leave，因此旧评估结果不能与修复后
的同种子轨迹直接比较。

这不是游戏内部每个微事件的战斗日志：一次 `env.step()` 内连锁发生的
伤害、抽牌与被动触发，只表现为该策略动作前后的状态变化。当前工具运行
Python 模拟器，不录制实机 Bridge 会话。产物在被 Git 忽略的 `output/`
目录，不应提交 checkpoint 或大批轨迹。
