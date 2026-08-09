# O3 候选实现计划

日期：2026-08-09
上游架构：[`V2_I1_STOCHASTIC_PERMANENCE_PLAN.md`](V2_I1_STOCHASTIC_PERMANENCE_PLAN.md) §5（权威，本文件不重述）
预注册门：[预注册 C 节](V2_L0_PERMANENCE_DIAGNOSIS_AND_PREREGISTRATION_DRAFT.md#c-永久性预注册2026-08-08-重写版)
待办编号：**O3**——[待办清单](../review/OPEN_ITEMS.md)中唯一剩余的技术阻断项

本文件只回答三个问题：**已有什么、缺什么、按什么顺序补**。

---

## 一、一个决定成本的结构事实

**候选是注入式的，不是被锁定入口 import 的。**

```python
run_candidate_lifecycle(factory, *, train_sensor_streams,
                        evaluation_sensor_streams, run_episode)
```

`CandidateFactory` 是 `runtime_checkable` Protocol，只要求两个方法：

| 方法 | 签名（`validate_candidate_factory` 逐参数名强校验） |
| --- | --- |
| `fit_kernel` | `(train_sensor_streams)` → `Mapping` |
| `new_episode` | `(frozen_kernel, model_seed)` → 候选实例 |

**后果**：候选模块不出现在 Phase-0/Phase-R 的 22 模块 import 闭包里，因此

- 不触发协议 V3；
- 不需要重跑 3 小时 41 分的 phase0；
- `test_source_lock_covers_the_whole_permanence_import_closure` 保持绿。

这不是绕开锁，而是锁的范围本来就正确：候选决定的是**未来的确认证据**，不是当前的开发产物。
**但候选自己冻结时需要它自己的锁**，正如 O19 守卫那条注记所说。

> **反向约束**：如果实现过程中发现必须修改 22 个锁定模块中的任何一个，
> 成本立刻变成协议 V3 + 两份产物重跑。**遇到这种情况先停下来报告，不要顺手改。**

其他硬约束：

- `fit_kernel` **只收 sensor stream**，环境 seed 只作外部键，永不进入 `new_episode`；
- frozen kernel 在评估期间被深冻结并逐次重编码比对，任何回调改动即抛异常；
- 每个 episode 必须是**全新实例**（`new_episode` 返回同一对象即拒）；
- `model_seed` 固定 `17011`（`MODEL_SEED`），多初始化需另立协议。

---

## 二、已有 vs 缺失

### 已有且 Phase-R 已验证

| 组件 | 位置 | 状态 |
| --- | --- | --- |
| 稀疏运动学后验 + 定容 packed 剪枝 | `PackedKinematicFilter` | Phase-R V6 通过；TV ≤ 6.3e-8，剪枝质量 0 |
| 不确定 static 下的转移核（§5.2 三项分解） | `autonomous_successors` / `_probabilistic_bounce_distribution_validated` | 穷举 38 400 例对齐真值核，L1 = 0；**实测最大后继数 4**（界 12） |
| 定容多假设池 | `PackedPosteriorPool` | 5×11×96 槽位，原子替换，资源门全过 |
| 存在性 / no-detection 更新（§5.3） | `bayesian_no_detection_update` | **算术已实现且有测试，但零调用方**——本轮已在 docstring 标注为预留 |
| 关联假设 bank、birth/death、self posterior | `cal/model/entity_graph.py`、`integrated_agent.py` | 历史 I1 实现，作为 `old_i1` 非劣性对照在用 |

**结论：§5 里最难的两块（有界后验传播、不确定拓扑下的精确核）已经存在并被独立核验过。**

### 缺失

| # | 组件 | 依据 | 备注 |
| --- | --- | --- | --- |
| **M1** | `SensorOnlyFrontEnd`：从 sensor history 维护逐格 Bernoulli `m_t(c)=P(static_c)` | §5.2 | **一切下游的前提**，也是"候选不得读真值 static"的执行点。界外格 `m=1` |
| **M2** | 存在性通道接线：把 `bayesian_no_detection_update` 接进逐步传播 | §5.3 | 算术已在，缺的是生命周期与 branch weight 记账 |
| **M3** | branch-local self identity `π_h`（含显式 null 类）与至多一个 self 的联合边缘化 | §5.4 | 历史 I1 有 self posterior，但**没有 branch-local 版本**；不得把跨假设聚合值灌回各 branch |
| **M4** | occupancy 合成 `P_occ = 1-(1-m_t)(1-P_dyn)` | §5.5 | M1–M3 就位后是简单组合 |
| **M5** | `CandidateFactory` 实现：`fit_kernel` 从 train stream 估核、`new_episode` 建实例 | §7 | 核参数只能来自 train stream |

---

## 三、实施顺序

每一步都要求：**可独立测试、不改锁定模块、留下可证伪的断言**。

| 增量 | 内容 | 完成判据 |
| --- | --- | --- |
| **I1** | M1 `SensorOnlyFrontEnd` | 从 sensed patch 序列重建 `m_t`；对已观察为自由的格 `m→0`、未观察格保持先验、界外恒 1；**与真值 static 的一致性只作诊断断言，绝不作为输入** |
| **I2** | M2 存在性通道 | 单实体、无关联歧义下，逐步 `e_t` 与 `branch_log_weight` 与 §5.3 公式逐项吻合 |
| **I3** | M3 branch-local self + 联合边缘化 | 单实体 marginal 等于 `π·P_action+(1-π)·P_auto`；多实体时至多一个 self 响应 action |
| **I4** | M4 + M5，产出可被 `run_candidate_lifecycle` 接受的 factory | `validate_candidate_factory` 通过；每 episode 新实例；frozen kernel 未被改动 |
| **I5** | 在 development split 上跑通并与既有五个参照同表比较 | 产出 development 报告；**此时才第一次知道 12 个确认门在真实候选上的表现** |

**I5 之前不得声称任何门控结论。** 开发集上的数字是开发数字，不是确认证据。

---

## 四、必须预先说清的失败语义

预注册 C.1 已写明：候选未能在 6+ 遮挡区间闭合 `oracle` 与 `belief_free` 之间差距的
40% 即为**失败**。本计划不改变该判据，也不得因候选表现不佳而回头调低闭合比例
——那正是评审 F1 要防的事。

**`belief_free` 是下限参照**：红队用"反射外推 + 125 条目误差表、零信念滤波"通过了
当时全部 18 个门。候选必须显示它在**校准平滑之上**多做了什么，而不是跑赢点质量。
