# 未完成项清单

截至 2026-08-10。**总判定仍为 `block`。**

| | |
| --- | --- |
| 当前分支 | `permanence-candidate-static-map`（PR #4，**未合并**） |
| 最新提交 | `24fd028` |
| 已合并 | PR #3 → `main`（`a40bc6c`），即 O1/O2/O4 + P2/P3 修正轮 |
| 源码锁协议 | **V3**（22 模块），`experiments/V2_P1_PERMANENCE_STACK_SOURCE_LOCK_V3.json` |
| 开发产物 | Phase-0 **V12** `phase0_go`；Phase-R **V6** `phase_r_go`；两份 `drifted=[]` |
| 测试 | `uv run pytest` → **474 passed** |

本文件汇总**已识别但尚未处理**的项。已处理项移入文末台账并附验证方式
——**没有验证方式的"已解决"不算解决**。

来源标记：`P#` = [永久性评审](REVIEW_PERMANENCE_FREEZE_2026_08_08.md) 的发现编号；
`G#` = 该报告 §4.3 的系统性缺口编号。2026-08-09 修正轮的完整改动见
[完成报告](REVIEW_PERMANENCE_FREEZE_COMPLETION_2026_08_09.md)；候选实现的分解与
进度见[候选实现计划](../experiments/V2_P1_CANDIDATE_IMPLEMENTATION_PLAN.md)。

---

## 换机器继续前必读

1. **`git pull` 后先跑 `uv run pytest`**（应 474 passed）与
   `verify_locked_sources(root=".")`（应报 V3 / 22 files）。任一不符先查清楚，
   不要在漂移状态上继续。
2. **改动 22 个锁定模块中的任何一个 = 协议新版本 + 两份产物重跑**。
   Phase-0 实测**单核 3 小时 41 分**，`--simulation-trials 1024` 是锁定常量
   `POWER_LOCKED_SIMULATION_TRIALS`，**不得为省时间调低**。
3. **重跑必须脱离终端**（`nohup … & disown`）。曾有一次在 110 分钟处被外部进程
   管理终止、全部作废；原子写入设计经受住了那次终止（无残留临时文件）。
4. 候选代码**在锁外**（通过 `CandidateFactory` 注入），所以写候选本身不触发协议
   改动。只有改共享的转移核/滤波器才会。

---

## 一、阻断冻结

### O3 候选实现（进行中，非「尚无」）

**状态已变**：候选本体已存在，12 项确认门**仍未在其上跑过**。O3 未完成。

已完成的增量（细节见候选实现计划）：

| 增量 | 组件 | 结果 |
| --- | --- | --- |
| I1 ✅ | `cal/model/sensor_only_static_map.py` | 恢复 24/31 场内静态格，零假阳性 |
| I2 ✅ | `cal/model/permanence_track.py` | 存在性通道接线，§5.3 逐项吻合 |
| I3 ✅ | `cal/model/branch_self_identity.py` | branch-local self + 联合边缘化 |
| I4 ✅ | `cal/model/permanence_candidate.py` | 通过 §7 生命周期契约 |
| I4b ✅ | `cal/model/association_bank.py` | 多假设 `w_h`，一对一指派 |
| **I5 ◐** | `cal/evaluation/permanence_candidate_development.py` | 框架就绪、诊断已出；**全量未跑** |

#### 剩下要做的两步

**第一步：改进跟踪。** 这是当前主要瓶颈，由下面的特权诊断定位。

**第二步：全量 development split 跑一次**（`cal-…candidate_development`，
40 train + 150 eval seed，**实测外推约 4.3 小时**），产出报告，此时才第一次知道
12 项确认门在真实候选上的表现。

#### 定位瓶颈的关键诊断（已完成，务必先读再动手）

参照 `_belief_map` / `_geometric_map` 直接读 `sample.hidden_tracks`——采集器把
**末见格、观测速度、遮挡时长**白送给它们；候选必须自己推断。为把"信念不行"与
"跟踪不行"分开，`privileged_belief_maps` 用候选**自己的**信念机制跑参照的轨迹
（12 eval seed，916 样本，候选自己拟合的核 τ=0.55）：

| 对象 | top1 | NLL | 6+ |
| --- | ---: | ---: | ---: |
| `belief_free` | 0.4065 | 2.103 | 0.1179 |
| **诊断（候选信念 + 白送轨迹）** | 0.4460 | **1.453** | **0.2411** |
| `oracle` | 0.4743 | 1.402 | 0.2874 |

**6+ 闭合 0.727，预注册下限 0.40。** 而 sensor-only 候选整体仅 0.2391。
**结论：差距几乎全部来自跟踪，不是信念。**

> 没有这个诊断，下一步的自然动作是改进信念滤波器——**那基本是白费**。

#### 已知的具体缺陷（改跟踪时的入手点）

- 关联仅按预测 MAP 位置做 1 格门控 + 贪心/枚举，速度估计来自 1 步位移匹配，
  可能误配到别的实体；
- 新生轨道速度用四方向均匀先验（已修掉原先硬塞 `(1,0)` 的问题），但再次匹配前
  仍无方向信息；
- 转向概率估计量**方差偏大**：同一估计量在 10 个训练 seed 上给 0.95、30 个上给
  0.55（真值 0.45）。**拟合须用完整训练集，不能采样**。

---

## 二、非阻断但影响正确性（P2）

**本节已清空。** O5–O11 七项全部落地，见文末台账。

---

## 三、记录在案，不阻断（P3）

**本节已清空。** O12–O15 全部落地，见文末台账。其中 O14 的永久禁令由
`test_pairwise_negative_construction_stays_out_of_every_gate` 持续强制，
不再依赖人记得。

---

## 四、架构级缺口（评审 §4.3）

**实测**：以下三条在当前代码上仍然成立。

| # | 缺口 | 证据 | 本轮变化 |
| --- | --- | --- | --- |
| G1 | `_verify_locked_sources` 只有 `v2_m1_m3_confirmation.py` 一个调用点 | 全库唯一调用点 | **已缓解但未消除**：永久性栈现在有独立的 `verify_locked_sources`，由 `run_phase0` / `run_phase_r_diagnostic` 调用。M1–M3 侧不变 |
| G2 | `v2_m2.py` 省略 `--split` 时静默走无哈希验证的 legacy 路径 | `v2_m2.py:484` `split or "legacy_development"` | 不变。`v2_m2.py` 在 V7 确认协议的 `locked_source_sha256` 内，本轮不得触碰 |
| G5 | M4 体系整体无源码锁 | 已消费留出的机制被后续修复改变过行为（对照余量 0.075 → 0.012） | 不变 |
| G6 | `capture_provenance` 的 `source_sha256` 只描述不校验 | 无任何校验方 | 不变。后果已按 O9 记入[已知限制](../experiments/V2_KNOWN_PROVENANCE_LIMITATIONS.md) |

G3（`v2_m1.py` 无锁）：`v2_m1.py` 被 V7 确认协议覆盖，但仅在确认阶段运行时校验，
单独运行不校验。不变。

G4（锁未覆盖 `v2_m3.py` 与真值模拟器）：**M1–M3 侧不变**；永久性侧已消除——
`permanence_stack_source_paths` 锁的是**传递 import 闭包**而非手工清单，
并由 `test_source_lock_covers_the_whole_permanence_import_closure` 每次重算。
手工清单正是 G4 的成因，永久性栈不再用手工清单。

> **注**：上一版本清单把 G4 的修复范围写成"14 个永久性模块 + `cal/env/` 的真值
> 模拟器"。实测 `cal/env/` **不在永久性栈的 import 闭包内**——那是 M1/V1 的世界。
> 本栈的真值模拟器是 `randomized_occlusion_world.py` 与 `v2_i1_integration.py`，
> 两者都已锁定。此事写入协议的 `out_of_scope` 字段，是有意省略而非遗漏。

---

## 五、待定夺（需人类判断，非技术阻塞）

| # | 事项 | 现状 |
| --- | --- | --- |
| O16 | 闭合比例是否沿用 **0.40** | 当前实现即 0.40，理由是与既有 `6+/top1_closure_0.40` 同值、不新造阈值。**建议与 O20 一并定夺** |
| **O20** | **闭合门的下参照被白送了跟踪解**（2026-08-10 新增） | `belief_free` 由采集器提供 `hidden_tracks`，被门的候选没有。**候选可能在信念机制完全正常的情况下，纯因关联误差挂掉这道门，而门无法区分**——上面的诊断实测了这一点（信念侧闭合 0.727，整体却远低于 `belief_free`）。要么把下参照放到与候选同等的信息条件下评分，要么把门明确写成"跟踪与信念的合成能力"。**不得为迁就候选而调低阈值——那正是 F1 要防的事**，故只记录、不动手 |
| O17 | 留出规模 | `recommended_holdout_seed_count` = **11078**。**新增约束**：O2 实测 `identity_scrambled` 只在 **16.7%**（2081/12473）的事件上可构造，留出规模须按该比例折算 |
| O18 | 冻结授权 | 本轮改动了预注册的锁定常量与门定义；按 [`REVIEW_PLAN.md`](REVIEW_PLAN.md) §8，冻结需评审 `pass` 且**人类 Gatekeeper 签署** |
| O19 | 留出密盐 | F9 的设计要求保管人持有密盐。**必须在生成任何留出 seed 之前就位**，事后改代码无法补救已生成的留出。`DEFAULT_HIDDEN_STREAM_SALT` 是公开的开发盐，**不得**用于留出。**代码侧的前置守卫已就位**（见下），但"谁持有那个秘密"仍是人的事，不是代码能替代的 |

---

## 本轮已解决（2026-08-09）

每条附**怎么验证**，而不只是"改了"。

### 阻断项

| # | 事项 | 验证 |
| --- | --- | --- |
| O1 | 永久性栈的 source lock + 运行时拒绝 | 新协议 `experiments/V2_P1_PERMANENCE_STACK_SOURCE_LOCK_V1.json`（22 模块，import 闭包）。**反向验证**：向 `stochastic_motion_filter.py` 追加一行注释，`cal-v2-i1-permanence-phase-r` 以 `refusing to produce evidence. changed=[...]` 拒绝运行；复原后恢复 |
| O2 | 对照可构造性实证 | `cal-v2-p1-permanence-controls` 在 40+150 seed / 12 473 样本上实跑五个对照，全部构造成功；输出落盘 `results/V2-P1-permanence-control-constructability-development.json` |
| O4 | 计划文档与预注册的权威关系 | 计划 §9 整章加注被预注册 C 节取代；C.9 反向记录。实质分叉：§9.3 闭合下参照用 `geometric`，C 节用 `belief_free`——A4 通过的正是前者 |

### P2

| # | 事项 | 验证 |
| --- | --- | --- |
| O5 | seed 重叠断言 | `run_benchmark` / `gru_capacity_sweep` / `run_diagnostic` / 对照普查全部调用 `validate_disjoint_seed_sets`；实测三个入口对重叠 seed 均抛 `seed collision between streams` |
| O6 | CLI 覆盖不再盖注册表 provenance | `--steps/--warmup/--turn-probability` 默认改为 `None` 以区分"显式指定"；与 `--seed-registry` 同用即拒。软默认 `registry.get("selected_turn_probability", _DEFAULT)` 改为必需键 |
| O7 | `cal-index` 不再截断 | 干净检出上运行 `uv run cal-index --results results` 现在退出码 1 并拒绝写入，报告将丢失 680 条；`INDEX.json` 仍为 702 条。`--prune` 才允许丢弃 |
| O8 | `require_authorization` 支持 schema 2 | `SUPPORTED_RESULT_SCHEMA_VERSIONS = (1, 2)` |
| O9 | 27 份脏 provenance 产物 | 不可修复，已按可复现清单记入[已知限制](../experiments/V2_KNOWN_PROVENANCE_LIMITATIONS.md)，含复现脚本 |
| O10 | `autonomous_successors` 非二值混合 | 转向分支在非二值 `static_probability` 下**默认拒绝执行**。**2026-08-10 修订**：候选（O3）需要分数拓扑，而 §5.2 本就规定用这个逐方向近似并列入 candidate lock——守卫反对的是它**静默**，不是近似本身。故加 `marginal_turn_mixture` 显式 opt-in（默认仍拒绝，算术不变），并测试钉住**二值网格上两条路径逐项相等**。代价照付：协议 V2 → V3 + 两份产物重跑 |
| O11 | `GridSpec` 默认值陷阱 | 三个默认值删除，改为必填；评估侧唯一定义 `EVALUATION_GRID_SPEC` 由世界常量推导 |

### P3

| # | 事项 | 验证 |
| --- | --- | --- |
| O12 | 四个恒真门 | `fully_detached_safe` 改为比对**实际分配的数组尺寸**；`shared_expansion_workspace_safe` 增加穷举 38 400 例实测分支因子（**实测 4**，界为 12）；`branch_evidence_accounting` 改为交叉校验两个独立累加器（残差由恒 0.0 变为**实测 2.44e-15**）；`formal_research_budget_declared` → `formal_research_budget_respected`，加入实测工作量。产物 schema 3 → 4 |
| O13 | 容差不对称 | 门与验证器统一为 `abs_tol 1e-12`；registry turn probability 失配从"抛异常"降为**门为 False**，诚实的 no-go 现在可以写出来 |
| O14 | pairwise 永久禁令 | `test_pairwise_negative_construction_stays_out_of_every_gate` AST 扫描六个门控模块与 11 个门控评分函数；向 `_rank` 注入一次 `.negative` 读取即被检出 |
| O15 | 零散项 | 零质量状态改记 `retained_positive_support`（与参照可比）；`maximum_step_pruned_mass` 进入 step 返回值与逐 episode 记录；`bayesian_no_detection_update` 标注为预留未接线；`replace_factor_atomic` 校验 code 唯一性；`maximum_tv_checkpoint` 改为记录**最先**达到最大值的步（`>` 而非 `>=`）；`run_scan` 改读 `coverage_contract()`；RNG 流间距 50 000 加断言 `_require_stream_separation` |

### 产物

Phase-0 **V12**（`phase0_go`，150 seed）与 Phase-R **V6**（`phase_r_go`，
15 门全真，schema 4，1 615 episode）在全部改动落地后重生成，
两份的 `audit_artifact_source_lock` 实测 `matched=True`、`drifted=[]`。
`ACKNOWLEDGED_SOURCE_DRIFT` 已清空为空集合。V11 / V5 作为被取代的历史环节保留。

**注（2026-08-10）**：这两份产物此后**又重跑过一次**——O3 的 I4 增量改动了
`stochastic_motion_filter.py`（为 F16 守卫加显式 `marginal_turn_mixture` opt-in），
触发协议 **V2 → V3**，按约定重跑而非声明漂移。当前有效的就是 V12 / V6，
但它们是 V3 下的版本。协议共铸三版：V1（初次）→ V2（O19 密盐守卫入 custody）
→ V3（F16 opt-in）。**每一版都带 `amendment_record`**，`build_permanence_stack_source_lock`
强制 V1 之后的版本必须带。

**统计结论未移动**：`recommended_holdout_seed_count` 仍为 11078，与 V11 一致——
本轮修的是验证与记账，不是效应量。

回归：`uv run pytest` **413 passed**（本轮之前 396）。

### O19 的代码侧前置守卫

`cal/evaluation/stochastic_permanence_custody.py` 新增
`require_custodian_salt`（拒绝公开开发盐、过短盐、单字节占位盐）与
`open_reserved_split_world`（一次性 split 生成应当走的门）。

放在 custody 是因为它就是一条 custody 义务，而且是这个模块里**唯一无法事后补救**
的一条：其余部分管的是什么可以被**消费**，这一条管的是什么可以被**生成**——
等 episode 生成出来，盐已经烙进它们每一条隐藏轨迹里了。

**它关的是哪个坑**：`RandomizedOcclusionWorld.__init__` 的
`hidden_stream_salt` 有默认值（公开开发盐），所以生成脚本**漏传参数不会报错**，
只会安静地产出一批可反演的留出。走这道门则漏传即 `TypeError`、传公开盐即
`InvertibleSplitError`。测试 `test_the_constructor_default_is_the_trap_this_closes`
**同时钉住两半**——"构造函数确实会静默接受漏传"与"这道门不会"；
只钉后者的话，哪天默认值被去掉、这道门变成多余，测试也不会告诉你。

**代价照付**：custody.py 在 22 个锁定模块内，所以这次改动走了完整的修订流程——
铸出协议 **V2**（含 `amendment_record`，指明前身 V1、其 digest 与改动理由），
并**重跑了 phase0 与 phase-R 两份产物**。这段守卫零调用方、不可能影响任何已记录
的数字，本可以用"声明漂移"糊过去；没有那样做，因为那正是本轮要消除的
"锁只是描述性的"状态。**这也是永久性栈第一次真正走通修订链**——此前
`mint_permanence_stack_source_lock` 的多版本路径从未被执行过。

**重跑成本记录**：Phase-0 实测单核约 **3 小时 40 分**，且
`--simulation-trials 1024` 是锁定常量不得调低。第一次尝试在 110 分钟处被外部
终止、全部作废；原子写入设计经受住了那次终止（无残留临时文件、无半成品产物）。
后续任何人重跑必须在不会被中途终止的环境里安排。

---

## 2026-08-10 这一轮（O3 候选实现）

产出全部在 **PR #4，未合并**。八个提交，`a632077` → `24fd028`。

| # | 事项 | 验证 |
| --- | --- | --- |
| I1 | sensor-only 静态拓扑推断 | 观测模型对本世界**精确而非假设**：可见且空 ⟹ 必非静态（`P(sensed=0∣static)=0`）。3 seed 拟合、5 seed 评估恢复 24/31 场内静态格、**零假阳性**。真值只作被比对对象 |
| I2 | 存在性通道 | 关键是**不双重计数**：滤波器报告的 `observation_evidence` 就是 `L_no`，其 `branch_log_weight` 是无存在性版本；轨道只叠加存在性感知证据 + 剪枝损失 |
| I3 | branch-local self | `{null} ∪ entities` 类别分布让"至多一个 self"成为**构造性质**。联合式一并实现，**因为独立 Bernoulli 设计在每个单实体边缘上都与正确模型一致、只在联合上分叉** |
| I4 | 组装 + `CandidateFactory` | 通过 §7 生命周期契约。**发现**：`_maybe_hidden_turn` 在实体可见时返回，**转向只在遮挡期发生**，故 τ 无法从可见轨迹估计，只能从再现区间做极大似然 |
| I4b | 多假设关联 bank | 关联不确定性只存在于 `w_h`；每 branch 一对一指派；两个界以 `discarded_hypothesis_mass` 审计。**全程未触发协议改动**，坐实"候选在锁外" |
| I5 ◐ | 比较框架 + 特权诊断 | 重放与采集器逐样本对齐已验证（297/297）。诊断定位瓶颈在跟踪（见上）。**全量未跑** |

**修掉的三个缺陷**：新生轨道硬塞速度 `(1,0)` → 四方向均匀先验；Bernoulli 并集
溢出一个 ULP → `as_probability_field` 只裁剪舍入、**超出即抛错**；转向概率估计量
拟合出 0.05（真值 0.45）→ 两个 bug（**跨 episode 池化拓扑**、**似然未对非检测
条件化**），修后恢复测试通过。

**一次自我更正**：曾声称"信念机制是在错了一个数量级的核上清过下限的"——当时诊断
被传了真值 τ，该说法不成立。修掉特权泄漏后重测，结论仍成立且更干净。

---

## 上一轮已解决（2026-08-08，仅供对照）

P0：无信念候选过门（`belief_free` 成为参照下限）、草案门表与实现零重合、
草案无数值阈值与决策规则。
P1：phase0 验证盲区（验证器从原始数据重算）、恒真 floor 门（已删）、
绑定过期注册表（V4）、种子反演（HMAC 派生）。
P2：README/RESEARCH_STATUS 不提永久性专案。
方案自身评审的 F1–F8 全部落地（见
[补充说明](REVIEW_REVIEW_PLAN_SUPPLEMENT_2026_08_08.md) 的复核台账）。
