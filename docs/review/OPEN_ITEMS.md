# 未完成项清单

截至 2026-08-09（分支 `permanence-belief-free-gate-redesign`，PR #2）。
上一版截至 2026-08-08；本版记录 2026-08-09 修正轮之后的状态，逐条实测复核。

本文件汇总**已识别但尚未处理**的项。已处理项移入文末的「本轮已解决」，
并附验证方式——**没有验证方式的"已解决"不算解决**。

来源标记：`P#` = [永久性评审](REVIEW_PERMANENCE_FREEZE_2026_08_08.md) 的发现编号；
`G#` = 该报告 §4.3 的系统性缺口编号。修正轮的完整改动见
[完成报告](REVIEW_PERMANENCE_FREEZE_COMPLETION_2026_08_09.md)。

---

## 一、阻断冻结

### O3 尚无候选实现

12 项确认门从未在真实候选上运行过。当前只有 `oracle` / `belief_free` /
`geometric` / `uniform` / `old_i1` 五个参照。

**这是唯一剩下的技术性阻断项，也是最重的一项。** O1/O2/O4 的完成不会让程序更接近
通过——它们只是让"冻结"这个词名副实。红队攻击 A4 的教训仍然完整成立：门系统
是否能测出目标机制，只有在真实候选上跑过才知道。

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
| O16 | 闭合比例是否沿用 **0.40** | 当前实现即 0.40，理由是与既有 `6+/top1_closure_0.40` 同值、不新造阈值 |
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
| O10 | `autonomous_successors` 非二值混合 | 转向分支在非二值 `static_probability` 下直接抛异常，不再静默近似 |
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
15 门全真，schema 4，1 615 episode）已在全部改动落地后一次性重生成，
两份的 `audit_artifact_source_lock` 实测 `matched=True`、`drifted=[]`。
`ACKNOWLEDGED_SOURCE_DRIFT` 已清空为空集合。V11 / V5 作为被取代的历史环节保留。

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

## 上一轮已解决（2026-08-08，仅供对照）

P0：无信念候选过门（`belief_free` 成为参照下限）、草案门表与实现零重合、
草案无数值阈值与决策规则。
P1：phase0 验证盲区（验证器从原始数据重算）、恒真 floor 门（已删）、
绑定过期注册表（V4）、种子反演（HMAC 派生）。
P2：README/RESEARCH_STATUS 不提永久性专案。
方案自身评审的 F1–F8 全部落地（见
[补充说明](REVIEW_REVIEW_PLAN_SUPPLEMENT_2026_08_08.md) 的复核台账）。
