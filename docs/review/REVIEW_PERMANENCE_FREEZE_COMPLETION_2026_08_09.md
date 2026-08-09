# 补充评审：待办清单修正轮 2026-08-09

上游：[`REVIEW_PERMANENCE_FREEZE_2026_08_08.md`](REVIEW_PERMANENCE_FREEZE_2026_08_08.md)
（判定 `block`）与其
[F4/F8 补充](REVIEW_PERMANENCE_FREEZE_FOLLOWUP_2026_08_08.md)。两份均为不可在位
修改的历史产物，故本轮以补充文档追加。

本轮处理 [`OPEN_ITEMS.md`](OPEN_ITEMS.md) 上除 O3 与 O16–O19 之外的全部条目：
**4 条阻断项中的 3 条、7 条 P2、4 条 P3、以及 G4 在永久性侧的消除。**

> **总判定仍为 `block`。**
>
> 本轮**没有**让程序更接近通过，也不该被这样读。O1/O2/O4 做的事情是让"冻结"
> 这个词名副实：在此之前，永久性栈的"锁"只是描述性的，"对照可构造"只是论证。
> 唯一剩下的技术阻断项 **O3（尚无候选实现）恰恰是最重的一条**——红队攻击 A4
> 的结论只有在真实候选上跑过门系统之后才可能被推翻或确认。

---

## 一、顺序不是可选的

本轮最先确定的不是改什么，而是**顺序**。实测：Phase-0 V11 与 Phase-R V5 的
`source_lock` 覆盖 20 / 19 个文件，其中包含 `v2_artifacts.py`、
`cal/infra/provenance.py`、`pyproject.toml`，**以及
`docs/experiments/V2_I1_STOCHASTIC_PERMANENCE_PLAN.md`**。

也就是说 O4（文档合并）、O8（schema 修复）与全部代码修复，**每一条都会让当时
产物的 source lock 漂移**。由此只有一个自洽顺序：

```
全部代码/文档修复  →  一次性重生成产物  →  加运行时源码锁
```

先加锁再改代码等于自己锁死自己；先重生成再改代码等于白跑。这一点在动手前就已
写明，并按此执行。

## 二、已完成项与验证

逐条验证方式见 [`OPEN_ITEMS.md`](OPEN_ITEMS.md) 的台账，此处只记**本轮产生了
新事实**的几条。

### O1（P1 / F8 / G7）源码锁：从"能发现"到"拒绝执行"

新增冻结协议
`experiments/V2_P1_PERMANENCE_STACK_SOURCE_LOCK_V1.json`（+ `.sha256`），
`run_phase0` 与 `run_phase_r_diagnostic` 在做任何工作之前调用
`verify_locked_sources`。

**锁定范围是 import 闭包，不是手工清单**，这是本条最实质的设计决定。
G4 的成因正是手工清单——M1–M3 的 `locked_source_sha256` 至今漏掉
`v2_m3_hypotheses.py` 所 import 的 `v2_m3.py`。因此
`permanence_stack_source_paths` 的内容由 Phase-0 / Phase-R / 扫描 / 注册表四个
入口的传递 import 闭包决定，并由
`test_source_lock_covers_the_whole_permanence_import_closure` 每次重算比对：
**新 import 逃出锁 = 测试失败**。共 22 个模块。

**双向验证**

| 方向 | 结果 |
| --- | --- |
| 未改动时 | `verify_locked_sources` 通过，22 文件 |
| 向 `stochastic_motion_filter.py` 追加一行注释 | `cal-v2-i1-permanence-phase-r` 拒绝运行：`refusing to produce evidence. changed=['cal/model/stochastic_motion_filter.py'] missing=[] unlocked=[]` |
| 复原 | 恢复通过 |

**一处对上一版清单的更正**：清单要求覆盖"14 个永久性模块 + `cal/env/` 的真值
模拟器"。实测 **`cal/env/` 不在本栈的 import 闭包内**——那是 M1/V1 的世界。
本栈的真值模拟器是 `randomized_occlusion_world.py` 与 `v2_i1_integration.py`，
两者都已锁定。该省略写入协议的 `out_of_scope` 字段，是有意的记录而非遗漏。

**这道锁的代价要说清楚**：从现在起，改动 22 个模块中的任何一个，
`cal-v2-i1-permanence-phase0` / `-phase-r` 以及调用它们的测试都会立刻拒绝运行，
直到用 `mint_permanence_stack_source_lock` 铸出**新的协议版本**（`V2`、`V3`……，
保留旧版作为修订链）。这不是副作用而是目的——M1–M3 侧的
`_verify_locked_sources` 是同样的代价，`CLAUDE.md` 已写明"编辑六个锁定文件之一
会让下一次运行抛异常，by design"。不接受这个代价就等于不接受冻结。

### O2（P1 / F6 后半条）对照可构造性：论证换成了数字

`cal-v2-p1-permanence-controls`（新增，非门控）在完整 development split
（40 train / 150 eval seed、train 3 506 / eval **12 473** 个样本）上实跑了五个
V8 对照的构造代码。**五个全部构造成功。**

| 对照 | 可构造样本 | top-1 | categorical NLL |
| --- | ---: | ---: | ---: |
| `raw_sensor` | 12 473 | 0.0520 | 5.253 |
| `assume_all_visible` | 12 473 | 0.0000 | 13.816 |
| `time_shuffled` | 12 473 | 0.1362 | 11.829 |
| **`identity_scrambled`** | **2 081** | 0.1802 | 11.279 |
| `random_labels` | 12 473 | 0.0461 | 6.627 |

**本条的产出不是那五个"是"，而是 `identity_scrambled` 的 2 081。** 身份打乱需要
同时至少两个被遮挡且轨迹已知的物体，只有 **16.7%** 的事件满足。V5 留出正是在
这一点上中途停止且不可重试——同一个数字，在 development 上的代价是零，在留出上
的代价是那批留出。它现在是 O17（留出规模）的硬约束。

附带一条可解释性检查：`assume_all_visible` 的 top-1 **恰为** 0.0000、NLL **恰为**
上限 13.816。这不是 bug——该对照把全部质量放在可见格上，而隐藏正例按定义全在
不可见格，所以它在场内质量为零，被评分框架记为显式 miss。

### O12（P3 / F19）四个恒真门：两个变成了真实测量

「删掉恒真门」不是本轮的做法；四条都被**换成可失败的形式**，其中两条因此产生了
此前不存在的证据：

| 门 | 改前 | 改后 | 新事实 |
| --- | --- | --- | --- |
| `fully_detached_safe` | `s_max >= H*E*K`，而 `s_max` 定义即 `H*E*K` | 比对**实际分配的数组尺寸** | — |
| `shared_expansion_workspace_safe` | `12*k_max` 与同一公式比较 | 加入穷举 38 400 例实测的分支因子 | **实测最大后继数 = 4**，界为 12（3× 余量）。此前无人知道真实分支因子 |
| `branch_evidence_accounting` | 用 filter 自己报告的两项重算 filter 自己，残差恒为 `0.0` | 交叉校验 `branch_log_weight` 与独立维护的 `cumulative_retained_probability` | 残差**实测 2.44e-15**，仍以三个数量级余量通过 1e-12。这是恒零门给不出的证据 |
| `formal_research_budget_declared` | 模块常量与同模块字面量比较 | 改名 `formal_research_budget_respected`，加入实测工作量 | — |

产物 schema 因门键集合改变而 3 → 4。

### O13（P3 / F20）诚实的 no-go 现在写得出来

`registry_turn_probability` 失配此前在**验证器里抛异常**，于是一次未被注册表绑定
的运行会以 `registry provenance mismatch` 崩溃，而不是写出 `phase_r_no_go`。
现改为：结构性字段仍然强校验，**是否用了注册表绑定的概率降为门**，门与验证器
统一用 `abs_tol 1e-12`（此前 runner 用 `np.isclose` 的 rtol 1e-5）。

### O14（P3 / F18）永久禁令改由测试强制

评审给 pairwise 正/负构造附了一条永久禁令。禁令此前只存在于文档里。
`test_pairwise_negative_construction_stays_out_of_every_gate` 用 AST 扫描六个门控
模块与 `permanence_forward_benchmark` 中 11 个门控评分函数，任何对 `.negative`
的读取即失败。**反向验证**：向 `_rank` 注入一次 `.negative` 读取即被检出。

非门控的 GRU 基线仍然读它，这是允许的——测试的范围就是"门控路径"。

---

## 三、产物重生成

全部改动落地后一次性重生成，两份均通过：

| 产物 | 结果 | 摘要 SHA-256 |
| --- | --- | --- |
| Phase-0 **V12** | `phase0_go`，2 项 phase-0 门全真，150 seed 全覆盖 | `90d3e5da91613cd7` |
| Phase-R **V6** | `phase_r_go`，**15 门全真**，schema 4，1 615 episode | `7adbe8369e032f6c` |

`ACKNOWLEDGED_SOURCE_DRIFT` 清空为空集合，并在注释中写明：再次添加名字等于决定
发布一份与产生它的代码不一致的产物。两份产物的 `audit_artifact_source_lock`
实测 `matched=True`、`drifted=[]`、`missing=[]`。V11 / V5 作为被取代的历史环节
保留。

**统计结论未因本轮改动而移动**：`recommended_holdout_seed_count` 仍为 **11078**，
与 V11 一致。这是一个有意义的负面对照——它说明本轮修的是**验证与记账**，不是
效应量，四个恒真门换成可失败形式之后它们依然全部为真。

回归：`uv run pytest` **413 passed**（本轮之前 396，新增 17 个测试）。

### 两条关于重跑成本的记录

**成本**：Phase-0 的计算量是 4 个效应场景 + 4 个零假设场景 × 1024 trials ×
2000 bootstrap × 150 seed ≈ **1 638 万次 bootstrap 重采样、约 24.6 亿次元素抽取**，
另有最多 27 个对比量与 56 步 bounded-moment 二分。**实测单核约 3 小时 40 分。**
`--simulation-trials` 不得为省时间调低：1024 是锁定常量
`POWER_LOCKED_SIMULATION_TRIALS`，调低的产物无法通过校验。

**必须在不会被中途终止的环境里跑**。本轮第一次尝试在 110 分钟处被外部进程管理
终止，未写出产物，那 110 分钟完全作废。值得记下的正面结果是：
**原子写入设计经受住了这次意外终止**——无残留临时文件、无 `.write.lock`、
工作树干净、没有半成品产物。重跑改用 `nohup` + `disown` 完全脱离后，越过 110
分钟点并正常完成。

---

## 四、副作用与未做的事

**副作用**：本轮改动了机制代码，因此 Phase-0 / Phase-R 的开发产物必须重跑，
上一代 V11 / V5 的数字不再是当前代码的产物。**没有任何已发表的终局证据受影响**
——V8 与 I1 v4 holdout 不经过本栈。

**未做**：

- **O3（尚无候选实现）**：研究工作量级，本轮未动，也是唯一剩下的技术阻断项。
- **O16–O19**：需要人类判断（闭合比例、留出规模、冻结签署、留出密盐保管人），
  非技术阻塞。其中 **O19 有硬顺序约束**：密盐必须在生成任何留出 seed 之前就位。
  代码侧的前置守卫已补上（`stochastic_permanence_holdout.py`，见下），但
  **"谁持有那个秘密"不是代码能替代的**。

### 附：O19 的前置守卫

一次性 split 的生成路径尚不存在（没有候选就没有要预留的东西），但它的一个前提
**事后补不了**，所以先把门立在那里。

`RandomizedOcclusionWorld.__init__` 的 `hidden_stream_salt` 默认为公开的开发盐
——开发集要可复现，这个默认值本身是对的。但它同时意味着：一个生成留出的脚本
**漏传该参数不会报任何错**，只会安静地产出一批从诞生起就可被反演的留出。而
换盐重跑不是"把那批留出加固了"，是产出了**另一批**留出；在一次性语义下第一批
已经花掉了。

`open_one_shot_world` 是这条路径应当走的门：漏传即 `TypeError`，传公开盐、过短
盐或单字节占位盐即 `InvertibleSplitError`。

**放置位置是有意选择**。语义上它属于 `custody.py`，但那是 22 个锁定模块之一，
改动它需要铸协议 V2 并重跑 3 小时 40 分的产物——而这段代码零调用方，
不可能影响 V12/V6 的任何数字。放在锁外的依据是：本锁的范围是"决定当前门控证据
的代码"，未来的一次性生成器不决定其中任何一项。为了让这不退化成遗漏，
`test_the_guard_is_not_inside_the_gated_source_lock` 钉住了这个边界：
一旦永久性入口开始 import 它，它就必须随之入锁。**留出程序冻结时，
这个文件需要它自己的锁。**
- **G2 / G5 / G6 与 M1–M3 侧的 G1 / G3 / G4**：不变。`v2_m2.py` 等在 V7 确认协议
  的 `locked_source_sha256` 内，按 `CLAUDE.md` 不得在位编辑；修它们需要新的协议
  版本，属于独立的一轮工作。

---

本报告按 [`REVIEW_PLAN.md`](REVIEW_PLAN.md) §9 为**不可在位修改**的历史产物。
