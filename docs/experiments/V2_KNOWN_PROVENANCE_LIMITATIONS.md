# 已知 provenance 限制：27 份已提交结果不可归因到干净工作树

日期：2026-08-09
来源：[2026-08-08 冻结前评审](../review/REVIEW_PERMANENCE_FREEZE_2026_08_08.md) 发现 **F12**
（评级 P2）；待办编号 **O9**。

## 结论

`results/` 下有 **27 份已提交的结果 JSON**，其 `provenance` 记录
`git_dirty=true`，其中 **20 份连 `git_commit` 都是 `null`**。这些产物**无法被
重跑复现**——重跑需要知道当时的确切工作树状态，而这一信息没有被记录下来。

**这不可修复，只能记录。** 重新生成会得到不同的数字（同一份代码在不同工作树
状态下产出的产物本就不该被当作同一份证据），而伪造一个干净的 commit 字段等于
制造虚假可追溯性。因此本文件的作用是：**任何引用下列产物的人必须知道它的可追溯
性到此为止。**

## 影响范围与不影响范围

**不受影响（关键终局证据是干净的）**：

| 产物 | 状态 |
| --- | --- |
| `results/V2-L0-language-readout-holdout-v8.json` | 干净，有 commit |
| I1 v4 holdout 系列 | 干净，有 commit |
| `experiments/V2_I1_P1_PHASE0_*` / `PHASE_R_*`（永久性栈全部产物） | 干净 |

也就是说：**已发表的终局判定不依赖任何一份脏产物**。

**受影响**：下列 27 份，含**两份已消费 holdout 摘要**
（`V2-M2-probabilistic-holdout-summary.json`、
`V2-M3-body-graph-holdout-summary.json`）与多份 `authorize_*` 链路上的授权产物。
已消费 holdout 按一次性语义**不可重跑**，因此这两份的限制是永久的。

| # | 路径 | `git_commit` |
| ---: | --- | --- |
| 1 | `results/V2-I1-v2-development-v3-review-baseline.json` | `d235e888` |
| 2 | `results/V2-L0-language-readout-development.json` | `d6064e9b` |
| 3 | `results/V2-L0-language-readout-development-v2.json` | `d6064e9b` |
| 4 | `results/V2-L0-language-readout-development-v3.json` | `d6064e9b` |
| 5 | `results/V2-M1-M3-integrated-confirmation-review-summary.json` | `0745bae9` |
| 6 | `results/V2-M4-unprivileged-development-v1-summary.json` | `8fb4e545` |
| 7 | `results/V2-M4-unprivileged-development-v2-summary.json` | `bff14b9d` |
| 8 | `results/V2-stage-summary.json` | `976fb894` |
| 9 | `results/V2-M1-M3-integrated-development-summary.json` | **无** |
| 10 | `results/V2-M1-M3-integrated-v2-confirmation-summary.json` | **无** |
| 11 | `results/V2-M1-summary.json` | **无** |
| 12 | `results/V2-M2-summary.json` | **无** |
| 13 | `results/V2-M2-hard-map-development-summary.json` | **无** |
| 14 | `results/V2-M2-nearest-development-summary.json` | **无** |
| 15 | `results/V2-M2-probabilistic-development-summary.json` | **无** |
| 16 | `results/V2-M2-probabilistic-holdout-summary.json` | **无**（已消费 holdout） |
| 17 | `results/V2-M2-probabilistic-review-summary.json` | **无** |
| 18 | `results/V2-M3-summary.json` | **无** |
| 19 | `results/V2-M3-body-graph-development-summary.json` | **无** |
| 20 | `results/V2-M3-body-graph-development-no-causal-likelihood.json` | **无** |
| 21 | `results/V2-M3-body-graph-holdout-summary.json` | **无**（已消费 holdout） |
| 22 | `results/V2-M3-body-graph-review-summary.json` | **无** |
| 23 | `results/V2-M4-summary.json` | **无** |
| 24 | `results/V2-audit-summary.json` | **无** |
| 25 | `results/V2-causal-sufficiency-summary.json` | **无** |
| 26 | `results/V2-diagnostic-ceiling-summary.json` | **无** |
| 27 | `results/V2-identifiability-summary.json` | **无** |

复现该清单：

```bash
uv run python - <<'PY'
import json, subprocess
from pathlib import Path
for f in subprocess.run(["git","ls-files","*.json"],capture_output=True,text=True).stdout.split():
    try:
        payload = json.loads(Path(f).read_text())
    except Exception:
        continue
    provenance = payload.get("provenance") if isinstance(payload, dict) else None
    if isinstance(provenance, dict) and provenance.get("git_dirty") is True:
        print(f, provenance.get("git_commit"))
PY
```

## 对后续工作的约束

1. **不得**据这些产物主张"可复现"。引用它们时应同时引用本文件。
2. 上表第 16、21 行是**已消费的一次性 holdout**，其限制不可补救。
3. 新产物必须在干净工作树上生成。`capture_provenance` 已记录 `git_dirty`，
   但**没有任何校验方会因为它为 true 而拒绝**——这是 G6 的一部分，
   见 [待办清单](../review/OPEN_ITEMS.md) 的架构级缺口表。
