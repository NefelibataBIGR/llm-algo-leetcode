# 07 Visual Assets

## 页面目标

这页收口后训练与对齐专题的图册资产。第一阶段先固定“该有什么图、图用来回答什么问题”，后续再逐步补成正式 SVG。

## 图册顺序

建议按下面顺序补图：

1. `alignment_lifecycle`
- 从 `SFT model -> alignment gap -> method choice -> eval -> project decision`

![Post-Training Alignment Lifecycle](/topic_discussion/post_training_alignment/alignment_lifecycle.svg)

2. `rlhf_ppo_system_loop`
- policy / reference / reward / rollout / update 的完整链路

![RLHF and PPO System Loop](/topic_discussion/post_training_alignment/rlhf_ppo_system_loop.svg)

3. `dpo_pairwise_objective`
- chosen / rejected / reference 的关系图

![DPO Pairwise Objective](/topic_discussion/post_training_alignment/dpo_pairwise_objective.svg)

4. `grpo_groupwise_candidates`
- group candidates、relative comparison 和候选组评测

![GRPO Groupwise Candidates](/topic_discussion/post_training_alignment/grpo_groupwise_candidates.svg)

5. `preference_eval_matrix`
- win-rate / pairwise accuracy / judge score 的口径对照

![Preference Data and Evaluation Matrix](/topic_discussion/post_training_alignment/preference_eval_matrix.svg)

6. `project_decision_board`
- adopt / tune / reject 的项目决策图

![Post-Training Project Decision Board](/topic_discussion/post_training_alignment/project_decision_board.svg)

## 图的职责

这些图不追求花哨，重点是降低抽象感：

- `01` 负责总入口图
- `02 / 03 / 04` 负责方法区别
- `05` 负责评测口径统一
- `06` 负责项目收口

## 当前状态

第一批与第二批方法图已经补齐；当前图册已覆盖 `01-06` 的主要入口。
