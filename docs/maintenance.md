# 维护与发布手册

只看三件事：怎么同步、怎么验证、脚本各管哪一层。正文模板规则见 [template_guidelines.md](./template_guidelines.md)。

## 源文件优先

- 先改源文件，再同步 `docs/` 镜像，不能直接手改 `docs/` 作为最终提交。
- 正文类改动统一从源 notebook 或源 markdown 出发，完成后再运行对应同步脚本。
- 如果镜像页和源文件出现不一致，以源文件为准，后续同步会覆盖镜像。

## 脚本分层

| 层 | 脚本 | 作用 |
|---|---|---|
| `verify` | `verify.py` | 统一验证入口 |
| `convert` | `tools/convert_notebook.py` | 正文镜像主链路 |
| `sync` | `tools/sync_docs_index.py`、`tools/sync_docs_navigation.py` | 首页 / 导学页 / 组页同步 |
| `check` | `tools/check_source_docs_mirror.py`、`tools/check_chapter_links.py` | 镜像和链接检查 |
| `test` | `tools/test_chapter0_1_notebooks.py`、`tools/test_notebook_answers.py` | Notebook 校验 |
| `audit` | `tools/audit_chapter0_1_notebooks.py` | Part 0 / Part 1 执行验证 + 结构审计 |
| `migration` | `tools/md_to_notebook.py` | markdown -> notebook 迁移辅助 |

`tools/convert_chapter0_1.py` 只保留 legacy 兼容。

## Part 0-4 维护分工

- `Part 00` 和 `Part 01` 一起作为前置知识层，重点是基础语言、张量、系统视角和性能边界。
- `Part 02` 是主干实现层，重点是 PyTorch 里的训练、推理、并行、量化和项目收口。
- `Part 02` 的项目建设按“核心项目 + 扩展项目 + 延伸方向”组织：`2.9` 是项目收口层，核心项目优先覆盖训练落地、推理选型和训练分析，扩展项目优先覆盖 profiling 闭环、并行基准和量化部署；`36-42` 则作为更细的延伸方向，继续补推理服务、cache、量化家族和通信 profiling。项目页的 TODO 仍保持 notebook-first 的统一结构，但职责从“补算法”转为“组织实验、输出对比和沉淀结论”。
- `Part 03` 是 Triton / kernel 过渡层，重点是把框架级实现继续下沉到高性能算子。
- `Part 04` 是 CUDA / 系统优化层，重点是继续向硬件、通信、调度和架构收口。
- 维护时可以按下面的验证分段理解：
  - `verify.py part0_1`：检查 `Part 00 / Part 01`
  - `verify.py part2`：检查 `Part 02`
  - `verify.py part3`：检查 `Part 03`
  - `verify.py part4`：检查 `Part 04`
- 横向专题主要横切 `Part 00 / Part 01 / Part 02`，后续若继续下探性能和实现，可以逐步接到 `Part 03 / Part 04`。

## 教程信息架构口径

后续写导学页、组导航页、专题页和 `Part 01` 正文导读时，统一使用下面四层概念，不再混写：

- `纵向主线`
  - 指教程按能力递进展开的默认学习顺序
  - 固定理解为：`Part 00 -> Part 01 -> Part 02 -> Part 03 -> Part 04`
  - 作用是回答“整个教程先学什么，后学什么”
- `学习路线`
  - 指面向任务目标的跨 Part 阅读路径
  - 当前固定为三条主路线：`训练微调路线`、`推理优化路线`、`显存优化路线`
  - 作用是回答“为了做成什么任务，应该重点串哪些页”
- `横向专题`
  - 指跨多个 Part 反复出现的方法轴或技术轴
  - 当前固定包括：`量化与压缩`、`通信与并行`、`Profiling`、`后训练与对齐`
  - 作用是回答“同一类技术问题分散在不同 Part 时，应该怎么按主题重组来看”
- `基础支撑专题`
  - 指被多条学习路线共同依赖的底层知识底座
  - 当前固定包括：`反向传播与训练机制`、`大模型架构`、`编译与图优化`
  - 作用是回答“哪些基础认知会在多条路线里被反复依赖”

边界约束：

- 不把技术领域直接写成 `学习路线`
- 不把 `基础支撑专题` 混写成 `横向专题`
- 不把组导航页或专题页写成目录复述
- `学习路线` 强调目标和顺序，`横向专题` 强调主题和重组，`基础支撑专题` 强调底座和共性依赖

## Part 01 导读写法口径

`Part 01` 的 `本节导读` 第二段优先按下面顺序写：

1. 这一节在 `纵向主线` 里属于哪一类基础页。
2. 它优先服务哪条 `学习路线`。
3. 学完这里后面更顺地进入哪些具体小节、项目页或判断任务。
4. 如果这里没学明白，后面通常会卡在哪些实现、判断或项目验证上。
5. 它同时归属于哪个 `横向专题` 或 `基础支撑专题`。

推荐模板：

```text
这一节在整个教程的纵向主线里属于 Part01 的基础页，主要为「某条学习路线」提供前置支撑。学完这里，后面可以更顺地进入「A / B / C 小节或项目页」；如果这里没学明白，通常会卡在「哪些判断、哪些实现或哪些项目验证」上。按专题归类，它同时属于「某个横向专题」或「某个基础支撑专题」。
```

补充约束：

- 优先写具体后续页，不只写“服务训练”或“服务推理”这种泛表述。
- 如果一页同时服务多条路线，只写主服务路线，再补一句次要关联，不要把三条路线全部堆上去。
- 如果一页更像底层机制页、很难强挂主路线，可以写“当前主要作为 Part01 基础支撑页”，再补具体后续去向。

## 日常流程

1. 先改 source。
2. 首页改动后跑 `python tools/sync_docs_index.py`。
3. 导学页 / 组页改动后跑 `python tools/sync_docs_navigation.py`。
4. 正文改动后跑 `python tools/convert_notebook.py`。
5. 最后跑 `cd docs && npm run docs:build`。

## 图片资产规则

图片资产后续统一按“先分级、再入正文”的原则处理，不再边写正文边临时插图。

- `Part02` 的正文图先看 [part02_visual_assets_audit.md](./part02_visual_assets_audit.md)。
- `topic_discussion` 的专题图先看 [topic_discussion_visual_assets_audit.md](./topic_discussion_visual_assets_audit.md)。

固定规则：

- 未经过审核的图片，不进入正文主叙事位置。
- 未审核图如果必须先保留，只能放在 `visual_assets` 页或附录型页面，不直接承担正文主解释职责。
- 图片审核至少要回答三件事：
  - 这张图是不是核心教学图、路线收束图，还是结构占位图
  - 这张图应该说明什么
  - 这张图不应该说明什么
- 图片进入正文前，至少完成：
  - 职责分级
  - 可读性初审
  - 是否需要减字 / 中文化 / 重画的判断

## Part02 图解格式收口

`Part02` 当前更大的问题不是某一张图本身，而是正文里长期混用了三种表达：

- 正式 `SVG`
- notebook 内的 `ASCII / text block`
- 尚未稳定模板化的 `Mermaid`

后续统一按下面的职责边界执行：

- `SVG`
  - 这是 `Part02` 正文正式主图的唯一默认格式
  - 只要一张图承担核心机制解释，就应进入 `SVG` 体系并先审计
- `ASCII / text block`
  - 只保留为局部辅助结构
  - 适合维度流向、短流程、图前骨架提示
  - 不再承担正文唯一主图职责
- `Mermaid`
  - 当前先冻结为“非默认正文主图格式”
  - 在没有统一模板、职责边界和维护结论前，不继续向 `Part02` 正文扩散

执行顺序固定为：

1. 先清职责边界，不先扩写新图。
2. 先盘点 `ASCII / text block`，看哪些只是辅助，哪些已经和 `SVG` 重复。
3. 先复核高价值 `SVG` 的可读性和信息密度。
4. 页面职责稳定后，再决定是否中文化或重画。

当前结论：

- `Part02` 这轮优先级是“统一图解体系”，不是“先把所有图翻成中文”。
- 没审过的图，不入正文；没定职责的格式，也不继续扩散进正文。

## 常用命令

```bash
python verify.py part0_1 --no-build
python verify.py part0_1_audit
python verify.py part2 --no-build
python verify.py part3 --no-build
python verify.py part4 --no-build
python verify.py all --no-build
python tools/sync_docs_index.py
python tools/sync_docs_navigation.py
python tools/convert_notebook.py
cd docs && npm run docs:build
```

## 测试脚本索引

| 层 | 脚本 | 作用 |
|---|---|---|
| `verify` | `verify.py` | 统一验证入口 |
| `convert` | `tools/convert_notebook.py` | 正文镜像主链路 |
| `sync` | `tools/sync_docs_index.py`、`tools/sync_docs_navigation.py` | 首页 / 导学页 / 组页同步 |
| `check` | `tools/check_source_docs_mirror.py`、`tools/check_chapter_links.py` | 镜像和链接检查 |
| `check` | `tools/check_docs_links.py`、`tools/check_math_formula_symbols.py`、`tools/check_part01_code_blocks.py` | docs 链接、公式与代码块检查 |
| `test` | `tools/test_chapter0_1_notebooks.py`、`tools/test_notebook_answers.py` | Notebook 校验 |
| `migration` | `tools/md_to_notebook.py` | markdown -> notebook 迁移辅助 |

## 验证模型对照

`Part 0 / Part 1` 和 `Part 2 / Part 3` 的 notebook 结构不同，不能用同一套脚本假设去验证。

| 范围 | 主脚本 | 默认验证逻辑 | 适用结构 |
|---|---|---|---|
| `Part 0 / Part 1` | `tools/test_chapter0_1_notebooks.py` | 顺序执行每个非空 `code cell`，只要执行过程中不抛异常就算通过 | 讲解型 / 逐代码块验证型 notebook |
| `Part 0 / Part 1` | `tools/audit_chapter0_1_notebooks.py` | 分开看 `codecell_run` 和 `structure_only`：前者检查代码块执行，后者检查 `本节导读 / 前置阅读 / 相关阅读`、链接数量、`cell id` | 验证收尾、结构审计、warning 归档 |
| `Part 2 / Part 3` | `tools/test_notebook_answers.py` | 按 `import -> 题目区 -> STOP HERE -> 参考代码与解析` 抽取代码，再分别验证题目区 / 答案区 | 练习型 / 题目区答案区双结构 notebook |

结论上要区分两类“通过”：

- `Part 0 / Part 1` 的“代码通过”主要表示：代码块能顺序运行且不报错。
- `Part 2 / Part 3` 的“代码通过”主要表示：题目区或答案区在既定抽取规则下能被正确提取并验证。
- 因此，`Part 0 / Part 1` 不应用 `Part 2` 的题目区 / 答案区脚本强套；如果需要收尾审计，应优先用 `tools/audit_chapter0_1_notebooks.py`。

## Part 0 / Part 1 固定口径

`Part 0 / Part 1` 后续统一按下面这套入口理解，不再临时拼脚本：

- 主入口：`python verify.py part0_1 --no-build`
  - 这是 `Part 0 / Part 1` 的标准验证命令
  - 职责是：镜像转换、source/docs 链接检查、逐 `code cell` 执行验证
- 补充审计：`python verify.py part0_1_audit`
  - 这是收尾审计命令，不替代主入口
  - 职责是把执行验证和结构检查拆开归档

`part0_1_audit` 下的两个 profile 固定解释为：

- `codecell_run`
  - 顺序执行 notebook 中每个非空 `code cell`
  - 目标是确认讲解型 notebook 的代码块在当前环境下能否连续运行且不报错
  - 适合定位“哪一页 / 哪一个 cell 执行失败”
- `structure_only`
  - 不执行代码
  - 只检查 `本节导读 / 前置阅读 / 相关阅读`、链接数量、`cell id`、基础 notebook 结构
  - 适合做页头收尾、warning 分类和结构回归检查

推荐理解方式：

- 日常验证只跑 `verify.py part0_1`
- 需要验证收尾、warning 归档、结构追责时，再补跑 `verify.py part0_1_audit`
- 不再直接用 `Part 2` 的 `tools/test_notebook_answers.py` 套 `Part 0 / Part 1`

## 推荐用法

```bash
python verify.py part0_1 --no-build
python verify.py part2 --no-build
python verify.py part3 --no-build
python verify.py all --no-build
python tools/check_math_formula_symbols.py
python tools/check_part01_code_blocks.py
python tools/audit_chapter0_1_notebooks.py --profile all
```

无 GPU 时，`verify.py` 会跳过 Part 2 / 3 的 GPU-only 答案验证，但仍保留转换、镜像和链接检查。单独排查时直接用底层脚本。
`tools/md_to_notebook.py` 仅用于历史迁移，不进入日常主流程。

## 说明

- `Part 0 / Part 1` 用 `tools/test_chapter0_1_notebooks.py`
- `Part 0 / Part 1` 如需把代码执行和结构/warning 分开看，用 `tools/audit_chapter0_1_notebooks.py`
- `Part 2 / Part 3` 用 `tools/test_notebook_answers.py`
- 先改源，再同步 `docs/`
- 导学页、组页、正文页分开同步
- `tools/convert_chapter0_1.py` 只保留兼容用途
- `tools/md_to_notebook.py` 只保留迁移辅助用途
