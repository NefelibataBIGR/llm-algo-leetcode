# 编译与图优化深入阅读

## 主故事线

这条专题最适合按下面这条顺序来读：

`graph looks clean -> cost still high -> lowering changes form -> backend constraints appear -> benchmark decides whether the compiler story was actually useful`

重点不是记住某个编译名词，而是理解：为什么一张“语义正确”的图，离“执行高效”的系统还隔着很多层。

## 1. 先承认图和执行不是一回事

很多误解都从这里开始：

- 图结构很规整
- 算子数量看起来不多
- 依赖图也没有明显异常

但 benchmark 仍然可能很差。  
这意味着“图级合理”并不自动推出“执行级高效”。

## 2. 先看图级，再看 lowering

进入 `02` 后，先判断：

- 哪些中间张量真的在拖成本
- 哪些 fuse 看起来合理但实际上会引入别的约束

然后进入 `03`，再看：

- lowering 后的形式是否真的更适合执行
- schedule 是否把原来的图级判断贯彻下去

## 3. 再看 backend 约束

进入 `04` 和 `05` 时，要把注意力放到：

- execution model 是否允许这种 fuse / layout / schedule
- 不同 backend 为什么会把同一张图导向不同方案
- 成本模型为什么会让“同样正确”的 lowering 产生不同收益

## 4. 最后回到 benchmark

进入 `06` 后，要回答：

- backend 假设是否真的改善了最终系统指标
- benchmark 结论能否支撑项目里的 adopt / keep / switch

如果不能，这条编译故事还没有闭环。

## 阅读建议

1. 第一次读，按 `01 -> 02 -> 03 -> 04 -> 05 -> 06`。
2. 已经懂 graph optimization，但不懂 backend 差异，先读 `04 -> 05`。
3. 已经有 benchmark 结果，但说不清为什么，先读 `06` 再回查前面几页。
