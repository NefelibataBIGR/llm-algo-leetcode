# 24. SGLang RadixAttention | SGLang 基数注意力
**难度：** Hard | **环境：** CPU-first | **标签：** `推理优化`, `KV Cache`, `RadixAttention` | **目标人群：** 推理优化学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/24_SGLang_RadixAttention.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

真实推理服务里，请求往往不是彼此独立的：多轮对话会反复带上历史上下文，Agent 和工具调用也会共享很长的 system prompt。问题在于，如果每个请求都重新计算这些公共前缀，KV Cache 会被重复占用，首 token 延迟也会被拖高。

RadixAttention 要解决的就是前缀复用问题：把已经算过的 prompt 组织成一棵可检索的前缀树，新请求到来时先找最长公共前缀，命中的部分直接复用，没命中的后缀再交给模型计算。本节用一个简化的 Python 结构模拟这件事，重点看清“缓存命中长度”如何转化为少算多少 token。

**关键词：** `RadixAttention`, `prefix tree`, `multi-turn`

---
## 前置阅读

- [22. vLLM PagedAttention | vLLM 分页注意力](./22_vLLM_PagedAttention.md)
- [21. Decoding Strategies | 解码策略](./21_Decoding_Strategies.md)
- [P1: 11. KV Cache and Memory Growth | KV Cache 与显存增长](../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)

## 相关阅读

- [34. Prefix Caching and Chunked Prefill | 前缀缓存与分块预填充](./34_Prefix_Caching_and_Chunked_Prefill.md)
- [37. KV Cache Scheduling | KV Cache 调度](./37_KV_Cache_Scheduling.md)
- [69. Prefix Caching Benchmark | 前缀缓存基准项目](./69_Prefix_Caching_Benchmark.md)

---

### Step 1: 核心机制对比

> **vLLM PagedAttention：线性的分页存储**
> vLLM 就像操作系统的虚拟内存。每个请求有一个页表，它能很好地解决碎片，但请求与请求之间的页表是隔离的。就算两个请求的 Prompt 一模一样，它们也会各占一份显存，各算一次。

> **SGLang RadixAttention：基于基数树的共享路由**
> 系统维护了一棵全局的树。树的每一条边代表一段 Token 序列，节点里存着这段序列对应的 KV Cache 物理块指针。
> 当新请求到来时，SGLang 会用它的 Prompt 去这棵树里做**最长前缀匹配 (Longest Prefix Match)**。
> 匹配到的部分直接拿来用，没匹配到的部分再去计算并作为新分支挂在树上。
> **这里的核心不是“树有多复杂”，而是“共享前缀只存一份，后续请求沿着同一条路径复用同一份 KV Cache”。**

> **最长前缀匹配公式**
> 对于一个新的请求前缀 `prompt_tokens`，在所有已缓存路径中找到共享前缀长度最大的那一条：
> $$H = \max_j \operatorname{lcp}(prompt\_tokens, cached\_path_j)$$
> 其中 `H` 就是可以直接复用的 Hit Length。
> `H` 越大，说明当前请求命中的公共前缀越长，可直接复用的 KV Cache 越多，后续重计算越少。
> 如果 `H > 0`，说明前 `H` 个 token 的 KV Cache 可以直接复用；如果 `H = 0`，说明没有命中任何缓存路径。

> **为什么它适合多轮对话？**
> 因为多轮对话里，不同请求往往共享很长的 System Prompt 或历史上下文。Radix Tree 会把这些公共前缀只存一份，所有请求都能沿着同一条前缀路径复用 KV Cache，而不是像按请求隔离的页表那样重复保存。

![RadixAttention 前缀树图](/02_PyTorch_Algorithms/24_radix_attention_tree.svg)

### Step 2: 动手实战 —— 模拟 Radix Tree 前缀匹配

为了让你深刻理解 SGLang 的调度思想，我们将用 Python 原生数据结构，亲手模拟一个非常简化的 Radix Tree 路由管理器。
代码要做的事情很直接：遍历树中已有的缓存路径，计算和当前 `prompt_tokens` 的共享前缀长度，最后返回可以省去的重计算长度（Hit Length）。
这一步本质上不是在做“数值计算”，而是在做“前缀索引”：先找到能复用的最长公共前缀，再把后面的新 token 留给模型重新计算。**换句话说，代码返回的不是一个普通长度，而是“这段请求可以省掉多少 KV Cache 计算”；而 `split_prompt` 则把这段命中长度真正拆成“可复用前缀 + 待重算后缀”。**

**要求**：完成 `match_prefix` 函数，在全局 KV 树中寻找当前请求的最长前缀，返回可以省去的重计算长度（Hit Length）。
### Step 3: 机制边界

RadixAttention 和 vLLM PagedAttention 的核心差异不在“有没有 cache”，而在“cache 的组织方式”。前者强调共享前缀和最长前缀匹配，后者强调分页块表和按块分配。把边界说清楚，后面的代码就不会把“前缀复用”误写成“分页管理”。
### Step 4: 动手实战

完成 `match_prefix`、`split_prompt` 和最长公共前缀匹配后，确认三件事：最长命中长度是否正确、可复用前缀是否正确拆出、没有命中的 prompt 是否能回退到完整重算。
### 提示

- `match_prefix` 只允许从 prompt 开头连续命中。
- `split_prompt` 要把命中部分和未命中部分明确拆开。
- Radix Tree 这里是教学简化版，重点看最长前缀匹配逻辑，不要被树结构本身带偏。
### 测试

运行下面的测试单元，确认最长前缀命中、拆分和回退逻辑都正确。

```python
import torch
```


```python
class TreeNode:
    def __init__(self, key_tokens):
        self.key_tokens = key_tokens  # 这条边上的 Token 序列 (如 [101, 532, 789])
        self.children = []            # 子节点列表
        self.kv_cache_ptr = None      # 模拟指向物理 KV Cache 的指针

class SimpleRadixCache:
    def __init__(self):
        # 根节点是空的
        self.root = TreeNode([])
        
    def insert(self, tokens):
        """简单模拟向基数树中插入完整的请求。"""
        # 简化的插入逻辑（不涉及分裂裂变，仅为演示层级添加）
        # 假设当前系统只有一个 System Prompt，我们将其挂在根节点下
        node = TreeNode(tokens)
        self.root.children.append(node)
        
    def _lcp_len(self, cached_tokens, prompt_tokens):
        """计算两段 token 序列的最长公共前缀长度。"""
        match_len = 0
        # ==========================================
        # TODO 1: 逐个 token 计算最长公共前缀长度
        # 提示: 遇到不相等时立刻停止
        # ==========================================
        # i = ???
        # match_len = ???
        return match_len
        
    def match_prefix(self, prompt_tokens):
        """
        在现有树中，为新的 prompt_tokens 寻找最长的匹配前缀。
        如果前 N 个 token 完全一致，说明这 N 个 token 的 KV Cache 可以直接复用！
        """
        # 为了教学，我们只做单层子节点的暴力匹配
        best_match_len = 0
        
        # ==========================================
        # TODO 2: 遍历 self.root.children，更新最长匹配前缀长度
        # ==========================================
        # 提示: 逐个比较候选路径，取最大的命中长度
        return best_match_len

    def split_prompt(self, prompt_tokens):
        """把 prompt 拆成可复用前缀和需要重算的后缀。"""
        # ==========================================
        # TODO 3: 先找命中长度，再拆出前缀和后缀
        # 提示: hit_len 是可复用的前缀长度
        # ==========================================
        # hit_len = ???
        # hit_prefix = ???
        # miss_suffix = ???
        return hit_prefix, miss_suffix, hit_len
```


```python
# 测试你的实现
def test_radix_attention():
    try:
        cache = SimpleRadixCache()
        cache.insert([0, 1, 2, 3])
        cache.insert([0, 1, 2, 3, 4])
        cache.insert([9, 9, 9])

        # 1. 基础 LCP 检查
        assert cache._lcp_len([1, 2, 3], [1, 2, 4]) == 2, "LCP 计算失败！"
        assert cache._lcp_len([7, 8], [7, 8, 9, 10]) == 2, "完整前缀匹配失败！"
        print("✅ 最长公共前缀计算正确！")

        # 2. 多候选路径下，应该选择最长命中前缀
        match_len = cache.match_prefix([0, 1, 2, 3, 4, 5])
        assert match_len == 5, "匹配失败！应该命中最长的 5 个 token 前缀。"
        assert cache.match_prefix([7, 6, 5]) == 0, "错误匹配！不该匹配到任何东西。"
        print("✅ 多路径前缀命中选择正确！")

        # 3. 前缀拆分验证
        hit_prefix, miss_suffix, hit_len = cache.split_prompt([0, 1, 2, 3, 4, 5])
        assert hit_len == 5, "Hit Length 计算错误！"
        assert hit_prefix == [0, 1, 2, 3, 4], "可复用前缀拆分错误！"
        assert miss_suffix == [5], "待重算后缀拆分错误！"

        hit_prefix2, miss_suffix2, hit_len2 = cache.split_prompt([7, 6, 5])
        assert hit_len2 == 0, "无命中时 Hit Length 应为 0！"
        assert hit_prefix2 == [], "无命中时前缀应为空！"
        assert miss_suffix2 == [7, 6, 5], "无命中时后缀应保持原样！"
        print("✅ 前缀拆分与回退逻辑正确！")

        print("\n 所有测试通过！这正是 SGLang 让大模型推理首字响应飞升 10 倍的底层秘密！")

    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError, RuntimeError) as e:
        if isinstance(e, AttributeError):
            print("代码未完成，无法找到必要的属性")
        elif isinstance(e, NameError):
            print("代码可能未完成，导致了变量未定义")
        elif isinstance(e, TypeError):
            print("代码可能未完成，导致了操作错误")
        elif isinstance(e, ValueError):
            print("代码可能未完成，导致了张量维度错误")
        elif isinstance(e, AssertionError):
            print("代码可能未完成，导致了断言失败")
        elif isinstance(e, RuntimeError):
            print("代码可能未完成，导致了运行时错误")
        else:
            print("代码可能未完成，导致了断言失败")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except Exception as e:
        print(f"❌ 发生未知异常: {e}")
        raise


test_radix_attention()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---
## 参考代码与解析

### 代码

```python
class TreeNode:
    def __init__(self, key_tokens):
        self.key_tokens = key_tokens
        self.children = []
        self.kv_cache_ptr = None

class SimpleRadixCache:
    def __init__(self):
        self.root = TreeNode([])
        
    def insert(self, tokens):
        """简单模拟向基数树中插入完整的请求。"""
        node = TreeNode(tokens)
        self.root.children.append(node)
        
    def _lcp_len(self, cached_tokens, prompt_tokens):
        # TODO 1: 逐个 token 计算最长公共前缀长度
        match_len = 0
        while match_len < len(cached_tokens) and match_len < len(prompt_tokens):
            if cached_tokens[match_len] == prompt_tokens[match_len]:
                match_len += 1
            else:
                break
        return match_len
        
    def match_prefix(self, prompt_tokens):
        """
        在现有树中，为新的 prompt_tokens 寻找最长的匹配前缀。
        """
        best_match_len = 0
        
        # TODO 2: 遍历 self.root.children，更新最长匹配前缀长度
        for child in self.root.children:
            match_len = self._lcp_len(child.key_tokens, prompt_tokens)
            if match_len > best_match_len:
                best_match_len = match_len
        return best_match_len

    def split_prompt(self, prompt_tokens):
        """把 prompt 拆成可复用前缀和需要重算的后缀。"""
        # TODO 3: 先找命中长度，再拆出前缀和后缀
        hit_len = self.match_prefix(prompt_tokens)
        hit_prefix = prompt_tokens[:hit_len]
        miss_suffix = prompt_tokens[hit_len:]
        return hit_prefix, miss_suffix, hit_len
```

### 解析

**1. TODO 1（逐个 token 计算最长公共前缀长度）**
- `match_len` 的本质就是两个 token 序列的最长公共前缀长度。
- 用 `while` 逐位比较，遇到不相等就立即停止。
- 两个边界都要检查：缓存路径是否结束、当前 prompt 是否结束。

**2. TODO 2（遍历所有缓存路径，更新最长命中长度）**
- 先对树中的每个 child 计算局部前缀长度。
- 再在所有候选里取最大值，得到最终的 `best_match_len`。
- 这个值就是可以直接复用的 KV Cache 长度，也就是 Hit Length。
- 从工程上看，Hit Length 不是“命中一个 token”，而是“命中一段可共享的前缀缓存”。

**3. TODO 3（拆出可复用前缀与待重算后缀）**
- 先调用 `match_prefix` 得到 `hit_len`。
- 再把 `prompt_tokens` 切成 `hit_prefix` 和 `miss_suffix`。
- 这一步把“命中长度”真正变成“复用前缀 + 新增后缀”的工程操作。

**4. 进阶思考**
- 多轮对话和系统提示词通常有很长的公共前缀。
- Radix Tree 把这些公共前缀只保存一次，多个请求共享同一份缓存。
- 这比按请求隔离的线性页表更适合前缀高度重复的推理场景。
- 在工程上，它能显著降低首字响应时间并减少重复计算。
