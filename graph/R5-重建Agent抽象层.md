# R5 · 重建 Agent 抽象层

> 📍 **深入篇 · 路线 B(图运行时)** —— 本章原为两层教程的第 6 章,现编号 R5;章内小节号(6.x)保持原状。文中「第 0 章」「第 1 章」指主线章节;本篇代码使用 `code/tinygraph` + `code/tinyagent`(与主线 `tinycore` 概念同构,仅 usage 等生产字段有出入)。从主线来的读者建议先读 [第 10 章·桥](../harness/10-第二条路线-图运行时.md) 与 [R0-深入篇导读](R0-深入篇导读.md)。


> 前五章造出了通用运行时 `tinygraph`。但它太底层——每写一个 Agent 都要手动搭"model 节点 ↔ tools 节点"的图。本章在运行时之上建造 `tinyagent`(对应 LangChain 1.0):一个 `create_agent(model, tools)` 工厂,自动搭好那张标准 Agent 图,并直接继承前五章的全部能力(检查点、中断、流式)。本章示例对接真实模型,需配好 `TINYAGENT_MODEL` 与 key。

## 6.1 为什么需要上面这一层

回忆R2 章示例 B,我们手搭了一张 Agent 图:

```python
g = StateGraph({"messages": add_reducer})
g.add_node("model", call_model)
g.add_node("tools", call_tools)
g.add_edge(START, "model")
g.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
g.add_edge("tools", "model")
```

这张图的结构对**几乎所有 Agent 都一样**——model 节点调模型,tools 节点执行工具,条件边判断"还要不要继续"。每写一个 Agent 都重敲这十行,既啰嗦又容易错。

所以需要一个工厂把它封装起来:

```python
agent = create_agent(model=model, tools=[get_weather], system_prompt="...")
result = agent.invoke({"messages": [HumanMessage("北京天气?")]})
```

一行搞定。这就是 LangChain 1.0 的 `create_agent` 做的事——它是 LangGraph 之上的"Agent 工厂"。

这里有个关键认知:`tinyagent` 不是一个新框架,它**只是 `tinygraph` 的一个使用模式**。它没有自己的执行引擎、没有自己的持久化,全部复用底层。这正是第 0 章"两层架构"的落地:底层管"怎么可靠地跑图"(通用),上层管"Agent 长什么样"(业务)。

(对照真实框架:`create_agent` 是 LangChain 1.0 标准建 Agent 方式,取代了旧的 `langgraph.prebuilt.create_react_agent`。官方文档明确它"is built on LangGraph",所以自动获得 persistence / streaming / human-in-the-loop / time-travel——和我们这里一致。)

## 6.2 state/messages 契约

`create_agent` 定义了一个统一的接口约定,叫 **state/messages 契约**:

- **输入**:一个 dict,核心是 `messages` 键——`{"messages": [HumanMessage("...")]}`。
- **输出**:也是 dict,核心仍是 `messages`(完整对话历史),需要时还有 `structured_response`(结构化输出,见 6.5)。

为什么以 `messages` 为中心?因为 Agent 的本质就是"在消息历史上反复迭代"。状态 schema 只有一个 channel:

```python
schema = {"messages": add_reducer}    # 用累加 reducer,新消息追加而非覆盖
```

`add_reducer`(对应 LangGraph 的 `add_messages`)保证每个节点返回的 `{"messages": [新消息]}` 都**追加**到历史末尾。这是整个 Agent 能"记住前文"的根基。

(对照真实框架:对应 LangChain 的 `AgentState`,核心字段就是 `messages`(带 `add_messages` reducer)。真实的 `AgentState` 还有 `remaining_steps` 等;invoke 的返回也是 `{"messages": [...], "structured_response": ...}`。)

## 6.3 create_agent 内部:自动搭那张图

`create_agent` 的核心就是把R2 章那张图自动搭出来。看 `tinyagent/agent.py`:

```python
def create_agent(model, tools=(), system_prompt=None, middleware=(), checkpointer=None):
    tools_by_name = {t.name: t for t in tools}
    model = model.bind_tools(list(tools))
    schema = {"messages": add_reducer}

    def call_model(state):
        messages = list(state["messages"])
        # 自动注入 system_prompt(若历史里还没有)
        if system_prompt and not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_prompt)] + messages
        ai = model.invoke(messages)
        return {"messages": [ai]}

    def call_tools(state):
        last = state["messages"][-1]
        out = []
        for call in last.tool_calls:
            result = tools_by_name[call.name].invoke(call.args)
            out.append(ToolMessage(content=str(result),
                                   tool_call_id=call.id, name=call.name))
        return {"messages": out}

    def should_continue(state):
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    g = StateGraph(schema)
    g.add_node("model", call_model)
    g.add_node("tools", call_tools)
    g.add_edge(START, "model")
    g.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "model")
    return Agent(g.compile(checkpointer=checkpointer), list(middleware))
```

(上面省略了中间件钩子,R6 章细讲;这里先看主干。)

几个值得注意的设计:

- **`system_prompt` 自动注入**:只在历史里还没有 `SystemMessage` 时插到最前面,避免多轮对话里重复插。
- **`tools` 节点遍历所有 `tool_calls`**:模型一次请求多个工具,这里全部执行,各自包成 `ToolMessage` 返回。
- **返回的 `Agent` 对象**只是对底层 `CompiledGraph` 的薄包装,转发 `invoke`/`stream`/`get_state`。

最后那个 `Agent` 包装类有多薄,看一眼就知道——它几乎只是把调用透传给底层编译好的图(`before_agent`/`after_agent` 是给R6 章中间件留的钩子,现在是空操作):

```python
class Agent:
    """create_agent 的返回物:包装底层 CompiledGraph,提供 invoke/stream。"""
    def __init__(self, compiled, middleware):
        self._compiled = compiled
        self._middleware = middleware

    def invoke(self, input, config=None):
        for mw in self._middleware:
            input = mw.before_agent(input) or input          # R6 章钩子
        result = self._compiled.invoke(input, config)
        for mw in reversed(self._middleware):
            result = mw.after_agent(result) or result        # R6 章钩子
        return result

    def stream(self, input, config=None, stream_mode="updates"):
        return self._compiled.stream(input, config, stream_mode=stream_mode)

    def get_state(self, config):
        return self._compiled.get_state(config)
```

它没有自己的执行逻辑——`invoke` 直接转发给 `self._compiled`(第 3-5 章那个跑图的引擎),`get_state` 也是。这从代码上印证了 6.1 的话:`tinyagent` 没有自己的运行时,它就是 `tinygraph` 的一个使用模式。

把这段和第 1 章那个 `while` 循环对照:逻辑完全一致(调模型 → 判断 → 执行工具 → 回到调模型),但这次它是一张跑在 `tinygraph` 上的图。于是它**自动**拥有了第 1 章那个循环求之不得的一切——下一节验证。

## 6.4 直接继承的能力:多轮记忆

`create_agent` 自己**一行持久化代码都没写**,但只要传个 `checkpointer`,它就有了多轮对话记忆。看示例 B:

```python
agent = create_agent(model=model, tools=[], checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "chat-1"}}

r1 = agent.invoke({"messages": [HumanMessage("你好,我叫小明")]}, config)
# → "你好,小明!很高兴认识你。"

r2 = agent.invoke({"messages": [HumanMessage("我叫什么名字?")]}, config)
# → "你叫小明。"   ← 第二轮记得第一轮的内容
```

第二轮为什么能记得"小明"?这背后是R3 章检查点机制的一个延伸,值得讲清楚,因为它是"对话记忆"的真正实现方式。

### 引擎的"接着对话"逻辑

当你用**同一个 `thread_id`**、传入一个**新的 messages dict** 时,引擎不会从零开始。它会:

```python
# pregel.py(节选)
elif isinstance(input, dict) and self.checkpointer is not None:
    prior = self.checkpointer.get(thread_id)
    if prior is not None and not prior.next_nodes:   # 上轮已执行完
        prior_cp = prior

# ...
if prior_cp is not None:
    channels = self._channels_from_checkpoint(prior_cp)   # 载入上轮结束状态
    ...
init = {}; self._collect_writes(input, init)
self._commit(channels, init)    # 把本轮新消息经 reducer【追加】进旧历史
```

三步:**载入上轮结束时的状态 → 把本轮新输入经 reducer 追加 → 从 START 重新跑**。因为 `messages` 用的是 `add_reducer`,本轮的 `HumanMessage("我叫什么名字?")` 被**追加**到 `[..., "我叫小明", "你好小明..."]` 之后,而不是覆盖掉。模型于是看到完整历史,自然记得"小明"。

这就是"对话记忆"的全部:记忆 = 检查点里持续累积的 `messages` channel。没有单独的"记忆模块",记忆就是状态本身的持久化。换个 `thread_id`,就是另一段完全独立、互不串味的对话。

### 验证一下:历史真的在累积吗

上面是机制,跑一下就看得见。为了不依赖联网模型(也让这个验证可复现),我们塞一个桩模型——它什么也不"想",只把"自己这一轮看到了多少条历史"回显出来:

```python
from tinygraph.models import BaseChatModel

class EchoModel(BaseChatModel):
    def bind_tools(self, tools): self._tools = list(tools); return self
    def invoke(self, messages):
        human = [m for m in messages if m.role == "user"]
        return AIMessage(content=f"我看到了 {len(messages)} 条历史,最后一句是「{human[-1].content}」")

agent = create_agent(model=EchoModel(), tools=[], checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "chat-1"}}

r1 = agent.invoke({"messages": [HumanMessage("第一句话")]}, config)
print("第1轮模型看到:", r1["messages"][-1].content)

r2 = agent.invoke({"messages": [HumanMessage("第二句话")]}, config)   # 同一 thread_id
print("第2轮模型看到:", r2["messages"][-1].content)
print("第2轮完整历史:", [f"{m.role}:{m.content[:12]}" for m in r2["messages"]])

# 换个 thread_id —— 一段全新对话,不含上面任何内容
r3 = agent.invoke({"messages": [HumanMessage("我是新对话")]},
                  {"configurable": {"thread_id": "chat-2"}})
print("新 thread 历史长度:", len(r3["messages"]))
```

输出:

```
第1轮模型看到: 我看到了 1 条历史,最后一句是「第一句话」
第2轮模型看到: 我看到了 3 条历史,最后一句是「第二句话」
第2轮完整历史: ['user:第一句话', 'assistant:我看到了 1 条历史,最', 'user:第二句话', 'assistant:我看到了 3 条历史,最']
新 thread 历史长度: 2
```

三处都印证了上面的机制:**第一轮**模型只看到 1 条(自己的输入);**第二轮**模型看到了 3 条——第一轮的 user + assistant 加上本轮的 user,正是"载入旧状态 + reducer 追加新输入"的结果,所以它"记得"前文不是因为有记忆模块,而是因为旧消息真的还在 channel 里;**换 thread_id** 后历史长度归 2(只有本轮),证明不同 thread 的状态彻底隔离。`EchoModel` 一个字都没"记",记忆完全来自 `tinygraph` 那层的检查点——这就是"create_agent 不写一行持久化代码却有多轮记忆"的真相。

(把 `EchoModel` 换成 `get_model()` 读的真实模型,行为一字不变,只是回答从"回显历史长度"变成真正理解上文——示例 B 就是这么跑的。)

这里能看出两层架构的复利:我们在 `tinygraph` R3 章实现的检查点,在 `tinyagent` 这层**不费一行代码**就变成了"多轮对话记忆"这个 Agent 特性。底层的通用能力,在上层自动具象成业务价值。同理,R4 章的 `interrupt` 在这层就是"工具执行前人工审批",R2 章的流式就是"实时显示 Agent 进度"。

把这种复利画成一张映射表,你会看清 `tinyagent` 为什么"几乎没写新代码却功能齐全"——它做的全部事情,就是把底层的通用机制**翻译**成 Agent 场景下的说法:

```
   tinygraph 通用能力(第3-5章)            tinyagent 业务特性(本章及R6章)
   ───────────────────────────         ───────────────────────────────
   检查点:channel 值随超步累积    ─────▶  多轮对话记忆(同 thread_id 续接 messages)
   thread_id 隔离执行线          ─────▶  会话隔离(不同用户/对话互不串味)
   interrupt() 存档+退出+重跑     ─────▶  工具执行前人工审批(HITL)
   每超步 yield(步骤间缝隙)       ─────▶  实时显示 Agent 进度("正在查天气…")
   时间旅行(不可变检查点链)        ─────▶  对话倒带、改一句重答
   BSP 超步内并行 + 确定性合并     ─────▶  一次多个工具调用并行执行
   ───────────────────────────         ───────────────────────────────
        "怎么可靠地跑一张图"(通用)         "图里跑的是个会调工具的 Agent"(业务)
```

回看第 0 章那个边界判据(这能力跟跑的是不是 Agent 有关吗?):左列每一条都**与 Agent 无关**——任何带状态的图都用得上,所以它们沉在底层;右列每一条都**只在"图跑的是 Agent"时才有意义**,所以浮在上层。`create_agent` 的全部价值,就是搭好那张特定的"model↔tools"图,从而把左列一次性兑换成右列。底层做厚一分,上层就免费长出一片。

## 6.5 结构化输出:在主循环内完成

一个常见需求:让 Agent 最终返回**结构化数据**(如一个 JSON 对象 `{"city": ..., "temp": ...}`),而不是一段自然语言。

旧做法是 Agent 跑完后**再加一次 LLM 调用**把文本转成结构化——多一次往返、多花钱。LangChain 1.0 的改进是:**在主循环内完成结构化输出**。具体做法是把"返回结构化结果"也建模成一个特殊工具(`ToolStrategy`):模型在每一轮可以选择"调普通工具"或"调这个结构化输出工具"。当它调后者时,Agent 就知道"可以结束了,且结果是结构化的"。

我们的简化框架不展开实现(它需要给条件边加一个"收到结构化工具调用就结束并解析"的分支),但你已经具备实现它的全部零件:加一个名为 `__response__` 的特殊工具、在 `should_continue` 里判断"最后一条 AI 消息是不是调了它"、是则解析其 args 存进 `structured_response` 并走向 END。这是个很好的练习。

把这个练习的关键决策摊开,你会发现它纯粹是"复用已有零件"的拼装,没有任何新机制:

```python
# 思路示意,讲清机制(非框架现有代码):
# ① 把你要的输出 schema 注册成一个特殊工具的参数,绑给模型
#    schema = {"city": str, "temp": int} → 生成一个名为 __response__ 的工具
#    模型现在 "可调的工具" = 普通工具 + 这个 __response__
# ② state schema 多一个 channel(它不是 messages,用 last_value 覆盖即可)
schema = {"messages": add_reducer, "structured_response": last_value}
# ③ should_continue 多判一种情况:
def should_continue(state):
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None)
    if not calls:
        return END
    if calls[0].name == "__response__":      # ← 模型选择了 "结构化收尾"
        return "respond"                      #    走解析分支,而非普通 tools
    return "tools"
# ④ respond 节点:解析那次调用的 args,存进 structured_response,走 END
def respond(state):
    last = state["messages"][-1]
    return {"structured_response": last.tool_calls[0].args}
```

三个设计点值得品味:

- **为什么省掉了一次 LLM 调用?** 旧做法是"Agent 跑完拿到自然语言 → 再调一次模型把它转成 JSON"。新做法里,"产出结构化结果"和"调普通工具"是模型在**同一轮**里二选一的动作——它决定收尾时,直接以"调 `__response__` 工具"的形式**一次性**给出结构化数据,没有额外往返。
- **为什么用工具调用来表达而不是另写 prompt?** 因为原生工具调用的 args 是 API 保证的结构化字段(回看第 0 章),比"求模型吐 JSON 再正则解析"可靠得多——结构化输出复用了工具调用那套"结构由 API 保证"的好处。
- **解析失败怎么办?** 真实 `ToolStrategy(handle_errors=...)` 会在 schema 校验失败时,把错误作为 `ToolMessage` 喂回模型让它重填(和第 1 章"工具错误喂回模型"同一个套路),而不是直接崩。

整个结构化输出没引入任何新概念,全是前几章零件的重新组合——这正是好抽象的标志。

(对照真实框架:对应 `create_agent(response_format=ToolStrategy(SchemaModel))`。官方文档强调它"is now generated in the main loop instead of requiring an additional LLM call",正是为了省掉那次额外调用。`handle_errors` 参数处理解析失败和模型一次发多个结构化调用的情况。)

## 6.6 为下一章埋的伏笔:中间件钩子

你可能注意到 `create_agent` 有个 `middleware` 参数,而 6.3 的代码我故意省略了钩子。其实完整版的 `call_model` / `call_tools` 在关键位置都留了插入点:

```python
# 完整版 call_model 里(tinyagent/agent.py)
for mw in mws:
    messages = mw.before_model(messages) or messages      # 调模型【前】
handler = base_call
for mw in reversed(mws):
    handler = _wrap_model(mw, handler)                     # 【包裹】模型调用
ai = handler(messages)
for mw in mws:
    ai = mw.after_model(ai) or ai                          # 调模型【后】
```

这些 `before_model` / `wrap_model_call` / `after_model` 就是**中间件钩子**。现在它们只是空操作(基类默认什么都不做),所以本章示例感觉不到它们存在。下一章我们就用这些钩子实现摘要压缩、PII 脱敏、人工审批——现代 Agent 工程(上下文工程)的核心全在这里。

## 6.7 本章小结

- `create_agent` 是 `tinygraph` 之上的 **Agent 工厂**:自动搭好"model ↔ tools"那张标准图。它不是新框架,而是底层运行时的一个使用模式。
- **state/messages 契约**:输入 `{"messages": [...]}`,输出同样以 `messages` 为中心;`messages` channel 用累加 reducer(`add_messages`)。
- `create_agent` 自己不写持久化,却**直接继承**了检查点(多轮记忆)、中断(工具审批)、流式——这是两层架构的复利。
- **对话记忆 = 检查点里持续累积的 `messages` channel**:同一 `thread_id` 续接历史靠"载入旧状态 + reducer 追加新输入";换 thread_id 即换一段独立对话。
- **结构化输出**在主循环内完成(建模成特殊工具),省掉额外的 LLM 调用。
- `create_agent` 在关键位置预留了**中间件钩子**(before/after_model、wrap_*),下一章兑现。

---

← [R4-流式与人在回路](R4-流式与人在回路.md) | 下一章 → [R6-中间件系统](R6-中间件系统.md)
