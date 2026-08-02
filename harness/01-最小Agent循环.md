# 第 1 章 · 最小 Agent 循环

> 这一章用大约 100 行代码写出一个能对接真实大模型、能自主调用工具的 Agent。它由四个模块组成:消息、模型接口、工具、循环。请认真对待这一章——**这不是玩具,也不是通往"真正架构"的脚手架**:2026 年你每天在用的 Claude Code、Codex,内核就是这个循环。后面十章做的事,是给它配上工具、上下文、事件流和一层壳,而循环本身,今天就写完了。代码在 `code/tinycore/` 下;本章示例离线可跑,接真实模型需按第 0 章配好 `TINYAGENT_MODEL`。

## 1.1 四个模块,以及它们各自解决什么

第 0 章给 Agent 的定义是:让模型在循环里反复决策,并调用工具改变世界。翻译成代码需要四个东西,缺一不可:

1. **消息(Message)**。模型的输入输出不是裸字符串,而是带角色的结构化对象——谁说的(系统/用户/助手/工具),一条助手消息里是纯文本还是夹着工具调用请求,工具结果对应哪一次调用。消息是 Agent 的记忆载体。
2. **模型接口(ChatModel)**。一个与厂商无关的统一接口:输入一串消息,输出一条 AI 消息。底层接 Anthropic、OpenAI、DeepSeek 还是本地模型,上层代码完全不变。
3. **工具(Tool)**。把一个普通 Python 函数包装成"模型可以请求调用的能力",并自动从函数签名生成参数说明。
4. **循环(loop)**。把上面三者串起来:调模型 → 看模型要不要调工具 → 调了就执行、结果喂回去 → 再调模型,直到模型给出最终答案。

一个贴身的类比,后面九章都用得上:把 Agent 想成你雇的一位**远程助理**,你们只通过聊天软件沟通。聊天记录是**消息**;助理本人是**模型**——今天 A 家的人明天换 B 家的人,聊天软件不用换;你交到他手里的设备(查询系统、文件柜钥匙)是**工具**;而"你说需求 → 他去查 → 查到的贴回聊天 → 接着想 → 给你答复"这个一来一回的节奏,就是**循环**。全书后面做的事,不过是给这位助理配更好的装备、立更严的规章。

按顺序逐个实现。

## 1.2 消息:Agent 的数据骨架

### 为什么必须用结构化消息

如果用裸字符串列表存对话,三个问题立刻无解:

- **分不清谁说的**。"我帮你查一下"是助手说的还是工具返回的?模型 API 要求每条消息标明角色,否则无法理解对话结构。
- **一条助手消息可能同时含文本和工具调用**。模型说"我来查一下"的同时发起 `get_weather("北京")`——给人看的文本和给框架执行的指令是两类东西,挤在一个字符串里无法区分。
- **工具结果要配对到具体某次调用**。模型一次发三个调用(查北京、上海、广州),三个结果回来必须各自说明"我回应的是哪一次"。

### 四种消息 + 一个调用结构

`tinycore/messages.py` 的核心定义。最关键的是 `ToolCall`:

```python
# tinycore/messages.py
@dataclass
class ToolCall:
    """模型发出的一次工具调用请求。"""
    name: str                 # 调用哪个工具,如 "get_weather"
    args: Dict[str, Any]      # 参数,如 {"city": "北京"}
    id: str = field(default_factory=_new_id)   # 用于和结果配对
```

注意 `ToolCall` 是**结构化的调用请求**——`name` 和 `args` 是独立字段,不是一段要解析的文本。这是现代 Agent 与 2023 年那批"靠 prompt 教模型输出 `Action: xxx` 再用正则去抠"的 Agent 最本质的区别(1.3 节展开)。

`ToolCall` 之外还差一个小结构:`Usage`,每次模型调用的 token 账单。现在就把它定下来,因为后面两个机制都依赖它——第 3 章的压缩触发要知道"上下文现在多大",第 6 章的预算控制要知道"已经烧了多少"。(代码注释里的 provider 指模型服务商——Anthropic、OpenAI 这些,后文沿用这个叫法。)

```python
# tinycore/messages.py
@dataclass
class Usage:
    """一次模型调用的 token 用量(provider 返回什么就记什么,记不到就是 0)。"""
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":     # 支持 total += ai.usage 逐轮累计
        return Usage(self.input_tokens + other.input_tokens,
                     self.output_tokens + other.output_tokens)
```

然后是消息基类。所有消息共享两样东西:文本内容 `content` 和唯一 `id`(id 在第 5 章的日志去重、UI 的消息定位里都有用,现在先按惯例带上):

```python
# tinycore/messages.py
def _new_id() -> str:
    return uuid.uuid4().hex[:12]

@dataclass
class BaseMessage:
    content: str = ""
    id: str = field(default_factory=_new_id)

    @property
    def role(self) -> str:            # 子类各自返回 system/user/assistant/tool
        raise NotImplementedError

    def __repr__(self) -> str:        # 调试友好:打印时截断长文本
        text = self.content if len(self.content) <= 60 else self.content[:57] + "..."
        return f"{type(self).__name__}({text!r})"
```

四种消息完整定义如下——注意这**就是全部**,没有省略号,抄完这一段你的 `messages.py` 就有了主体:

```python
# tinycore/messages.py
@dataclass
class SystemMessage(BaseMessage):
    """系统指令。注意:在 tinycore 里它不进入历史,而是每轮由
    ContextManager 现场装配(第 3 章讲为什么)。"""
    @property
    def role(self) -> str:
        return "system"


@dataclass
class HumanMessage(BaseMessage):
    """用户输入(也用于压缩摘要、子代理指令等"以用户身份注入"的内容)。"""
    @property
    def role(self) -> str:
        return "user"


@dataclass
class AIMessage(BaseMessage):
    """模型输出。tool_calls 非空 = 请求调用工具;为空 = 最终回答。"""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)   # 本次调用的 token 账单
    stop_reason: str = ""    # provider 报告的停止原因:end_turn / tool_use / max_tokens…

    @property
    def role(self) -> str:
        return "assistant"

    def __repr__(self) -> str:
        if self.tool_calls:
            calls = ", ".join(f"{c.name}({c.args})" for c in self.tool_calls)
            return f"AIMessage(tool_calls=[{calls}])"
        return super().__repr__()


@dataclass
class ToolMessage(BaseMessage):
    """工具执行结果;tool_call_id 回指发起它的那次 ToolCall.id。"""
    tool_call_id: str = ""
    name: str = ""
    is_error: bool = False   # 这是一次失败的执行吗(第 2 章:错误即消息)

    @property
    def role(self) -> str:
        return "tool"


AnyMessage = BaseMessage
Messages = List[BaseMessage]
```

停一下,回答一个"为什么不"的问题——**为什么用 dataclass 而不是 dict / Pydantic / TypedDict?** 这类选型题贯穿全书,第一次遇到就把判法立起来:

| 方案 | 优点 | 代价 | 判定 |
|---|---|---|---|
| 裸 dict | 零定义成本 | 字段名靠记忆,typo 静默出错;`msg["tool_cals"]` 不报错只报废 | ✗ 教学和生产都不该用 |
| TypedDict | 有类型提示 | 运行时不校验;没有方法(repr/role) | ✗ 差口气 |
| Pydantic | 校验强、序列化全 | 引入第三方依赖;校验开销;魔法多 | 生产可选;教程要零依赖 ✗ |
| **dataclass** | 标准库;字段显式;可挂方法;`asdict` 够用 | 无运行时校验 | ✓ 本书选择 |

真实框架的选择也各不相同(LangChain 用 Pydantic,Claude Code 内部是 TS interface),但**"消息必须是显式 schema 的结构化对象"这一点没有分歧**——分歧只在校验强度。

三个设计点值得停下来看,它们贯穿全书:

**第一,`AIMessage` 同时持有 `content` 和 `tool_calls`。** 模型每次输出只有两种形态:`tool_calls` 非空——"我还需要调这些工具";为空——"我可以回答了"。**整个循环要不要继续,判据只有这一个:`tool_calls` 是否为空。** 请记住这句话,第 6、8 章乃至深入篇的条件边,用的都是同一个判据。

**第二,`ToolMessage.tool_call_id` 回指 `ToolCall.id`。** 模型一轮发多个调用、框架并行执行完,靠这个 id 配对"哪条结果回应哪次调用"。它的作用和取餐号一样:一次点了三杯咖啡,取餐凭号对单,不然分不清谁是谁的。丢了它,三个城市的天气结果回来,模型不知道哪个是北京的。

**第三,`usage` 与 `stop_reason` 不是锦上添花。** `usage` 记录每次调用花了多少 token——第 3 章的上下文压缩、第 6 章的预算控制都要靠它决策,**测不到就管不了**;`stop_reason` 区分"模型说完了"和"模型被 max_tokens 掐断了",后者你往往需要特殊处理。旧教程和很多入门实现都省了这两个字段,生产 harness 一个都不能省。

### 序列化:消息必须能无损往返

`messages.py` 的最后一块是一对序列化函数。现在看平平无奇,第 5 章它是会话持久化的地基——**消息要能写进 JSONL 文件、重启后读回来还原成同样的对象**。难点只有一个:JSON 里没有"类型"概念,`json.dumps(asdict(msg))` 之后,你分不清这个 dict 原来是 HumanMessage 还是 ToolMessage。解法是老手艺:**打类型标记**——像搬家打包,纸箱长得都一样,所以在箱皮上写"厨房""卧室",拆包时照标签归位。`__type__` 字段就是箱皮上那行字。

```python
# tinycore/messages.py
_MESSAGE_TYPES = {
    "SystemMessage": SystemMessage,
    "HumanMessage": HumanMessage,
    "AIMessage": AIMessage,
    "ToolMessage": ToolMessage,
}

def to_jsonable(obj: Any) -> Any:
    """消息/ToolCall/Usage → 可 JSON 化的 dict(带 __type__ 标记),递归处理容器。"""
    if isinstance(obj, BaseMessage):
        d = asdict(obj)
        d["__type__"] = type(obj).__name__        # ★ 类型标记
        return {k: to_jsonable(v) for k, v in d.items()}
    if isinstance(obj, (ToolCall, Usage)):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj

def from_jsonable(obj: Any) -> Any:
    """逆操作:凭 __type__ 还原对象;tool_calls/usage 两个嵌套字段要手工还原。"""
    if isinstance(obj, list):
        return [from_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        t = obj.get("__type__")
        if t in _MESSAGE_TYPES:
            data = {k: from_jsonable(v) for k, v in obj.items() if k != "__type__"}
            if "tool_calls" in data:
                data["tool_calls"] = [ToolCall(**tc) if isinstance(tc, dict) else tc
                                      for tc in data["tool_calls"]]
            if "usage" in data and isinstance(data["usage"], dict):
                data["usage"] = Usage(**data["usage"])
            return _MESSAGE_TYPES[t](**data)
        return {k: from_jsonable(v) for k, v in obj.items()}
    return obj
```

看一眼往返的实际样子(在 REPL 里跑一遍,建立体感):

```python
>>> ai = AIMessage(content="查一下", tool_calls=[ToolCall("get_weather", {"city": "北京"})])
>>> to_jsonable(ai)
{'content': '查一下', 'id': 'a3f8c21b9d04',
 'tool_calls': [{'name': 'get_weather', 'args': {'city': '北京'}, 'id': '7e2f90c1aa53'}],
 'usage': {'input_tokens': 0, 'output_tokens': 0}, 'stop_reason': '',
 '__type__': 'AIMessage'}
>>> m2 = from_jsonable(to_jsonable(ai))
>>> type(m2).__name__, m2.tool_calls[0].name, m2.tool_calls[0].id == ai.tool_calls[0].id
('AIMessage', 'get_weather', True)      # ← 无损:连 id 都原样回来了
```

两个容易踩的坑,现在记下第 5 章少调半天试:一是 `tool_calls` 里的 `ToolCall` 经 `asdict` 变成了嵌套 dict,还原时必须手工转回对象(否则后续代码 `call.name` 直接 `AttributeError`);二是**id 必须原样保留**——如果还原时重新生成 id,`ToolMessage.tool_call_id` 和 `ToolCall.id` 的配对就断了,provider 会拒绝这段历史。**"无损"的标准不是"内容一样",而是"配对关系一样"。**

到此 `messages.py` 完整了:一个 `Usage`、一个 `ToolCall`、一个基类、四个消息类、一对序列化函数,约 150 行。

(对照真实框架:这套消息对应 Anthropic API 的 messages + content blocks、OpenAI 的 chat messages,也与 LangChain 的 `langchain_core.messages` 同构。真实 API 的 content 是"块列表"(文本块、图片块、思考块混排),我们用"content 字符串 + tool_calls 列表"作最小可用版,概念一致。)

## 1.3 模型接口:一个 invoke,屏蔽所有厂商差异

### 接口契约

```python
# tinycore/models.py
class BaseChatModel:
    def bind_tools(self, tools):      # 告诉模型有哪些工具可用
        self._tools = list(tools)
        return self

    def invoke(self, messages) -> AIMessage:      # 一次拿全量
        raise NotImplementedError

    def stream(self, messages):                    # 流式:先吐增量,最后一项是完整消息
        ai = self.invoke(messages)                 # 默认实现:退化为一次性
        if ai.content:
            yield ai.content
        yield ai
```

上层(循环、harness)只依赖这三个方法,从头到尾不知道底层是谁。换厂商,上层一行不改——这是整个框架保持厂商无关的根基。`stream` 的契约("先 `str` 增量,最后一项是完整 `AIMessage`")第 4 章才真正用到,但默认实现值得现在看一眼:**不支持流式的模型自动退化为一次性输出**,消费者的写法永远统一。

### 关键:为什么是"原生工具调用"而不是"解析文本"

早期做法(ReAct 论文时代)是在 system prompt 里教模型按固定文本格式输出:

```
Thought: 我需要查北京的天气
Action: get_weather
Action Input: 北京
```

然后用正则从文本里抠出 action 名和参数。致命问题:模型少打个冒号、换个措辞,正则就崩,"解析模型输出"本身成了一个高频失败环节。

现代做法是**原生工具调用**:把工具的 JSON Schema 通过 API 的 `tools` 参数传给模型,模型直接返回结构化字段:

```json
{"name": "get_weather", "args": {"city": "北京"}, "id": "toolu_abc"}
```

框架直接读字段,零解析。格式由 API 层保证(模型被专门训练过、API 会校验),"解析失败"这个环节整个消失了。注意:ReAct 的**思想**(思考-行动-观察循环)一个字没变,变的只是"行动"的载体——从格式化文本变成结构化字段。这正是 1.2 节 `ToolCall` 是两个字段而不是一段文本的原因。

### 先看线上到底传什么:同一段对话的两种线格式

对接真实模型的难点不是"调 API",而是**厂商消息格式不同**。空谈"格式差异"没有体感,直接看线上传输的 JSON。同一段对话——用户问天气、模型调了一次工具、结果回来了——发给两家 API 时分别长这样:

```jsonc
// ―― Anthropic /v1/messages 的请求体 ――
{
  "model": "claude-sonnet-5",
  "max_tokens": 4096,
  "system": "你是生活助手",                    // ① system 是独立参数,不在 messages 里
  "tools": [{
    "name": "get_weather",
    "description": "查询指定城市的实时天气。city 传中文城市名。",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}},
                     "required": ["city"]}
  }],
  "messages": [
    {"role": "user", "content": "北京今天穿什么?"},
    {"role": "assistant", "content": [                     // ② 内容是"块列表"
        {"type": "text", "text": "我查一下。"},
        {"type": "tool_use", "id": "toolu_01A", "name": "get_weather",
         "input": {"city": "北京"}}                        //    工具调用是 tool_use 块
    ]},
    {"role": "user", "content": [                          // ③ 工具结果装在 user 消息里!
        {"type": "tool_result", "tool_use_id": "toolu_01A",
         "content": "5°C, 有风", "is_error": false}
    ]}
  ]
}

// ―― OpenAI /chat/completions 的请求体(同一段对话)――
{
  "model": "gpt-5.1",
  "tools": [{"type": "function", "function": {
      "name": "get_weather", "description": "查询指定城市的实时天气。…",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                     "required": ["city"]}}}],
  "messages": [
    {"role": "system", "content": "你是生活助手"},          // ① system 是普通消息
    {"role": "user", "content": "北京今天穿什么?"},
    {"role": "assistant", "content": "我查一下。",
     "tool_calls": [{"id": "call_9x", "type": "function",   // ② 调用在 tool_calls 字段
        "function": {"name": "get_weather",
                     "arguments": "{\"city\": \"北京\"}"}}]}, //   参数是 JSON【字符串】!
    {"role": "tool", "tool_call_id": "call_9x",             // ③ 工具结果是独立 tool 角色
     "content": "5°C, 有风"}
  ]
}
```

三处差异一目了然:**system 的位置**(独立参数 vs 普通消息)、**工具调用的载体**(assistant 内容里的 `tool_use` 块 vs `tool_calls` 字段,且后者的参数是 JSON 字符串)、**工具结果的角色**(装进 user 消息的 `tool_result` 块 vs 独立的 `tool` 角色)。这就是为什么不能有一个"通用转换",必须每家一个翻译函数。

### Anthropic 实现:翻译函数 + 模型类,一行不少

先写翻译函数——它把我们的消息列表翻成上面第一种线格式:

```python
# tinycore/models.py
def _to_anthropic(messages: Messages) -> Tuple[str, List[Dict[str, Any]]]:
    """内部消息 → (system 文本, Anthropic messages 列表)。"""
    system_text = ""
    out: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            system_text += ("\n" if system_text else "") + m.content   # ① 抽出去当参数
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            blocks: List[Dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for c in m.tool_calls:                                     # ② 调用 → tool_use 块
                blocks.append({"type": "tool_use", "id": c.id,
                               "name": c.name, "input": c.args})
            out.append({"role": "assistant", "content": blocks})
        elif isinstance(m, ToolMessage):
            out.append({                                               # ③ 结果 → user 里的
                "role": "user",                                        #    tool_result 块
                "content": [{"type": "tool_result",
                             "tool_use_id": m.tool_call_id,
                             "content": m.content,
                             "is_error": m.is_error}],
            })
    return system_text, out
```

然后是模型类。把"拼参数"和"解响应"拆成两个小方法,是为了第 4 章的流式版能复用它们:

```python
# tinycore/models.py
class AnthropicChatModel(BaseChatModel):
    """Anthropic 原生 API(pip install anthropic + ANTHROPIC_API_KEY)。"""

    def __init__(self, model="claude-sonnet-5", max_tokens=4096,
                 api_key=None, **kwargs):
        import anthropic                       # 延迟导入:没装也不影响其他实现
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model, self.max_tokens, self.kwargs = model, max_tokens, kwargs
        self._tools: List[Tool] = []

    def _params(self, messages):               # 拼请求参数(invoke 与 stream 共用)
        system_text, msgs = _to_anthropic(messages)
        tools_param = [{"name": t.name, "description": t.description,
                        "input_schema": t.parameters} for t in self._bound_tools]
        p = dict(model=self.model, max_tokens=self.max_tokens,
                 messages=msgs, **self.kwargs)
        if system_text:
            p["system"] = system_text
        if tools_param:
            p["tools"] = tools_param
        return p

    @staticmethod
    def _from_response(resp) -> AIMessage:     # 解响应(invoke 与 stream 共用)
        text, tool_calls = "", []
        for block in resp.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(block.name, dict(block.input), block.id))
        return AIMessage(content=text, tool_calls=tool_calls,
                         stop_reason=resp.stop_reason or "",
                         usage=Usage(resp.usage.input_tokens, resp.usage.output_tokens))

    def invoke(self, messages) -> AIMessage:
        return self._from_response(self.client.messages.create(**self._params(messages)))
```

两个不起眼但重要的习惯:**延迟导入**(`import anthropic` 在 `__init__` 里),只装了 openai 的用户不会因为没装 anthropic 而崩;**key 只从环境变量读**,永远不进代码、不进日志、更不进第 5 章的会话文件。

### OpenAI 兼容实现:一份代码接大半个生态

```python
# tinycore/models.py
def _to_openai(messages: Messages) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            out.append({
                "role": "assistant",
                "content": m.content or None,
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.name,
                                             "arguments": json.dumps(c.args, ensure_ascii=False)}}
                               for c in m.tool_calls],       # ★ 参数要序列化成字符串
            })
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "tool_call_id": m.tool_call_id,
                        "content": m.content})
        else:
            out.append({"role": m.role, "content": m.content})   # system/user 都是普通消息
    return out


class OpenAIChatModel(BaseChatModel):
    """任何 OpenAI 兼容服务(pip install openai),换 base_url 即换 provider。"""

    def __init__(self, model="gpt-5.1", base_url=None, api_key=None,
                 api_key_env="OPENAI_API_KEY", **kwargs):
        from openai import OpenAI              # 延迟导入
        self.client = OpenAI(api_key=api_key or os.environ.get(api_key_env),
                             base_url=base_url)
        self.model, self.kwargs = model, kwargs
        self._tools: List[Tool] = []

    def invoke(self, messages) -> AIMessage:
        tools_param = [{"type": "function",
                        "function": {"name": t.name, "description": t.description,
                                     "parameters": t.parameters}}
                       for t in self._bound_tools]
        resp = self.client.chat.completions.create(
            model=self.model, messages=_to_openai(messages),
            tools=tools_param or None, **self.kwargs)
        msg = resp.choices[0].message
        tool_calls = [ToolCall(tc.function.name,
                               json.loads(tc.function.arguments or "{}"),  # ★ 字符串→dict
                               tc.id)
                      for tc in (msg.tool_calls or [])]
        usage = (Usage(resp.usage.prompt_tokens, resp.usage.completion_tokens)
                 if resp.usage else Usage())
        return AIMessage(content=msg.content or "", tool_calls=tool_calls,
                         stop_reason=resp.choices[0].finish_reason or "", usage=usage)
```

两个 ★ 标记的是同一件事的两面,也是新手第一坑:OpenAI 线格式里工具参数是 **JSON 字符串**——发出去要 `json.dumps`,收回来要 `json.loads`。漏掉后者,`call.args` 是个字符串,`tools_by_name[call.name].invoke(call.args)` 会在 `func(**args)` 处炸出一个莫名其妙的 `TypeError`。另外注意 `arguments or "{}"`:个别兼容服务在无参调用时返回空串,裸 `json.loads("")` 直接异常。

两家实现并排看,骨架完全一致:**拼工具 schema → 翻译消息 → 调 API → 解回统一的 AIMessage**。差异被关在 `_to_anthropic`/`_to_openai` 两个纯函数里。翻译函数的角色像出国用的电源转换头:各国插座标准不同,但你的设备只有一种插头——设备不改,换国家只换转换头。以后要接新厂商,照这个骨架抄一份翻译函数即可。

### FakeModel:全书的第三个"厂商"

第三个实现是本教程的特色装备:一个照剧本演戏的**假模型**(测试行话叫"桩",stub——顶替真实组件的替身)。完整代码就这么长:

```python
# tinycore/models.py
class FakeModel(BaseChatModel):
    """按剧本依次返回响应,用于离线示例、测试与教学。

    剧本元素可以是:
    - str                        → 纯文本回答
    - AIMessage                  → 原样返回(可带 tool_calls)
    - list[ToolCall]             → 一条只带工具调用的 AIMessage
    - callable(messages) -> 上述 → 按当轮输入动态决定(断言/复读机都靠它)
    剧本耗尽后固定回答 "(剧本已结束)"——循环因此必然终止。
    """

    def __init__(self, script: Sequence[Any] = ()) -> None:
        self.script = list(script)
        self.calls: List[Messages] = []     # 记录每轮实际收到的消息,测试用
        self._tools: List[Tool] = []

    def invoke(self, messages: Messages) -> AIMessage:
        self.calls.append(list(messages))                 # ① 留证据:模型看到了什么
        item = self.script.pop(0) if self.script else "(剧本已结束)"
        if callable(item):
            item = item(messages)                         # ② 动态台词
        if isinstance(item, AIMessage):
            ai = item
        elif isinstance(item, list):
            ai = AIMessage(content="", tool_calls=item)
        else:
            ai = AIMessage(content=str(item))
        if not ai.stop_reason:
            ai.stop_reason = "tool_use" if ai.tool_calls else "end_turn"
        est_in = sum(len(m.content or "") for m in messages) // 4    # ③ 假账单
        ai.usage = Usage(input_tokens=est_in, output_tokens=len(ai.content) // 4)
        return ai
```

三处设计各有用途:① `self.calls` 记录每轮**模型实际收到的完整输入**——第 3 章验证"装配出来的 system 长什么样"、第 5 章验证"恢复后的历史对不对",全靠翻这本账;② 剧本元素允许 `callable`,让假模型能按当轮收到的输入决定台词(回显收到的历史条数、断言某条消息存在);③ 连 usage 都按字符数伪造,于是压缩、预算这些"靠账单决策"的机制**离线也能被触发**。

用法:

```python
model = FakeModel(script=[
    [ToolCall("get_weather", {"city": "北京"})],   # 第 1 轮:请求调工具
    "北京晴,25 度,适合出门。",                      # 第 2 轮:给最终回答
])
```

别把它当成"没有 key 的妥协"。它是理解本章最重要的一件教具:**模型在循环里无非就是"给一串消息,回一条消息"的函数**——FakeModel 把这个函数的智能拧到零,循环、配对、终止判据全都照常工作。智能是这个函数的品质,不是循环的结构。这也是工程上的测试哲学(第 11 章展开):机制用桩模型钉死,能力才用真模型评。

### 工厂:字符串选型 + 环境变量收口

最后把"构造哪个类、base_url 填什么、key 从哪个环境变量读"这些散碎知识收进一个工厂,完整代码:

```python
# tinycore/models.py
_OPENAI_COMPATIBLE = {          # 要接新厂商,在这里加一行就够
    "openai":   (None, "OPENAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "qwen":     ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "kimi":     ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
}

def init_chat_model(spec: str, **kwargs) -> BaseChatModel:
    """按 "provider:model" 构造模型,如 anthropic:claude-sonnet-5。"""
    provider, _, model = spec.partition(":")
    if provider == "fake":
        return FakeModel()
    if not model:
        raise ValueError(f"spec 应形如 'provider:model',收到: {spec!r}")
    if provider == "anthropic":
        return AnthropicChatModel(model, **kwargs)
    if provider in _OPENAI_COMPATIBLE:
        base_url, key_env = _OPENAI_COMPATIBLE[provider]
        kwargs.setdefault("base_url", base_url)
        kwargs.setdefault("api_key_env", key_env)
        return OpenAIChatModel(model, **kwargs)
    return OpenAIChatModel(model, **kwargs)   # 未知厂商按兼容网关处理(自带 base_url)

def get_model(**kwargs) -> BaseChatModel:
    """从环境变量 TINYAGENT_MODEL 读 spec(全书示例的统一入口)。"""
    spec = os.environ.get("TINYAGENT_MODEL", "").strip()
    if not spec:
        raise RuntimeError("未设置 TINYAGENT_MODEL。"
                           "示例:export TINYAGENT_MODEL='anthropic:claude-sonnet-5'")
    return init_chat_model(spec, **kwargs)
```

全书示例都经 `get_model()` 拿模型:换 provider 只改一行环境变量,示例代码零改动。`models.py` 到此完整:一个基类、三个实现、两个翻译函数、一个工厂,约 300 行——这就是 L0 的全部。

## 1.4 工具:把函数变成模型能调用的能力

模型怎么知道有哪些工具、每个工具怎么用?它看不到函数体,唯一依据是你提供的 schema——名字、用途描述、参数定义。手写 schema 烦且易错,所以用装饰器从签名自动生成:

```python
# tinycore/tools.py
@dataclass
class Tool:
    name: str                    # 模型用它指定调用谁
    description: str             # 模型靠它判断"何时该用"
    parameters: Dict[str, Any]   # JSON Schema:参数名/类型/必填
    func: Callable               # 真实现,藏在框架侧

    def invoke(self, args):
        return str(self.func(**args))    # 真正执行——永远由框架做

@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气。city 传中文城市名,如 "北京"。"""
    return f"{city} 晴, 25°C"
```

装饰器的完整实现——核心是用标准库 `inspect` 读签名、`get_type_hints` 读注解,把 Python 类型映射成 JSON Schema 类型:

```python
# tinycore/tools.py
_PY_TO_JSON = {str: "string", int: "integer", float: "number",
               bool: "boolean", list: "array", dict: "object"}

def _build_parameters(func: Callable) -> Dict[str, Any]:
    """从函数签名 + 类型注解构造 JSON Schema。"""
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)          # {"city": str, "return": str}
    except Exception:
        hints = {}                            # 注解写错不至于崩,退化为全 string
    properties, required = {}, []
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        json_type = _PY_TO_JSON.get(hints.get(pname, str), "string")
        properties[pname] = {"type": json_type}
        if param.default is inspect.Parameter.empty:   # 没有默认值 → 必填
            required.append(pname)
    return {"type": "object", "properties": properties, "required": required}

def tool(func=None, *, name=None, max_result_chars=20_000):
    """装饰器:@tool 或 @tool(name=..., max_result_chars=...) 两种写法都支持。"""
    def wrap(f):
        return Tool(name=name or f.__name__,
                    description=(f.__doc__ or "").strip(),   # docstring 即说明书
                    parameters=_build_parameters(f),
                    func=f,
                    max_result_chars=max_result_chars)
    return wrap(func) if func is not None else wrap
```

(`tool(func=None, *, ...)` 这个写法是"可带参装饰器"的标准套路:`@tool` 时 func 直接传进来走 `wrap(func)`;`@tool(name="x")` 时先收关键字参数、返回 wrap 等着接函数。第一次见记住模式即可。`max_result_chars` 字段第 2 章讲。)

在 REPL 里验证生成的说明书,并注意它**恰好就是 1.3 节线格式里 `tools` 参数的内容**:

```python
>>> get_weather.name, get_weather.description
('get_weather', '查询指定城市的实时天气。city 传中文城市名,如 "北京"。')
>>> get_weather.parameters
{'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}
>>> get_weather.invoke({"city": "上海"})
'上海 晴, 25°C'
```

`name + description + parameters` 三件套发给模型(经 `bind_tools`),`func` 留在框架侧,`invoke` 负责真正执行——**docstring 是写给模型的接口文档**,不是给人看的注释,它的质量直接决定模型"调不调、怎么传参"的正确率。工具设计的完整方法论放在第 2 章(那是工具的主场),这里先立住一条贯穿全书的边界:

> **模型永远不直接执行工具。** 它只能输出一个 `ToolCall`——"我想调用"的请求;真正执行 `func(**args)` 的是框架的 `Tool.invoke`。模型说它想删库,不等于库被删了——中间隔着框架这一道。第 6 章的权限、审批、沙箱,全部建立在这道缝隙上。如果模型能直接执行,这些安全机制就无处安放。

## 1.5 循环:把三个模块拼成 Agent

四个模块齐了三个,写那个把它们串起来的循环。教学版十几行:

```python
# 教学版主循环(examples/k01_loop.py 里可直接跑)
def run_agent(model, tools, user_input, system_prompt=None, max_turns=10):
    tools_by_name = {t.name: t for t in tools}
    model = model.bind_tools(tools)

    messages = [HumanMessage(content=user_input)]
    for _ in range(max_turns):
        working = ([SystemMessage(content=system_prompt)] if system_prompt else []) + messages
        ai = model.invoke(working)                   # ① 模型决策
        messages.append(ai)

        if not ai.tool_calls:                        # ② 无工具调用 → 结束
            return ai.content, messages

        for call in ai.tool_calls:                   # ③ 执行工具,结果喂回
            result = tools_by_name[call.name].invoke(call.args)
            messages.append(ToolMessage(
                content=result, tool_call_id=call.id, name=call.name))

    return "(达到最大轮数)", messages
```

三步与 ReAct 思想严丝合缝:①决策(Thought)、②终止判断(判据就是 1.2 节那一个:`tool_calls` 空不空)、③行动 + 观察(Action + Observation),然后回到 ①。

两个容易被略过、实则重要的细节:

**system prompt 不放进 `messages`,而是每轮临时拼在最前面。** 教学版里这只是个小讲究,第 3 章它会成为一个正式原则:历史里只存 user/assistant/tool 三种消息,system 每轮由上下文层**现场装配**——这样记忆文件改了立刻生效、恢复会话不会带出陈旧指令。

**`max_turns` 是安全阀,不是摆设。** Agent 真的会死循环,生产里三种根源很常见:工具反复返回模型无法满足的结果(查不存在的订单号,不死心换着花样查)、模型看不懂报错以为重试就行、任务本身无解但模型不认输。上限只是兜底,生产 harness 还会配重复调用检测和预算控制(第 6 章)。

### 跑起来,看清"循环"到底在循环什么

先离线跑(FakeModel 照剧本出牌,零依赖零 key):

```python
model = FakeModel([
    [ToolCall("get_weather", {"city": "北京"})],
    "北京晴 25°C,穿单衣即可。",
])
answer, history = run_agent(model, [get_weather], "北京今天穿什么?")
```

把 `messages` 的演化摊开,是一次完整的 ReAct 轨迹:

```
第 1 轮:
  喂给模型 → [Human]
  模型返回 → AIMessage(tool_calls=[get_weather(北京)])   ← 有 tool_calls,继续
  框架执行 → 追加 ToolMessage("北京 晴, 25°C")
第 2 轮:
  喂给模型 → [Human, AI(调工具), Tool]                   ← 比上轮多两条
  模型返回 → AIMessage("北京晴 25°C,穿单衣即可。")        ← 无 tool_calls,结束

最终历史(4 条): [Human, AI(调工具), Tool, AI(回答)]
```

这里有个后面整个"记忆"与"持久化"话题的根:**每一轮都把截止当前的完整历史重新喂给模型。** 模型本身无状态,它不"记得"上一轮——是我们靠不断追加、整卷重发,才营造出连续思考的效果。"对话记忆"这个听起来高级的东西,本质就是持续累积的 `messages` 列表(第 5 章把它落到磁盘上)。

然后换真实模型,逻辑一字不改:

```bash
export TINYAGENT_MODEL="anthropic:claude-sonnet-5"   # 或 deepseek:deepseek-chat
export ANTHROPIC_API_KEY="sk-ant-..."
python k01_loop.py
```

真实模型有自主性:同样的问题,它可能先调工具、可能直接回答、可能反问你。**"没调工具"不是 bug**——我们的循环对所有情况都正确处理。这正是 Agent 与写死脚本的区别:控制流由模型在运行时决定。

### 包里的正式版:Agent 类

教学版讲清了原理,`tinycore/loop.py` 里的正式版把它长成一个类:

```python
from tinycore import Agent, FakeModel

agent = Agent(model=model, tools=[get_weather], system_prompt="你是生活助手")
messages = agent.invoke("北京今天穿什么?")       # 简单入口:只要结果
```

正式版的全部字段先亮出来,建立"后面几章各自负责哪个参数"的地图:

```python
# tinycore/loop.py(字段全景;run() 的完整实现在第 4 章逐行写)
@dataclass
class Agent:
    model: object                      # L0:BaseChatModel(本章)
    tools: Sequence[Tool] = ()         # 工具箱(本章 + 第 2 章)
    system_prompt: str = ""            # 身份指令(第 3 章讲装配)
    max_turns: int = 40                # 安全阀(本章)
    context: Optional[ContextManager] = None   # 上下文策略(第 3 章)
    gate: Optional[Gate] = None        # 权限/钩子接缝(第 6、7 章)
    parallel_tools: bool = True        # 并行执行(第 2 章)
    stream_text: bool = True           # 逐 token 流式(第 4 章)
    name: str = "agent"                # 多代理时标识自己(第 8 章)

    def invoke(self, user_input, prior_messages=None) -> Messages:
        """便捷入口:吞掉事件流,只要最终历史。"""
        gen = self.run(user_input, prior_messages)
        for _ in gen:
            pass
        return self.last_messages

    def run(self, user_input, prior_messages=None):
        """真正的主循环:事件生成器。第 4 章完整实现。"""
        ...
```

骨架与教学版逐行对应,多出来的字段各有一章去讲。**先记住:它们都是这个循环的参数,不是新框架**——全书学完,`Agent` 也只有这九个字段。

用正式版跑 `examples/k01_loop.py`,实际输出(离线,FakeModel):

```
使用 FakeModel(离线剧本;加 --real 切真实模型)

用户: 北京今天适合穿什么?
  [工具调用] get_weather({'city': '北京'})
  [工具结果] 5°C, 有风

Agent 回答: 北京今天 5°C 且有风,建议穿厚外套加围巾。

最终历史(4 条):
  user: 北京今天适合穿什么?
  assistant(调工具):
  tool: 5°C, 有风
  assistant: 北京今天 5°C 且有风,建议穿厚外套加围巾。
```

加 `--real` 换真实模型,同一段代码、同一个循环——差别只在"那个函数"的智能品质。

### 第一次跑常见的报错

| 报错或现象 | 根因 | 怎么解 |
|---|---|---|
| `ModuleNotFoundError: tinycore` | 没从 `code/examples/` 目录跑 | 示例首行 `import _config` 会把上层包加进路径 |
| `ModuleNotFoundError: anthropic/openai` | 没装对应 SDK | `pip install anthropic` 或 `openai`(离线示例不需要) |
| 401 / `AuthenticationError` | key 没设或环境变量名不对 | 对照 `TINYAGENT_MODEL` 的 provider 检查对应 key |
| 模型名 404 | 拼错或该厂商无此模型 | 检查 `TINYAGENT_MODEL` 冒号后的部分 |
| 连接超时 | 网络/代理,或 base_url 不对 | 兼容服务确认 `base_url` |
| 模型没调工具 | 它判断不需要 | 合法行为,不是 bug(见上) |

## 1.6 这个循环已经是主流内核——差的是壳

盘点一下:100 来行,四个模块,一个能对接真实模型、自主调用工具、换厂商只改环境变量的 Agent。

现在回看第 0 章的判断:真实产品的模型决策循环只是骨架,上下文、工具、事件和控制面才让它从"能跑"走到"能用、敢用"。这四样东西各有专章:

- **工具还是玩具**。查天气改变不了世界。给模型一台"电脑"——读写文件、执行命令、检索代码,以及随之而来的并行、报错、越界问题 → **第 2 章**。
- **上下文会爆、会腐**。历史无限追加,几十轮后窗口撑爆、注意力稀释,而且 system prompt 该放什么、项目知识从哪来,都还没答案 → **第 3 章**。
- **过程是黑盒**。`run_agent` 只在最后 return 一次,中间十几秒发生了什么外面看不见,更别说打断它、中途补一句话 → **第 4 章**。
- **无法托管**。进程一退历史就没了;模型想删文件没人拦得住;想挂个自动化逻辑无处可挂 → **第 5-9 章(Harness 篇)**。

顺带交代另一条路:这个循环还有一组更深的结构性局限——并行写入的确定性合并、回到历史任意一步、细粒度重试。Harness 路线用"事件日志 + 消息边界"解决了其中一部分,另一部分是**图运行时**(家族 B)的主场。第 10 章把这笔账算清楚;性急的读者可以从 [[R1-循环的局限|深入篇 R1]] 直接进入那条线——它只依赖本章的四个模块。

## 1.7 动手:把本章抄成代码(验收标准)

本章读完的验收标准不是"看懂了",是**合上书能写出来**。三份文件,建议按此顺序:

1. **`tinycore/messages.py`**(~150 行):`Usage` → `ToolCall` → `BaseMessage` → 四个消息类 → `to_jsonable`/`from_jsonable`。
   自查:REPL 里 `from_jsonable(to_jsonable(ai))` 往返后,`tool_calls[0].id` 与原对象一致。
2. **`tinycore/tools.py` 的前半**(~60 行):`_PY_TO_JSON` → `_build_parameters` → `Tool` → `@tool`。
   自查:`get_weather.parameters` 打印出 1.4 节那个 schema;`invoke({"city":"上海"})` 能执行。
3. **`tinycore/models.py`**(~300 行):`BaseChatModel` → `FakeModel` → `_to_anthropic` + `AnthropicChatModel` → `_to_openai` + `OpenAIChatModel` → `init_chat_model`/`get_model`。
   自查:用 FakeModel 跑通教学版 `run_agent`;有 key 的话换 `get_model()` 再跑一遍,代码零改动。

写不出来的部分,回到对应小节**再读一遍代码旁边的"为什么"**——背不下来是正常的,推不出来才说明没懂。三份文件都过自查后,跑 `python examples/k01_loop.py`,输出应与 1.5 节贴的一致。

两个加深理解的改造练习(可选):

- 给 `@tool` 加 `Optional[int]` 类型支持(提示:`typing.get_origin/get_args` 拆 Optional,JSON Schema 里体现为非必填)。
- 给 `FakeModel` 加一个 `on_call` 回调,每次被调用时打印"第 N 轮,收到 M 条消息"——你会对"整卷重发"有更直观的体感。

## 1.8 源码解析与代码逻辑

> 动手是"抄得出来";这一节是"讲得清楚"。
> 正文 1.2–1.5 已经把四个模块的实现大段贴过——这里**不重贴整文件**,只收成**跨模块数据流** + 几处铰链 + 失败态。
> 打开仓库对照:`tinycore/messages.py`、`tinycore/models.py`、`tinycore/tools.py`(前半)、`examples/k01_loop.py`。
> 正式的 `Agent.run` 完整生成器在第 4 章展开;本章先把**教学版循环**和数据骨架钉死。

### 先用一张生活图把四个模块钉住

把最小 Agent 想成**一桌麻将**:

- **消息**(`messages.py`)是牌面——桌上有什么,所有人(模型、工具、以后的日志)都只认这几种牌。
- **模型**(`models.py`)是庄家——看一桌子牌,决定出牌(最终回答)还是叫牌(工具调用)。
- **工具**(`tools.py` 前半)是点菜单——把厨房能力写成庄家能读的说明书;`func` 还在后厨。
- **循环**(`k01` 的 `run_agent` / 以后的 `Agent.run`)是规则——庄家出完,若叫了牌就去后厨端菜,再把菜摆回桌上继续。

```
用户一句话
    │
    ▼
HumanMessage 进 messages[]          ← 记忆载体(历史)
    │
    ▼
每轮: working = [SystemMessage?] + messages   ← system 现场装,不进历史
    │
    ▼
model.bind_tools(tools).invoke(working) → AIMessage
    │
    ├─ tool_calls 为空  → 最终回答,结束
    └─ tool_calls 非空 → 对每个 ToolCall:
              tools_by_name[name].invoke(args) → str
              包成 ToolMessage(tool_call_id=call.id) 追加进 messages
              再进入下一轮 invoke(整卷重发)
```

### 本章源码地图(读顺序)

| 阅读顺序 | 路径 | 体量 | 你要带走的一件事 |
|---|---|---|---|
| 1 | `tinycore/messages.py` | ~170 行 | 四种消息 + `ToolCall` + `Usage`;`tool_calls` 空/非空 = 循环判据 |
| 2 | `tinycore/tools.py` 前半 | ~90 行 | `@tool` 把函数变成说明书;`Tool.invoke` 才执行 |
| 3 | `tinycore/models.py` | ~370 行 | 统一 `invoke`/`stream`/`bind_tools`;厂商差异关进翻译函数 |
| 4 | `examples/k01_loop.py` | ~70 行 | 教学版 `run_agent`:与正文逐行一致的最小循环 |
| 5 | `tinycore/loop.py`(预告) | 字段即可 | 正式 `Agent` 多了事件流/gate/context;第 4 章再拆 `run()` |

### 主路径走读:k01 的"北京穿什么"

剧本:

1. 第一轮模型返回 `[ToolCall("get_weather", {"city": "北京"})]`
2. 工具返回天气字符串
3. 第二轮模型返回纯文本建议

**① 注册工具:函数 → Tool**

```python
# examples/k01_loop.py + tinycore/tools.py
@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气。..."""
    return {...}.get(city, ...)
```

装饰之后 `get_weather` 已是 `Tool` 对象,不是函数:

- `name` = `"get_weather"`
- `description` = docstring(写给模型的接口文档)
- `parameters` = `_build_parameters` 扫签名得到的 JSON Schema
- `func` = 原来的函数

**② 组循环:`run_agent` 的三拍**

```python
# examples/k01_loop.py —— 教学版主循环铰链
ai = model.invoke(working)          # ① 模型决策
messages.append(ai)
if not ai.tool_calls:               # ② 无工具 → 结束
    return ai.content, messages
for call in ai.tool_calls:          # ③ 执行并喂回
    result = tools_by_name[call.name].invoke(call.args)
    messages.append(ToolMessage(content=result, tool_call_id=call.id, name=call.name))
```

对照正式内核(`loop.py`),多出来的以后是:事件 `yield`、gate、并行 `run_tool_calls`、上下文压缩。  
**判停条件一字不差**:`if not ai.tool_calls`。

**③ FakeModel 怎么"演"第一轮**

```python
# tinycore/models.py —— 剧本元素 list[ToolCall] → AIMessage
elif isinstance(item, list):
    ai = AIMessage(content="", tool_calls=item)
# …
ai.stop_reason = "tool_use" if ai.tool_calls else "end_turn"
```

- 剧本第一项是工具列表 → 合成一条带 `tool_calls` 的 AI 消息。
- `stop_reason` 自动标成 `tool_use` / `end_turn`,与真模型字段对齐。
- 每轮 `self.calls.append(list(messages))`:测试可断言"模型到底看见了什么"。

**④ 工具结果如何配对**

`ToolMessage.tool_call_id = call.id`。  
一轮若有多个调用,provider 和模型都靠这个 id 对齐结果;省掉 id,多城市天气会串台,API 也会拒收。

**⑤ 第二轮:整卷重发**

`working = [SystemMessage(...)] + messages` 再次 `invoke`。  
历史里此时已有:Human → AI(调工具) → Tool → (本轮新 AI)。  
**模型本身无状态**;记忆 = 每次完整重发的消息列表。这是后面上下文爆炸(第 3 章)的根源,也是 FakeModel 能测机制的原因——它只认消息进、消息出。

### 消息模块:该盯的生产字段

| 类型/字段 | 逻辑要点 |
|---|---|
| `ToolCall.name/args/id` | 结构化请求,不是正则抠文本;`id` 默认 12 位 hex |
| `AIMessage.tool_calls` | **空 = 结束,非空 = 继续**——全书唯一正常终止判据 |
| `AIMessage.usage` | 为第 3 章压缩、第 6 章预算留的计量器 |
| `AIMessage.stop_reason` | provider 原样记下;`end_turn` / `tool_use` / `max_tokens`… |
| `ToolMessage.is_error` | 第 2 章"错误即消息"的旗标;本章教学循环还没设,正式执行器会设 |
| `SystemMessage` | **约定不进长期历史**;每轮现场拼到 `working` 最前 |
| `to_jsonable` / `from_jsonable` | 带 `__type__` 的往返;第 5 章 JSONL 会话靠它复活对象 |

序列化铰链(读懂即可,第 5 章重用):

```python
# tinycore/messages.py —— 消息 → 可 JSON 的 dict
if isinstance(obj, BaseMessage):
    d = asdict(obj)
    d["__type__"] = type(obj).__name__
    return {k: to_jsonable(v) for k, v in d.items()}
```

`from_jsonable` 看到 `__type__` 再把 `tool_calls` / `usage` 字典还原成对象。  
动手自查:`from_jsonable(to_jsonable(ai)).tool_calls[0].id` 必须等于原 id。

### 模型模块:契约比实现重要

上层(循环、harness)只依赖:

| 方法 | 契约 |
|---|---|
| `bind_tools(tools)` | 挂上工具说明书,返回 self,便于链式调用 |
| `invoke(messages) -> AIMessage` | 一次拿全量 |
| `stream(messages)` | 先 yield 若干 `str` 增量,**最后一项必须是完整 `AIMessage`** |

默认 `stream` 退化成"整段当一个增量再 yield ai"——所以 FakeModel 也能被流式消费者驱动(第 4 章)。

真模型两条翻译路径:

- `_to_anthropic`:system 提出为独立参数;工具调用变 `tool_use` block;工具结果变 user 里的 `tool_result`(没有独立 tool 角色)。
- `_to_openai`:更接近 Chat Completions;`tool` 角色独立;流式参数还要按 index 拼 JSON(第 4 章细讲)。

`init_chat_model("provider:model")` / `get_model()` 只是工厂:循环代码零改动切换厂商。

### 分支与失败态

| 若出现… | 落点 | 变成什么 | 体感 |
|---|---|---|---|
| `tool_calls` 为空 | `run_agent` / 正式 `run` 判停 | 返回最终文本 | 正常结束 |
| 剧本耗尽 | `FakeModel.invoke` | `"(剧本已结束)"` 纯文本 AI | 循环下一轮必停(无 tool_calls) |
| 剧本项是 callable | 同上 | 动态决定本轮输出 | 测试里写断言/复读机 |
| 调用不存在的工具名 | 教学版 `tools_by_name[name]` | **KeyError 炸循环** | 第 2 章执行器改成错误消息;对比刻意留着 |
| 真模型网络/鉴权失败 | `Anthropic/OpenAI.invoke` | 异常上抛 | 教学循环不吞;正式 `run` 会先 `yield ERROR` 再 raise(第 4 章) |
| 达到 `max_turns` | for 结束 | 教学版返回 `"(达到最大轮数)"` | 安全阀,防死循环 |
| system_prompt 为 None | `working` 不加 System | 只发历史 | 合法;有些任务不需要系统词 |
| `Usage` 两轮相加 | `Usage.__add__` | 累加 in/out tokens | 正式循环 `total += ai.usage` |

### 教学循环 vs 正式 `Agent` 差在哪

| | `k01.run_agent` | `loop.Agent.run`(第 4 章) |
|---|---|---|
| 输出 | 最终 `(text, messages)` | **事件生成器** + 返回值 messages |
| 工具执行 | 手写 for + `invoke` | `run_tool_calls`(gate/并行/错误即消息) |
| 上下文 | 每次简单拼 system | 可选 `ContextManager.assemble/compact` |
| 外部控制 | 无 | `interrupt` / `inject` |
| 终止 | `not tool_calls` 或 max_turns | 同左,并写入 `stop_reason` 事件 |

本章把**数据骨架 + 判停 + 整卷重发**学透;壳上的能力都是后加的消费者与接缝,不改这三件事。

### 与前后章的接缝

- **上游 · 第 0 章四层架构**——本章交出 L0(模型接口)与 L1 循环的最小切片;L2/L3 还是空的。
- **下游 · 第 2 章执行器**——把 k01 里裸 `tools_by_name[name].invoke` 升级为 `_execute_one` / `run_tool_calls`;错误不再 KeyError。
- **下游 · 第 3 章上下文**——`Usage` 与"整卷重发"逼出压缩;system 继续每轮装配。
- **下游 · 第 4 章事件流**——同一循环改成 `yield Event`;`stream()` 契约在此兑现。
- **下游 · 第 5 章会话**——`to_jsonable`/`from_jsonable` 变成 JSONL 里的消息复活术。

### 只要记住的 5 行逻辑

1. **历史是 `messages` 列表;模型无状态,每轮把能看见的整卷再发一遍。**
2. **`AIMessage.tool_calls` 空不空,是循环要不要继续的唯一正常判据。**
3. **`ToolCall.id` ↔ `ToolMessage.tool_call_id` 配对;多调用场景省不得。**
4. **模型接口只认 invoke/stream/bind_tools;厂商差异关在 `_to_*` 翻译里。**
5. **`@tool` 产出说明书 + 隐藏 `func`;模型只请求,框架(现在是教学循环,以后是执行器)才执行。**

### 对照验收

1. REPL:`from_jsonable(to_jsonable(ai))` 往返后 `tool_calls[0].id` 不变。  
2. 打印 `get_weather.parameters`,应看到 `city` 为 required string。  
3. 跑 `python examples/k01_loop.py`:两轮历史 = user → assistant(调工具) → tool → assistant(最终)。  
4. 有 key 时加 `--real`,循环代码零改动——只换 `model` 对象。

## 1.9 自测

**Q1. 循环的终止判据是什么?它写在代码的哪一行?**

<details><summary>想清楚再展开</summary>

`AIMessage.tool_calls` 是否为空。空 = 模型认为可以回答了,循环从 `if not ai.tool_calls: return` 退出。这是唯一的正常出口;另一个出口 `max_turns` 是防死循环的安全阀。
</details>

**Q2. 为什么 `ToolCall.id` 不能省?省了会在什么场景下出错?**

<details><summary>想清楚再展开</summary>

模型一轮发多个工具调用时,多个 `ToolMessage` 回来必须靠 `tool_call_id == ToolCall.id` 配对。省掉它,模型无法区分"哪条结果回应哪次调用"——三个城市的天气全混在一起;而且 provider API 会直接拒绝没有配对 id 的工具结果。
</details>

**Q3. FakeModel 能替代真模型做哪些验证、不能做哪些?**

<details><summary>想清楚再展开</summary>

能:一切**机制**验证——循环结构、工具执行与配对、并行、权限拦截、会话读写、子代理编排,因为这些只依赖"消息进、消息出"的契约。不能:一切**智能**验证——模型会不会选对工具、参数填得对不对、回答质量如何。机制用桩模型测(快、稳、免费),智能用真模型评(第 11 章谈 eval)。
</details>

## 1.10 本章小结

- 四个模块:**消息、模型接口、工具、循环**。消息是记忆载体;`AIMessage.tool_calls` 空不空是循环唯一判据;`usage`/`stop_reason`/`is_error` 是生产字段,不能省。
- **模型接口**只暴露 invoke/stream/bind_tools;厂商差异关进各自翻译函数;原生工具调用消灭了"解析模型输出"这个失败环节;`FakeModel` 让一切机制离线可验。
- **工具**由 `@tool` 从签名自动生成说明书;**模型只请求、框架才执行**,这道缝隙是全部安全机制的地基。
- **循环 = ReAct 的代码化**。system prompt 每轮装配不进历史;`max_turns` 是安全阀;模型无状态,记忆 = 不断重发的完整历史。
- 这个循环就是主流 harness 的内核;从能跑到敢用,差的是工具、上下文、事件流和壳——正是接下来十章。
- **源码阅读顺序**:`messages` 定数据 → `tools` 前半定说明书 → `models` 定契约 → `k01.run_agent` 把三拍跑通;正式 `Agent.run` 留到第 4 章。

---

← [[00-全景-Agent框架与Harness]] | 下一章 → [[02-工具系统与执行环境]]
