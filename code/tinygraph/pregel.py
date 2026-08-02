"""Pregel -- BSP(整体同步并行)执行引擎,tinygraph 的心脏。

对应 LangGraph 内部的 PregelLoop。它把一张 StateGraph 一步步地、确定性地
执行出来。核心是 "超步(super-step)" 循环:

    每个超步 =  规划(选出要跑的节点)
              -> 并行执行(每个节点拿到状态的独立副本)
              -> 提交(把所有写入按确定性顺序经 reducer 合并进 channel,版本+1)
              -> 存档(若配了 checkpointer,把状态快照存成检查点)

这种结构让我们天然获得:确定性并发、对环的支持、以及 "每步之间" 这个万能
插入点(检查点 / 流式 / 中断都挂在这里)。

阅读顺序建议:本文件的主角是 stream()(主循环 / 唯一执行入口),其余下划线
方法都是被它在各个阶段调用的 "零件"。先看 stream,再随调用展开各零件即可。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, Optional

from .channels import Channel
from .checkpoint import Checkpoint, new_checkpoint_id
from .graph import END, START, StateGraph
from .interrupt import GraphInterrupt
from .messages import revive


class CompiledGraph:
    """编译后的可执行图。StateGraph.compile() 的产物,真正跑图的就是它。

    持有三样东西:
      - graph       : 声明好的 StateGraph(节点 / 边 / 状态 schema)
      - checkpointer: 检查点持久化器,None 表示不存盘(进程内跑完即弃)
      - recursion_limit: 超步数硬上限,防止带环图(如 Agent 反复调工具)无限循环
    """

    def __init__(self, graph: StateGraph, checkpointer: Any = None) -> None:
        self.graph = graph
        self.checkpointer = checkpointer
        self.recursion_limit = 25

    # ---- channel 的建立 / 快照 / 从检查点恢复 ----------------------------
    # channel 是状态的存储单位(见 channels.py):每个状态 key 一个 Channel,
    # 持有 当前值 + 单调递增的版本号 + reducer(合并策略)。下面这组方法负责
    # channel 的 "建空 / 从检查点还原 / 拍快照 / 打包成检查点"。

    def _fresh_channels(self) -> Dict[str, Channel]:
        """全新开始时用:按 schema 给每个状态 key 建一个空 Channel。

        graph.schema 形如 {"messages": add_reducer, "count": last_value},
        每个 value 就是该 channel 的 reducer。Channel(reducer) 初始 value=None、version=0。
        """
        return {k: Channel(reducer) for k, reducer in self.graph.schema.items()}

    def _channels_from_checkpoint(self, cp: Checkpoint) -> Dict[str, Channel]:
        """从检查点还原 channels:先建空,再把存盘的值/版本灌回去。

        关键在 revive():检查点经 JSON 序列化后,消息对象变成了普通 dict;
        revive() 递归地把带 __type__ 标记的 dict 还原成 HumanMessage / AIMessage
        等真实对象。这是 "可移植检查点" 的前提--状态要能 序列化->存盘->读回->还原。
        版本号也一并还原,使恢复后的状态和存盘那一刻逐位相同。
        """
        channels = self._fresh_channels()
        for key, ch in channels.items():
            if key in cp.channel_values:
                ch.value = revive(cp.channel_values[key])   # JSON dict -> 消息对象
                ch.version = cp.channel_versions.get(key, 0)  # 还原版本,没有则 0
        return channels

    @staticmethod
    def _state_snapshot(channels: Dict[str, Channel]) -> Dict[str, Any]:
        """把所有 channel 的当前值拍成一个普通 dict。

        这就是节点函数能读到的状态视图。Channel.snapshot() 直接返回 self.value。
        静态方法:不依赖 self,纯函数式地读 channels。
        """
        return {k: ch.snapshot() for k, ch in channels.items()}

    def _make_checkpoint(
        self,
        thread_id: str,
        channels: Dict[str, Channel],
        next_nodes: List[str],
        parent_id: Optional[str],
        step: int,
    ) -> Checkpoint:
        """把当前 channels 打包成一个 Checkpoint 对象,用于存盘。

        - channel_values / channel_versions: 各取一遍 channel 的值与版本
        - next_nodes=list(next_nodes): 拷一份,防止外部修改原 list
        - checkpoint_id: 现场生成(时间戳前缀 + uuid,同 thread 内大致按时间有序)
        - parent_id / step: 透传,用于把检查点串成链(时间旅行)与记录步数
        """
        return Checkpoint(
            thread_id=thread_id,
            checkpoint_id=new_checkpoint_id(),
            channel_values={k: ch.value for k, ch in channels.items()},
            channel_versions={k: ch.version for k, ch in channels.items()},
            next_nodes=list(next_nodes),
            parent_id=parent_id,
            step=step,
        )

    # ---- 写入收集与提交 --------------------------------------------------
    # 一个超步里,多个节点可能写同一个 channel。这两步解决 "怎么把这些写入
    # 收拢、再按确定性顺序合并进旧值" 的问题。确定性是可复现性的根基。

    @staticmethod
    def _collect_writes(updates: Optional[Dict[str, Any]], sink: Dict[str, List[Any]]) -> None:
        """把一个节点返回的 updates({key: value}) 收进 sink。

        sink 形如 {key: [值1, 值2, ...]}--一个 key 可攒多个值,因为同一超步里
        多个并行节点可能写同一个 channel。setdefault(key, []) 在 key 不存在时
        建空 list,再 append。`not updates` 兜住 None 和空 dict。
        """
        if not updates:
            return
        for key, value in updates.items():
            sink.setdefault(key, []).append(value)

    @staticmethod
    def _commit(channels: Dict[str, Channel], writes: Dict[str, List[Any]]) -> None:
        """把收集到的 writes 提交进 channels。

        按 sorted(writes.keys()) 的字典序遍历--这是 "确定性顺序":并行节点谁先
        完成不可控,但提交顺序固定,从而同输入永远得同结果。Channel.update(一组值)
        会把这组值依次经 reducer 合并进当前值,然后 version += 1。
        """
        for key in sorted(writes.keys()):          # 确定性顺序:字典序固定
            if key in channels:
                channels[key].update(writes[key])

    def _plan(self, active: List[str]) -> List[str]:
        """规划:从待激活节点里只留图中真实存在的节点名。

        过滤掉 START / END 等虚拟节点名,以及可能的拼写错误。空列表表示没东西可跑。
        """
        return [n for n in active if n in self.graph.nodes]

    # ---- 执行一个超步 ----------------------------------------------------
    def _run_superstep(self, tasks: List[str], channels: Dict[str, Channel]) -> Dict[str, List[Any]]:
        """执行一个超步的所有任务,返回收集到的 writes。

        - 先拍状态快照;每个节点都拿 dict(snapshot)--一份浅拷贝副本,保证并行
          节点互不干扰地读状态。
        - 单节点直接同步跑;多节点用线程池并行跑。
        - 收集结果时按 tasks 原始顺序(而非完成顺序)遍历--再次保证可复现。
        - 若节点内部调了 interrupt() 抛 GraphInterrupt,异常会一路冒泡到主循环
          被捕获(本方法不处理)。
        """
        snapshot = self._state_snapshot(channels)
        writes: Dict[str, List[Any]] = {}
        if len(tasks) == 1:
            # 单节点:直接取出节点函数,传一份状态浅拷贝,拿回部分更新
            updates = self.graph.nodes[tasks[0]](dict(snapshot))
            self._collect_writes(updates, writes)
        else:
            # 多节点:开线程池并行。worker 数 = 任务数。
            with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
                # 字典推导:{Future 对象: 节点名}。pool.submit(fn, arg) 立即异步
                # 提交任务并返回 Future;每个节点各拿一份 dict(snapshot) 浅拷贝。
                futures = {
                    pool.submit(self.graph.nodes[name], dict(snapshot)): name
                    for name in tasks
                }
                # f.result() 阻塞等待该任务完成并取返回值;futures[f] 把它映射回节点名。
                # 这里 results 已是 {节点名: updates}。
                results = {futures[f]: f.result() for f in futures}
                for name in tasks:                  # 按固定顺序收集 -> 可复现
                    self._collect_writes(results[name], writes)
        return writes

    # ---- 解析边 ----------------------------------------------------------
    # 这组方法决定 "下一批跑谁"。本简化引擎用【边】决定触发(不走 LangGraph
    # 的 "订阅 channel + 比较版本号" 触发模型)。

    def _resolve_next(self, node: str, state: Dict[str, Any]) -> List[str]:
        """求一个节点的全部后继 = 固定边 + 条件边。

        - 固定边(add_edge):edges[node] = [to, ...],无条件激活。
        - 条件边(add_conditional_edges):branches[node] = (router, mapping)。
          调 router(state) 拿路由结果(单个值或列表,支持扇出);返回值经 mapping
          映射成真实节点名(mapping 没命中或无 mapping 则直接当节点名用)。
          返回 END 表示该分支结束。
        """
        nxt: List[str] = []
        # 1) 固定边:取该节点的所有无条件后继
        for target in self.graph.edges.get(node, []):
            nxt.append(target)
        # 2) 条件边:若有,调路由函数决定去向
        if node in self.graph.branches:
            router, mapping = self.graph.branches[node]
            result = router(state)
            # 单值也包成 list,统一迭代;支持 router 一次返回多个目标(扇出)
            for c in (result if isinstance(result, list) else [result]):
                nxt.append(mapping.get(c, c) if mapping else c)
        return nxt

    def _next_active(self, tasks: List[str], state: Dict[str, Any]) -> List[str]:
        """汇总所有刚执行节点的后继,得到下一批待激活节点。

        去掉 END(到终点),并去重(多个节点可能指向同一后继)。返回空 = 全部分支
        已结束,主循环将退出。
        """
        out: List[str] = []
        for name in tasks:
            for target in self._resolve_next(name, state):
                if target != END and target not in out:
                    out.append(target)
        return out

    # ---- 主循环 ----------------------------------------------------------
    def stream(
        self,
        input: Optional[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
        stream_mode: str = "updates",
    ) -> Iterator[Dict[str, Any]]:
        """执行图(生成器)。config={"configurable": {"thread_id": ...}} 用于检查点。

        input 的三种形态:
          - dict              : 全新开始执行
          - None              : 从最新检查点恢复(断点续跑)
          - Command(resume=x) : 从中断处恢复,并把 x 注入给 interrupt() 的返回值

        stream_mode:
          - "values"  : 每步 yield 完整状态快照
          - "updates" : 每步 yield {__step__, nodes, state} 结构化进度
        """
        from .interrupt import Command, set_resume

        # 1) 取线程 id(一条独立执行线,如一个用户会话);缺省 "__default__"
        thread_id = self._thread_id(config)

        # 2) Command(resume=...) -> 把恢复值塞进全局盒子,再按 "从检查点恢复" 处理。
        #    这个值稍后会被被中断的节点重跑时,从 interrupt() 里取到。
        if isinstance(input, Command):
            set_resume(input.resume)
            input = None

        # 3) 初始化:判出两个检查点标志,决定 "从哪起步"。
        #    resume_cp:断点 / 中断恢复时,要接着跑的检查点
        #    prior_cp  :input=dict 且该 thread 已有【已完成】检查点 -> 视为 "接着对话"
        resume_cp = None
        prior_cp = None
        if input is None and self.checkpointer is not None:
            # input=None:断点续跑 / 中断恢复。取最新检查点;没有就报错(没法恢复)。
            resume_cp = self.checkpointer.get(thread_id)
            if resume_cp is None:
                raise ValueError(f"thread {thread_id} 没有可恢复的检查点")
        elif isinstance(input, dict) and self.checkpointer is not None:
            # input=dict:看是否已有历史。仅当历史检查点已执行完(next_nodes 为空)
            # 才视为 "接着对话"。
            prior = self.checkpointer.get(thread_id)
            if prior is not None and not prior.next_nodes:
                prior_cp = prior

        # 4) 根据上面判出的标志,走三条路之一,确定 channels / active / steps / parent_id。
        if resume_cp is not None:
            # 路 A:断点 / 中断恢复。从检查点重建 channels;active 取它记录的 next_nodes
            # (中断恢复时 next_nodes 正是被中断的节点 -> 会重跑);steps / parent_id 还原。
            channels = self._channels_from_checkpoint(resume_cp)
            active = list(resume_cp.next_nodes)
            steps = resume_cp.step
            parent_id = resume_cp.checkpoint_id
        else:
            if prior_cp is not None:
                # 路 B:接着对话。从上轮结束状态起步,沿用 parent_id / steps。
                channels = self._channels_from_checkpoint(prior_cp)
                parent_id = prior_cp.checkpoint_id
                steps = prior_cp.step
            else:
                # 路 C:全新开始。建空 channels,从第 0 步开始。
                channels = self._fresh_channels()
                parent_id = None
                steps = 0

            # 路 B/C 都是从一个新的用户输入开始，必须先把 input 合并进历史，
            # 再从 START 解析本轮 active；否则多轮对话没有下一步可执行。
            init: Dict[str, List[Any]] = {}
            self._collect_writes(input, init)
            self._commit(channels, init)
            active = self._resolve_next(START, self._state_snapshot(channels))

            # 初始检查点覆盖路 B/C，保证第一步执行前也能恢复。
            if self.checkpointer is not None:
                cp = self._make_checkpoint(thread_id, channels, active, parent_id, steps)
                self.checkpointer.put(cp)
                parent_id = cp.checkpoint_id

        # 5) 超步循环:每轮 = 规划 -> 执行 -> (中断处理) -> 提交 -> 求下一批 -> 存档 -> 产出
        while active:
            steps += 1
            if steps > self.recursion_limit:
                raise RecursionError(f"超过最大超步数 {self.recursion_limit}")
            tasks = self._plan(active)                  # 规划:滤出真实节点
            if not tasks:
                break

            try:
                writes = self._run_superstep(tasks, channels)     # 执行这批节点
            except GraphInterrupt as gi:
                # 节点抛出中断:不当错误处理。存检查点--其 next_nodes 仍是 tasks
                # (指向被中断的节点),所以下次恢复会重跑该节点;然后 yield 中断值
                # 给外部(如等人审批),return 结束本次执行。
                if self.checkpointer is not None:
                    cp = self._make_checkpoint(thread_id, channels, tasks, parent_id, steps)
                    self.checkpointer.put(cp)
                yield {"__interrupt__": gi.value}
                return

            self._commit(channels, writes)                        # 提交:按确定性顺序合并
            state_now = self._state_snapshot(channels)            # 拍当前状态
            active = self._next_active(tasks, state_now)          # 求下一批(空则循环结束)

            # 存档:每个超步存一个检查点,parent_id 指向上一个 -> 串成链,支持时间旅行
            if self.checkpointer is not None:
                cp = self._make_checkpoint(thread_id, channels, active, parent_id, steps)
                self.checkpointer.put(cp)
                parent_id = cp.checkpoint_id

            # 流式产出:每跑完一个超步 yield 一次,调用方能边跑边看
            if stream_mode == "values":
                yield state_now
            else:
                yield {"__step__": steps, "nodes": tasks, "state": state_now}

    def invoke(
        self,
        input: Optional[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """同步执行并返回最终状态。是 stream() 的薄包装:"只要最终结果,不要中间步"。

        用 values 模式跑完生成器,取最后一个快照;遇中断则透传中断信号;
        若一个快照都没产出(从已结束的检查点恢复),从 get_state 回填当前状态。
        """
        last: Dict[str, Any] = {}
        for snap in self.stream(input, config, stream_mode="values"):
            if "__interrupt__" in snap:
                return snap        # 把中断信号透传给调用方,而非抛异常
            last = snap
        if not last:
            # 从检查点恢复但没产出新超步(已结束)时,回填当前状态,避免返回空 dict
            cp = self.get_state(config)
            if cp is not None:
                last = cp.channel_values
        return last

    # ---- 状态查询 / 时间旅行 --------------------------------------------
    # 主循环只管 "写" 检查点;这组方法负责 "读":查最新 / 查指定历史点 / 列全部。
    # 因为每个超步都存了检查点,所以能回到任意历史点重跑或分支--即时间旅行。

    def get_state(self, config: Optional[Dict[str, Any]]) -> Optional[Checkpoint]:
        """取某 thread 的检查点:默认最新;config 里带 checkpoint_id 则取指定历史点。"""
        if self.checkpointer is None:
            return None
        cfg = (config or {}).get("configurable", {})
        return self.checkpointer.get(self._thread_id(config), cfg.get("checkpoint_id"))

    def get_state_history(self, config: Optional[Dict[str, Any]]) -> List[Checkpoint]:
        """列出某 thread 的全部检查点,最新的在前(用于时间旅行浏览历史)。"""
        if self.checkpointer is None:
            return []
        return self.checkpointer.list(self._thread_id(config))

    @staticmethod
    def _thread_id(config: Optional[Dict[str, Any]]) -> str:
        """从 config 取 thread_id;config 缺失或没配 thread_id 都回退到 "__default__"。"""
        if not config or "configurable" not in config:
            return "__default__"
        return config["configurable"].get("thread_id", "__default__")
