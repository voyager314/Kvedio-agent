# 系统设计

本文把架构主张落到代码必须确定的部分：模块边界、数据形状、任务推进方式、失败如何收敛。
需求与选型论证不在此重复；凡本文未写的，按仓库根目录 `AGENTS.md` 的规则处理。

## 1. 不变量

四条约束贯穿所有模块，任何实现都不得违反。

1. **转录全文不进提示词。** 送入模型的永远是检索到的片段，带时间戳、有上限。整份转录只存在于库里。
2. **产物绑定输入身份。** 一份转录、一帧描述，只在产生它的视频内容、阶段参数和模型版本都相同时才可复用。身份变了，产物失效，而不是静默沿用。
3. **界面只有一个写入者。** 控件树只由界面线程修改。工作线程的进展先变成事件，再由界面线程渲染。
4. **密钥只以间接名出现。** 配置文件可存 `env:NAME` 形式的引用；界面、日志、异常只显示"已配置/未配置"，不显示密钥的任何片段。

## 2. 运行时结构

### 2.1 执行体

单进程，三类执行体，职责互不重叠。

| 执行体 | 拥有 | 不得做 |
| --- | --- | --- |
| 界面线程（Textual 事件循环） | 控件树、事件泵、用户输入 | 阻塞 IO、解码、网络请求 |
| 工作线程（有界线程池） | 一个任务的一次执行 | 触碰控件、静默吞掉异常 |
| 主线程 | 装配、启动、退出 | 承载长任务 |

媒体解码、文件哈希、模型调用都在工作线程里同步执行。异步只出现在界面侧；流水线内部不用协程——取消靠令牌与超时收敛，进度靠事件发布。这样做的原因是取消语义：线程无法被强杀，只能靠令牌在检查点退出，而协程取消会把"已发出的付费请求"变成不可观测状态。

### 2.2 事件流

```
工作线程 ──publish──► EventBus（线程安全队列） ──drain──► 界面泵（120ms）
                                             └────────► 追踪写入（作业目录 JSONL）
```

事件是工作线程向外传递状态的唯一通道，队列由标准库提供，任意线程可写。界面泵在界面线程里非阻塞取空队列，因此不存在"工作线程更新控件"的路径。

事件结构固定为四项：种类、发生时刻、所属作业、所属任务，外加一个自由负载（阶段名、进度、计数、耗时）。已定义种类：

- `video.registered` — 一条视频被引用并完成探测
- `job.queued` `job.started` `job.succeeded` `job.failed` `job.canceled`
- `task.ready` `task.started` `task.progress` `task.succeeded` `task.failed` `task.canceled` `task.skipped`
- `models.changed` — 某个角色的当前绑定被切换
- `log.line` — 面向用户的一行说明，不是调试日志

调试日志走标准库 `logging`，与事件流分离：日志可以丢弃，事件不能。

### 2.3 一次引用的生命周期

1. 输入框提交 `@相对路径 问题`，`pipeline` 解析路径为绝对路径。
2. 计算指纹（毫秒级），按指纹查库：命中则复用视频行，未命中则插入并探测。
3. 探测写入时长、容器、视频编解码与分辨率、音轨参数。
4. 建立作业与任务图并交给调度器。M0 的任务图只有一个探测后的登记任务；M1 起扩展为探测 → 转录 → 索引。
5. 任务推进以事件呈现，界面按事件更新任务面板、时间轴与证据区。

## 3. 目录与角色

目录即角色。一个文件放在哪里，只由它承担的角色决定。「阶段」列表示该目录在哪一步出现，未出现的目录此刻不存在于代码树中。

| 目录 | 角色 | 阶段 |
| --- | --- | --- |
| `core/` | 引擎总入口：只含系统内部实现，不导入界面 | M0 |
| `core/domain/` | 跨层传递的值对象与枚举：视频记录、作业、任务、事件 | M0 |
| `core/config/` | 用户配置的发现、模板生成、校验、脱敏 | M0 |
| `core/storage/` | SQLite 连接、模式迁移、仓储 | M0 |
| `core/tasks/` | 任务状态机、事件总线、DAG 调度、工作线程池 | M0 |
| `core/pipeline/` | 把一次请求翻译成任务图并驱动阶段推进 | M0 |
| `core/media/` | 解码级媒体处理：探测、音轨提取、抽帧、场景边界 | M0 起，逐步补齐 |
| `core/workspace/` | 作业目录与产物布局、检查点键 | M1 |
| `core/llm/` | 模型接入：OpenAI 兼容客户端、重试、用量记账 | M1 |
| `core/asr/` | 转录流水线：切段、并发、合并、检查点 | M1 |
| `core/supervisor/` | 意图路由、计划生成、结论合成、质量评估 | M2 |
| `core/memory/` | 索引与检索：关键词、向量、证据选取 | M3 |
| `core/video/` | 视觉理解：抽帧策略、帧描述、事件抽取 | M4 |
| `interface/` | 终端界面：子命令、屏幕、面板、事件泵、输入语法 | M0 |
| `tools/` | 暴露给监督者的工具实现，一族一模块 | M2 |
| `prompts/` | 提示词模板 | M2 |
| `default/` | 随包分发的默认文件：配置样例 | M0 |
| `temp/` | 一次性脚本与试验产物，被 `.gitignore` 覆盖，不入库 | — |

依赖方向单向：`interface` → `core`；`core` 内部 `supervisor` → (`asr` `video` `memory`) → (`media` `llm` `storage` `workspace`) → (`domain` `config`)。`tools` 依赖 `core` 的检索与存储，反向不成立。`prompts` 是数据，不被导入。任何一个模块都不导入 `interface`。

## 4. 数据模型

SQLite 单文件库，位置 `~/.kvedio_agent/kvedio.db`。开启外键约束；写操作按作业串行，读操作不限。

### 4.1 表

```sql
-- 一条被引用的源视频。路径只用于定位，不参与判断"是不是同一个文件"。
CREATE TABLE video (
  id              INTEGER PRIMARY KEY,
  path            TEXT    NOT NULL,          -- 绝对路径
  size_bytes      INTEGER NOT NULL,
  mtime_ns        INTEGER NOT NULL,
  fingerprint     TEXT    NOT NULL UNIQUE,   -- 见 §8
  content_hash    TEXT,                      -- 全文件 sha256；NULL 表示尚未整读
  duration_ms     INTEGER,                   -- NULL 表示探测未完成
  container       TEXT,
  video_codec     TEXT,
  width           INTEGER,
  height          INTEGER,
  fps             REAL,
  audio_codec     TEXT,
  sample_rate     INTEGER,
  channels        INTEGER,
  created_at      TEXT    NOT NULL,
  updated_at      TEXT    NOT NULL
);

-- 一次分析请求。question 是用户原话，不改写。
CREATE TABLE job (
  id              TEXT    PRIMARY KEY,
  video_id        INTEGER NOT NULL REFERENCES video(id),
  question        TEXT    NOT NULL,
  status          TEXT    NOT NULL,          -- queued|running|succeeded|failed|canceled|partial
  created_at      TEXT    NOT NULL,
  started_at      TEXT,
  finished_at     TEXT,
  error           TEXT,                      -- 成功为 NULL
  plan_json       TEXT                       -- 计划快照，作业开始后不改写
);

-- 任务图的一个节点。依赖以同表 id 的 JSON 数组表示，值而非实体。
CREATE TABLE task (
  id              TEXT    PRIMARY KEY,
  job_id          TEXT    NOT NULL REFERENCES job(id),
  agent           TEXT    NOT NULL,          -- media|asr|video|supervisor
  action          TEXT    NOT NULL,
  status          TEXT    NOT NULL,          -- 见 §5.1
  depends_on      TEXT    NOT NULL,          -- JSON 数组
  dependency_mode TEXT    NOT NULL,          -- all|any|partial
  attempt         INTEGER NOT NULL DEFAULT 0,
  progress        REAL    NOT NULL DEFAULT 0,
  started_at      TEXT,
  finished_at     TEXT,
  error           TEXT,
  result_ref      TEXT                       -- 产物引用，仓库内路径或表内 id
);

CREATE INDEX idx_task_job ON task(job_id);
```

`transcript_segment`（M1）、`artifact`（M1）、`checkpoint`（M1）随对应阶段建立，字段定义在建立该表时补入本文档。

### 4.2 引用规则

模型之间只存 id，不嵌套记录。`job` 不内嵌 `video`，`task` 不内嵌 `job`；读取时由仓储按 id 解析。一个记录不因被引用而被复制。

### 4.3 迁移

`schema_version` 单行表记录当前版本。启动时若版本低于代码期望，按顺序执行迁移函数；不引入迁移框架，迁移是纯 SQL 加必要的数据搬运。库文件损坏或版本高于代码时，拒绝启动并给出路径。

## 5. 任务模型

### 5.1 状态与迁移

```
pending ──► ready ──► running ──► succeeded
   │         │          │  └───► failed ──► ready（重试）
   │         │          └──────► canceled
   └─────────┴─────────────────► skipped
```

合法迁移写在一张表里，由单一函数校验，非法迁移抛出异常而不是静默忽略：

| 起点 | 可到 |
| --- | --- |
| `pending` | `ready` `canceled` `skipped` |
| `ready` | `running` `canceled` `skipped` |
| `running` | `succeeded` `failed` `canceled` |
| `failed` | `ready`（`attempt` 未达上限） `canceled` |
| `succeeded` `canceled` `skipped` | 无 |

重试是新的一次 `running`，`attempt` 自增，之前的 `error` 保留到下一次开始时清空。

### 5.2 依赖语义

每个节点声明依赖模式，调度器据此判断就绪与失败：

| 模式 | 就绪条件 | 失败条件 |
| --- | --- | --- |
| `all` | 全部依赖成功 | 任一依赖失败或被取消 |
| `any` | 任一依赖成功 | 全部依赖失败或被取消 |
| `partial` | 全部依赖进入终态 | 永不因依赖失败；成功数为零时视为无输入 |

`partial` 是"ASR 成了但视觉失败仍要出结论"的表达方式，也是长视频里最常见的收敛路径。

### 5.3 取消、中断与恢复

- 取消请求写令牌，调度器不再派发新节点，已在跑的节点在下一个检查点退出；退出后的状态是 `canceled`，不是 `failed`。
- 检查点的最小粒度是一个媒体分片或一次模型调用，取消在两次调用之间生效。
- 进程退出时仍为 `running` 的任务，在下次启动时被重置为 `ready` 并保留 `attempt`，由调度器重新派发。这是"恢复"而非"重来"：已完成的任务不会被重算。

### 5.4 部分成功

作业的终态由任务图决定：全部成功为 `succeeded`；有成功也有失败为 `partial`；全部失败为 `failed`。`partial` 的作业照常产出结论，结论里必须标明缺失的那部分证据。

## 6. 并发

- 全局上限来自 `runtime.max_global_workers`，作用于同时运行的任务数。
- 单个任务内部的并发（转录分片、视觉批）由各自阶段限制，不占全局名额之外的额度：阶段内部再开线程池时，其大小由该阶段配置决定，且不超过全局上限。
- 依赖未满足的节点不占用线程。
- 排队不做公平性承诺：同一作业内按拓扑序，不同作业之间按提交顺序。

## 7. 配置与密钥

### 7.1 位置与创建

用户配置固定在 `~/.kvedio_agent/config.yml`，程序只读。首次运行时若文件不存在，按随包模板创建并在界面提示路径与"需要填写密钥"。程序自身的状态（当前选中的模型绑定等）写 `~/.kvedio_agent/state.yml`，与用户配置分离——写回用户文件会丢掉注释与排版。

### 7.2 结构

```yaml
active:                        # 每个角色当前使用的绑定名
  supervisor: deepseek
  asr: dashscope
  video: dashscope

roles:
  supervisor:
    - name: deepseek
      model: deepseek-chat
      api_base: https://api.deepseek.com/v1
      api_key: env:DEEPSEEK_API_KEY
  asr:
    - name: dashscope
      model: qwen3-asr-flash
      api_base: https://dashscope.aliyuncs.com/compatible-mode/v1
      api_key: env:DASHSCOPE_API_KEY
  video:
    - name: dashscope
      model: qwen-vl-max
      api_base: https://dashscope.aliyuncs.com/compatible-mode/v1
      api_key: env:DASHSCOPE_API_KEY

runtime:
  max_video_duration_seconds: 7200
  retention_days: 30
  max_global_workers: 6
  request_timeout_seconds: 120
  workspace_dir: null          # null 表示 ~/.kvedio_agent/workspace
```

三类角色共用一种绑定形状：`name` 在角色内唯一；`api_key` 允许为空，表示尚未填写，此时绑定"不可用"而不是"非法"——模板刚生成时必然如此。`env:NAME` 形式的引用在使用时才解析，配置展示时原样显示引用名。

### 7.3 脱敏

展示层拿到的配置对象只有 `configured: bool`，没有密钥字符串。脱敏不靠"打印时打码"，而是让密钥从不出现在可被打印的对象里：绑定对象把密钥存在私有字段，`repr` 与序列化都不包含它。

### 7.4 角色绑定的切换

界面命令 `/model` 列出三个角色的全部绑定与可用状态，选择后写入 `state.yml` 的 `active`，发 `models.changed` 事件。绑定名在配置里消失时，回退到该角色第一个可用绑定，并在界面说明回退原因。

## 8. 视频身份、指纹与检查点

- **指纹** = `sha256(size ‖ mtime_ns ‖ 头 4 MiB ‖ 尾 4 MiB)`。一次引用只读 8 MiB，毫秒级完成，足以发现替换、截断与改动。
- **内容哈希** = 全文件 sha256，在首次顺序解码时顺带算出（解码本来就要读完整份文件），写入 `content_hash`。此后同一内容换路径、改名，都能凭哈希命中已有产物。
- **复用键** = `content_hash + stage + model_id + params_hash`。`content_hash` 为空时退回指纹，并在检查点记录里标注"可能重复计算"。
- 源文件变动不做自动跟踪：打开作业前重算指纹，与库里不一致时该视频的既有产物视为失效，界面提示"源文件已变化"，作业需重新开始。

## 9. 上下文策略

长视频的上下文问题不靠"塞更多"，靠三级收窄：

1. **覆盖图**：先给模型一份区间清单（话题或场景 → 起止时间 → 一句话），回答"视频里讲了什么"只需这张图。
2. **检索窗口**：按问题取回命中的段，每段保留起止时间与原文，合并后单次证据不超过 6000 token。
3. **下钻**：模型认为证据不足时，通过工具请求指定区间的更多上下文，每次请求同样有上限。

工具返回值是片段进入上下文的唯一入口。工具包括：按时间取段、按关键词检索、按说话人取段、取区间摘要（M2 起逐步补齐）。

两种记忆分开：会话记忆是当前对话与任务状态，随会话消失；视频记忆是库里的转录、场景与索引，跨会话存在，只通过工具访问。二者不共享上下文窗口。

## 10. 界面

四区加一条状态栏：

| 区域 | 内容 | 空态 |
| --- | --- | --- |
| 对话 | 用户输入、系统回复、流式输出 | 显示用法提示与配置状态 |
| 任务 | 任务 ID、角色、动作、状态、进度、耗时 | 显示"尚未引用视频" |
| 时间轴 | 时长刻度与区间标记（M1 起填充转录与场景） | 显示已知时长 |
| 证据 | 当前结论引用的时间戳、文本、关键帧路径 | 显示"暂无证据" |

状态栏常驻：当前数据库路径、三个角色的绑定名与可用状态、运行中任务数。

输入语法：`@<相对路径> <问题>` 引用视频并起作业；`/model` 切换绑定；`/help` 列出语法；`/quit` 退出。路径解析以进程工作目录为基准，解析失败在对话区报错并保留原文，不清空输入。

## 11. 里程碑

每一步以可验证的行为收尾，不以下一步的铺垫收尾。

**M0 地基（本次）**
- 引用本地 mp4：指纹、探测、入库，重复引用不产生第二条记录。
- 作业与任务图落库并调度，状态迁移由状态机校验。
- 工作台四区渲染，任务面板随事件实时更新。
- 首次运行生成配置模板；`/model` 查看与切换绑定并写回状态文件。
- 启动时把上次残留的 `running` 任务恢复为 `ready`。

验收：一条真实 mp4 被引用后，时长、分辨率、音轨在界面可见；改动文件后再次引用产生新记录并在对话区说明；非法迁移、坏配置、密钥泄漏各有测试；`uv run pytest` 与 `ruff check` 通过。

**M1 转录**：在线 ASR 分段时间戳、分片并发、检查点、断点续跑、时间轴填充。
验收：同一视频二次分析零转录开销；中断后重跑只补未完成分片。

**M2 监督者**：规则与模型双层路由、计划生成、工具调用、结论合成、阶段轨迹、质量评估。
验收：每条结论都带可点开的时间戳证据；计划与轨迹可在界面查看。

**M3 记忆与检索**：关键词索引、向量索引、覆盖图、按问题取证。
验收：小时级视频的提问在固定证据预算内作答，并记录定位误差。

**M4 视觉**：场景边界、抽帧策略、帧描述、事件抽取、跨模态对齐。
验收：画面类问题给出时间戳与关键帧。

**M5 硬化**：取消与恢复全链路、用量记录、保留策略、并发调优。性能基准与评测集不在实现范围内。

## 12. 不在范围内

本地 ASR 模型部署与推理优化、GPU 显存调度与动态批处理、图数据库与 GraphRAG、分布式编排框架、评测数据集构建。这些不进入代码，配置里也不留占位字段。
