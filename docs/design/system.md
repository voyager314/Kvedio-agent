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
- `job.queued` `job.started` `job.succeeded` `job.partial` `job.failed` `job.canceled`
- `task.ready` `task.started` `task.progress` `task.succeeded` `task.failed` `task.canceled` `task.skipped`
- `log.line` — 面向用户的一行说明，不是调试日志

调试日志走标准库 `logging`，与事件流分离：日志可以丢弃，事件不能。

### 2.3 一次引用的生命周期

1. 输入框提交 `@相对路径 问题`，`pipeline` 解析路径为绝对路径。
2. 计算指纹（毫秒级），按指纹查库：命中则复用视频行，未命中则插入一条只带身份的记录。
3. 按已有产物裁剪任务图，只保留缺的节点：探测 → 提取音轨 → 转录 → 作答。首次引用四个节点都跑；已转写过的视频再问一次，只跑作答。
4. 调度器按依赖推进节点，每次状态变化先落库再发事件。
5. 作答节点内部再分三步：让模型给出工具计划 → 按计划检索转录 → 让模型写结论并校验每条引用。计划与检索轨迹以 `log.line` 出现在对话区。
6. 界面按事件更新任务面板；探测完成后时间轴显示时长，转录完成后显示段数、覆盖区间与模型，作答完成后证据区换成本次结论引用的时间点。

## 3. 目录与角色

目录即角色。一个文件放在哪里，只由它承担的角色决定。「阶段」列表示该目录由哪一步引入。

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
| `core/llm/` | 模型接入：对话、转写、嵌入三个客户端，共用重试规则 | M1 起，M3 补齐嵌入 |
| `core/asr/` | 转录流水线：切段、并发、合并、检查点 | M1 |
| `core/supervisor/` | 计划生成、检索编排、结论合成、引用校验 | M2 |
| `core/memory/` | 覆盖图构建与混合检索：关键词、向量、证据选取 | M3 |
| `core/video/` | 视觉理解：抽帧策略、帧描述、事件抽取 | M4 |
| `interface/` | 终端界面：子命令、屏幕、面板、事件泵、输入语法 | M0 |
| `tools/` | 暴露给监督者的工具实现，一族一模块：转录检索 | M2 |
| `prompts/` | 提示词模板：Jinja2，纯英文，作答语言随提问 | M2 |
| `default/` | 随包分发的默认文件：配置样例 | M0 |
| `temp/` | 一次性脚本与试验产物，被 `.gitignore` 覆盖，不入库 | — |

依赖方向：`interface` → `core.pipeline` → (`asr` `supervisor`) → (`media` `llm` `storage` `workspace`) → (`domain` `config`)。`core.supervisor` 另外调用 `tools`，`tools` 只读取 `domain` 与 `storage`，不导入 `pipeline`、`supervisor` 与界面，因此这条边不构成环。`prompts` 是数据，不被导入。任何一个模块都不导入 `interface`。

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
  error           TEXT                       -- 成功为 NULL
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
  max_attempts    INTEGER NOT NULL DEFAULT 1, -- 允许的总运行次数，1 表示不重试
  progress        REAL    NOT NULL DEFAULT 0,
  started_at      TEXT,
  finished_at     TEXT,
  error           TEXT
);

CREATE INDEX idx_task_job ON task(job_id);
```

一个视频只有一份转录。`cache_key` 记录这份转录是在什么条件下产生的；`completed_at` 为空表示仍在写入，此时任何读取都不引用它。

```sql
CREATE TABLE transcript (
  id           INTEGER PRIMARY KEY,
  video_id     INTEGER NOT NULL UNIQUE REFERENCES video(id),
  cache_key    TEXT    NOT NULL,          -- 见 §8
  model        TEXT    NOT NULL,
  created_at   TEXT    NOT NULL,
  completed_at TEXT                       -- NULL 表示仍在写入
);

-- 分段不存序号：顺序由时间给出，并发写入不需要协调编号。
CREATE TABLE transcript_segment (
  id            INTEGER PRIMARY KEY,
  transcript_id INTEGER NOT NULL REFERENCES transcript(id),
  start_ms      INTEGER NOT NULL,
  end_ms        INTEGER NOT NULL,         -- 不含；恒大于 start_ms
  text          TEXT    NOT NULL
);

CREATE INDEX idx_segment_transcript ON transcript_segment(transcript_id, start_ms);

-- 一个分片一行。重复写入同一分片是覆盖，不是追加。
CREATE TABLE checkpoint (
  video_id    INTEGER NOT NULL REFERENCES video(id),
  stage       TEXT    NOT NULL,           -- 产生该分片的阶段，当前只有 transcribe
  cache_key   TEXT    NOT NULL,
  chunk_index INTEGER NOT NULL,
  start_ms    INTEGER NOT NULL,
  end_ms      INTEGER NOT NULL,
  created_at  TEXT    NOT NULL,
  PRIMARY KEY (video_id, stage, cache_key, chunk_index)
);
```

一份作业一条结论；引用单独成行，`quote` 存的是转录原文而不是模型转述，读结论的人不必回头再检索一次。

```sql
CREATE TABLE answer (
  id         INTEGER PRIMARY KEY,
  job_id     TEXT    NOT NULL UNIQUE REFERENCES job(id),  -- 一个作业只答一次
  video_id   INTEGER NOT NULL REFERENCES video(id),
  question   TEXT    NOT NULL,            -- 用户原话
  text       TEXT    NOT NULL,
  model      TEXT    NOT NULL,            -- 写出结论的对话模型
  created_at TEXT    NOT NULL
);

CREATE TABLE answer_citation (
  id        INTEGER PRIMARY KEY,
  answer_id INTEGER NOT NULL REFERENCES answer(id),
  start_ms  INTEGER NOT NULL,
  end_ms    INTEGER NOT NULL,             -- 不含
  quote     TEXT    NOT NULL              -- 该时刻的转录原文
);

CREATE INDEX idx_citation_answer ON answer_citation(answer_id, start_ms);
```

一份覆盖图，一条向量。两者都用缓存键标明"由哪份转录、哪个模型产生"；读的时候不看键，因为重建时会先把旧键的行删掉。

```sql
CREATE TABLE coverage_block (
  id         INTEGER PRIMARY KEY,
  video_id   INTEGER NOT NULL REFERENCES video(id),
  cache_key  TEXT    NOT NULL,          -- 转录键 + 对话模型 + 块长
  position   INTEGER NOT NULL,
  start_ms   INTEGER NOT NULL,
  end_ms     INTEGER NOT NULL,
  gist       TEXT    NOT NULL,          -- 一行摘要，可导航但不可引用
  created_at TEXT    NOT NULL,
  UNIQUE (video_id, cache_key, position)
);

CREATE TABLE segment_vector (
  video_id   INTEGER NOT NULL REFERENCES video(id),
  cache_key  TEXT    NOT NULL,          -- 转录键 + 嵌入模型
  segment_id INTEGER NOT NULL REFERENCES transcript_segment(id),
  start_ms   INTEGER NOT NULL,          -- 冗余，检索时不必回头排序
  dims       INTEGER NOT NULL,
  vector     BLOB    NOT NULL,          -- float32 小端
  PRIMARY KEY (video_id, cache_key, segment_id)
);
```

`artifact` 仍未建立：它要等视觉阶段产生独立文件，届时再随迁移加入。计划与轨迹刻意不落库：它们是本次作答的中间过程，界面上按事件显示，重放一段过期计划没有意义。

### 4.2 引用规则

模型之间只存 id，不嵌套记录。`job` 不内嵌 `video`，`task` 不内嵌 `job`；读取时由仓储按 id 解析。一个记录不因被引用而被复制。

### 4.3 迁移

`PRAGMA user_version` 记录当前版本，现为 4。启动时若版本低于代码期望，按顺序执行各版本的迁移脚本；不引入迁移框架，迁移是纯 SQL 加必要的数据搬运。版本高于代码时拒绝打开：那些列没有反向迁移，继续写会静默丢数据。

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

- 取消请求写令牌，调度器不再派发新节点，已在跑的节点在下一个检查点退出；退出后的状态是 `canceled`，不是 `failed`。检查点的最小粒度是一个媒体分片或一次模型调用。（M5 实现）
- 进程退出时仍为 `running` 的作业，在下次启动时连同其未完成节点一起判为 `failed`，原因写明"进程退出时作业未完成"。没有检查点的阶段无从判断做到哪一步，把它报告为运行中或待执行都是对读者的误导。
- 实现了检查点的阶段（M1 起）在重启后按 `ready` 重新派发：已完成的分片不会重算，这是"恢复"；未实现检查点的阶段只能"重来"，两种行为在界面上必须可区分。

### 5.4 部分成功

作业的终态由任务图决定：全部成功为 `succeeded`；有成功也有失败为 `partial`；全部失败为 `failed`。`partial` 的作业照常产出结论，结论里必须标明缺失的那部分证据。作答失败也是失败：给出一个没有引用的"结论"比给不出更糟，所以此时作业停在 `partial` 或 `failed`，界面上写明原因。

## 6. 并发

- 全局上限来自 `runtime.max_global_workers`，作用于同时运行的任务数。
- 单个任务内部的并发（转录分片、视觉批）由各自阶段配置：转录用 `runtime.asr_concurrency`，默认 2——一次上传几个几分钟的音频已经足够占满上行带宽，再多只会同时撞上对方限流。
- 依赖未满足的节点不占用线程。
- 排队不做公平性承诺：同一作业内按拓扑序，不同作业之间按提交顺序。

## 7. 配置与密钥

### 7.1 位置与创建

用户配置固定在 `~/.kvedio_agent/config.yml`，程序只读，也不另存一份选择：每个角色只有一个模型，更换模型是编辑这个文件并重启，程序不回写用户文件（回写会丢掉注释与排版）。首次运行时若文件不存在，按随包模板创建并在界面提示路径与"需要填写密钥"。

### 7.2 结构

```yaml
supervisor:
  model: deepseek-chat
  url: https://api.deepseek.com/v1/chat/completions
  api_key: env:DEEPSEEK_API_KEY
asr:
  model: qwen3-asr-flash
  url: https://dashscope.aliyuncs.com/compatible-mode/v1/audio/transcriptions
  api_key: env:DASHSCOPE_API_KEY
video:
  model: qwen-vl-max
  url: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
  api_key: env:DASHSCOPE_API_KEY
embedding:
  model: text-embedding-v3
  url: https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings
  api_key: env:DASHSCOPE_API_KEY

runtime:
  max_video_duration_seconds: 7200
  retention_days: 30
  max_global_workers: 6
  request_timeout_seconds: 120
  asr_chunk_seconds: 300       # 单次转写请求承载的音频长度
  asr_concurrency: 2           # 同时发出的转写请求数
  coverage_block_seconds: 180  # 覆盖图每块的时长
  workspace_dir: null          # null 表示 ~/.kvedio_agent/workspace
```

四类角色共用一种形状：`model`、`url`、`api_key`。

`url` 是**完整地址**，调用时原样使用：不拼接路径、不改写末尾斜杠、不补版本段。供应商的路径各不相同（阿里云的兼容路径是 `/compatible-mode/v1/...`，有的供应商把版本放在别处），由用户照文档填写；缺少 `http(s)://` 在载入时就拒绝，因为半截地址要到第一次请求才暴露。

`api_key` 允许为空，表示尚未填写，此时该角色"不可用"而不是配置非法——模板刚生成时必然如此。`env:NAME` 形式的引用在使用时才解析，配置展示时只显示引用名。

### 7.3 脱敏

展示层拿到的配置对象只有"是否可用"，没有密钥字符串。脱敏不靠"打印时打码"，而是让密钥从不出现在可被打印的对象里：密钥字段不参与 `repr` 与序列化，展示时只报模型名、地址与环境变量名。

### 7.4 更换模型

编辑 `config.yml` 里对应角色的 `model` 与 `url`，重启工作台。界面命令 `/model` 只列出三个角色当前使用的模型、完整端点与密钥来源，不提供切换，也不写任何文件。

## 8. 视频身份、指纹与检查点

- **指纹** = `sha256(size ‖ mtime_ns ‖ 头 4 MiB ‖ 尾 4 MiB)`。一次引用只读 8 MiB，毫秒级完成，足以发现替换、截断与改动。
- **转录的缓存键** = `fingerprint + model + url + asr_chunk_seconds`，取 sha256 前 16 位存入 `transcript.cache_key`。任一项变化就重建整份转录，不把两次结果拼在一起；密钥不参与——换密钥不会改变答案，秘密也不该进键。
- **内容哈希尚未计算。** 全文件 sha256 能让同一内容换路径后命中旧产物，但要额外读一整份文件；当前引用只读 8 MiB，复制或改名会重新转写。等出现真实的重命名场景再加，加的时候是 `video` 表的一次迁移。
- **检查点**按 `(video_id, stage, cache_key, chunk_index)` 记录：一个分片写完，另起一次运行直接跳过它。写入分段与写入检查点在同一个事务里，所以"模型调用成功但标记丢失"这类中断只会重放该片，不会留下两份。
- 源文件变动不做自动跟踪：打开作业前重算指纹，与库里不一致时该视频的既有产物视为失效，界面提示"源文件已变化"，作业需重新开始。

## 9. 上下文策略

长视频的上下文问题不靠"塞更多"，靠三级收窄：

1. **覆盖图**：超过 40 段的录像，转写后按 `runtime.coverage_block_seconds`（默认 3 分钟）切块，逐块让对话模型写一行摘要存入 `coverage_block`。`overview` 先给这张图，再附 6 段原文作为可引用的锚点；短录音不建图，直接均匀抽样 24 段。
2. **检索窗口**：`search` 同时跑关键词与向量两条排名，用倒序秩融合（RRF）合并——两种分数不同尺度，按值相加会让一方淹没另一方。向量来自 `segment_vector`，没有嵌入模型时退回关键词；中文整句无命中时退回 2 字窗口。
3. **下钻**：模型可以按秒指定区间再取一段，单次最多 40 段；这一步同样消耗总预算。

工具返回值是片段进入上下文的唯一入口。当前三个工具：`overview`、`search`、`read_range`。说话人分离与区间摘要要等说话人识别和场景切分，不在 M3。

覆盖图的摘要只作导航：它出现在 `overview` 结果里时标为 `(map)`，引用校验只认 `verbatim` 的片段，所以结论不会落在"摘要说"这种无人说过的话上。

计划由模型给出，最多 4 步；计划不可解析时退回固定两步（overview + 用问题检索），并在轨迹里说明。没有"反思循环"：一次计划、一次作答，引用不合格时只允许一次纠正重试。

提示词是英文的 Jinja2 模板，工具清单、证据行、纠正示例都由模板循环渲染，时间戳统一走 `moment` 过滤器——格式只有一处定义，提示词与解析器不会各说一套。只有监督者与用户对话，所以作答模板把语言交给提问：指令英文，回答随用户。检索为空时仍调用一次模型，也是为这条规则服务——一句写死的中文说明在英文提问下必然答错语言。

两种记忆分开：会话记忆是当前对话与任务状态，随会话消失；视频记忆是库里的转录、场景与索引，跨会话存在，只通过工具访问。二者不共享上下文窗口。

## 10. 界面

四区加一条状态栏：

| 区域 | 内容 | 空态 |
| --- | --- | --- |
| 对话 | 用户输入、系统回复、计划与检索轨迹 | 显示用法提示与配置状态 |
| 任务 | 任务 ID、角色、动作、状态、进度、耗时 | 显示"尚未引用视频" |
| 时间轴 | 时长、容器、转录段数与覆盖区间、记忆规模 | 显示"尚未引用视频" |
| 证据 | 当前结论引用的时间点与转录原文 | 显示"暂无证据" |

状态栏常驻：数据库文件名、三个角色使用的模型与可用状态、运行中任务数。

输入语法：`@<相对路径> <问题>` 引用视频并起作业；直接输入一句话，就是对当前视频的追问；`/model` 列出模型与端点；`/help` 列出语法；`/quit` 退出。路径解析以进程工作目录为基准，解析失败在对话区报错并保留原文，不清空输入。

## 11. 里程碑

每一步以可验证的行为收尾，不以下一步的铺垫收尾。

**M0 地基（已实现）**
- 引用本地 mp4：指纹、探测、入库，重复引用不产生第二条记录。
- 作业与任务图落库并调度，状态迁移由状态机校验。
- 工作台四区渲染，任务面板随事件实时更新。
- 首次运行生成配置模板；`/model` 列出三个角色使用的模型与完整端点。
- 启动时把上次残留的 `running` 作业连同未完成节点判为失败，并写明原因。

验收：一条真实 mp4 被引用后，时长、分辨率、音轨在界面可见；改动文件后再次引用产生新记录并在对话区说明；非法迁移、坏配置、密钥泄漏各有测试；`uv run pytest` 与 `ruff check` 通过。

**M1 转录（已实现）**
- 音轨提取：PyAV 解码首条音轨，重采样为 16 kHz 单声道 PCM，写 WAV；先写 `.part` 再改名，文件存在即代表完整。
- 分片：按目标时长切分，边界吸附到前后四分之一片长内能量最低的窗口；末尾短于 1 秒的片段并入前片；切片只拷帧不重编码。
- 转写：向配置里的完整 URL 发 multipart（`file` + `model` + `verbose_json`），把响应的分段秒转换为毫秒；超时、429、5xx 与网络错误按 1s、2s… 退避重试，其余 4xx 立即失败。
- 检查点：每片成功后一个事务写入分段与该片标记；同一片重复写入覆盖而不追加。
- 界面：时间轴显示段数、覆盖区间与模型；无音轨、无语音各有明确文案。

验收：同一视频二次分析不再发起任何请求（`test_a_finished_transcript_costs_nothing_on_a_second_run`）；中断后重跑只补未完成分片（`test_a_crashed_run_keeps_the_chunks_it_finished`）；换模型重建而不混两份（`test_changing_the_model_rebuilds_instead_of_mixing`）；无音轨视频不产生转录，作业停在作答节点并写明"没有音轨"。真实 socket 端到端验证过：10 秒视频切 3 片、3 次请求都命中配置路径、6 段落在 0–9200ms。

不做 VAD 与说话人分离：分片边界用能量最低点近似，足够避免切断句子；需要区分"谁在说"时再引入检测模型。

**M2 监督者（已实现）**
- 追问：直接输入一句话就是对当前视频的提问；任务图按已有产物裁剪，已转写过的视频只跑作答节点。
- 提示词：英文 Jinja2 模板（`prompts/*.j2`）。工具清单由模板循环渲染，证据行与时间戳走同一个 `moment` 过滤器，引用格式在模板里写、由解析器读，两者有测试对拍。模板里没有一句中文（有测试守着），也没有一段提示词写在 Python 里。
- 语言：只有监督者直接与用户对话，因此作答与"没找到"两个模板都要求按提问的语言回答；英文只用于指令本身。
- 计划：模型输出 JSON 步骤（最多 4 步），校验工具名与参数；不可解析时退回 overview + 关键词检索，并在轨迹里说明。
- 工具：`overview` / `search` / `read_range`，共享 6000 字符证据预算，用尽时返回"预算不足"而不是继续塞。
- 结论：必须用 `[分:秒]` 或 `[分:秒-分:秒]` 标注依据，时间点要落在已检索到的片段内。
- 引用校验：不合格先带着错误时间点纠正重试一次，仍不合格则作业失败——不保存无法追溯的结论。引用的 `quote` 取转录原文，不取模型转述。
- 存储：`answer` 一作业一条，`answer_citation` 保存时间点与该处原文。

验收：已转写视频追问只跑 `answer`（`test_a_second_question_reuses_every_finished_stage`）；编造时间点被拒（`test_one_repair_is_allowed_when_a_moment_was_invented`、`test_a_second_unsupported_answer_is_refused`）；检索为空时模型用提问的语言说明没有找到（`test_the_loop_says_so_when_nothing_matched`）。本地假服务同时提供转写与对话时，界面依次出现"转录完成 2 段""按计划执行 2 步""结论引用了 1 处"与证据区的一条引用。

**M3 记忆与检索（已实现）**
- 角色：新增 `embedding`（完整 URL、模型、密钥）。未配置时检索退回关键词，其余流程照常。
- 覆盖图：按时间窗切块，逐块写一行摘要；超过 40 段才建，短录音不付这笔钱。写入按块进行，中断后从缺的块继续。
- 向量：每段一条 float32 向量，32 段一批写入；换嵌入模型或换转录键即整体重建（旧键先行删除，不会两份并存）。
- 混合检索：关键词与余弦各自排名，RRF 合并去重；只有一个排名时按原序截断。
- 任务图：probe → audio → transcribe → memory → answer。记忆已就绪时，追问只跑 answer；转写刚更新时记忆必随之重建。
- 界面：时间轴显示覆盖图块数与向量条数。

验收：46 段的录像建图 8 块、向量 46 条，二次构建零调用（`test_a_second_build_costs_nothing`）；第二批向量中断后重跑只补一批（`test_an_interrupted_build_resumes_at_the_first_missing_batch`）；换嵌入模型后向量仍是 46 条而不是 92 条（`test_a_new_embedding_model_replaces_the_old_vectors`）；"资金"这种说法在转写里不存在时，只有向量能把它召回到预算段落（`test_hybrid_memory_finds_what_keywords_alone_miss`）。

跨视频检索尚未实现：结论的引用目前属于"一条结论一个视频"，跨视频要先给 `answer_citation` 加视频归属、扩展引用格式，再谈多视频取证。这是下一个增量，不塞进本阶段。

**M4 视觉**：场景边界、抽帧策略、帧描述、事件抽取、跨模态对齐。
验收：画面类问题给出时间戳与关键帧。

**M5 硬化**：取消与恢复全链路、用量记录、保留策略、并发调优。性能基准与评测集不在实现范围内。

## 12. 不在范围内

本地 ASR 模型部署与推理优化、GPU 显存调度与动态批处理、图数据库与 GraphRAG、分布式编排框架、评测数据集构建。这些不进入代码，配置里也不留占位字段。
