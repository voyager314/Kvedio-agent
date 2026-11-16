# kvedio-agent1

本地长视频理解 agent，界面是终端。用户 `@` 引用磁盘上的视频并提出问题，系统把分析拆成可追踪的任务图，结论的每一条都指向录像中的具体时刻。

当前实现到第一阶段（地基）：引用视频、按内容指纹登记、用 PyAV 自带解码器探测流信息，把每次引用作为作业与任务图落库，工作台随事件实时显示。转录、监督者与视觉理解尚未接入，见 [系统设计](docs/design/system.md) 的里程碑。

## 运行

```bash
uv sync
uv run kvedio-agent1            # 工作台（默认子命令）
uv run kvedio-agent1 config     # 配置位置与当前绑定，不显示密钥
uv run kvedio-agent1 doctor     # 检查配置、数据库与解码器
```

首次运行会在 `~/.kvedio_agent/config.yml` 生成模板，并创建 `~/.kvedio_agent/kvedio.db`。模板里三类角色（`supervisor`、`asr`、`video`）都按 OpenAI 兼容协议接入，`api_key` 推荐写成 `env:变量名`，让密钥留在配置文件之外。

不需要系统安装 ffmpeg：解码器随 `av` 一起安装。

## 在工作台里

```
@videos/weekly.mp4 这场会议讨论了什么   引用视频并起一个作业；路径含空格时加引号
/model                                  列出各角色的绑定与可用状态
/model asr dashscope                    切换某个角色使用的绑定，选择写入 state.yml
/help                                   语法
/quit                                   退出
```

四区一栏：对话、任务、时间轴、证据；底部状态栏显示数据库、三个角色的绑定与运行中任务数。

## 目录

- `src/kvedio_agent1/core/` — 引擎：领域值、配置、存储、任务调度、媒体探测、流水线
- `src/kvedio_agent1/interface/` — 终端界面、输入语法、进程入口
- `docs/design/system.md` — 模块职责、数据模型、任务状态机、并发与上下文策略
- `AGENTS.md` — 仓库规则与源码目录分类
