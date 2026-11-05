# Kvedio Agent

Kvedio Agent is a single-user terminal workbench for analyzing local videos up
to two hours long. It preprocesses media with PyAV before model calls, stores
timestamped transcript and visual evidence, and keeps jobs and follow-up
conversations recoverable after restart.

## Requirements

- Python 3.13 or later
- `uv`
- A PyAV wheel available for the platform and Python version
- Model endpoints and credentials configured by the user

PyAV wheels bundle FFmpeg libraries, so a separate FFmpeg executable is not
required. The installed wheel's codec support determines which inputs can be
opened. The application leaves the original video in place and stores generated
artifacts under `~/.kvedio_ai/workspace/`.

## Install and configure

```powershell
uv sync
uv run kvedio-agent
```

The first run creates `~/.kvedio_ai/settings.json` with no provider, model, or
credential defaults. Add model entries and bind the roles required by the video
and analysis. `api_key_env` contains an environment variable name; the key value
must be set in the process environment. The application does not read `.env`.

```json
{
  "models": {
    "my-chat": {
      "model": "provider/model-id",
      "api_base": "https://provider.example/v1",
      "api_key_env": "MY_CHAT_TOKEN",
      "roles": ["chat"]
    },
    "my-asr": {
      "model": "provider/asr-model-id",
      "api_base": "https://provider.example/v1",
      "api_key_env": "MY_ASR_TOKEN",
      "roles": ["asr"]
    },
    "my-vision": {
      "model": "provider/vision-model-id",
      "api_base": "https://provider.example/v1",
      "api_key_env": "MY_VISION_TOKEN",
      "roles": ["vision"]
    }
  },
  "active": {
    "chat": "my-chat",
    "asr": "my-asr",
    "vision": "my-vision"
  },
  "runtime": {
    "max_video_duration_seconds": 7200,
    "retention_days": 30,
    "max_segment_duration_seconds": 120,
    "segment_overlap_seconds": 2,
    "max_media_workers": 2,
    "max_asr_workers": 4,
    "max_vision_workers": 2
  }
}
```

Role adapters currently require OpenAI-compatible transcription and chat/image
endpoints. Provider model identifiers, endpoint compatibility, request limits,
timestamps, and prices must be checked with the provider before use. See
[reference.md](reference.md) for the status of provider comparisons.

## Commands

- `kvedio-agent workbench` opens the prompted terminal menu.
- `kvedio-agent analyze VIDEO --objective "Find the decision and its evidence"`
  preprocesses and analyzes a local file.
- `kvedio-agent jobs` lists persisted jobs.
- `kvedio-agent resume JOB_ID` continues a job using its source fingerprint and
  completed artifacts.
- `kvedio-agent show JOB_ID` displays timestamped evidence.
- `kvedio-agent locate JOB_ID START_MS END_MS` returns original transcript and
  visual evidence overlapping a source-relative time range.
- `kvedio-agent ask JOB_ID "What changed later?"` continues the latest session;
  pass `--session SESSION_ID` to select another one.
- `kvedio-agent models` lists model roles and whether their environment keys are
  available. It never prints key values.
- `kvedio-agent bind ROLE MODEL` changes a persisted role binding.
- `kvedio-agent cancel JOB_ID` marks work cancelled; `delete JOB_ID` removes
  generated job data and leaves the source video untouched.

The workbench is a Textual application. It restores the durable job list,
stage milestones, timestamped evidence, and follow-up input from the local
SQLite workspace. Follow-up answers run outside the UI thread and are persisted
with their evidence links. Analysis instructions are structured Jinja2
templates shipped under the package `prompts/` directory.

Jobs, stages, segment artifacts, evidence, scratchpad checkpoints, conversations,
messages, and events are stored in SQLite below the workspace. Default artifact
retention is 30 days and can be changed with `runtime.retention_days`. Expiry
cleanup removes generated data; source files are never managed by the program.

## Development checks

```powershell
uv run pytest -q
uv run ruff check .
uv run mypy src
uv build
```

Tests use fake model providers and a small generated media fixture. They do not
make paid provider calls.

