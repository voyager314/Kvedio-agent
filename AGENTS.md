# AGENTS.md

Instructions for AI coding agents working in this repository.

## What this repository is

A command-line agent that analyzes long videos held on the local disk. The user
uploads an mp4, states what to find out about it, and receives a conclusion whose
every claim points at the moment in the recording that supports it.

The product owns five capabilities:

- **Content location.** Given a point or a range in the recording, report what is
  being said there — the transcript segment that covers the requested interval.
- **Transcription.** Speech-to-text over the whole recording. Everything else
  reads this output, so its timing accuracy and its segment boundaries are
  contract, not implementation detail.
- **Analysis.** Summarize and answer questions about the content. A conclusion
  arrives with timestamped evidence, an execution plan, a stage-by-stage trace,
  and a quality assessment; a bare paragraph of prose is an incomplete result.
- **Job management.** A workbench view lists uploaded videos and their analysis
  jobs, shows progress while a job runs, and keeps a finished result open for
  further questions about the same video. A follow-up is answered from the
  multi-turn context at hand; where that history is persisted is undecided.
- **Ingestion.** Chunked upload with resume. A transfer interrupted partway
  continues from the last acknowledged chunk instead of starting over.

The terminal is the interface. Rendering uses panels, tables, and live progress
rather than plain text output, and interaction is prompted rather than typed as
raw command lines.

### What the domain forces

Long video is long-running, resource-hungry, and billed per call. The following
are structural constraints, not tuning knobs:

- **Wall clock.** An hour of recording is minutes of transcription and many model
  calls. Progress must be observable while a job runs, and work must be resumable
  rather than restartable.
- **External cost.** Every transcription and model call is paid for. Caching,
  batching, and refusing to redo finished work are cheaper than any latency win.
- **Failure.** Upstream calls fail, rate-limit, and degrade. Retry policy and
  fallback belong to each stage that calls out, not to a wrapper around the run.
- **Parallelism.** Independent segments run concurrently, under limits that keep
  a single machine and a rate-limited API usable.
- **Context.** A full transcript does not fit in a prompt. Selecting, compressing,
  and retrieving from the transcript is core to the pipeline; passing the
  transcript whole is a defect, not a shortcut.
- **Secrets.** Keys and tokens never reach a log line, an error message, a
  traceback, or a build artifact. The driver ships no provider, key, or default
  model: connections, role bindings, and permission to switch them come from
  `~/.kvedio_ai/settings.json`, which names environment variables instead of
  holding keys. A `.env` at the repository root is a fallback lookup only.


## Where the rules live

This file is the only copy of these rules. `CLAUDE.md` includes it rather than
restating it, so an edit here reaches every tool at once.

Architecture decisions belong in the design documents they are made in. Do not
restate a decision here, and do not paraphrase one in a commit message. This file
covers only what no other document states.

## Source layout

A file's directory states the role it plays, and nothing else decides it: not what
the file imports, not who calls it, not when it was written, not how long it is. A
reader who knows a module's role knows where to open it without searching.

### The root

Only a tool's requirement puts a file at the repository root — the build manifest,
the lock file, the readme, the license, the agent instructions, the ignore file.
A file the work produces goes to `docs/` when it outlives the task and to `temp/`
when it does not. Nothing is created at the root "for now".

The root is the one directory every reader scans before anything else. Every file
there is charged to everyone who opens the repository, and a file that is merely
convenient to reach sits next to one the build cannot run without.

### Inside the package

`src/kvedio_agent/` is split by role. Four directories, and the set is closed:

| Directory    | Role                                                                    |
| ------------ | ----------------------------------------------------------------------- |
| `interface/` | The terminal edge: parsing intent, prompting, panels, tables, progress   |
| `pipeline/`  | One run in order: decoding, scheduling, selecting evidence, concluding   |
| `platform/`  | What is replaced to run the same run elsewhere                            |
| `defaults/`  | Data the wheel ships: templates and reference text                        |

`platform/` holds the vendor contracts and their adapter, the pacing and cost
reservations attached to those calls, the database, the settings file, and the
validated runtime parameters. It answers where a run lives; `pipeline/` answers
what the run does.

Dependency direction is fixed: `interface` may import `pipeline` and `platform`,
`pipeline` may import `platform`, and no edge reverses. `defaults/` is data,
readable by anyone.

- `platform/` never imports `pipeline/`. Where it appears to need to, what it
  needs is a value handed in by the caller, not the module that stores it.
- `pipeline/` never imports `interface/`. A stage reports progress through a
  callable it was given; it does not reach for a console, a table, or a prompt.
- A module that seems to serve two roles marks a boundary drawn in the wrong
  place: move the shared part down until each side owns exactly one role.

**Placing a new module.** Name the thing you would have to replace to test it. A
terminal, a prompt, a rendered panel → `interface/`. An HTTP endpoint, a database
file, a settings file, a codec → `platform/`. Nothing outside the process, only
the values the previous stage handed over → `pipeline/`.

**A new directory edits this file.** There is no `utils/`, `helpers/`, `common/`,
or `models/`. A directory named after a shape collects whatever arrives and
answers no question; a directory named after a role answers where a file belongs
before it exists. If no row above fits, the role is missing — add the row and the
directory in one change, never a directory alone.

**A subpackage `__init__.py` carries a docstring and nothing else.** A re-export
gives one definition two names, and the shorter name hides which layer the caller
crossed to reach it.

**Module file names are unique across the package.** `tests/` stays flat, and its
names stay unambiguous only while `storage.py` means one file.

## Code

Four rules govern the code itself: comments carry intent, types carry shape,
every definition is used, and functions are cut only where a cut earns its keep.
Each is a review criterion, not a preference.

### Comments

A comment explains what the code cannot say about itself. It never narrates what
the next line already says. Write in the register of a specification: no first
person, no hedging, no restating the obvious.

**Comments stand alone.** No comment may cite a document: no design document
name, no section number, no task number, no requirement ID, no ticket. A comment
states what the code does in terms the file itself supplies, so a reader who has
never opened a planning document still understands it. A rule that came from a
design decision is stated as the rule; a requirement is stated as the behavior
that satisfies it.

The reason is not taste. Documents get rewritten, renumbered, and archived,
while the comment is read for as long as the code exists. A citation either
breaks or pulls the explanation out of the file into one the reader may not be
able to open.

**Modules, classes, protocols.** The block comment states responsibility: what
the unit owns, what it guarantees to its callers, and what it must not do.
Describe the contract, never the implementation path. Use bullets when the unit
carries more than one independent guarantee; one clause is enough when it
carries one.

**Fields of a data model.** Every field of a persisted entity, and of any type
that crosses a layer boundary — command payloads, export JSON, repository query
specs — carries a comment giving its meaning, its unit, and its permitted
domain: enum membership, what `null` means, what a sentinel value means. A
field whose name and type look self-explanatory still needs it when a scale, a
sign convention, or a time zone is attached to the value.

**Functions.** Document a function when its signature does not state its
contract. That covers any function which takes more than one parameter, returns
a value whose meaning is not carried by its type, requires a precondition from
its caller, or contains a branch that encodes a business rule. State, in this
order:

- one clause for what the function does, phrased as a guarantee rather than a
  recipe;
- every parameter: meaning, unit, accepted domain, and whether it is mutated;
- the return value: meaning, and what an empty or failed result means.

A function whose contract is fully visible in its signature gets no doc comment.
Repeating `(path: str) -> list[Segment]` in prose is noise.

### Types

**Nest values, not entities.** A model may embed another model only when the
nested value has no identity of its own: no primary key, no independent
lifecycle, no row of its own. A value that carries an ID is stored as a
reference and resolved by the caller. Nesting an entity duplicates rows, makes
the nested field unqueryable under single-field indexes, and turns one write
into several. A model inside a model inside a model is a view rather than a
record; flatten it, or resolve the references at read time.

**One model per concept.** Search the existing models before defining a type.
A type that differs from an existing one by an optional field, a narrowed union,
or a renamed field is a variation: widen the original, or derive it by
subclassing it or narrowing it with `TypedDict` or `Protocol`. Two definitions of
one concept diverge, and every consumer then pays for knowing which one to trust.

### Reuse

**Every definition is used.** A function, class, protocol, or type with nothing
calling it and nothing referring to it is deleted, not parked for later. Dead
code is not free: it is read, trusted, and eventually edited by someone who
assumes it runs.

**Search before defining.** Before adding a helper, a type, or a module,
search for one that already does the job. A near-duplicate costs more than a
slightly general original: the two drift apart, and the reader has to diff them
to find out which one the rest of the code follows.

### Decomposition

**Do not extract what reads straight through.** A short block — a few
conditionals and assignments, read once, top to bottom — stays in its caller.
Splitting it sends the reader out of the function to learn what the function
already showed.

**Extract only for a reason.** Three reasons qualify: the block is called from
more than one place; it carries a rule that needs its own test; or the name says
more than the body does. Without one of them, the extraction adds a jump and
returns nothing.

**A single call site is not a defect.** A long block, or one that carries a
rule, still deserves a name. What does not is a helper that only forwards
arguments, renames a variable, or wraps a single call without adding a
condition.

**No step-per-line pipelines.** Do not split a function into one-line stages
whose only reader is the next stage. A sequence of transformations reads better
as one function than as five.

### Self-check before opening a PR

1. Does any comment cite a document — a path, a section number, a requirement
   ID, a task number?
2. Can every field of every exported model be explained without opening another
   file?
3. Is there a function, class, protocol, or type that nothing calls or refers
   to?
4. Is there a helper that forwards its arguments without adding a rule?
5. Does a doc comment restate the signature instead of the contract?
6. Is there a test outside `tests/`, or a scratch script still in the tree or
   in the index?
7. Does any module sit in a directory its role does not match, or import across
   a direction the layout fixes?
8. Is there a directory, or a module name, that the layout table does not list?

## Tests

Every test lives under `tests/`.

The placement is asserted, not requested: a test outside that directory fails
`uv run pytest`. The reason is that a test directory is only useful as a complete
answer to "where are the tests". One stray file breaks that answer, and the file
it hides in gets read by nobody.

Two consequences follow from that directory being the only home for tests:

- Import the module under test by its absolute package path
  (`from kvedio_agent.pipeline.analyze import AnalysisService`), never
  relatively. A relative import keeps resolving against the old depth and
  reaches the wrong module or raises.
- Shared fixtures, seeds, transcript samples, and stubs live in `tests/` beside
  the tests, as `conftest.py` or a helper module — not inside the package, which
  ships without them.

## Scratch scripts

A script written to carry out one task — a migration loop, a file rewriter, a
one-off generator — is gone before that task is reported done, and it does not
enter git: not staged, not committed, not carried into a later commit that
touches the same files.

The rule is about the reader, not about tidiness. A surviving script is read as
code that runs, and nothing calls it; what documents it is a filename like
`fix_imports.py`, written for an audience of one. In history the cost is higher
still: `git log` reports the file as part of the change, so the record claims the
repository needs something only the task needed.

- Stage explicit paths while a scratch script exists in the tree. `git add -A`
  and `git add .` are how such a file gets in.
- If project paths force it into the tree, keep it under a path `.gitignore`
  already covers, and delete it before finishing.
- A script that should outlive its task is not scratch. It is code: it needs a
  caller before it is kept, and the rules above apply to it.

Check before reporting done: `git status` lists nothing the task created.

## Commit messages

A commit message has two jobs: let `git log --oneline` read as a changelog, and
give `git blame` a reason that the code cannot. Nothing else belongs in it.

A message outlives the working notes around it. A design document leaves the
working tree once the work ships, so any path into one stops resolving. Write from
the diff, not from the plan: state what the commit does, in a form that stays true
after the documents are gone.

### Title

```
<type>(<scope>): <what changed>
```

- `type`: `feat` `fix` `refactor` `perf` `test` `docs` `build` `chore` `ci` `style`
- `scope`: the area the diff touches — `upload` `asr` `analyze` `context` `cli`
  `llm` `storage` `config`. A diff that only edits documentation takes `docs`.
- `<what changed>`: one concrete clause stating the result rather than the
  activity. Required — never `fix`, `update`, `chore`, or `wip` on its own.
- English takes the imperative (`add`, not `added`); Chinese takes a plain
  statement of the result. One language per message, identifiers stay verbatim.
- 50 characters max, prefix included (a Chinese character counts as one). No
  trailing period.
- One logical change per commit. If the title needs "and", split the commit.

### Body

Omit it by default. Write one only when the diff cannot answer either question:

- **Trade-off** — you chose A over B, and the reason is not recoverable from the
  code.
- **Impact** — migration, compatibility, or a downstream change the reader must
  know about.

Then: 6 lines maximum, 3 bullets maximum, one line each. Anything longer is a
design document wearing a commit message's clothes.

A body carries reasoning, never evidence. Which task a commit closes, how it was
verified, and what the plan called it are all recoverable elsewhere. A reason
that exists only in a design document does not fit in a message: restate it as
the invariant it protects, or leave it out.

### No timeline

A message describes one state of the code, not a sequence of events. Words that
place the change on a timeline — `now`, `previously`, `currently`, `originally`,
`no longer`, `现在`, `目前`, `之前`, `原先`, `曾经` — are banned. State the
property instead of the transition:

```
✗ 原先报错落在后续的 put 调用上，现在落在构造点。
✓ The type error surfaces where the record is built, not where it is written.
```

When a message genuinely needs a point in time, write it as ISO 8601
(`2026-10-29`), never as a relative expression.

### Never put in a commit message

| Content                                                                         | Where it belongs             |
| ------------------------------------------------------------------------------- | ---------------------------- |
| Task numbers (`2.1–2.3`), acceptance criteria, section numbers, requirement IDs | The design document          |
| Paths into a design document, or any pointer to a file archived with the work   | Nowhere                      |
| File lists, "added X, modified Y"                                               | The diff                     |
| Dependency versions, what was not introduced                                    | `pyproject.toml` / `uv.lock` |
| Verification logs ("tested with 8 violation files, then deleted them")          | The PR description           |
| Restating what the code does, line by line                                      | Nowhere                      |

`Nowhere` means the content has no valid home in a message — delete it rather
than rephrase it. No `Co-authored-by` trailers for tools, no "generated with"
footers.

### Self-check before committing

1. Delete the body. Does the title still tell a reviewer what changed? If yes,
   the body was too long.
2. Delete every path and number the message cites. Does it still say something
   true? If not, the citation was carrying the message.
3. Does any line describe *what* the code does rather than *why*? Cut it.
4. Does any line place the change on a timeline? Rewrite it as a property.
5. Would the message still be correct once the change is archived? If not, it
   describes the plan rather than the commit.

### Example

Wrong — 26 lines, design section numbers, a path that stops resolving once the
work ships, and a "now" that dates the commit:

```
feat(upload): 分片上传、断点续传与分片校验

新增 upload/ 契约层（design.md D16）：分片大小、并发窗口、断点记录的
类型、取值范围、字段规格与校验纯函数……（下略 20 余行）

- 断点记录现在就写，全新上传与中断续传走同一条代码路径。
- 详见 docs/design/upload.md D15/D16。
```

The same commit, written from the diff — one language, no pointers, no timeline.
The defects are language-independent; these examples happen to be English:

```
feat(upload): resume interrupted transfers

- The resume record is written before the first chunk, so a fresh upload and a
  resumed one follow the same path and cannot diverge.
- Refuse a chunk whose digest does not match, otherwise a corrupted part is
  acknowledged and the failure surfaces later as a wrong transcript.
```

A one-line fix with no trade-off gets a title and nothing else:

```
fix(asr): key segment cache by file digest
```

A title that names the activity instead of the change, and the same commits
written as results:

```
project init        ->  chore: scaffold the typer cli and uv package
add MIT LICENSE     ->  docs: add MIT license
```
