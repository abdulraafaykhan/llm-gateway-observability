# llm-gateway

A small web server that sits between an app and its AI models and records the cost, speed and outcome of every request in a local database. It can route each question to a small or a big model, fall back to the big model when the small one fails, and comes with command-line tools to measure all of it.

## What it does

- **Pass-through chat API.** `POST /v1/chat` sends a conversation to a Groq model and returns the answer with token counts, latency and cost.
- **A record for every request.** Each call, including failures and cut-off answers, is saved as one row in SQLite. `GET /v1/requests` lists recent rows.
- **Stats.** A command reports request counts, error rate, tokens, cost, and latency at p50, p95 and p99 (p95 is the time 95% of answered requests finished under), overall and per model.
- **Routing with fallback.** Sending `"model": "auto"` lets simple scoring rules pick the small or the big model. If the chosen model fails (an error or an empty answer), the gateway retries once on the big model. A cut-off answer with text is returned as it is, flagged as cut off, unless `fallback_on_truncated = true` in `config\routing.toml`.
- **Fake mode and load tests.** A simulated provider with realistic delays, failures and a limit on parallel requests, plus a script that sends traffic at a planned rate to see how the gateway behaves under a spike.
- **Read-only dashboard and chat page.** `/dashboard` charts the request log of any database file in the data folder, including failure and routing insights; `/chat` sends messages through the gateway and shows how each was handled.
- **Evals and offline replay.** Two labelled question sets, a grading command, a comparison of two runs, and replay commands that estimate what a router or a cascade would have scored and cost using saved answers, with no new model calls.

## How a request flows

```
caller
  |
  |  POST /v1/chat  {"messages": [...], "model": "small" | "big" | "auto"}
  v
chat controller
  |
  v
ChatService --- model "auto"? ---> rule router (config/routing.toml) picks small or big
  |
  v
provider: Groq (real mode) or simulated (fake mode)
  |
  |-- "auto" only: first model failed --> one retry on big
  v
one row in SQLite (model used, tokens, latency, cost, status, routing reason, fallback flag)
plus one row per attempt in request_attempts
  |
  v
response: answer, model, tokens, latency_ms, cost_usd, route_reason, fallback_used
```

The controller only receives the request and returns the response. `ChatService` holds the logic: it chooses the model, calls the provider, decides on the fallback and writes the record. A request is always recorded once, whether it succeeds, is cut off or fails. If writing the record fails, the error is logged and the caller still gets the answer.

## Key findings so far

All eval numbers below are real Groq runs. The traffic spike is simulated in fake mode.

**Everyday question set** (100 questions, real Groq). The small model (`openai/gpt-oss-20b`) passed 99 of 100. Its one miss was an empty answer after it used all 4096 output tokens, not a wrong answer. The big model (`openai/gpt-oss-120b`) passed 100 of 100. This set is too easy to tell the models apart.

**Multiple-choice set** (100 MMLU-Pro questions, real Groq). These results are preliminary: the small run had rate-limit errors and a clean rerun is pending.

- Small passed 64 of 100 and big passed 69 of 100. Small had 4 errors (2 rate limits, 2 empty answers after using all 4096 tokens); big had 0.
- 58 questions were passed by both, 25 by neither, 11 only by big and 6 only by small. One of the 11 was a small rate-limit error. The difference is within normal noise at this size.
- Total cost of the run: small $0.019304, big $0.031826.

**Router experiments** (offline replay of the saved MMLU-Pro runs, 50 questions per split).

- Rule-based router: a threshold that looked slightly better on the tune split did not hold on the test split. Every MMLU-Pro prompt is long and full of numbers (the instruction plus ten options), so the surface rules fire on almost every question. Changing the threshold only moved results between "all small" and "all big".
- Token cascade (small answers first, escalate to big if it used more than a set number of tokens): no limit met the rule set in advance, so it is not built into the gateway.
- Small with fallback to big only on error or cut-off ("fallback only") cost about 81% (tune) and 66% (test) of sending everything to big. It scored 68.0% vs 68.0% for all big on tune, and 62.0% vs 70.0% on test.
- These replays used the older small run and may change after the clean rerun.

**Traffic spike** (simulated in fake mode, 262 requests, one run, same seed), all big vs `"auto"`:

| | all big | auto |
|---|---|---|
| errors | 1 | 0 |
| cut-off answers | 6 | 0 |
| p50 | 12824.4 ms | 628.4 ms |
| p95 | 25223.1 ms | 1050.6 ms |
| p99 | 26644.1 ms | 2741.8 ms |

In that run the router sent every request to small, so the gain comes mostly from small's higher capacity in the simulation plus the fallback, not from smart routing. These numbers come from a run that was not saved to `results/`. The commands to reproduce it are under "Fake mode and load test" below.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and a Groq API key.

```powershell
uv sync
Copy-Item .env.example .env
```

1. Put your key in `.env` as `GROQ_API_KEY=...`.
2. Check the prices in `config\models.toml` against Groq's pricing page.
3. Start the server and open http://127.0.0.1:8000/docs:

```powershell
uv run uvicorn app.main:app --reload
```

On the `/docs` page, open `POST /v1/chat`, click "Try it out" and send:

```json
{"messages": [{"role": "user", "content": "What is the capital of France?"}], "model": "auto"}
```

Tests and lint: `uv run pytest` and `uv run ruff check .`

## Using the tools

Each command has `--help`.

### Stats

```powershell
uv run python -m app.cli.stats_cli --minutes 60
```

Shows counts, error rate, fallbacks, tokens, cost and latency percentiles from the request log, overall and per model. `--db PATH` reads another database. It only reads the database, never writes it.

### Fake mode and load test

Start the gateway in fake mode with its own database, so simulated and real numbers never mix:

```powershell
$env:PROVIDER_MODE="fake"; $env:DATABASE_PATH="data\fake.db"; uv run uvicorn app.main:app
```

In a second terminal, run a scenario (`smoke`, `normal` or `spike`) and then the stats for it:

```powershell
uv run python -m app.cli.load_test_cli --scenario spike
uv run python -m app.cli.load_test_cli --scenario spike --model auto
uv run python -m app.cli.stats_cli --db data\fake.db --minutes 3 | Out-File -Encoding utf8 results\router_spike.txt
```

Sends requests on a random schedule without waiting for earlier answers, so a spike can build up a queue. Run one scenario per clean database: stop the gateway and run `Remove-Item data\fake.db*` between runs. `--questions config\load_questions_mixed.txt` uses a question file in which a third of the questions are routed to big. Non-smoke scenarios refuse to run against a real-mode gateway unless you pass `--allow-real`. The `$env:` settings last for the whole terminal session.

### Eval

```powershell
uv run python -m app.cli.eval_cli --validate-only
uv run python -m app.cli.eval_cli --model small
uv run python -m app.cli.eval_cli --eval-set eval\mmlu_pro_subset.jsonl --model big --label big-mmlu
```

Sends each question through a running real-mode gateway, grades the answer and prints a score, with failed questions listed. The full results are saved to `results\eval_<label>_<time>.json`. Use `--concurrency 1` if rate limits cause errors, and `--split tune|test` to run one half.

### Eval compare

```powershell
uv run python -m app.cli.eval_compare_cli results\eval_small-mmlu_20261126-102848.json results\eval_big-mmlu_20261126-102336.json
```

Groups the questions both runs share into: both passed, only A passed, only B passed, and neither passed. The "neither passed" list is where wrong answer keys show up.

### Router replay

```powershell
uv run python -m app.cli.router_replay_cli --small results\eval_small-mmlu_20261126-102848.json --big results\eval_big-mmlu_20261126-102336.json --split tune
```

Shows the score and cost the router would have had at each threshold, using the saved answers. Pick a threshold on the `tune` split, then report the `test` split. Options: `--thresholds 0,1,2,3`, `--no-fallback`. The replay tools still treat a cut-off answer as a reason to escalate; the gateway now does so only when fallback_on_truncated is true.

### Cascade replay

```powershell
uv run python -m app.cli.cascade_replay_cli --small results\eval_small-mmlu_20261126-102848.json --big results\eval_big-mmlu_20261126-102336.json
```

Shows what "small first, escalate to big on error, cut-off or more than T output tokens" would have scored and cost, for several values of T, next to all big, all small and fallback only. It prints a tune table and a test table. "Rescued" counts questions small got wrong that the final answer got right; "lost" counts the reverse.

### Gemini models

```powershell
uv run python -m app.cli.gemini_models_cli                      # list models your key can use
uv run python -m app.cli.gemini_models_cli --probe MODEL_ID     # one "Reply with the word OK." request each
```

Put your key in `.env` as `GEMINI_API_KEY=...`; the gateway refuses to start in real mode if a model uses `provider = "gemini"` without it. To add a model, copy a commented template at the end of `config\models.toml`, remove the `#` signs and fill in the model id and prices. Groq models can set `reasoning_effort = "low" | "medium" | "high"` (`small-low` and `big-low` templates); Gemini models cannot. Gemini free tier prompts and answers may be used by Google to improve its products. Use only non-private test data with it.

## Dashboard

While the gateway runs, open http://127.0.0.1:8000/dashboard. The page is read-only: it never writes to any database.

It lists every `.db` file directly in the data folder (`DASHBOARD_DATA_DIR`, default `data`) in a dropdown, and shows summary cards, latency by model, requests and p95 latency over time, cost by model, a per-model table and the 50 most recent requests. Range buttons limit it to the last 15 minutes, hour or 24 hours, and "Live" refreshes every 5 seconds. The numbers come from the same code as the stats command.

To keep a finished run for comparison, stop the gateway and copy the file:

```powershell
Copy-Item data\fake.db data\fake_baseline_spike.db
```

Use a file name that contains "fake" for simulated runs, so the page shows the "SIMULATED DATA" banner. The JSON behind the page is under `/api/dashboard/` (`sources`, `summary`, `timeseries`, `requests`, `insights`).

The Insights section at the bottom has four panels: failures by type (empty answer, rate limit, timeout, auth, other) with what they cost, the routing split for `auto` and explicit requests with fallback rescues, a latency spread histogram per model, and the 10 slowest answered requests.

Warnings come in a later part.

## Gateway and Experiments pages

The navigation on every page links Chat, Observability, Gateway and Experiments.

- **Gateway** (`/gateway`): the configured models with prices and live health from every attempt in the last 15 minutes, hour or 24 hours ("Healthy", "Degraded" at 10% errors or any rate-limit error, or "No recent traffic"). It also lists the non-secret settings and routing rules (read-only), and has a route preview that only runs the rules, without calling a model or recording anything.
- **Demo traffic** (on the Gateway page): it starts a smoke, normal or spike scenario inside the gateway, through the same router, fallback and recording as real requests, and shows progress with a Stop button. It only works when the gateway is in fake mode and its database file name contains "fake". Start it like this:

      $env:PROVIDER_MODE="fake"; $env:DATABASE_PATH="data/fake.db"; uv run uvicorn app.main:app

- **Experiments** (`/experiments`): it lists the saved `eval_*.json` runs in `RESULTS_DIR` and plots quality against cost. For a chosen pair of runs it shows the router and cascade replay tables (the same numbers as the command-line tools) and compares two runs. It never writes anything.
- **Settings:** `RESULTS_DIR` (default `results`) is where saved eval runs are read from. `LOAD_QUESTIONS_DIR` (default `config`) is where the demo's `load_questions*.txt` files are read from.

## Chat page

Open http://127.0.0.1:8000/chat (the site root redirects there). Pick a model or `auto`, and send messages; the whole conversation is sent with each new message. The Inspector shows how the selected answer was handled: the model, the route reason, fallback, tokens, latency and cost. "Compare two models" sends each question on its own, without earlier messages, to two models at once and shows the answers side by side. Requests made here are recorded in the database like any other request. A navigation bar on every page shows the gateway mode.

- **Attempts:** the Inspector and the dashboard's expanded rows list every attempt (for example small failed, then big answered) with its own tokens, time and cost, plus a total.
- **Concise answers** (on by default) puts this system message first in every request from the chat page: "Answer in plain text. Be brief unless I ask for detail. Do not use tables or LaTeX." It only affects the chat page. The database does not record whether it was on, so token and cost numbers from chats with it on and off are not comparable.
- **Formatting:** answers show headings, bold, lists, code and tables; "Show raw text" switches back to the original text. Links, images and LaTeX are not rendered.
- **Compare summary:** when both columns finish, a bar above them states the cost ratio, the time difference and the output-token difference, and says when a cut-off or failed column is not comparable.

## API

| Endpoint | Request | Response |
|---|---|---|
| `POST /v1/chat` | `messages`: list of `{role, content}` (role is `system`, `user` or `assistant`; at least one). `model`: optional, `small`, `big` or `auto`. If missing, `DEFAULT_MODEL_NAME` is used. | `answer`, `model` (the model that gave the final answer), `input_tokens`, `output_tokens`, `latency_ms`, `truncated`, `requested_model`, `route_reason`, `fallback_used`, `cost_usd`, `attempts` (one entry per provider call: model, status, tokens, time, cost, error) |
| `GET /v1/requests` | `limit`: 1 to 200, default 20 | `items`: the newest recorded requests, one per call, with the same fields as the database row |
| `GET /health` | none | `status`, `mode` (`real` or `fake`), `models`, `auto` |

`small` and `big` go straight to that model, and `route_reason` is `explicit`. `auto` runs the rule router, and `route_reason` explains the score, for example `score 2 (threshold 2): keywords +1 [calculate], many_numbers +1 [4 numbers]`. For `auto` requests, tokens, latency and cost in the response are totals over both attempts when a fallback happened. Each attempt is also stored on its own in the `request_attempts` table (created at startup, so older databases are upgraded); rows recorded before that have no attempt details.

Errors: an unknown model returns 400 (`unknown_model`), a provider failure returns 502 (`provider_error`), and an invalid body returns 422.

## Configuration

Settings are read from `.env` (copied from `.env.example`):

| Setting | Meaning |
|---|---|
| `GROQ_API_KEY` | Groq key. Required in real mode, not needed in fake mode. |
| `GEMINI_API_KEY` | Gemini key. Required in real mode only when a model uses `provider = "gemini"`. |
| `DEFAULT_MODEL_NAME` | Model used when a request has no `model` (`big` in `.env.example`). |
| `REQUEST_TIMEOUT_SECONDS` | Timeout for one provider call. |
| `MAX_OUTPUT_TOKENS` | Output token limit per call. Reasoning tokens count toward it. |
| `TEMPERATURE` | Sampling temperature (0 in `.env.example`). |
| `LOG_LEVEL` | Logging level. |
| `MODELS_CONFIG_PATH` | Path to `config\models.toml`. |
| `DATABASE_PATH` | SQLite file for the request log. |
| `PROVIDER_MODE` | `real` (Groq) or `fake` (simulated). |
| `FAKE_CONFIG_PATH`, `FAKE_SEED` | Fake-mode profiles and the random seed. |
| `ROUTING_CONFIG_PATH` | Path to `config\routing.toml`. |
| `DASHBOARD_DATA_DIR` | Folder whose `.db` files the dashboard can read. |
| `RESULTS_DIR`, `LOAD_QUESTIONS_DIR` | Folders the Experiments page and the demo controls read from. |

- `config\models.toml` holds one table per model (`small`, `big`): the provider, the provider's model id, and the input and output price per million tokens. The prices were taken from Groq's public pricing page: small $0.075 input and $0.30 output, big $0.15 input and $0.60 output.
- `config\routing.toml` holds the router settings: which models count as small and big, the score threshold, the fallback model, and the settings for each scoring rule (question length, keywords, how many numbers, how many question marks).
- `config\fake.toml` holds made-up timing, size, failure and capacity settings for each model in fake mode. The timings were sized against one real observation: the big model took about 2.2 seconds for 554 output tokens on one request.

## How things are measured

- **Cost** = (input tokens × input price + output tokens × output price) / 1,000,000, rounded to 10 decimals. A failed call that reports tokens is priced too. With a fallback, both attempts are added together.
- **Percentiles** use the nearest-rank method: sort the latencies and take the value at position ceil(pct × count / 100). They use answered requests only. Failed requests are counted separately, so a fast rejection or a slow timeout does not distort the speed numbers.
- **Latency** is measured inside the gateway around the provider call, and includes time spent waiting for a free slot in fake mode.
- **Temperature** is 0 in `.env.example`, but answers still vary a little between runs.
- **Eval grading** uses plain rules, not an AI judge. Text checks look for expected words, ignoring case. The number check is lenient: any number in the answer that equals the key counts. The multiple-choice check reads the last `Answer: X` line. A cut-off answer always counts as failed.
- **Fake mode** numbers are simulated from made-up settings. They are kept in a separate database and must always be labelled "simulated".

## Project layout

```
app/
  cli/            command-line tools (stats, load test, eval, compare, replays)
  clients/        HTTP client the eval uses to call the gateway
  controllers/    routes: receive a request, call a service, return a view
  models/         data shapes used inside the app (not AI models)
  providers/      model adapters: Groq and the simulated provider
  repositories/   storage: the SQLite request log behind an interface
  routing/        the router interface, scoring rules and rule router
  services/       logic: chat, routing config, stats, eval, replays
  static/         the chat, dashboard, gateway and experiments pages and shared navigation
  views/          response shapes for the API and the text reports
config/           models, routing rules, fake-mode profiles, load questions
eval/             the two labelled question sets
scripts/          one-off data builder for the MMLU-Pro subset
results/          saved eval runs and stats reports
tests/            tests with fake providers and an in-memory store (no network)
data/             the SQLite databases (created at startup, not tracked by git)
```

The app follows a model-view-controller layout with a service layer: models are data shapes, views are response shapes, controllers are routes, services hold the logic, providers adapt to each AI company, and repositories handle storage. Objects are wired together in one place, `app\dependencies.py`.

## Data and credits

- `eval\mmlu_pro_subset.jsonl` is a subset of MMLU-Pro by TIGER-Lab, MIT licence, https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro. It has 100 questions, about 7 per category, picked with seed 42 from 10-option questions only. Citation: Wang et al., "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark", 2024. A few answer keys in the public dataset may be wrong. It was built once with `uv run --no-project --with datasets python scripts\build_mmlu_pro_subset.py`; `datasets` is not a project dependency.
- `eval\eval_set.jsonl` (the everyday question set) and the load-test questions in `config\` were written for this project. Number answers in the everyday set were computed with Python.

## Limits

- The test sets are small, so differences of a few questions are noise.
- One provider (Groq) only.
- The router uses simple surface rules (length, keywords, numbers), not a trained model.
- Storage is a local SQLite file only.
- Number grading is lenient and can pass an answer that mentions the right number for the wrong reason.
- The cost formula ignores any cached-input discounts.
- The models are reasoning models that use hidden tokens, so token counts and costs vary between runs of the same question.

## Roadmap (not built yet)

- Exact-match cache, then a semantic cache.
- Quality and speed warnings.
- An assistant that answers questions about the project.
- Rate-limit simulation in fake mode.
