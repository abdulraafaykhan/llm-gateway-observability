# llm-gateway

A small web server that sits between an app and AI models.
Right now it forwards a chat request to one Groq model and returns the answer, token counts, and latency.

## Setup

1. `uv sync`
2. Copy `.env.example` to `.env`: `Copy-Item .env.example .env`
3. Put your Groq API key in `.env` as `GROQ_API_KEY=...`
4. Fill the prices in `config/models.toml` from the Groq pricing page

If Groq returns a 404 for a model, run models.list() with your key and update the model_id values in config/models.toml.

## Run

- Server: `uv run uvicorn app.main:app --reload`, then open http://127.0.0.1:8000/docs
- Tests: `uv run pytest`
- Lint: `uv run ruff check .`
- Recorded requests: `GET /v1/requests?limit=20` lists the newest rows from the request log.

## Stats

Print counts, failures, tokens, cost, and latency (avg, p50, p95, p99, max) from the request log:

- All time: `uv run python -m app.cli.stats_cli`
- `--minutes N` only counts requests from the last N minutes (N >= 1)
- `--db PATH` reads a different database file (default: `DATABASE_PATH`)

The command only reads the database; it never creates or changes it.

## Fake mode and load tests

Fake mode numbers are simulated from made-up settings in `config/fake.toml`. Always label them "simulated" when reporting them; they are not real Groq numbers.

Start the gateway in fake mode (separate database so fake and real numbers never mix):

    $env:PROVIDER_MODE="fake"; $env:DATABASE_PATH="data/fake.db"; uv run uvicorn app.main:app

Run a scenario (second terminal; `smoke`, `normal`, or `spike`):

    uv run python -m app.cli.load_test_cli --scenario normal

Show the report and save it:

    uv run python -m app.cli.stats_cli --db data/fake.db --minutes 3 | Out-File -Encoding utf8 results/baseline_normal.txt

Real-mode check (20 requests only):

    uv run uvicorn app.main:app
    uv run python -m app.cli.load_test_cli --scenario smoke

Between scenarios: stop the gateway, delete `data\fake.db`, start it again.

The `$env:` settings stay for the whole terminal session. Before the real-mode check, open a new terminal or run `Remove-Item Env:PROVIDER_MODE, Env:DATABASE_PATH`. The load script prints a stats command using `DATABASE_PATH` from its own terminal, so for fake runs use `--db data/fake.db` as above.

Other load options: `--model small|big`, `--seed N`, `--base-url URL`, `--questions PATH`. Non-smoke scenarios refuse to run against a real-mode gateway unless you pass `--allow-real`.

## Eval

`eval/eval_set.jsonl` holds 100 labelled questions (easy/hard, tune/test). The gateway must run in real mode; the eval refuses a fake-mode gateway.

    uv run python -m app.cli.eval_cli --validate-only
    uv run python -m app.cli.eval_cli --model small --limit 5
    uv run python -m app.cli.eval_cli --model small
    uv run python -m app.cli.eval_cli --model big
    uv run python -m app.cli.eval_compare_cli results\eval_small_<stamp>.json results\eval_big_<stamp>.json

Each run saves its per-question results to `results/eval_<label>_<stamp>.json`. Other options: `--split tune|test`, `--concurrency N` (use 1 if rate limits cause errors), `--retries N`, `--label NAME`.

- Eval requests are recorded in the database like any other request, so use `--minutes` in the stats command to separate them.
- Grading uses plain rules, not an AI judge. The number check is lenient: any number in the answer that equals the key counts.
- Results vary slightly between runs, even at temperature 0.

## Routing

Send `"model": "auto"` to let the router pick `small` or `big`. Sending `"small"` or `"big"` bypasses the router, so baselines stay clean. The rules live in `config/routing.toml`: each rule (long question, keywords, many numbers, several question marks) adds points, and a score at or above `threshold` goes to big.

Fallback: for `auto` requests only, if the chosen model fails or its answer is cut off, the gateway retries once on `fallback_model` (big). The response and the request log show `requested_model`, `route_reason`, `fallback_used` and `cost_usd`. Tokens, latency and cost are totals over both attempts.

Replay the saved eval runs through the router at several thresholds, with no model calls:

    uv run python -m app.cli.router_replay_cli --small results\eval_small_<stamp>.json --big results\eval_big_<stamp>.json --split tune

Options: `--split all|tune|test`, `--thresholds 0,1,2,3,4,5`, `--no-fallback`. Workflow: pick the threshold on the `tune` split, set it in `config/routing.toml`, then report the score from the `test` split.

Fake-mode comparison against the baselines (start from a clean `data\fake.db` as above):

    uv run python -m app.cli.load_test_cli --scenario spike --model auto
    uv run python -m app.cli.stats_cli --db data/fake.db --minutes 3 | Out-File -Encoding utf8 results/router_spike.txt

Real-mode eval through the router: `uv run python -m app.cli.eval_cli --model auto --split test` (the summary adds a `routing:` line).

## Layout

- `config/` model entries (`models.toml`); the table key (`small`, `big`) is the model name; also routing rules (`routing.toml`), fake-mode profiles (`fake.toml`) and load test questions (`load_questions.txt`)
- `app/routing/` the `Router` interface, the scoring rules, and the rule router
- `app/models/` plain data shapes used inside the app (Pydantic only)
- `app/views/` request and response shapes for the API, including errors
- `app/controllers/` FastAPI routers; receive a request, call a service, return a view
- `app/services/` the logic: chat, model registry, routing config, router replay, stats, fake profiles, load plan, and eval (loader, checker, runner, report)
- `app/cli/` command-line entry points (stats, load test, eval, eval compare, router replay)
- `app/clients/` HTTP client the eval uses to call the gateway, behind the `GatewayClient` interface
- `eval/` the labelled eval set (`eval_set.jsonl`)
- `results/` saved stats reports and eval runs
- `app/providers/` adapters to AI companies behind the `LLMProvider` interface
- `app/repositories/` storage for request records behind the `RequestRepository` interface (SQLite)
- `app/dependencies.py` the only place where objects are wired together
- `data/` the SQLite request log (`gateway.db`), created at startup and git-ignored
- `tests/` tests that use a fake provider and an in-memory repository (no network)

Planned for later parts (not built yet): cache service, dashboard.
