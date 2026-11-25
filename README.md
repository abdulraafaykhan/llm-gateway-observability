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

## Layout

- `config/` model entries (`models.toml`); the table key (`small`, `big`) is the model name
- `app/models/` plain data shapes used inside the app (Pydantic only)
- `app/views/` request and response shapes for the API, including errors
- `app/controllers/` FastAPI routers; receive a request, call a service, return a view
- `app/services/` the logic: `ChatService` and `ModelRegistry`
- `app/providers/` adapters to AI companies behind the `LLMProvider` interface
- `app/repositories/` storage for request records behind the `RequestRepository` interface (SQLite)
- `app/dependencies.py` the only place where objects are wired together
- `data/` the SQLite request log (`gateway.db`), created at startup and git-ignored
- `tests/` tests that use a fake provider and an in-memory repository (no network)

Planned for later parts (not built yet): stats script, load test, eval set, router service, cache service, dashboard.
