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

## Layout

- `config/` model entries (`models.toml`); the table key (`small`, `big`) is the model name
- `app/models/` plain data shapes used inside the app (Pydantic only)
- `app/views/` request and response shapes for the API, including errors
- `app/controllers/` FastAPI routers; receive a request, call a service, return a view
- `app/services/` the logic: `ChatService` and `ModelRegistry`
- `app/providers/` adapters to AI companies behind the `LLMProvider` interface
- `app/dependencies.py` the only place where objects are wired together
- `tests/` tests that use a fake provider (no network)

Planned for later parts (not built yet): request logging to a database (a repositories/ folder behind an interface), stats script, load test, eval set, router service, cache service, dashboard.
