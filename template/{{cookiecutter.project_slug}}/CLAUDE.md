# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**{{ cookiecutter.project_name }}** is a FastAPI application generated with [Full-Stack FastAPI + Next.js Template](https://github.com/vstorm-co/full-stack-fastapi-nextjs-llm-template).

**Stack:**
- FastAPI + Pydantic v2
{%- if cookiecutter.use_postgresql %}
- PostgreSQL (async with asyncpg + SQLAlchemy 2.0)
{%- endif %}
{%- if cookiecutter.use_mongodb %}
- MongoDB (async with motor)
{%- endif %}
{%- if cookiecutter.use_sqlite %}
- SQLite (sync with SQLAlchemy)
{%- endif %}
{%- if cookiecutter.use_jwt %}
- JWT authentication (access + refresh tokens)
{%- endif %}
{%- if cookiecutter.enable_redis %}
- Redis (caching, sessions)
{%- endif %}
{%- if cookiecutter.enable_ai_agent and cookiecutter.use_pydantic_ai %}
- PydanticAI (AI agents with tool support)
{%- endif %}
{%- if cookiecutter.enable_ai_agent and cookiecutter.use_langchain %}
- LangChain (AI agents with tool support)
{%- endif %}
{%- if cookiecutter.use_celery %}
- Celery (background tasks)
{%- endif %}
{%- if cookiecutter.use_taskiq %}
- Taskiq (async background tasks)
{%- endif %}
{%- if cookiecutter.use_frontend %}
- Next.js 15 + React 19 + TypeScript + Tailwind CSS v4
{%- endif %}

## Commands

### Backend

```bash
cd backend

# Install dependencies
uv sync

# Run development server
uv run uvicorn app.main:app --reload --port {{ cookiecutter.backend_port }}

# Or use project CLI
uv run {{ cookiecutter.project_slug }} server run --reload

# Run tests
pytest
pytest tests/test_file.py::test_name -v

# Linting and formatting
ruff check .
ruff check . --fix
ruff format .

# Type checking
mypy app
```
{%- if cookiecutter.use_postgresql or cookiecutter.use_sqlite %}

### Database

```bash
cd backend

# Run all migrations
uv run alembic upgrade head

# Create new migration
uv run alembic revision --autogenerate -m "Description"

# Or use project CLI
uv run {{ cookiecutter.project_slug }} db upgrade
uv run {{ cookiecutter.project_slug }} db migrate -m "Description"
```
{%- endif %}
{%- if cookiecutter.use_jwt %}

### User Management

```bash
cd backend

# Create admin user
uv run {{ cookiecutter.project_slug }} user create-admin --email admin@example.com

# List users
uv run {{ cookiecutter.project_slug }} user list
```
{%- endif %}
{%- if cookiecutter.use_frontend %}

### Frontend

```bash
cd frontend

# Install dependencies
bun install

# Run development server
bun dev

# Run tests
bun test
bun test:e2e
```
{%- endif %}
{%- if cookiecutter.enable_docker %}

### Docker

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```
{%- endif %}

## Architecture

This project follows a **Repository + Service** layered architecture:

```
API Routes → Services → Repositories → Database
```

### Directory Structure (`backend/app/`)

| Directory | Purpose |
|-----------|---------|
| `api/routes/v1/` | HTTP endpoints, request validation, auth |
| `api/deps.py` | Dependency injection (db session, current user) |
| `services/` | Business logic, orchestration |
| `repositories/` | Data access layer, database queries |
| `schemas/` | Pydantic models for request/response |
| `db/models/` | SQLAlchemy/MongoDB models |
| `core/config.py` | Settings via pydantic-settings |
{%- if cookiecutter.use_auth %}
| `core/security.py` | JWT/API key utilities |
{%- endif %}
{%- if cookiecutter.enable_ai_agent %}
| `agents/` | PydanticAI agents and tools |
{%- endif %}
| `commands/` | Django-style CLI commands |
{%- if cookiecutter.use_celery or cookiecutter.use_taskiq %}
| `worker/` | Background task definitions |
{%- endif %}

### Adding New Features

**1. Add a new API endpoint:**
```
1. Create schema in `schemas/`
2. Create model in `db/models/` (if new entity)
3. Create repository in `repositories/`
4. Create service in `services/`
5. Create route in `api/routes/v1/`
6. Register route in `api/routes/v1/__init__.py`
```

**2. Add a custom CLI command:**
```python
# app/commands/my_command.py
from app.commands import command, success
import click

@command("my-command", help="Description")
@click.option("--option", "-o", help="Some option")
def my_command(option: str):
    # Logic here
    success(f"Done with {option}")
```
Commands are auto-discovered. Run with: `{{ cookiecutter.project_slug }} cmd my-command`
{%- if cookiecutter.enable_ai_agent and cookiecutter.use_pydantic_ai %}

**3. Add an AI agent tool (PydanticAI):**
```python
# app/agents/assistant.py
@agent.tool
async def my_tool(ctx: RunContext[Deps], param: str) -> dict:
    """Tool description for LLM."""
    # Tool logic
    return {"result": param}
```
{%- endif %}
{%- if cookiecutter.enable_ai_agent and cookiecutter.use_langchain %}

**3. Add an AI agent tool (LangChain):**
```python
# app/agents/langchain_assistant.py
from langchain.tools import tool

@tool
def my_tool(param: str) -> dict:
    """Tool description for LLM."""
    # Tool logic
    return {"result": param}
```
{%- endif %}

## Key Patterns

### Dependency Injection

```python
# In routes
from app.api.deps import get_db, get_current_user

@router.get("/items")
async def list_items(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ItemService(db)
    return await service.get_multi()
```

### Service Layer

```python
# Services contain business logic
class ItemService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, item_in: ItemCreate) -> Item:
        # Business validation
        # Repository calls
        return await item_repo.create(self.db, **item_in.model_dump())
```

### Repository Layer

```python
# Repositories handle data access only
class ItemRepository:
    async def get_by_id(self, db: AsyncSession, id: UUID) -> Item | None:
        return await db.get(Item, id)

    async def create(self, db: AsyncSession, **kwargs) -> Item:
        item = Item(**kwargs)
        db.add(item)
        await db.flush()
        await db.refresh(item)
        return item
```

### Custom Exceptions

```python
from app.core.exceptions import NotFoundError, AlreadyExistsError

# In services
if not item:
    raise NotFoundError(message="Item not found", details={"id": str(id)})
```
{%- if cookiecutter.use_frontend %}

## Frontend Patterns

### Authentication

Tokens stored in HTTP-only cookies. Use the auth hook:

```typescript
import { useAuth } from '@/hooks/use-auth';

function Component() {
  const { user, isAuthenticated, login, logout } = useAuth();
}
```

### State Management (Zustand)

```typescript
import { useAuthStore } from '@/stores/auth-store';

const { user, setUser, logout } = useAuthStore();
```
{%- if cookiecutter.enable_ai_agent %}

### WebSocket Chat

```typescript
import { useChat } from '@/hooks/use-chat';

function ChatPage() {
  const { messages, sendMessage, isStreaming } = useChat();
}
```
{%- endif %}
{%- endif %}

## Environment Variables

Key variables in `.env`:

```bash
ENVIRONMENT=local  # local, staging, production
{%- if cookiecutter.use_postgresql %}
POSTGRES_HOST=localhost
POSTGRES_PASSWORD=secret
{%- endif %}
{%- if cookiecutter.use_jwt %}
SECRET_KEY=change-me-use-openssl-rand-hex-32
{%- endif %}
{%- if cookiecutter.enable_ai_agent %}
OPENAI_API_KEY=sk-...
{%- endif %}
{%- if cookiecutter.enable_logfire %}
LOGFIRE_TOKEN=your-token
{%- endif %}
```

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=app --cov-report=term-missing

# Specific test
pytest tests/api/test_health.py -v

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/
```

## Key Design Decisions

{%- if cookiecutter.use_postgresql or cookiecutter.use_mongodb %}
- Database operations are async
{%- endif %}
- Use `db.flush()` in repositories (not `commit`) - let the dependency manage transactions
- Services raise domain exceptions (`NotFoundError`, etc.) - routes convert to HTTP
- Schemas are separate for Create, Update, and Response
{%- if cookiecutter.enable_ai_agent and cookiecutter.use_pydantic_ai %}
- AI Agent uses PydanticAI `iter()` for WebSocket streaming
{%- endif %}
{%- if cookiecutter.enable_ai_agent and cookiecutter.use_langchain %}
- AI Agent uses LangChain `stream()` for WebSocket streaming
{%- endif %}
- Custom commands auto-discovered from `app/commands/`

## Documentation

- [Template Repository](https://github.com/vstorm-co/full-stack-fastapi-nextjs-llm-template)
- [Architecture Guide](https://github.com/vstorm-co/full-stack-fastapi-nextjs-llm-template/blob/main/docs/architecture.md)
{%- if cookiecutter.use_frontend %}
- [Frontend Guide](https://github.com/vstorm-co/full-stack-fastapi-nextjs-llm-template/blob/main/docs/frontend.md)
{%- endif %}
{%- if cookiecutter.enable_ai_agent %}
- [AI Agent Guide](https://github.com/vstorm-co/full-stack-fastapi-nextjs-llm-template/blob/main/docs/ai-agent.md)
{%- endif %}
- [Deployment Guide](https://github.com/vstorm-co/full-stack-fastapi-nextjs-llm-template/blob/main/docs/deployment.md)
