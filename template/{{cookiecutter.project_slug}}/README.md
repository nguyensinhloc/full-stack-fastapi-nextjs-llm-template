# {{ cookiecutter.project_name }}

{{ cookiecutter.project_description }}

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI">
{%- if cookiecutter.use_frontend %}
  <img src="https://img.shields.io/badge/Next.js-15-black?logo=next.js&logoColor=white" alt="Next.js">
{%- endif %}
{%- if cookiecutter.use_postgresql %}
  <img src="https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql&logoColor=white" alt="PostgreSQL">
{%- endif %}
{%- if cookiecutter.use_mongodb %}
  <img src="https://img.shields.io/badge/MongoDB-7-47A248?logo=mongodb&logoColor=white" alt="MongoDB">
{%- endif %}
{%- if cookiecutter.enable_redis %}
  <img src="https://img.shields.io/badge/Redis-7-DC382D?logo=redis&logoColor=white" alt="Redis">
{%- endif %}
</p>

<p align="center">
  <sub>Generated with <a href="https://github.com/vstorm-co/full-stack-fastapi-nextjs-llm-template">Full-Stack FastAPI + Next.js Template</a></sub>
</p>

---

## Features

{%- if cookiecutter.enable_ai_agent %}
- 🤖 **AI Agent** - PydanticAI with WebSocket streaming
{%- endif %}
{%- if cookiecutter.use_jwt %}
- 🔐 **JWT Authentication** - Access + Refresh tokens
{%- endif %}
{%- if cookiecutter.use_api_key %}
- 🔑 **API Key Auth** - Header-based authentication
{%- endif %}
{%- if cookiecutter.enable_oauth %}
- 🌐 **OAuth2** - Social login (Google)
{%- endif %}
{%- if cookiecutter.use_postgresql %}
- 🐘 **PostgreSQL** - Async with SQLAlchemy 2.0
{%- endif %}
{%- if cookiecutter.use_mongodb %}
- 🍃 **MongoDB** - Async with Motor
{%- endif %}
{%- if cookiecutter.enable_redis %}
- 📦 **Redis** - Caching and sessions
{%- endif %}
{%- if cookiecutter.use_celery %}
- 🥬 **Celery** - Background task processing
{%- endif %}
{%- if cookiecutter.use_taskiq %}
- ⚡ **Taskiq** - Async task queue
{%- endif %}
{%- if cookiecutter.enable_rate_limiting %}
- 🚦 **Rate Limiting** - Request throttling
{%- endif %}
{%- if cookiecutter.enable_admin_panel %}
- 🗄️ **Admin Panel** - SQLAdmin with automatic model discovery
{%- endif %}
{%- if cookiecutter.enable_logfire %}
- 📊 **Logfire** - Full-stack observability (see [Logfire section](#logfire-observability))
{%- endif %}
{%- if cookiecutter.enable_sentry %}
- 🛡️ **Sentry** - Error tracking
{%- endif %}
{%- if cookiecutter.use_frontend %}
- 🎨 **Next.js 15** - React 19 + TypeScript + Tailwind
{%- endif %}
{%- if cookiecutter.enable_docker %}
- 🐳 **Docker** - Containerized development
{%- endif %}

---

## Quick Start

### Prerequisites

- Python 3.11+ ([uv](https://docs.astral.sh/uv/) recommended)
{%- if cookiecutter.use_frontend %}
- [Bun](https://bun.sh) (for frontend)
{%- endif %}
{%- if cookiecutter.enable_docker %}
- Docker & Docker Compose
{%- endif %}

### 1. Setup Backend

```bash
cd backend

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your settings
```

{%- if cookiecutter.use_postgresql %}

### 2. Start PostgreSQL

```bash
# Using Docker (recommended)
docker run -d \
  --name {{ cookiecutter.project_slug }}-db \
  -e POSTGRES_PASSWORD=secret \
  -e POSTGRES_DB={{ cookiecutter.project_slug }} \
  -p 5432:5432 \
  postgres:16-alpine

# Or use existing PostgreSQL - update .env accordingly
```
{%- endif %}

{%- if cookiecutter.enable_redis %}

### {% if cookiecutter.use_postgresql %}3{% else %}2{% endif %}. Start Redis

```bash
docker run -d \
  --name {{ cookiecutter.project_slug }}-redis \
  -p 6379:6379 \
  redis:7-alpine
```
{%- endif %}

{%- if cookiecutter.use_postgresql or cookiecutter.use_sqlite %}

### {% if cookiecutter.enable_redis %}4{% elif cookiecutter.use_postgresql %}3{% else %}2{% endif %}. Initialize Database

```bash
cd backend

# Run migrations
uv run alembic upgrade head
{%- if cookiecutter.use_jwt %}

# Create admin user
uv run {{ cookiecutter.project_slug }} user create-admin --email admin@example.com
{%- endif %}
```
{%- endif %}

### {% if cookiecutter.use_postgresql and cookiecutter.enable_redis %}5{% elif cookiecutter.use_postgresql or cookiecutter.enable_redis %}4{% elif cookiecutter.use_sqlite %}3{% else %}2{% endif %}. Run Development Server

```bash
cd backend
uv run uvicorn app.main:app --reload --port {{ cookiecutter.backend_port }}
```

{%- if cookiecutter.use_frontend %}

### {% if cookiecutter.use_postgresql and cookiecutter.enable_redis %}6{% elif cookiecutter.use_postgresql or cookiecutter.enable_redis %}5{% elif cookiecutter.use_sqlite %}4{% else %}3{% endif %}. Setup Frontend

```bash
cd frontend
bun install
bun dev
```
{%- endif %}

### Access Points

| Service | URL |
|---------|-----|
| API | http://localhost:{{ cookiecutter.backend_port }} |
| API Docs (Swagger) | http://localhost:{{ cookiecutter.backend_port }}/docs |
| API Docs (ReDoc) | http://localhost:{{ cookiecutter.backend_port }}/redoc |
{%- if cookiecutter.enable_admin_panel %}
| Admin Panel | http://localhost:{{ cookiecutter.backend_port }}/admin |
{%- endif %}
{%- if cookiecutter.use_frontend %}
| Frontend | http://localhost:{{ cookiecutter.frontend_port }} |
{%- endif %}

---

{%- if cookiecutter.enable_docker %}

## Docker Development

### Start All Services

```bash
# Development mode
docker compose up -d

# Production mode
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# View logs
docker compose logs -f
```

### Services Started

| Service | Description |
|---------|-------------|
| `backend` | FastAPI application |
{%- if cookiecutter.use_postgresql %}
| `db` | PostgreSQL database |
{%- endif %}
{%- if cookiecutter.enable_redis %}
| `redis` | Redis cache |
{%- endif %}
{%- if cookiecutter.use_celery %}
| `celery-worker` | Celery worker |
| `celery-beat` | Celery scheduler |
| `flower` | Celery monitoring (port 5555) |
{%- endif %}
{%- if cookiecutter.use_taskiq %}
| `taskiq-worker` | Taskiq worker |
{%- endif %}
{%- if cookiecutter.use_frontend %}
| `frontend` | Next.js application |
{%- endif %}

---
{%- endif %}

## Project CLI

This project includes a Django-style CLI for common tasks:

### Server Commands

```bash
# Start development server
{{ cookiecutter.project_slug }} server run --reload

# Show all registered routes
{{ cookiecutter.project_slug }} server routes
```

{%- if cookiecutter.use_postgresql or cookiecutter.use_sqlite %}

### Database Commands

```bash
# Initialize database (run all migrations)
{{ cookiecutter.project_slug }} db init

# Create new migration
{{ cookiecutter.project_slug }} db migrate -m "Add new table"

# Apply pending migrations
{{ cookiecutter.project_slug }} db upgrade

# Rollback last migration
{{ cookiecutter.project_slug }} db downgrade

# Show current revision
{{ cookiecutter.project_slug }} db current
```
{%- endif %}

{%- if cookiecutter.use_jwt %}

### User Management

```bash
# Create user (interactive)
{{ cookiecutter.project_slug }} user create

# Create admin user
{{ cookiecutter.project_slug }} user create-admin --email admin@example.com

# List all users
{{ cookiecutter.project_slug }} user list

# Change user role
{{ cookiecutter.project_slug }} user set-role user@example.com --role admin
```
{%- endif %}

{%- if cookiecutter.use_celery %}

### Celery Commands

```bash
# Start worker
{{ cookiecutter.project_slug }} celery worker

# Start beat scheduler
{{ cookiecutter.project_slug }} celery beat

# Start Flower monitoring
{{ cookiecutter.project_slug }} celery flower
```
{%- endif %}

{%- if cookiecutter.use_taskiq %}

### Taskiq Commands

```bash
# Start worker
{{ cookiecutter.project_slug }} taskiq worker

# Start scheduler
{{ cookiecutter.project_slug }} taskiq scheduler
```
{%- endif %}

### Custom Commands

Create your own commands in `app/commands/`:

```python
# app/commands/seed.py
from app.commands import command, success
import click

@command("seed", help="Seed database with test data")
@click.option("--count", "-c", default=10, type=int)
def seed_database(count: int):
    # Your seeding logic here
    success(f"Created {count} records!")
```

```bash
# Run custom command
{{ cookiecutter.project_slug }} cmd seed --count 100
```

> **Note:** Commands are auto-discovered from `app/commands/`. Just create a file with the `@command` decorator.

---

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies |
| `make run` | Start dev server with hot reload |
| `make test` | Run tests |
| `make test-cov` | Run tests with coverage |
| `make lint` | Check code with ruff |
| `make format` | Format code with ruff |
| `make typecheck` | Run mypy type checking |
{%- if cookiecutter.use_postgresql or cookiecutter.use_sqlite %}
| `make db-migrate` | Create new migration |
| `make db-upgrade` | Apply migrations |
| `make db-downgrade` | Rollback migration |
{%- endif %}
{%- if cookiecutter.enable_docker %}
| `make docker-up` | Start all Docker services |
| `make docker-down` | Stop all Docker services |
| `make docker-logs` | View Docker logs |
{%- endif %}

---

## Project Structure

```
{{ cookiecutter.project_slug }}/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app with lifespan
│   │   ├── api/
│   │   │   ├── routes/v1/       # API endpoints
│   │   │   ├── deps.py          # Dependency injection
│   │   │   └── router.py        # Route aggregation
│   │   ├── core/
│   │   │   ├── config.py        # Settings (pydantic-settings)
{%- if cookiecutter.use_auth %}
│   │   │   ├── security.py      # Auth utilities
{%- endif %}
{%- if cookiecutter.enable_logfire %}
│   │   │   └── logfire_setup.py # Observability
{%- endif %}
│   │   ├── db/
│   │   │   ├── models/          # Database models
│   │   │   └── session.py       # Connection management
{%- if cookiecutter.enable_admin_panel %}
│   │   ├── admin.py             # SQLAdmin with auto-discovery
{%- endif %}
│   │   ├── schemas/             # Pydantic schemas
│   │   ├── repositories/        # Data access layer
│   │   ├── services/            # Business logic
{%- if cookiecutter.enable_ai_agent %}
│   │   ├── agents/              # PydanticAI agents
{%- endif %}
│   │   ├── commands/            # Custom CLI commands
{%- if cookiecutter.use_celery or cookiecutter.use_taskiq %}
│   │   └── worker/              # Background tasks
{%- endif %}
│   ├── cli/                     # Project CLI
│   ├── tests/                   # Test suite
{%- if cookiecutter.use_postgresql or cookiecutter.use_sqlite %}
│   └── alembic/                 # Database migrations
{%- endif %}
{%- if cookiecutter.use_frontend %}
├── frontend/
│   ├── src/
│   │   ├── app/                 # Next.js App Router
│   │   ├── components/          # React components
│   │   ├── hooks/               # Custom hooks
│   │   ├── stores/              # Zustand stores
│   │   └── lib/                 # Utilities
│   └── e2e/                     # Playwright tests
{%- endif %}
{%- if cookiecutter.enable_docker %}
├── docker-compose.yml
├── docker-compose.prod.yml
{%- endif %}
├── Makefile
└── README.md
```

---

## Architecture

This project follows a **Repository + Service** pattern:

```
API Routes → Services → Repositories → Database
```

| Layer | Location | Responsibility |
|-------|----------|----------------|
| **Routes** | `app/api/routes/` | HTTP handling, validation |
| **Services** | `app/services/` | Business logic |
| **Repositories** | `app/repositories/` | Data access |
| **Schemas** | `app/schemas/` | Request/Response models |
| **Models** | `app/db/models/` | Database models |

> 📚 For detailed architecture documentation, see the [template repository](https://github.com/vstorm-co/full-stack-fastapi-nextjs-llm-template/blob/main/docs/architecture.md).

---
{%- if cookiecutter.enable_admin_panel %}

## Admin Panel

The admin panel provides a web-based interface for managing database records. It uses [SQLAdmin](https://aminalaee.dev/sqladmin/) with **automatic model discovery** - all SQLAlchemy models are automatically registered without manual configuration.

### Access

- URL: `http://localhost:{{ cookiecutter.backend_port }}/admin`
{%- if cookiecutter.admin_require_auth %}
- **Authentication required**: Login with superuser credentials
{%- endif %}

### Features

| Feature | Description |
|---------|-------------|
| **Auto-Discovery** | All models from `Base.registry` are automatically registered |
| **Smart Defaults** | Searchable columns (String types), sortable columns, form exclusions |
| **Sensitive Data Protection** | Password, token, secret fields auto-excluded from forms |
| **Custom Overrides** | Per-model configuration in `CUSTOM_MODEL_CONFIGS` |

### Customizing Model Views

To customize a model's admin view, add it to `CUSTOM_MODEL_CONFIGS` in `app/admin.py`:

```python
CUSTOM_MODEL_CONFIGS: dict[type, dict[str, Any]] = {
    User: {
        "icon": "fa-solid fa-user",
        "form_excluded_columns": [User.hashed_password],
        "can_delete": False,  # Prevent deletion
    },
    Order: {
        "name": "Customer Order",
        "name_plural": "Customer Orders",
        "column_list": [Order.id, Order.status, Order.created_at],
        "can_create": False,  # Read-only
    },
}
```

### Available Options

| Option | Type | Description |
|--------|------|-------------|
| `name` | `str` | Display name in admin |
| `name_plural` | `str` | Plural name for list view |
| `icon` | `str` | Font Awesome icon class |
| `column_list` | `list` | Columns to show in list view |
| `column_searchable_list` | `list` | Columns to enable search |
| `column_sortable_list` | `list` | Columns to enable sorting |
| `form_excluded_columns` | `list` | Columns to hide in forms |
| `can_create` | `bool` | Allow creating records |
| `can_edit` | `bool` | Allow editing records |
| `can_delete` | `bool` | Allow deleting records |
| `can_view_details` | `bool` | Allow viewing record details |

### Excluding Models

To exclude a model from auto-registration:

```python
# In app/admin.py setup_admin()
register_models_auto(
    admin,
    Base,
    exclude_models=[InternalLog, TempData],  # These won't appear in admin
    custom_configs=CUSTOM_MODEL_CONFIGS,
)
```

---
{%- endif %}

## Configuration

All configuration via environment variables in `.env`:

### Core Settings

```bash
ENVIRONMENT=local          # local, staging, production
DEBUG=true
PROJECT_NAME={{ cookiecutter.project_name }}
```

{%- if cookiecutter.use_postgresql %}

### Database (PostgreSQL)

```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=secret
POSTGRES_DB={{ cookiecutter.project_slug }}

# Pool settings
DB_POOL_SIZE={{ cookiecutter.db_pool_size }}
DB_MAX_OVERFLOW={{ cookiecutter.db_max_overflow }}
```
{%- endif %}

{%- if cookiecutter.use_mongodb %}

### Database (MongoDB)

```bash
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_DB={{ cookiecutter.project_slug }}
MONGO_USER=
MONGO_PASSWORD=
```
{%- endif %}

{%- if cookiecutter.use_jwt %}

### Authentication

```bash
# Generate with: openssl rand -hex 32
SECRET_KEY=change-me-in-production

ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_MINUTES=10080  # 7 days
```
{%- endif %}

{%- if cookiecutter.enable_redis %}

### Redis

```bash
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
```
{%- endif %}

{%- if cookiecutter.enable_ai_agent %}

### AI Agent

```bash
OPENAI_API_KEY=sk-...
AI_MODEL=gpt-4o-mini
AI_TEMPERATURE=0.7
```
{%- endif %}

{%- if cookiecutter.enable_logfire %}

### Logfire

```bash
# Get token at https://logfire.pydantic.dev
LOGFIRE_TOKEN=your-token
LOGFIRE_SERVICE_NAME={{ cookiecutter.project_slug }}
```
{%- endif %}

{%- if cookiecutter.enable_sentry %}

### Sentry

```bash
SENTRY_DSN=https://xxx@sentry.io/xxx
```
{%- endif %}

---
{%- if cookiecutter.enable_logfire %}

## Logfire Observability

[Logfire](https://logfire.pydantic.dev) provides complete observability for your application. Built by the Pydantic team, it offers first-class support for the Python ecosystem.

### What Gets Instrumented

| Component | What You See |
|-----------|-------------|
{%- if cookiecutter.enable_ai_agent %}
| **PydanticAI** | Agent runs, tool calls, LLM requests, token usage |
{%- endif %}
| **FastAPI** | Request/response traces, latency, status codes |
{%- if cookiecutter.use_postgresql %}
| **PostgreSQL** | Query execution time, slow queries, connection pool |
{%- endif %}
{%- if cookiecutter.use_mongodb %}
| **MongoDB** | Collection operations, query filters, execution time |
{%- endif %}
{%- if cookiecutter.enable_redis %}
| **Redis** | Cache hits/misses, command latency, key patterns |
{%- endif %}
{%- if cookiecutter.use_celery %}
| **Celery** | Task execution, queue depth, worker performance |
{%- endif %}
{%- if cookiecutter.logfire_httpx %}
| **HTTPX** | External API calls, response times, error rates |
{%- endif %}

### Custom Instrumentation

```python
import logfire

# Manual spans for important operations
with logfire.span("process_order", order_id=str(order.id)):
    await validate_order(order)
    await charge_payment(order)
    await send_confirmation(order)

# Structured logging
logfire.info("User registered", user_id=user.id, email=user.email)
```

> 📚 For detailed Logfire documentation, see the [template observability guide](https://github.com/vstorm-co/full-stack-fastapi-nextjs-llm-template/blob/main/docs/observability.md).
{%- endif %}

---

## API Examples

{%- if cookiecutter.use_jwt %}

### Authentication

```bash
# Register
curl -X POST http://localhost:{{ cookiecutter.backend_port }}/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123"}'

# Login
curl -X POST http://localhost:{{ cookiecutter.backend_port }}/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=password123"

# Protected endpoint
curl http://localhost:{{ cookiecutter.backend_port }}/api/v1/users/me \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```
{%- endif %}

### Health Check

```bash
curl http://localhost:{{ cookiecutter.backend_port }}/api/v1/health
```

{%- if cookiecutter.enable_ai_agent %}

### AI Agent (WebSocket)

Connect to `ws://localhost:{{ cookiecutter.backend_port }}/api/v1/agent/ws` and send:

```json
{"type": "message", "content": "Hello!", "history": []}
```

Response events:
- `start` - Stream started
- `token` - Text token (streaming)
- `tool_call` - Tool invocation
- `end` - Stream complete
{%- endif %}

---

## Testing

```bash
cd backend

# Run all tests
pytest

# With coverage
pytest --cov=app --cov-report=term-missing

# Specific test
pytest tests/api/test_health.py -v
```

{%- if cookiecutter.use_frontend %}

### Frontend Tests

```bash
cd frontend

# Unit tests (Vitest)
bun test

# E2E tests (Playwright)
bun test:e2e
```
{%- endif %}

---

## Deployment

> 📚 For detailed deployment guide, see the [template documentation](https://github.com/vstorm-co/full-stack-fastapi-nextjs-llm-template/blob/main/docs/deployment.md).

### Quick Docker Deploy

```bash
# Build production images
docker compose -f docker-compose.yml -f docker-compose.prod.yml build

# Start with production config
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Environment Checklist

- [ ] Set `ENVIRONMENT=production`
- [ ] Set `DEBUG=false`
- [ ] Change `SECRET_KEY` (use `openssl rand -hex 32`)
{%- if cookiecutter.use_postgresql %}
- [ ] Configure production database credentials
{%- endif %}
{%- if cookiecutter.enable_logfire %}
- [ ] Set `LOGFIRE_TOKEN` for production
{%- endif %}
{%- if cookiecutter.enable_sentry %}
- [ ] Configure `SENTRY_DSN`
{%- endif %}

---

## Documentation

| Resource | Link |
|----------|------|
| Template Repository | [github.com/vstorm-co/full-stack-fastapi-nextjs-llm-template](https://github.com/vstorm-co/full-stack-fastapi-nextjs-llm-template) |
| Architecture Guide | [docs/architecture.md](https://github.com/vstorm-co/full-stack-fastapi-nextjs-llm-template/blob/main/docs/architecture.md) |
{%- if cookiecutter.use_frontend %}
| Frontend Guide | [docs/frontend.md](https://github.com/vstorm-co/full-stack-fastapi-nextjs-llm-template/blob/main/docs/frontend.md) |
{%- endif %}
{%- if cookiecutter.enable_ai_agent %}
| AI Agent Guide | [docs/ai-agent.md](https://github.com/vstorm-co/full-stack-fastapi-nextjs-llm-template/blob/main/docs/ai-agent.md) |
{%- endif %}
{%- if cookiecutter.enable_logfire %}
| Observability Guide | [docs/observability.md](https://github.com/vstorm-co/full-stack-fastapi-nextjs-llm-template/blob/main/docs/observability.md) |
{%- endif %}
| Deployment Guide | [docs/deployment.md](https://github.com/vstorm-co/full-stack-fastapi-nextjs-llm-template/blob/main/docs/deployment.md) |
| Development Guide | [docs/development.md](https://github.com/vstorm-co/full-stack-fastapi-nextjs-llm-template/blob/main/docs/development.md) |

---

## License

MIT

---

<p align="center">
  <sub>Built with <a href="https://github.com/vstorm-co/full-stack-fastapi-nextjs-llm-template">Full-Stack FastAPI + Next.js Template</a></sub>
</p>
