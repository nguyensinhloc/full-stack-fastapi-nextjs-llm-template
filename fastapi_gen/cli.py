"""CLI interface for FastAPI project generator."""

from pathlib import Path

import click
from rich.console import Console

from . import __version__
from .config import (
    AIFrameworkType,
    AuthType,
    CIType,
    DatabaseType,
    FrontendType,
    ProjectConfig,
)
from .generator import generate_project, post_generation_tasks
from .prompts import confirm_generation, run_interactive_prompts, show_summary

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="fastapi-gen")
def cli() -> None:
    """FastAPI Project Generator with Logfire observability."""


@cli.command()
@click.option(
    "-o",
    "--output",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Output directory for the generated project",
)
@click.option(
    "--no-input",
    is_flag=True,
    default=False,
    help="Use default values without prompts",
)
@click.option("--name", type=str, help="Project name (for --no-input mode)")
def new(output: Path | None, no_input: bool, name: str | None) -> None:
    """Create a new FastAPI project interactively."""
    try:
        if no_input:
            if not name:
                console.print("[red]Error:[/] --name is required when using --no-input")
                raise SystemExit(1)

            config = ProjectConfig(project_name=name)
        else:
            config = run_interactive_prompts()
            show_summary(config)

            if not confirm_generation():
                console.print("[yellow]Project generation cancelled.[/]")
                return

        project_path = generate_project(config, output)
        post_generation_tasks(project_path, config)

    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/]")
        raise SystemExit(0) from None
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        raise SystemExit(1) from None


@cli.command()
@click.argument("name", type=str)
@click.option(
    "-o",
    "--output",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Output directory",
)
@click.option(
    "--database",
    type=click.Choice(["postgresql", "mongodb", "sqlite", "none"]),
    default="postgresql",
    help="Database type",
)
@click.option(
    "--auth",
    type=click.Choice(["jwt", "api_key", "both", "none"]),
    default="jwt",
    help="Authentication method",
)
@click.option("--no-logfire", is_flag=True, help="Disable Logfire integration")
@click.option("--no-docker", is_flag=True, help="Disable Docker files")
@click.option("--no-env", is_flag=True, help="Skip .env file generation")
@click.option("--minimal", is_flag=True, help="Create minimal project (no extras)")
@click.option("--no-example-crud", is_flag=True, help="Skip example CRUD endpoint")
@click.option(
    "--frontend",
    type=click.Choice(["none", "nextjs"]),
    default="none",
    help="Frontend framework",
)
@click.option(
    "--backend-port",
    type=int,
    default=8000,
    help="Backend server port (default: 8000)",
)
@click.option(
    "--frontend-port",
    type=int,
    default=3000,
    help="Frontend server port (default: 3000)",
)
@click.option(
    "--db-pool-size",
    type=int,
    default=5,
    help="Database connection pool size (default: 5)",
)
@click.option(
    "--db-max-overflow",
    type=int,
    default=10,
    help="Database max overflow connections (default: 10)",
)
@click.option(
    "--ai-agent",
    is_flag=True,
    default=False,
    help="Enable AI agent with WebSocket streaming",
)
@click.option(
    "--ai-framework",
    type=click.Choice(["pydantic_ai", "langchain"]),
    default="pydantic_ai",
    help="AI framework (default: pydantic_ai)",
)
def create(
    name: str,
    output: Path | None,
    database: str,
    auth: str,
    no_logfire: bool,
    no_docker: bool,
    no_env: bool,
    minimal: bool,
    no_example_crud: bool,
    frontend: str,
    backend_port: int,
    frontend_port: int,
    db_pool_size: int,
    db_max_overflow: int,
    ai_agent: bool,
    ai_framework: str,
) -> None:
    """Create a new FastAPI project with specified options.

    NAME is the project name (e.g., my_project)
    """
    try:
        if minimal:
            config = ProjectConfig(
                project_name=name,
                database=DatabaseType.NONE,
                auth=AuthType.NONE,
                enable_logfire=False,
                enable_redis=False,
                enable_caching=False,
                enable_rate_limiting=False,
                enable_pagination=False,
                enable_admin_panel=False,
                enable_websockets=False,
                enable_docker=False,
                enable_kubernetes=False,
                ci_type=CIType.NONE,
                generate_env=not no_env,
                include_example_crud=False,
                frontend=FrontendType(frontend),
                backend_port=backend_port,
                frontend_port=frontend_port,
            )
        else:
            config = ProjectConfig(
                project_name=name,
                database=DatabaseType(database),
                auth=AuthType(auth),
                enable_logfire=not no_logfire,
                enable_docker=not no_docker,
                generate_env=not no_env,
                include_example_crud=not no_example_crud,
                frontend=FrontendType(frontend),
                backend_port=backend_port,
                frontend_port=frontend_port,
                db_pool_size=db_pool_size,
                db_max_overflow=db_max_overflow,
                enable_ai_agent=ai_agent,
                ai_framework=AIFrameworkType(ai_framework),
            )

        console.print(f"[cyan]Creating project:[/] {name}")
        console.print(f"[dim]Database: {config.database.value}[/]")
        console.print(f"[dim]Auth: {config.auth.value}[/]")
        if config.frontend != FrontendType.NONE:
            console.print(f"[dim]Frontend: {config.frontend.value}[/]")
        if config.enable_ai_agent:
            console.print(f"[dim]AI Agent: {config.ai_framework.value}[/]")
        console.print()

        project_path = generate_project(config, output)
        post_generation_tasks(project_path, config)

    except ValueError as e:
        console.print(f"[red]Invalid configuration:[/] {e}")
        raise SystemExit(1) from None
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        raise SystemExit(1) from None


@cli.command()
def templates() -> None:
    """List available template options."""
    console.print("[bold cyan]Available Options[/]")
    console.print()

    console.print("[bold]Databases:[/]")
    console.print("  - postgresql  PostgreSQL with asyncpg (async)")
    console.print("  - mongodb     MongoDB with Motor (async)")
    console.print("  - sqlite      SQLite with SQLAlchemy (sync)")
    console.print("  - none        No database")
    console.print()

    console.print("[bold]Authentication:[/]")
    console.print("  - jwt         JWT + User Management")
    console.print("  - api_key     API Key (header-based)")
    console.print("  - both        JWT with API Key fallback")
    console.print("  - none        No authentication")
    console.print()

    console.print("[bold]Background Tasks:[/]")
    console.print("  - none        FastAPI BackgroundTasks only")
    console.print("  - celery      Celery (classic)")
    console.print("  - taskiq      Taskiq (async-native)")
    console.print("  - arq         ARQ (lightweight)")
    console.print()

    console.print("[bold]Frontend:[/]")
    console.print("  - none        API only (no frontend)")
    console.print("  - nextjs      Next.js 15 (App Router, TypeScript, Bun)")
    console.print()

    console.print("[bold]AI Frameworks:[/]")
    console.print("  - pydantic_ai  PydanticAI (recommended)")
    console.print("  - langchain    LangChain")
    console.print()

    console.print("[bold]Optional Features:[/]")
    console.print("  - Logfire integration")
    console.print("  - Redis (caching/sessions)")
    console.print("  - Rate limiting (slowapi)")
    console.print("  - Pagination (fastapi-pagination)")
    console.print("  - Admin Panel (SQLAdmin)")
    console.print("  - WebSockets")
    console.print("  - File Storage (S3/MinIO)")
    console.print("  - AI Agent (--ai-agent --ai-framework pydantic_ai|langchain)")
    console.print("  - Example CRUD (Item model)")
    console.print("  - Docker + docker-compose")
    console.print("  - GitHub Actions / GitLab CI")
    console.print("  - Kubernetes manifests")


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
