"""Tests for fastapi_gen.cli module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from fastapi_gen.cli import cli, create, main, new, templates
from fastapi_gen.config import AuthType, CIType, DatabaseType, FrontendType, ProjectConfig


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


class TestCli:
    """Tests for main CLI group."""

    def test_cli_help(self, runner: CliRunner) -> None:
        """Test CLI help displays."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "FastAPI Project Generator" in result.output

    def test_cli_version(self, runner: CliRunner) -> None:
        """Test CLI version displays."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "fastapi-gen" in result.output


class TestNewCommand:
    """Tests for 'new' command."""

    def test_new_help(self, runner: CliRunner) -> None:
        """Test new command help displays."""
        result = runner.invoke(new, ["--help"])
        assert result.exit_code == 0
        assert "Create a new FastAPI project interactively" in result.output

    @patch("fastapi_gen.cli.generate_project")
    @patch("fastapi_gen.cli.post_generation_tasks")
    def test_new_with_no_input(
        self,
        mock_post_gen: MagicMock,
        mock_generate: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test new command with --no-input flag."""
        mock_generate.return_value = tmp_path / "myproject"

        result = runner.invoke(new, ["--no-input", "--name", "myproject"])

        assert result.exit_code == 0
        mock_generate.assert_called_once()
        mock_post_gen.assert_called_once()

    def test_new_no_input_requires_name(self, runner: CliRunner) -> None:
        """Test --no-input requires --name."""
        result = runner.invoke(new, ["--no-input"])
        assert result.exit_code == 1
        assert "--name is required" in result.output

    @patch("fastapi_gen.cli.run_interactive_prompts")
    @patch("fastapi_gen.cli.show_summary")
    @patch("fastapi_gen.cli.confirm_generation")
    @patch("fastapi_gen.cli.generate_project")
    @patch("fastapi_gen.cli.post_generation_tasks")
    def test_new_interactive(
        self,
        mock_post_gen: MagicMock,
        mock_generate: MagicMock,
        mock_confirm: MagicMock,
        mock_summary: MagicMock,
        mock_prompts: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test new command in interactive mode."""
        mock_prompts.return_value = ProjectConfig(project_name="testproject")
        mock_confirm.return_value = True
        mock_generate.return_value = tmp_path / "testproject"

        result = runner.invoke(new, [])

        assert result.exit_code == 0
        mock_prompts.assert_called_once()
        mock_summary.assert_called_once()
        mock_confirm.assert_called_once()
        mock_generate.assert_called_once()

    @patch("fastapi_gen.cli.run_interactive_prompts")
    @patch("fastapi_gen.cli.show_summary")
    @patch("fastapi_gen.cli.confirm_generation")
    def test_new_cancelled(
        self,
        mock_confirm: MagicMock,
        mock_summary: MagicMock,
        mock_prompts: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Test new command cancelled by user."""
        mock_prompts.return_value = ProjectConfig(project_name="testproject")
        mock_confirm.return_value = False

        result = runner.invoke(new, [])

        assert result.exit_code == 0
        assert "cancelled" in result.output.lower()

    @patch("fastapi_gen.cli.run_interactive_prompts")
    def test_new_keyboard_interrupt(
        self,
        mock_prompts: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Test new command handles keyboard interrupt."""
        mock_prompts.side_effect = KeyboardInterrupt()

        result = runner.invoke(new, [])

        assert result.exit_code == 0
        assert "Cancelled" in result.output

    @patch("fastapi_gen.cli.run_interactive_prompts")
    def test_new_generic_exception(
        self,
        mock_prompts: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Test new command handles generic exceptions."""
        mock_prompts.side_effect = RuntimeError("Something went wrong")

        result = runner.invoke(new, [])

        assert result.exit_code == 1
        assert "Error" in result.output
        assert "Something went wrong" in result.output


class TestCreateCommand:
    """Tests for 'create' command."""

    def test_create_help(self, runner: CliRunner) -> None:
        """Test create command help displays."""
        result = runner.invoke(create, ["--help"])
        assert result.exit_code == 0
        assert "Create a new FastAPI project" in result.output

    @patch("fastapi_gen.cli.generate_project")
    @patch("fastapi_gen.cli.post_generation_tasks")
    def test_create_with_defaults(
        self,
        mock_post_gen: MagicMock,
        mock_generate: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test create with default options."""
        mock_generate.return_value = tmp_path / "myproject"

        result = runner.invoke(create, ["myproject"])

        assert result.exit_code == 0
        mock_generate.assert_called_once()
        config = mock_generate.call_args[0][0]
        assert config.project_name == "myproject"
        assert config.database == DatabaseType.POSTGRESQL
        assert config.auth == AuthType.JWT

    @patch("fastapi_gen.cli.generate_project")
    @patch("fastapi_gen.cli.post_generation_tasks")
    def test_create_with_mongodb(
        self,
        mock_post_gen: MagicMock,
        mock_generate: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test create with MongoDB."""
        mock_generate.return_value = tmp_path / "myproject"

        result = runner.invoke(create, ["myproject", "--database", "mongodb"])

        assert result.exit_code == 0
        config = mock_generate.call_args[0][0]
        assert config.database == DatabaseType.MONGODB

    @patch("fastapi_gen.cli.generate_project")
    @patch("fastapi_gen.cli.post_generation_tasks")
    def test_create_with_api_key_auth(
        self,
        mock_post_gen: MagicMock,
        mock_generate: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test create with API key authentication."""
        mock_generate.return_value = tmp_path / "myproject"

        result = runner.invoke(create, ["myproject", "--auth", "api_key"])

        assert result.exit_code == 0
        config = mock_generate.call_args[0][0]
        assert config.auth == AuthType.API_KEY

    @patch("fastapi_gen.cli.generate_project")
    @patch("fastapi_gen.cli.post_generation_tasks")
    def test_create_with_no_logfire(
        self,
        mock_post_gen: MagicMock,
        mock_generate: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test create with Logfire disabled."""
        mock_generate.return_value = tmp_path / "myproject"

        result = runner.invoke(create, ["myproject", "--no-logfire"])

        assert result.exit_code == 0
        config = mock_generate.call_args[0][0]
        assert config.enable_logfire is False

    @patch("fastapi_gen.cli.generate_project")
    @patch("fastapi_gen.cli.post_generation_tasks")
    def test_create_with_no_docker(
        self,
        mock_post_gen: MagicMock,
        mock_generate: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test create with Docker disabled."""
        mock_generate.return_value = tmp_path / "myproject"

        result = runner.invoke(create, ["myproject", "--no-docker"])

        assert result.exit_code == 0
        config = mock_generate.call_args[0][0]
        assert config.enable_docker is False

    @patch("fastapi_gen.cli.generate_project")
    @patch("fastapi_gen.cli.post_generation_tasks")
    def test_create_minimal(
        self,
        mock_post_gen: MagicMock,
        mock_generate: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test create with --minimal flag."""
        mock_generate.return_value = tmp_path / "myproject"

        result = runner.invoke(create, ["myproject", "--minimal"])

        assert result.exit_code == 0
        config = mock_generate.call_args[0][0]
        assert config.database == DatabaseType.NONE
        assert config.auth == AuthType.NONE
        assert config.enable_logfire is False
        assert config.enable_docker is False
        assert config.ci_type == CIType.NONE

    @patch("fastapi_gen.cli.generate_project")
    @patch("fastapi_gen.cli.post_generation_tasks")
    def test_create_with_nextjs_frontend(
        self,
        mock_post_gen: MagicMock,
        mock_generate: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test create with Next.js frontend."""
        mock_generate.return_value = tmp_path / "myproject"

        result = runner.invoke(create, ["myproject", "--frontend", "nextjs"])

        assert result.exit_code == 0
        assert "Frontend: nextjs" in result.output
        config = mock_generate.call_args[0][0]
        assert config.frontend == FrontendType.NEXTJS

    def test_create_invalid_project_name(self, runner: CliRunner) -> None:
        """Test create with invalid project name."""
        result = runner.invoke(create, ["Invalid-Name"])
        assert result.exit_code == 1
        assert "Invalid configuration" in result.output or "Error" in result.output

    @patch("fastapi_gen.cli.generate_project")
    def test_create_generic_exception(
        self,
        mock_generate: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Test create command handles generic exceptions."""
        mock_generate.side_effect = RuntimeError("Generation failed")

        result = runner.invoke(create, ["myproject"])

        assert result.exit_code == 1
        assert "Error" in result.output
        assert "Generation failed" in result.output

    @patch("fastapi_gen.cli.generate_project")
    @patch("fastapi_gen.cli.post_generation_tasks")
    def test_create_with_ai_agent_pydantic_ai(
        self,
        mock_post_gen: MagicMock,
        mock_generate: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test create with AI agent (PydanticAI)."""
        mock_generate.return_value = tmp_path / "myproject"

        result = runner.invoke(create, ["myproject", "--ai-agent", "--ai-framework", "pydantic_ai"])

        assert result.exit_code == 0
        assert "AI Agent: pydantic_ai" in result.output
        config = mock_generate.call_args[0][0]
        assert config.enable_ai_agent is True
        assert config.ai_framework.value == "pydantic_ai"

    @patch("fastapi_gen.cli.generate_project")
    @patch("fastapi_gen.cli.post_generation_tasks")
    def test_create_with_ai_agent_langchain(
        self,
        mock_post_gen: MagicMock,
        mock_generate: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test create with AI agent (LangChain)."""
        mock_generate.return_value = tmp_path / "myproject"

        result = runner.invoke(create, ["myproject", "--ai-agent", "--ai-framework", "langchain"])

        assert result.exit_code == 0
        assert "AI Agent: langchain" in result.output
        config = mock_generate.call_args[0][0]
        assert config.enable_ai_agent is True
        assert config.ai_framework.value == "langchain"


class TestTemplatesCommand:
    """Tests for 'templates' command."""

    def test_templates_displays_options(self, runner: CliRunner) -> None:
        """Test templates command displays available options."""
        result = runner.invoke(templates, [])

        assert result.exit_code == 0
        assert "Databases" in result.output
        assert "postgresql" in result.output
        assert "mongodb" in result.output
        assert "Authentication" in result.output
        assert "jwt" in result.output
        assert "api_key" in result.output
        assert "Background Tasks" in result.output
        assert "celery" in result.output
        assert "taskiq" in result.output


class TestMainEntrypoint:
    """Tests for main entrypoint."""

    def test_main_calls_cli(self) -> None:
        """Test main function calls cli."""
        with patch("fastapi_gen.cli.cli") as mock_cli:
            main()
            mock_cli.assert_called_once()
