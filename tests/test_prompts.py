"""Tests for fastapi_gen.prompts module."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from fastapi_gen.config import (
    AdminEnvironmentType,
    AIFrameworkType,
    AuthType,
    BackgroundTaskType,
    CIType,
    DatabaseType,
    FrontendType,
    LogfireFeatures,
    OAuthProvider,
    WebSocketAuthType,
)
from fastapi_gen.prompts import (
    _check_cancelled,
    confirm_generation,
    prompt_admin_config,
    prompt_auth,
    prompt_background_tasks,
    prompt_basic_info,
    prompt_database,
    prompt_dev_tools,
    prompt_frontend,
    prompt_frontend_features,
    prompt_integrations,
    prompt_logfire,
    prompt_oauth,
    prompt_ports,
    prompt_python_version,
    prompt_websocket_auth,
    run_interactive_prompts,
    show_header,
    show_summary,
)


class TestCheckCancelled:
    """Tests for _check_cancelled helper."""

    def test_returns_value_when_not_none(self) -> None:
        """Test value is returned when not None."""
        assert _check_cancelled("test") == "test"
        assert _check_cancelled(123) == 123
        assert _check_cancelled([1, 2, 3]) == [1, 2, 3]

    def test_raises_keyboard_interrupt_when_none(self) -> None:
        """Test KeyboardInterrupt is raised when value is None."""
        with pytest.raises(KeyboardInterrupt):
            _check_cancelled(None)


class TestShowHeader:
    """Tests for show_header function."""

    def test_show_header_runs_without_error(self) -> None:
        """Test header displays without errors."""
        # Just verify it doesn't raise
        show_header()


class TestPromptBasicInfo:
    """Tests for prompt_basic_info function."""

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_all_fields(self, mock_questionary: MagicMock) -> None:
        """Test all basic info fields are returned."""
        mock_text = MagicMock()
        mock_text.ask.side_effect = [
            "myproject",
            "My description",
            "John Doe",
            "john@example.com",
        ]
        mock_questionary.text.return_value = mock_text

        result = prompt_basic_info()

        assert result == {
            "project_name": "myproject",
            "project_description": "My description",
            "author_name": "John Doe",
            "author_email": "john@example.com",
        }

    @patch("fastapi_gen.prompts.questionary")
    def test_raises_on_cancel(self, mock_questionary: MagicMock) -> None:
        """Test KeyboardInterrupt on cancel."""
        mock_text = MagicMock()
        mock_text.ask.return_value = None
        mock_questionary.text.return_value = mock_text

        with pytest.raises(KeyboardInterrupt):
            prompt_basic_info()


class TestPromptDatabase:
    """Tests for prompt_database function."""

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_selected_database(self, mock_questionary: MagicMock) -> None:
        """Test selected database is returned."""
        mock_select = MagicMock()
        mock_select.ask.return_value = DatabaseType.MONGODB
        mock_questionary.select.return_value = mock_select
        mock_questionary.Choice = MagicMock()

        result = prompt_database()

        assert result == DatabaseType.MONGODB

    @patch("fastapi_gen.prompts.questionary")
    def test_raises_on_cancel(self, mock_questionary: MagicMock) -> None:
        """Test KeyboardInterrupt on cancel."""
        mock_select = MagicMock()
        mock_select.ask.return_value = None
        mock_questionary.select.return_value = mock_select
        mock_questionary.Choice = MagicMock()

        with pytest.raises(KeyboardInterrupt):
            prompt_database()


class TestPromptAuth:
    """Tests for prompt_auth function."""

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_selected_auth(self, mock_questionary: MagicMock) -> None:
        """Test selected auth method is returned."""
        mock_select = MagicMock()
        mock_select.ask.return_value = AuthType.API_KEY
        mock_questionary.select.return_value = mock_select
        mock_questionary.Choice = MagicMock()

        result = prompt_auth()

        assert result == AuthType.API_KEY


class TestPromptLogfire:
    """Tests for prompt_logfire function."""

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_disabled_logfire(self, mock_questionary: MagicMock) -> None:
        """Test disabled Logfire returns False and default features."""
        mock_confirm = MagicMock()
        mock_confirm.ask.return_value = False
        mock_questionary.confirm.return_value = mock_confirm

        enabled, features = prompt_logfire()

        assert enabled is False
        assert isinstance(features, LogfireFeatures)

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_enabled_logfire_with_features(self, mock_questionary: MagicMock) -> None:
        """Test enabled Logfire returns selected features."""
        mock_confirm = MagicMock()
        mock_confirm.ask.return_value = True
        mock_questionary.confirm.return_value = mock_confirm

        mock_checkbox = MagicMock()
        mock_checkbox.ask.return_value = ["fastapi", "redis", "httpx"]
        mock_questionary.checkbox.return_value = mock_checkbox
        mock_questionary.Choice = MagicMock()

        enabled, features = prompt_logfire()

        assert enabled is True
        assert features.fastapi is True
        assert features.database is False
        assert features.redis is True
        assert features.celery is False
        assert features.httpx is True


class TestPromptBackgroundTasks:
    """Tests for prompt_background_tasks function."""

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_selected_task_type(self, mock_questionary: MagicMock) -> None:
        """Test selected background task type is returned."""
        mock_select = MagicMock()
        mock_select.ask.return_value = BackgroundTaskType.TASKIQ
        mock_questionary.select.return_value = mock_select
        mock_questionary.Choice = MagicMock()

        result = prompt_background_tasks()

        assert result == BackgroundTaskType.TASKIQ


class TestPromptIntegrations:
    """Tests for prompt_integrations function."""

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_selected_integrations(self, mock_questionary: MagicMock) -> None:
        """Test selected integrations are returned."""
        mock_checkbox = MagicMock()
        mock_checkbox.ask.return_value = ["redis", "caching", "websockets"]
        mock_questionary.checkbox.return_value = mock_checkbox
        mock_questionary.Choice = MagicMock()

        result = prompt_integrations()

        assert result["enable_redis"] is True
        assert result["enable_caching"] is True
        assert result["enable_websockets"] is True
        assert result["enable_pagination"] is False
        assert result["enable_sentry"] is False


class TestPromptDevTools:
    """Tests for prompt_dev_tools function."""

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_selected_dev_tools(self, mock_questionary: MagicMock) -> None:
        """Test selected dev tools are returned."""
        mock_checkbox = MagicMock()
        mock_checkbox.ask.return_value = ["pytest", "docker", "kubernetes"]
        mock_questionary.checkbox.return_value = mock_checkbox

        mock_select = MagicMock()
        mock_select.ask.return_value = CIType.GITLAB
        mock_questionary.select.return_value = mock_select
        mock_questionary.Choice = MagicMock()

        result = prompt_dev_tools()

        assert result["enable_pytest"] is True
        assert result["enable_precommit"] is False
        assert result["enable_makefile"] is False
        assert result["enable_docker"] is True
        assert result["enable_kubernetes"] is True
        assert result["ci_type"] == CIType.GITLAB


class TestPromptFrontend:
    """Tests for prompt_frontend function."""

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_selected_frontend(self, mock_questionary: MagicMock) -> None:
        """Test selected frontend is returned."""
        mock_select = MagicMock()
        mock_select.ask.return_value = FrontendType.NEXTJS
        mock_questionary.select.return_value = mock_select
        mock_questionary.Choice = MagicMock()

        result = prompt_frontend()

        assert result == FrontendType.NEXTJS

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_none_frontend(self, mock_questionary: MagicMock) -> None:
        """Test none frontend is returned."""
        mock_select = MagicMock()
        mock_select.ask.return_value = FrontendType.NONE
        mock_questionary.select.return_value = mock_select
        mock_questionary.Choice = MagicMock()

        result = prompt_frontend()

        assert result == FrontendType.NONE

    @patch("fastapi_gen.prompts.questionary")
    def test_raises_on_cancel(self, mock_questionary: MagicMock) -> None:
        """Test KeyboardInterrupt on cancel."""
        mock_select = MagicMock()
        mock_select.ask.return_value = None
        mock_questionary.select.return_value = mock_select
        mock_questionary.Choice = MagicMock()

        with pytest.raises(KeyboardInterrupt):
            prompt_frontend()


class TestPromptWebsocketAuth:
    """Tests for prompt_websocket_auth function."""

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_none_auth(self, mock_questionary: MagicMock) -> None:
        """Test none auth is returned."""
        mock_select = MagicMock()
        mock_select.ask.return_value = WebSocketAuthType.NONE
        mock_questionary.select.return_value = mock_select
        mock_questionary.Choice = MagicMock()

        result = prompt_websocket_auth()

        assert result == WebSocketAuthType.NONE

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_jwt_auth(self, mock_questionary: MagicMock) -> None:
        """Test JWT auth is returned."""
        mock_select = MagicMock()
        mock_select.ask.return_value = WebSocketAuthType.JWT
        mock_questionary.select.return_value = mock_select
        mock_questionary.Choice = MagicMock()

        result = prompt_websocket_auth()

        assert result == WebSocketAuthType.JWT

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_api_key_auth(self, mock_questionary: MagicMock) -> None:
        """Test API key auth is returned."""
        mock_select = MagicMock()
        mock_select.ask.return_value = WebSocketAuthType.API_KEY
        mock_questionary.select.return_value = mock_select
        mock_questionary.Choice = MagicMock()

        result = prompt_websocket_auth()

        assert result == WebSocketAuthType.API_KEY


class TestPromptAIFramework:
    """Tests for prompt_ai_framework function."""

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_pydantic_ai(self, mock_questionary: MagicMock) -> None:
        """Test PydanticAI framework is returned."""
        from fastapi_gen.prompts import prompt_ai_framework

        mock_select = MagicMock()
        mock_select.ask.return_value = AIFrameworkType.PYDANTIC_AI
        mock_questionary.select.return_value = mock_select
        mock_questionary.Choice = MagicMock()

        result = prompt_ai_framework()

        assert result == AIFrameworkType.PYDANTIC_AI

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_langchain(self, mock_questionary: MagicMock) -> None:
        """Test LangChain framework is returned."""
        from fastapi_gen.prompts import prompt_ai_framework

        mock_select = MagicMock()
        mock_select.ask.return_value = AIFrameworkType.LANGCHAIN
        mock_questionary.select.return_value = mock_select
        mock_questionary.Choice = MagicMock()

        result = prompt_ai_framework()

        assert result == AIFrameworkType.LANGCHAIN


class TestPromptAdminConfig:
    """Tests for prompt_admin_config function."""

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_dev_staging_with_auth(self, mock_questionary: MagicMock) -> None:
        """Test dev_staging environment with auth is returned."""
        mock_select = MagicMock()
        mock_select.ask.return_value = AdminEnvironmentType.DEV_STAGING
        mock_questionary.select.return_value = mock_select

        mock_confirm = MagicMock()
        mock_confirm.ask.return_value = True
        mock_questionary.confirm.return_value = mock_confirm
        mock_questionary.Choice = MagicMock()

        env, require_auth = prompt_admin_config()

        assert env == AdminEnvironmentType.DEV_STAGING
        assert require_auth is True

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_disabled_skips_auth(self, mock_questionary: MagicMock) -> None:
        """Test disabled environment skips auth question."""
        mock_select = MagicMock()
        mock_select.ask.return_value = AdminEnvironmentType.DISABLED
        mock_questionary.select.return_value = mock_select
        mock_questionary.Choice = MagicMock()

        env, require_auth = prompt_admin_config()

        assert env == AdminEnvironmentType.DISABLED
        assert require_auth is False
        # confirm should not have been called
        mock_questionary.confirm.assert_not_called()

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_all_without_auth(self, mock_questionary: MagicMock) -> None:
        """Test all environment without auth is returned."""
        mock_select = MagicMock()
        mock_select.ask.return_value = AdminEnvironmentType.ALL
        mock_questionary.select.return_value = mock_select

        mock_confirm = MagicMock()
        mock_confirm.ask.return_value = False
        mock_questionary.confirm.return_value = mock_confirm
        mock_questionary.Choice = MagicMock()

        env, require_auth = prompt_admin_config()

        assert env == AdminEnvironmentType.ALL
        assert require_auth is False


class TestPromptPythonVersion:
    """Tests for prompt_python_version function."""

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_python_312(self, mock_questionary: MagicMock) -> None:
        """Test Python 3.12 is returned."""
        mock_select = MagicMock()
        mock_select.ask.return_value = "3.12"
        mock_questionary.select.return_value = mock_select
        mock_questionary.Choice = MagicMock()

        result = prompt_python_version()

        assert result == "3.12"

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_python_311(self, mock_questionary: MagicMock) -> None:
        """Test Python 3.11 is returned."""
        mock_select = MagicMock()
        mock_select.ask.return_value = "3.11"
        mock_questionary.select.return_value = mock_select
        mock_questionary.Choice = MagicMock()

        result = prompt_python_version()

        assert result == "3.11"

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_python_313(self, mock_questionary: MagicMock) -> None:
        """Test Python 3.13 is returned."""
        mock_select = MagicMock()
        mock_select.ask.return_value = "3.13"
        mock_questionary.select.return_value = mock_select
        mock_questionary.Choice = MagicMock()

        result = prompt_python_version()

        assert result == "3.13"


class TestPromptPorts:
    """Tests for prompt_ports function."""

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_backend_port_only(self, mock_questionary: MagicMock) -> None:
        """Test only backend port is returned when no frontend."""
        mock_text = MagicMock()
        mock_text.ask.return_value = "8000"
        mock_questionary.text.return_value = mock_text

        result = prompt_ports(has_frontend=False)

        assert result == {"backend_port": 8000}
        # text should only be called once for backend port
        assert mock_questionary.text.call_count == 1

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_both_ports_with_frontend(self, mock_questionary: MagicMock) -> None:
        """Test both ports are returned when frontend is enabled."""
        mock_text = MagicMock()
        mock_text.ask.side_effect = ["8080", "3001"]
        mock_questionary.text.return_value = mock_text

        result = prompt_ports(has_frontend=True)

        assert result == {"backend_port": 8080, "frontend_port": 3001}
        # text should be called twice
        assert mock_questionary.text.call_count == 2

    @patch("fastapi_gen.prompts.questionary")
    def test_port_validator_valid_port(self, mock_questionary: MagicMock) -> None:
        """Test port validator accepts valid ports."""
        mock_text = MagicMock()
        mock_text.ask.return_value = "8000"
        mock_questionary.text.return_value = mock_text

        prompt_ports(has_frontend=False)

        # Get the validator function passed to questionary.text
        call_kwargs = mock_questionary.text.call_args[1]
        validate_port = call_kwargs["validate"]

        # Test valid ports
        assert validate_port("1024") is True
        assert validate_port("8000") is True
        assert validate_port("65535") is True

    @patch("fastapi_gen.prompts.questionary")
    def test_port_validator_invalid_port(self, mock_questionary: MagicMock) -> None:
        """Test port validator rejects invalid ports."""
        mock_text = MagicMock()
        mock_text.ask.return_value = "8000"
        mock_questionary.text.return_value = mock_text

        prompt_ports(has_frontend=False)

        # Get the validator function passed to questionary.text
        call_kwargs = mock_questionary.text.call_args[1]
        validate_port = call_kwargs["validate"]

        # Test invalid ports
        assert validate_port("1023") is False  # Below range
        assert validate_port("65536") is False  # Above range
        assert validate_port("invalid") is False  # Not a number
        assert validate_port("") is False  # Empty string


class TestPromptOAuth:
    """Tests for prompt_oauth function."""

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_google_provider(self, mock_questionary: MagicMock) -> None:
        """Test Google OAuth provider is returned."""
        mock_select = MagicMock()
        mock_select.ask.return_value = OAuthProvider.GOOGLE
        mock_questionary.select.return_value = mock_select
        mock_questionary.Choice = MagicMock()

        result = prompt_oauth()

        assert result == OAuthProvider.GOOGLE

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_none_provider(self, mock_questionary: MagicMock) -> None:
        """Test None OAuth provider is returned."""
        mock_select = MagicMock()
        mock_select.ask.return_value = OAuthProvider.NONE
        mock_questionary.select.return_value = mock_select
        mock_questionary.Choice = MagicMock()

        result = prompt_oauth()

        assert result == OAuthProvider.NONE


class TestPromptFrontendFeatures:
    """Tests for prompt_frontend_features function."""

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_i18n_enabled(self, mock_questionary: MagicMock) -> None:
        """Test i18n feature is returned when selected."""
        mock_checkbox = MagicMock()
        mock_checkbox.ask.return_value = ["i18n"]
        mock_questionary.checkbox.return_value = mock_checkbox
        mock_questionary.Choice = MagicMock()

        result = prompt_frontend_features()

        assert result["enable_i18n"] is True

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_no_features(self, mock_questionary: MagicMock) -> None:
        """Test empty features are returned when nothing selected."""
        mock_checkbox = MagicMock()
        mock_checkbox.ask.return_value = []
        mock_questionary.checkbox.return_value = mock_checkbox
        mock_questionary.Choice = MagicMock()

        result = prompt_frontend_features()

        assert result["enable_i18n"] is False


class TestRunInteractivePrompts:
    """Tests for run_interactive_prompts function."""

    @patch("fastapi_gen.prompts.questionary")
    @patch("fastapi_gen.prompts.prompt_ports")
    @patch("fastapi_gen.prompts.prompt_python_version")
    @patch("fastapi_gen.prompts.prompt_frontend")
    @patch("fastapi_gen.prompts.prompt_dev_tools")
    @patch("fastapi_gen.prompts.prompt_integrations")
    @patch("fastapi_gen.prompts.prompt_background_tasks")
    @patch("fastapi_gen.prompts.prompt_logfire")
    @patch("fastapi_gen.prompts.prompt_oauth")
    @patch("fastapi_gen.prompts.prompt_auth")
    @patch("fastapi_gen.prompts.prompt_database")
    @patch("fastapi_gen.prompts.prompt_basic_info")
    @patch("fastapi_gen.prompts.show_header")
    def test_builds_project_config(
        self,
        mock_header: MagicMock,
        mock_basic_info: MagicMock,
        mock_database: MagicMock,
        mock_auth: MagicMock,
        mock_oauth: MagicMock,
        mock_logfire: MagicMock,
        mock_background_tasks: MagicMock,
        mock_integrations: MagicMock,
        mock_dev_tools: MagicMock,
        mock_frontend: MagicMock,
        mock_python_version: MagicMock,
        mock_ports: MagicMock,
        mock_questionary: MagicMock,
    ) -> None:
        """Test ProjectConfig is built from prompts."""
        mock_basic_info.return_value = {
            "project_name": "test_project",
            "project_description": "Test",
            "author_name": "Test Author",
            "author_email": "test@test.com",
        }
        mock_database.return_value = DatabaseType.POSTGRESQL
        mock_auth.return_value = AuthType.JWT
        mock_oauth.return_value = OAuthProvider.NONE
        mock_logfire.return_value = (True, LogfireFeatures())
        mock_background_tasks.return_value = BackgroundTaskType.NONE
        mock_integrations.return_value = {
            "enable_redis": False,
            "enable_caching": False,
            "enable_rate_limiting": False,
            "enable_pagination": True,
            "enable_sentry": False,
            "enable_prometheus": False,
            "enable_admin_panel": False,
            "enable_websockets": False,
            "enable_file_storage": False,
            "enable_ai_agent": False,
            "enable_cors": True,
            "enable_orjson": True,
        }
        mock_dev_tools.return_value = {
            "enable_pytest": True,
            "enable_precommit": True,
            "enable_makefile": True,
            "enable_docker": True,
            "enable_kubernetes": False,
            "ci_type": CIType.GITHUB,
        }
        mock_frontend.return_value = FrontendType.NONE
        mock_python_version.return_value = "3.12"
        mock_ports.return_value = {"backend_port": 8000}

        # Mock session management confirm
        mock_confirm = MagicMock()
        mock_confirm.ask.return_value = False
        mock_questionary.confirm.return_value = mock_confirm

        config = run_interactive_prompts()

        assert config.project_name == "test_project"
        assert config.database == DatabaseType.POSTGRESQL
        assert config.auth == AuthType.JWT
        assert config.enable_logfire is True
        assert config.ci_type == CIType.GITHUB
        assert config.python_version == "3.12"
        assert config.backend_port == 8000

    @patch("fastapi_gen.prompts.questionary")
    @patch("fastapi_gen.prompts.prompt_ports")
    @patch("fastapi_gen.prompts.prompt_python_version")
    @patch("fastapi_gen.prompts.prompt_frontend")
    @patch("fastapi_gen.prompts.prompt_dev_tools")
    @patch("fastapi_gen.prompts.prompt_integrations")
    @patch("fastapi_gen.prompts.prompt_background_tasks")
    @patch("fastapi_gen.prompts.prompt_logfire")
    @patch("fastapi_gen.prompts.prompt_oauth")
    @patch("fastapi_gen.prompts.prompt_auth")
    @patch("fastapi_gen.prompts.prompt_database")
    @patch("fastapi_gen.prompts.prompt_basic_info")
    @patch("fastapi_gen.prompts.show_header")
    def test_auto_enables_redis_for_celery(
        self,
        mock_header: MagicMock,
        mock_basic_info: MagicMock,
        mock_database: MagicMock,
        mock_auth: MagicMock,
        mock_oauth: MagicMock,
        mock_logfire: MagicMock,
        mock_background_tasks: MagicMock,
        mock_integrations: MagicMock,
        mock_dev_tools: MagicMock,
        mock_frontend: MagicMock,
        mock_python_version: MagicMock,
        mock_ports: MagicMock,
        mock_questionary: MagicMock,
    ) -> None:
        """Test Redis is auto-enabled when Celery is selected."""
        mock_basic_info.return_value = {
            "project_name": "test_project",
            "project_description": "Test",
            "author_name": "Test Author",
            "author_email": "test@test.com",
        }
        mock_database.return_value = DatabaseType.POSTGRESQL
        mock_auth.return_value = AuthType.JWT
        mock_oauth.return_value = OAuthProvider.NONE
        mock_logfire.return_value = (False, LogfireFeatures())
        mock_background_tasks.return_value = BackgroundTaskType.CELERY
        mock_integrations.return_value = {
            "enable_redis": False,  # User didn't select Redis
            "enable_caching": False,
            "enable_rate_limiting": False,
            "enable_pagination": True,
            "enable_sentry": False,
            "enable_prometheus": False,
            "enable_admin_panel": False,
            "enable_websockets": False,
            "enable_file_storage": False,
            "enable_ai_agent": False,
            "enable_cors": True,
            "enable_orjson": True,
        }
        mock_dev_tools.return_value = {
            "enable_pytest": True,
            "enable_precommit": True,
            "enable_makefile": True,
            "enable_docker": True,
            "enable_kubernetes": False,
            "ci_type": CIType.GITHUB,
        }
        mock_frontend.return_value = FrontendType.NONE
        mock_python_version.return_value = "3.12"
        mock_ports.return_value = {"backend_port": 8000}

        # Mock session management confirm
        mock_confirm = MagicMock()
        mock_confirm.ask.return_value = False
        mock_questionary.confirm.return_value = mock_confirm

        config = run_interactive_prompts()

        # Redis should be auto-enabled for Celery
        assert config.enable_redis is True
        assert config.background_tasks == BackgroundTaskType.CELERY

    @patch("fastapi_gen.prompts.questionary")
    @patch("fastapi_gen.prompts.prompt_websocket_auth")
    @patch("fastapi_gen.prompts.prompt_ai_framework")
    @patch("fastapi_gen.prompts.prompt_ports")
    @patch("fastapi_gen.prompts.prompt_python_version")
    @patch("fastapi_gen.prompts.prompt_frontend")
    @patch("fastapi_gen.prompts.prompt_dev_tools")
    @patch("fastapi_gen.prompts.prompt_integrations")
    @patch("fastapi_gen.prompts.prompt_background_tasks")
    @patch("fastapi_gen.prompts.prompt_logfire")
    @patch("fastapi_gen.prompts.prompt_oauth")
    @patch("fastapi_gen.prompts.prompt_auth")
    @patch("fastapi_gen.prompts.prompt_database")
    @patch("fastapi_gen.prompts.prompt_basic_info")
    @patch("fastapi_gen.prompts.show_header")
    def test_ai_agent_with_conversation_persistence(
        self,
        mock_header: MagicMock,
        mock_basic_info: MagicMock,
        mock_database: MagicMock,
        mock_auth: MagicMock,
        mock_oauth: MagicMock,
        mock_logfire: MagicMock,
        mock_background_tasks: MagicMock,
        mock_integrations: MagicMock,
        mock_dev_tools: MagicMock,
        mock_frontend: MagicMock,
        mock_python_version: MagicMock,
        mock_ports: MagicMock,
        mock_ai_framework: MagicMock,
        mock_websocket_auth: MagicMock,
        mock_questionary: MagicMock,
    ) -> None:
        """Test AI agent prompts websocket auth and conversation persistence."""
        mock_basic_info.return_value = {
            "project_name": "test_project",
            "project_description": "Test",
            "author_name": "Test Author",
            "author_email": "test@test.com",
        }
        mock_database.return_value = DatabaseType.POSTGRESQL
        mock_auth.return_value = AuthType.JWT
        mock_oauth.return_value = OAuthProvider.NONE
        mock_logfire.return_value = (True, LogfireFeatures())
        mock_background_tasks.return_value = BackgroundTaskType.NONE
        mock_integrations.return_value = {
            "enable_redis": False,
            "enable_caching": False,
            "enable_rate_limiting": False,
            "enable_pagination": True,
            "enable_sentry": False,
            "enable_prometheus": False,
            "enable_admin_panel": False,
            "enable_websockets": False,
            "enable_file_storage": False,
            "enable_ai_agent": True,  # AI Agent enabled
            "include_example_crud": True,
            "enable_cors": True,
            "enable_orjson": True,
        }
        mock_dev_tools.return_value = {
            "enable_pytest": True,
            "enable_precommit": True,
            "enable_makefile": True,
            "enable_docker": True,
            "enable_kubernetes": False,
            "ci_type": CIType.GITHUB,
        }
        mock_frontend.return_value = FrontendType.NONE
        mock_python_version.return_value = "3.12"
        mock_ports.return_value = {"backend_port": 8000}
        mock_ai_framework.return_value = AIFrameworkType.PYDANTIC_AI
        mock_websocket_auth.return_value = WebSocketAuthType.JWT

        # Mock session management and conversation persistence confirm
        mock_confirm = MagicMock()
        mock_confirm.ask.side_effect = [False, True]  # session mgmt, conversation persistence
        mock_questionary.confirm.return_value = mock_confirm

        config = run_interactive_prompts()

        # WebSocket auth and conversation persistence should be set
        assert config.enable_ai_agent is True
        assert config.websocket_auth == WebSocketAuthType.JWT
        assert config.enable_conversation_persistence is True
        mock_websocket_auth.assert_called_once()

    @patch("fastapi_gen.prompts.questionary")
    @patch("fastapi_gen.prompts.prompt_admin_config")
    @patch("fastapi_gen.prompts.prompt_ports")
    @patch("fastapi_gen.prompts.prompt_python_version")
    @patch("fastapi_gen.prompts.prompt_frontend")
    @patch("fastapi_gen.prompts.prompt_dev_tools")
    @patch("fastapi_gen.prompts.prompt_integrations")
    @patch("fastapi_gen.prompts.prompt_background_tasks")
    @patch("fastapi_gen.prompts.prompt_logfire")
    @patch("fastapi_gen.prompts.prompt_oauth")
    @patch("fastapi_gen.prompts.prompt_auth")
    @patch("fastapi_gen.prompts.prompt_database")
    @patch("fastapi_gen.prompts.prompt_basic_info")
    @patch("fastapi_gen.prompts.show_header")
    def test_admin_panel_with_postgresql(
        self,
        mock_header: MagicMock,
        mock_basic_info: MagicMock,
        mock_database: MagicMock,
        mock_auth: MagicMock,
        mock_oauth: MagicMock,
        mock_logfire: MagicMock,
        mock_background_tasks: MagicMock,
        mock_integrations: MagicMock,
        mock_dev_tools: MagicMock,
        mock_frontend: MagicMock,
        mock_python_version: MagicMock,
        mock_ports: MagicMock,
        mock_admin_config: MagicMock,
        mock_questionary: MagicMock,
    ) -> None:
        """Test admin panel prompts config when PostgreSQL is selected."""
        mock_basic_info.return_value = {
            "project_name": "test_project",
            "project_description": "Test",
            "author_name": "Test Author",
            "author_email": "test@test.com",
        }
        mock_database.return_value = DatabaseType.POSTGRESQL
        mock_auth.return_value = AuthType.JWT
        mock_oauth.return_value = OAuthProvider.NONE
        mock_logfire.return_value = (True, LogfireFeatures())
        mock_background_tasks.return_value = BackgroundTaskType.NONE
        mock_integrations.return_value = {
            "enable_redis": False,
            "enable_caching": False,
            "enable_rate_limiting": False,
            "enable_pagination": True,
            "enable_sentry": False,
            "enable_prometheus": False,
            "enable_admin_panel": True,  # Admin panel enabled
            "enable_websockets": False,
            "enable_file_storage": False,
            "enable_ai_agent": False,
            "include_example_crud": True,
            "enable_cors": True,
            "enable_orjson": True,
        }
        mock_dev_tools.return_value = {
            "enable_pytest": True,
            "enable_precommit": True,
            "enable_makefile": True,
            "enable_docker": True,
            "enable_kubernetes": False,
            "ci_type": CIType.GITHUB,
        }
        mock_frontend.return_value = FrontendType.NONE
        mock_python_version.return_value = "3.12"
        mock_ports.return_value = {"backend_port": 8000}
        mock_admin_config.return_value = (AdminEnvironmentType.DEV_ONLY, True)

        # Mock session management confirm
        mock_confirm = MagicMock()
        mock_confirm.ask.return_value = False
        mock_questionary.confirm.return_value = mock_confirm

        config = run_interactive_prompts()

        # Admin config should be set
        assert config.enable_admin_panel is True
        assert config.admin_environments == AdminEnvironmentType.DEV_ONLY
        assert config.admin_require_auth is True
        mock_admin_config.assert_called_once()

    @patch("fastapi_gen.prompts.questionary")
    @patch("fastapi_gen.prompts.prompt_frontend_features")
    @patch("fastapi_gen.prompts.prompt_ports")
    @patch("fastapi_gen.prompts.prompt_python_version")
    @patch("fastapi_gen.prompts.prompt_frontend")
    @patch("fastapi_gen.prompts.prompt_dev_tools")
    @patch("fastapi_gen.prompts.prompt_integrations")
    @patch("fastapi_gen.prompts.prompt_background_tasks")
    @patch("fastapi_gen.prompts.prompt_logfire")
    @patch("fastapi_gen.prompts.prompt_oauth")
    @patch("fastapi_gen.prompts.prompt_auth")
    @patch("fastapi_gen.prompts.prompt_database")
    @patch("fastapi_gen.prompts.prompt_basic_info")
    @patch("fastapi_gen.prompts.show_header")
    def test_frontend_with_nextjs_prompts_frontend_features(
        self,
        mock_header: MagicMock,
        mock_basic_info: MagicMock,
        mock_database: MagicMock,
        mock_auth: MagicMock,
        mock_oauth: MagicMock,
        mock_logfire: MagicMock,
        mock_background_tasks: MagicMock,
        mock_integrations: MagicMock,
        mock_dev_tools: MagicMock,
        mock_frontend: MagicMock,
        mock_python_version: MagicMock,
        mock_ports: MagicMock,
        mock_frontend_features: MagicMock,
        mock_questionary: MagicMock,
    ) -> None:
        """Test frontend features are prompted when Next.js is selected."""
        mock_basic_info.return_value = {
            "project_name": "test_project",
            "project_description": "Test",
            "author_name": "Test Author",
            "author_email": "test@test.com",
        }
        mock_database.return_value = DatabaseType.POSTGRESQL
        mock_auth.return_value = AuthType.JWT
        mock_oauth.return_value = OAuthProvider.NONE
        mock_logfire.return_value = (False, LogfireFeatures())
        mock_background_tasks.return_value = BackgroundTaskType.NONE
        mock_integrations.return_value = {
            "enable_redis": False,
            "enable_caching": False,
            "enable_rate_limiting": False,
            "enable_pagination": True,
            "enable_sentry": False,
            "enable_prometheus": False,
            "enable_admin_panel": False,
            "enable_websockets": False,
            "enable_file_storage": False,
            "enable_ai_agent": False,
            "enable_cors": True,
            "enable_orjson": True,
        }
        mock_dev_tools.return_value = {
            "enable_pytest": True,
            "enable_precommit": True,
            "enable_makefile": True,
            "enable_docker": True,
            "enable_kubernetes": False,
            "ci_type": CIType.GITHUB,
        }
        mock_frontend.return_value = FrontendType.NEXTJS
        mock_python_version.return_value = "3.12"
        mock_ports.return_value = {"backend_port": 8000, "frontend_port": 3000}
        mock_frontend_features.return_value = {"enable_i18n": True}

        # Mock session management confirm
        mock_confirm = MagicMock()
        mock_confirm.ask.return_value = False
        mock_questionary.confirm.return_value = mock_confirm

        config = run_interactive_prompts()

        # Frontend features should be called and i18n enabled
        assert config.frontend == FrontendType.NEXTJS
        assert config.enable_i18n is True
        mock_frontend_features.assert_called_once()


class TestShowSummary:
    """Tests for show_summary function."""

    def test_show_summary_runs_without_error(self, minimal_config: Any) -> None:
        """Test summary displays without errors."""
        # Just verify it doesn't raise
        show_summary(minimal_config)

    def test_show_summary_with_features(self, full_config: Any) -> None:
        """Test summary displays with all features."""
        # Just verify it doesn't raise
        show_summary(full_config)


class TestConfirmGeneration:
    """Tests for confirm_generation function."""

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_true_on_confirm(self, mock_questionary: MagicMock) -> None:
        """Test True is returned on confirmation."""
        mock_confirm = MagicMock()
        mock_confirm.ask.return_value = True
        mock_questionary.confirm.return_value = mock_confirm

        result = confirm_generation()

        assert result is True

    @patch("fastapi_gen.prompts.questionary")
    def test_returns_false_on_decline(self, mock_questionary: MagicMock) -> None:
        """Test False is returned on decline."""
        mock_confirm = MagicMock()
        mock_confirm.ask.return_value = False
        mock_questionary.confirm.return_value = mock_confirm

        result = confirm_generation()

        assert result is False

    @patch("fastapi_gen.prompts.questionary")
    def test_raises_on_cancel(self, mock_questionary: MagicMock) -> None:
        """Test KeyboardInterrupt on cancel."""
        mock_confirm = MagicMock()
        mock_confirm.ask.return_value = None
        mock_questionary.confirm.return_value = mock_confirm

        with pytest.raises(KeyboardInterrupt):
            confirm_generation()
