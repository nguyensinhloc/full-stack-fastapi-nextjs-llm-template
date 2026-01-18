{%- if cookiecutter.enable_ai_agent and cookiecutter.use_deepagents %}
"""DeepAgents implementation with middleware stacking.

DeepAgents is a framework for building agentic coding assistants.
It uses LangGraph under the hood and comes with built-in tools for:
- File operations: ls, read_file, write_file, edit_file, glob, grep
- Task management: write_todos, task (subagent spawning)
- Shell execution: execute (when sandbox backend is enabled)
"""

import logging
from typing import Annotated, Any, TypedDict

from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
{%- if cookiecutter.use_openai %}
from langchain_openai import ChatOpenAI
{%- endif %}
{%- if cookiecutter.use_anthropic %}
from langchain_anthropic import ChatAnthropic
{%- endif %}

from app.agents.prompts import DEFAULT_SYSTEM_PROMPT
from app.core.config import settings

logger = logging.getLogger(__name__)


class AgentContext(TypedDict, total=False):
    """Runtime context for the agent.

    Passed via config parameter to the graph.
    """

    user_id: str | None
    user_name: str | None
    metadata: dict[str, Any]


class AgentState(TypedDict):
    """State for the DeepAgents agent.

    This is what flows through the agent graph.
    The messages field uses add_messages reducer to properly
    append new messages to the conversation history.
    """

    messages: Annotated[list[BaseMessage], add_messages]


class DeepAgentsAssistant:
    """Wrapper for DeepAgents with run() and stream() methods.

    DeepAgents creates a LangGraph-based agent with built-in tools for
    filesystem operations, task management, and code execution.
    """

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ):
        self.model_name = model_name or settings.AI_MODEL
        self.temperature = temperature or settings.AI_TEMPERATURE
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._graph = None
        self._checkpointer = MemorySaver()

    def _create_model(self):
        """Create the LLM model for DeepAgents."""
{%- if cookiecutter.use_openai %}
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=settings.OPENAI_API_KEY,
            streaming=True,
        )
{%- endif %}
{%- if cookiecutter.use_anthropic %}
        return ChatAnthropic(
            model=self.model_name,
            temperature=self.temperature,
            api_key=settings.ANTHROPIC_API_KEY,
            streaming=True,
        )
{%- endif %}

    @property
    def graph(self):
        """Get or create the compiled graph instance."""
        if self._graph is None:
            model = self._create_model()
            self._graph = create_deep_agent(
                model=model,
                system_prompt=self.system_prompt,
                checkpointer=self._checkpointer,
            )
        return self._graph

    @staticmethod
    def _convert_history(
        history: list[dict[str, str]] | None,
    ) -> list[HumanMessage | AIMessage | SystemMessage]:
        """Convert conversation history to LangChain message format."""
        messages: list[HumanMessage | AIMessage | SystemMessage] = []

        for msg in history or []:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                messages.append(SystemMessage(content=msg["content"]))

        return messages

    async def run(
        self,
        user_input: str,
        history: list[dict[str, str]] | None = None,
        context: AgentContext | None = None,
        thread_id: str = "default",
    ) -> tuple[str, list[Any], AgentContext]:
        """Run agent and return the output along with tool call events.

        Args:
            user_input: User's message.
            history: Conversation history as list of {"role": "...", "content": "..."}.
            context: Optional runtime context with user info.
            thread_id: Thread ID for conversation continuity.

        Returns:
            Tuple of (output_text, tool_events, context).
        """
        messages = self._convert_history(history)
        messages.append(HumanMessage(content=user_input))

        agent_context: AgentContext = context if context is not None else {}

        logger.info(f"Running DeepAgents with user input: {user_input[:100]}...")

        config = {
            "configurable": {
                "thread_id": thread_id,
                **agent_context,
            }
        }

        result = await self.graph.ainvoke({"messages": messages}, config=config)

        # Extract the final response and tool events
        output = ""
        tool_events: list[Any] = []

        for message in result.get("messages", []):
            if isinstance(message, AIMessage):
                if message.content:
                    output = message.content if isinstance(message.content, str) else str(message.content)
                if hasattr(message, "tool_calls") and message.tool_calls:
                    tool_events.extend(message.tool_calls)

        logger.info(f"DeepAgents run complete. Output length: {len(output)} chars")

        return output, tool_events, agent_context

    async def stream(
        self,
        user_input: str,
        history: list[dict[str, str]] | None = None,
        context: AgentContext | None = None,
        thread_id: str = "default",
    ):
        """Stream agent execution with message and state update streaming.

        Args:
            user_input: User's message.
            history: Conversation history.
            context: Optional runtime context.
            thread_id: Thread ID for conversation continuity.

        Yields:
            Tuples of (stream_mode, data) for streaming responses.
            - stream_mode="messages": (chunk, metadata) for LLM tokens
            - stream_mode="updates": state updates after each node
        """
        messages = self._convert_history(history)
        messages.append(HumanMessage(content=user_input))

        agent_context: AgentContext = context if context is not None else {}

        config = {
            "configurable": {
                "thread_id": thread_id,
                **agent_context,
            }
        }

        logger.info(f"Starting DeepAgents stream for user input: {user_input[:100]}...")

        async for stream_mode, data in self.graph.astream(
            {"messages": messages},
            config=config,
            stream_mode=["messages", "updates"],
        ):
            yield stream_mode, data


def get_agent() -> DeepAgentsAssistant:
    """Factory function to create a DeepAgentsAssistant.

    Returns:
        Configured DeepAgentsAssistant instance.
    """
    return DeepAgentsAssistant()


async def run_agent(
    user_input: str,
    history: list[dict[str, str]],
    context: AgentContext | None = None,
    thread_id: str = "default",
) -> tuple[str, list[Any], AgentContext]:
    """Run agent and return the output along with tool call events.

    This is a convenience function for backwards compatibility.

    Args:
        user_input: User's message.
        history: Conversation history.
        context: Optional runtime context.
        thread_id: Thread ID for conversation continuity.

    Returns:
        Tuple of (output_text, tool_events, context).
    """
    agent = get_agent()
    return await agent.run(user_input, history, context, thread_id)
{%- else %}
"""DeepAgents Assistant agent - not configured."""
{%- endif %}
