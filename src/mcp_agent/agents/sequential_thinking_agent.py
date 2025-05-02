import asyncio
import json
import re
from typing import Dict, Any, Optional, List, Union, TypeVar, cast

from mcp_agent.agents.base_agent import BaseAgent
from mcp_agent.core.request_params import RequestParams
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.core.prompt import Prompt
from mcp_agent.agents.agent import Agent
from mcp_agent.core.agent_types import AgentConfig, AgentType
from mcp_agent.human_input.types import HumanInputCallback
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.interfaces import AugmentedLLMProtocol
from mcp.types import PromptMessage  # Import from mcp.types instead

logger = get_logger(__name__)

# Define a TypeVar for AugmentedLLM and its subclasses
LLM = TypeVar("LLM", bound=AugmentedLLMProtocol)


class SequentialThinkingAgent(Agent):
    """
    A specialized Agent implementation that provides proper support for sequential thinking.

    This agent maintains state between interactions to ensure that sequential thinking
    works correctly, particularly with the nextThoughtNeeded parameter.
    """

    def __init__(
        self,
        config: AgentConfig,
        functions: Optional[List[Any]] = None,
        connection_persistence: bool = True,
        human_input_callback: Optional[HumanInputCallback] = None,
        context: Optional[Any] = None,
        max_thoughts: int = 10,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Initialize the Sequential Thinking Agent.

        Args:
            config: Agent configuration
            functions: Optional list of functions to register
            connection_persistence: Whether to persist connections
            human_input_callback: Optional callback for human input
            context: Optional context
            max_thoughts: Maximum number of thoughts to generate
            **kwargs: Additional arguments
        """
        # Initialize with base Agent constructor
        super().__init__(
            config=config,
            functions=functions,
            connection_persistence=connection_persistence,
            human_input_callback=human_input_callback,
            context=context,
            **kwargs,
        )

        # Initialize sequential thinking state
        self.max_thoughts = max_thoughts
        self.state = self._reset_state()

        # Ensure the agent has "sequential_thinking" in its servers
        if not any(s == "sequential_thinking" for s in self.config.servers):
            logger.warning(
                "SequentialThinkingAgent initialized without 'sequential_thinking' server. Adding it automatically."
            )
            self.config.servers.append("sequential_thinking")

    def _reset_state(self) -> Dict[str, Any]:
        """Reset the sequential thinking state to initial values."""
        return {
            "thought_number": 1,
            "total_thoughts": 5,  # Initial estimate
            "next_thought_needed": True,
            "branch_from_thought": None,
            "branch_id": None,
            "is_revision": False,
            "revises_thought": None,
            "needs_more_thoughts": False,
        }

    async def send(self, message: Union[str, PromptMessage, PromptMessageMultipart]) -> str:
        """
        Send a message to the agent, enhancing it with sequential thinking state.

        Args:
            message: The message to send

        Returns:
            The agent's response
        """
        # Check if this is a new conversation by safely accessing message history
        is_new_conversation = True

        # Use message_history property from BaseAgent which is safer than accessing _llm._history directly
        try:
            # If there's only the system message or no messages, it's a new conversation
            if self._llm is not None:
                history = self.message_history
                is_new_conversation = len(history) <= 1
        except (AttributeError, TypeError):
            # If we can't access history, assume it's a new conversation
            is_new_conversation = True

        if is_new_conversation:
            # Reset state for new conversations
            self.state = self._reset_state()

        # Only enhance string messages
        if isinstance(message, str):
            # Enhance the message with sequential thinking state information
            enhanced_message = self._enhance_message_with_state(message)
            # Send the enhanced message
            response = await super().send(enhanced_message)
        else:
            # Send the original message for non-string message types
            response = await super().send(message)

        # Parse the response to update our state
        self._update_state_from_response(response)

        return response

    def _enhance_message_with_state(self, message: str) -> str:
        """
        Enhance a message with sequential thinking state information.

        Args:
            message: The original message

        Returns:
            The enhanced message with state information
        """
        # Don't add state information for very short messages that are likely
        # just acknowledgments or follow-ups
        if len(message.strip()) < 10:
            return message

        # Format the state parameters for the message
        params = [
            f"thoughtNumber: {self.state['thought_number']}",
            f"totalThoughts: {self.state['total_thoughts']}",
            f"nextThoughtNeeded: {str(self.state['next_thought_needed']).lower()}",
        ]

        # Add optional parameters if they're set
        if self.state["is_revision"]:
            params.append(f"isRevision: true")
            params.append(f"revisesThought: {self.state['revises_thought']}")

        if self.state["branch_from_thought"] is not None:
            params.append(f"branchFromThought: {self.state['branch_from_thought']}")
            if self.state["branch_id"]:
                params.append(f'branchId: "{self.state["branch_id"]}"')

        if self.state["needs_more_thoughts"]:
            params.append("needsMoreThoughts: true")

        # Join parameters with commas
        params_str = ", ".join(params)

        # Add the state information to the message
        return f"""{message}

When using the sequential_thinking tool, please use these parameters:
{params_str}

Continue your reasoning from where you left off, showing your step-by-step thinking.
"""

    def _update_state_from_response(self, response: str) -> None:
        """
        Update the sequential thinking state based on the response.

        Args:
            response: The agent's response
        """
        # Check for tool call results in the response
        import re
        import json

        # Look for results from mcp_sequential-thinking tool calls
        pattern = r"Tool: (?:mcp_sequential-thinking_sequentialthinking|sequentialthinking), Result: (\{.*?\})"
        match = re.search(pattern, response, re.DOTALL)

        if match:
            try:
                # Parse the tool result
                result_str = match.group(1).strip()
                result = json.loads(result_str)

                # Update state with values from the tool result
                if "thoughtNumber" in result:
                    self.state["thought_number"] = result["thoughtNumber"]
                else:
                    # If thoughtNumber is missing, increment
                    self.state["thought_number"] += 1

                if "totalThoughts" in result:
                    self.state["total_thoughts"] = result["totalThoughts"]

                if "nextThoughtNeeded" in result:
                    self.state["next_thought_needed"] = result["nextThoughtNeeded"]

                # Update optional parameters if present
                if "isRevision" in result:
                    self.state["is_revision"] = result["isRevision"]

                if "revisesThought" in result:
                    self.state["revises_thought"] = result["revisesThought"]

                if "branchFromThought" in result:
                    self.state["branch_from_thought"] = result["branchFromThought"]

                if "branchId" in result:
                    self.state["branch_id"] = result["branchId"]

                if "needsMoreThoughts" in result:
                    self.state["needs_more_thoughts"] = result["needsMoreThoughts"]

            except json.JSONDecodeError:
                logger.warning("Failed to parse sequential thinking tool result as JSON")
            except Exception as e:
                logger.warning(f"Error updating sequential thinking state: {str(e)}")
        else:
            # No tool result found - increment thought number as fallback
            self.state["thought_number"] += 1

            # If we've reached max_thoughts, set next_thought_needed to False
            if (
                self.state["thought_number"] >= self.state["total_thoughts"]
                or self.state["thought_number"] >= self.max_thoughts
            ):
                self.state["next_thought_needed"] = False

    async def interactive(self, default_prompt: str = "") -> str:
        """
        Start an interactive prompt session with this sequential thinking agent.

        This overrides the base interactive method to provide a specialized
        experience for sequential thinking.

        Args:
            default_prompt: Default message to use when user presses enter

        Returns:
            The result of the interactive session
        """
        # Reset state before starting interactive mode
        self.state = self._reset_state()

        return await super().prompt(default_prompt=default_prompt)
