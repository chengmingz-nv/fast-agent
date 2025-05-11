from typing import Dict, List

from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
)
from openai import AuthenticationError, OpenAI
from openai.types.chat import ChatCompletion

# from openai.types.beta.chat import
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
)
from pydantic_core import from_json
from rich.text import Text

from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.core.prompt import Prompt
from mcp_agent.llm.augmented_llm import (
    AugmentedLLM,
    RequestParams,
)
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.multipart_converter_nim import NIMConverter, NIMMessage
from mcp_agent.llm.providers.multipart_converter_openai import OpenAIConverter, OpenAIMessage
from mcp_agent.llm.providers.sampling_converter_nim import NIMSamplingConverter
from mcp_agent.llm.providers.sampling_converter_openai import (
    OpenAISamplingConverter,
)
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
import json

_logger = get_logger(__name__)

DEFAULT_OPENAI_MODEL = r"meta/llama-3.3-70b-instruct"
DEFAULT_REASONING_EFFORT = "medium"


class NIMAugmentedLLM(AugmentedLLM[ChatCompletionMessageParam, ChatCompletionMessage]):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    This implementation uses OpenAI's ChatCompletion as the LLM.
    """

    # OpenAI-specific parameter exclusions
    NIM_EXCLUDE_FIELDS = {
        AugmentedLLM.PARAM_MESSAGES,
        AugmentedLLM.PARAM_MODEL,
        AugmentedLLM.PARAM_MAX_TOKENS,
        AugmentedLLM.PARAM_SYSTEM_PROMPT,
        AugmentedLLM.PARAM_PARALLEL_TOOL_CALLS,
        AugmentedLLM.PARAM_USE_HISTORY,
        AugmentedLLM.PARAM_MAX_ITERATIONS,
        AugmentedLLM.PARAM_RESPONSE_FORMAT,
    }

    def __init__(self, provider: Provider = Provider.NIM, *args, **kwargs) -> None:
        # Set type_converter before calling super().__init__
        if "type_converter" not in kwargs:
            kwargs["type_converter"] = NIMSamplingConverter

        super().__init__(*args, provider=provider, **kwargs)

        # Initialize logger with name if available
        self.logger = get_logger(f"{__name__}.{self.name}" if self.name else __name__)

        # Set up reasoning-related attributes
        self._reasoning_effort = kwargs.get("reasoning_effort", None)
        if self.context and self.context.config and self.context.config.nim:
            if self._reasoning_effort is None and hasattr(
                self.context.config.nim, "reasoning_effort"
            ):
                self._reasoning_effort = self.context.config.nim.reasoning_effort

        # Determine if we're using a reasoning model
        # TODO -- move this to model capabiltities, add o4.
        chosen_model = self.default_request_params.model if self.default_request_params else None
        self._reasoning = chosen_model and (
            chosen_model.startswith("o3") or chosen_model.startswith("o1")
        )
        if self._reasoning:
            self.logger.info(
                f"Using reasoning model '{chosen_model}' with '{self._reasoning_effort}' reasoning effort"
            )

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize OpenAI-specific default parameters"""
        chosen_model = kwargs.get("model", DEFAULT_OPENAI_MODEL)

        return RequestParams(
            model=chosen_model,
            systemPrompt=self.instruction,
            parallel_tool_calls=True,
            max_iterations=10,
            use_history=True,
        )

    def _parse_response(self, response: ChatCompletion, stream: bool) -> ChatCompletionMessage:
        if stream:
            # Initialize accumulation variables
            content = ""
            tool_calls = []
            role = "assistant"

            # Accumulate chunks
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content is not None:
                    content += delta.content
                    print(delta.content, end="")
                # Process tool calls if present in delta
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        # Handle accumulation of tool calls
                        self._accumulate_tool_call(tool_calls, tool_call)

            # Construct message from accumulated content
            return ChatCompletionMessage(
                role=role, content=content, tool_calls=tool_calls if tool_calls else None
            )
        else:
            if not response or isinstance(response, str):
                return ChatCompletionMessage(role="assistant", content="")
            elif not response.choices or len(response.choices) == 0:
                # No response from the model, we're done
                return ChatCompletionMessage(role="assistant", content="")
            return response.choices[0].message

    def _accumulate_tool_call(self, tool_calls, delta_tool_call):
        # If this is a new tool call
        if delta_tool_call.index >= len(tool_calls):
            tool_calls.append(
                {
                    "id": delta_tool_call.id or "",
                    "function": {"name": "", "arguments": ""},
                    "type": "function",
                }
            )

        # Update tool call with new delta information
        if delta_tool_call.id:
            tool_calls[delta_tool_call.index]["id"] = delta_tool_call.id

        if delta_tool_call.function:
            if delta_tool_call.function.name:
                tool_calls[delta_tool_call.index]["function"]["name"] = (
                    delta_tool_call.function.name
                )
            if delta_tool_call.function.arguments:
                tool_calls[delta_tool_call.index]["function"]["arguments"] += (
                    delta_tool_call.function.arguments
                )

    def _base_url(self) -> str:
        return self.context.config.nim.base_url if self.context.config.nim else None

    def _openai_client(self) -> OpenAI:
        try:
            _api_key = self._api_key()
            _base_url = self._base_url()
            return OpenAI(api_key=_api_key, base_url=_base_url)
        except AuthenticationError as e:
            raise ProviderKeyError(
                "Invalid OpenAI API key",
                "The configured OpenAI API key was rejected.\n"
                "Please check that your API key is valid and not expired.",
            ) from e

    async def _openai_completion(
        self,
        message: OpenAIMessage,
        request_params: RequestParams | None = None,
    ) -> List[TextContent | ImageContent | EmbeddedResource]:
        """
        Process a query using an LLM and available tools.
        The default implementation uses OpenAI's ChatCompletion as the LLM.
        Override this method to use a different LLM.
        """

        request_params = self.get_request_params(request_params=request_params)

        responses: List[TextContent | ImageContent | EmbeddedResource] = []

        # TODO -- move this in to agent context management / agent group handling
        messages: List[ChatCompletionMessageParam] = []
        system_prompt = self.instruction or request_params.systemPrompt
        if system_prompt:
            messages.append(ChatCompletionSystemMessageParam(role="system", content=system_prompt))

        messages.extend(self.history.get(include_completion_history=request_params.use_history))
        messages.append(message)

        response = await self.aggregator.list_tools()
        available_tools: List[ChatCompletionToolParam] | None = [
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": tool.name,
                    "description": tool.description if tool.description else "",
                    "parameters": tool.inputSchema,
                },
            )
            for tool in response.tools
        ]

        if not available_tools:
            available_tools = None  # deepseek does not allow empty array

        # we do NOT send "stop sequences" as this causes errors with mutlimodal processing
        for i in range(request_params.max_iterations):
            arguments = self._prepare_api_request(messages, available_tools, request_params)
            _stream = request_params.stream
            self.logger.debug(f"OpenAI completion requested for: {arguments}")

            self._log_chat_progress(self.chat_turn(), model=self.default_request_params.model)

            executor_result = await self.executor.execute(
                self._openai_client().chat.completions.create, **arguments
            )

            response = executor_result[0]

            self.logger.debug(
                "NIM OpenAI completion response:",
                data=response,
            )

            if isinstance(response, AuthenticationError):
                raise ProviderKeyError(
                    "Rejected OpenAI API key",
                    "The configured OpenAI API key was rejected.\n"
                    "Please check that your API key is valid and not expired.",
                ) from response
            elif isinstance(response, BaseException):
                self.logger.error(f"Error: {response}")
                break

            message = self._parse_response(response, stream=_stream)
            _finish_reason = response.choices[0].finish_reason
            # prep for image/audio gen models
            if message.content:
                responses.append(TextContent(type="text", text=message.content))

            converted_message = self.convert_message_to_message_param(message)
            messages.append(converted_message)

            message_text = converted_message.content
            if _finish_reason in ["tool_calls", "function_call"] and message.tool_calls:
                if message_text:
                    await self.show_assistant_message(
                        message_text,
                        message.tool_calls[
                            0
                        ].function.name,  # TODO support displaying multiple tool calls
                    )
                else:
                    await self.show_assistant_message(
                        Text(
                            "the assistant requested tool calls",
                            style="dim green italic",
                        ),
                        message.tool_calls[0].function.name,
                    )

                tool_results = []
                for tool_call in message.tool_calls:
                    self.show_tool_call(
                        available_tools,
                        tool_call.function.name,
                        tool_call.function.arguments,
                    )
                    tool_call_request = CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(
                            name=tool_call.function.name,
                            arguments={}
                            if not tool_call.function.arguments
                            or tool_call.function.arguments.strip() == ""
                            else from_json(tool_call.function.arguments, allow_partial=True),
                        ),
                    )
                    result = await self.call_tool(tool_call_request, tool_call.id)
                    self.show_oai_tool_result(str(result))

                    tool_results.append((tool_call.id, result))
                    responses.extend(result.content)
                messages.extend(OpenAIConverter.convert_function_results_to_openai(tool_results))

                self.logger.debug(
                    f"Iteration {i}: Tool call results: {str(tool_results) if tool_results else 'None'}"
                )
            elif _finish_reason == "length":
                # We have reached the max tokens limit
                self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'length'")
                if request_params and request_params.maxTokens is not None:
                    message_text = Text(
                        f"the assistant has reached the maximum token limit ({request_params.maxTokens})",
                        style="dim green italic",
                    )
                else:
                    message_text = Text(
                        "the assistant has reached the maximum token limit",
                        style="dim green italic",
                    )

                await self.show_assistant_message(message_text)
                break
            elif _finish_reason == "content_filter":
                # The response was filtered by the content filter
                self.logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'content_filter'"
                )
                break
            elif _finish_reason == "stop":
                self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'stop'")
                if message_text:
                    await self.show_assistant_message(message_text, "")
                break

        if request_params.use_history:
            # Get current prompt messages
            prompt_messages = self.history.get(include_completion_history=False)

            # Calculate new conversation messages (excluding prompts)
            new_messages = messages[len(prompt_messages) :]

            # Update conversation history
            self.history.set(new_messages)

        self._log_chat_finished(model=self.default_request_params.model)

        return responses

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List["PromptMessageMultipart"],
        request_params: RequestParams | None = None,
        is_template: bool = False,
    ) -> PromptMessageMultipart:
        last_message = multipart_messages[-1]

        # Add all previous messages to history (or all messages if last is from assistant)
        # if the last message is a "user" inference is required
        messages_to_add = (
            multipart_messages[:-1] if last_message.role == "user" else multipart_messages
        )
        converted = []
        for msg in messages_to_add:
            _ret = NIMConverter.convert_to_nim(msg)
            converted.append(_ret)

        # TODO -- this looks like a defect from previous apply_prompt implementation.
        self.history.extend(converted, is_prompt=is_template)

        if "assistant" == last_message.role:
            return last_message

        # For assistant messages: Return the last message (no completion needed)
        message_param: NIMMessage = NIMConverter.convert_to_nim(last_message)
        responses: List[
            TextContent | ImageContent | EmbeddedResource
        ] = await self._openai_completion(
            message_param,
            request_params,
        )
        return Prompt.assistant(*responses)

    async def pre_tool_call(self, tool_call_id: str | None, request: CallToolRequest):
        return request

    async def post_tool_call(
        self, tool_call_id: str | None, request: CallToolRequest, result: CallToolResult
    ):
        return result

    def prepare_provider_arguments(
        self,
        base_args: dict,
        request_params: RequestParams,
        exclude_fields: set | None = None,
    ) -> dict:
        """
        Prepare arguments for provider API calls by merging request parameters.

        Args:
            base_args: Base arguments dictionary with provider-specific required parameters
            params: The RequestParams object containing all parameters
            exclude_fields: Set of field names to exclude from params. If None, uses BASE_EXCLUDE_FIELDS.

        Returns:
            Complete arguments dictionary with all applicable parameters
        """
        # Start with base arguments
        arguments = base_args.copy()

        # Use provided exclude_fields or fall back to base exclusions
        exclude_fields = exclude_fields or self.BASE_EXCLUDE_FIELDS.copy()

        # Add all fields from params that aren't explicitly excluded
        params_dict = request_params.model_dump(exclude=exclude_fields)
        for key, value in params_dict.items():
            if value is not None and key not in arguments:
                arguments[key] = value

        # Finally, add any metadata fields as a last layer of overrides
        if request_params.metadata:
            arguments.update(request_params.metadata)
        if request_params.response_format:
            for _idx in range(len(arguments["messages"])):
                if arguments["messages"][_idx]["role"] == "system":
                    arguments["messages"][_idx]["content"] = (
                        arguments["messages"][_idx]["content"]
                        + f"\n\nresponse format must be aligned with the following:\n{json.dumps(request_params.response_format)}"
                    )
            arguments["response_format"] = {"type": "json_object"}
        return arguments

    def _prepare_api_request(
        self, messages, tools, request_params: RequestParams
    ) -> dict[str, str]:
        # Create base arguments dictionary

        # overriding model via request params not supported (intentional)
        base_args = {
            "model": self.default_request_params.model,
            "messages": messages,
        }

        if tools:
            base_args["tools"] = tools
        else:
            self.logger.warning("No tools provided, skipping tool calls")

        if self._reasoning:
            base_args.update(
                {
                    "max_completion_tokens": request_params.maxTokens,
                    "reasoning_effort": self._reasoning_effort,
                }
            )
        else:
            base_args["max_tokens"] = request_params.maxTokens

        if tools:
            base_args["parallel_tool_calls"] = request_params.parallel_tool_calls

        arguments: Dict[str, str] = self.prepare_provider_arguments(
            base_args, request_params, self.NIM_EXCLUDE_FIELDS.union(self.BASE_EXCLUDE_FIELDS)
        )

        return arguments
