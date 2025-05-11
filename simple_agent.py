import asyncio
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams
from mcp_agent.core.agent_types import AgentConfig
from mcp_agent.agents.sequential_thinking_agent import SequentialThinkingAgent

# Create the application
fast = FastAgent("sequential thinking demo")

# Configure parameters
_request_params = RequestParams(temperature=0.7, top_p=0.6, stream=False, use_history=True)
_servers = ["sequential_thinking"]

# Sequential thinking instruction
SEQUENTIAL_THINKING_INSTRUCTION = """You are a helpful AI Agent that uses sequential thinking to solve problems.

When thinking through a problem, follow these guidelines:
1. Use the sequential thinking tool for step-by-step reasoning
2. Break complex problems into manageable thoughts
3. Ensure each thought builds logically on previous ones
4. Provide clear conclusions when the sequence is complete
5. Your thinking process should be well-structured and show clear reasoning

Remember that sequential thinking is particularly effective for complex problems that require
careful analysis and consideration of multiple factors.
"""


# Define the agent with FastAgent decorator for compatibility
@fast.agent(
    name="default_agent",
    instruction=SEQUENTIAL_THINKING_INSTRUCTION,
    servers=_servers,
    request_params=_request_params,
)
async def main():
    """Run the sequential thinking agent interactively."""

    # Create configuration for specialized sequential thinking agent
    config = AgentConfig(
        name="sequential_thinker",
        instruction=SEQUENTIAL_THINKING_INSTRUCTION,
        servers=_servers,
        use_history=True,
    )

    async with fast.run() as agent_app:
        # Get the default agent by its name using AgentApp's dictionary access
        default_agent = agent_app["default_agent"]

        # Create sequential thinking agent
        sequential_agent = SequentialThinkingAgent(config=config, max_thoughts=10)

        # Get the LLM factory - only if the default agent's LLM is initialized
        if default_agent._llm is not None:
            # Attach the LLM
            await sequential_agent.attach_llm(
                # Use the same LLM class
                llm_factory=default_agent._llm.__class__,
                # Get the model safely using getattr
                model=getattr(default_agent._llm, "request_params", {}).get("model"),
                # Use our standard request params
                request_params=_request_params,
            )

        # Initialize connections
        await sequential_agent.initialize()

        try:
            print("\n=== Sequential Thinking Agent ===")
            print("This agent maintains its reasoning state between interactions.")
            print("Type 'exit' or 'quit' to end the session.\n")

            # Start interactive mode
            await sequential_agent.interactive()

        finally:
            # Clean up resources
            await sequential_agent.shutdown()


if __name__ == "__main__":
    # Fix asyncio.run issue by ensuring main returns a coroutine
    asyncio.run(main())
