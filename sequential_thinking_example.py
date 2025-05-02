import asyncio
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams
from mcp_agent.core.agent_types import AgentConfig
from mcp_agent.agents.sequential_thinking_agent import SequentialThinkingAgent

# Create the application
fast = FastAgent("sequential thinking demo")

# Configure request parameters
_request_params = RequestParams(temperature=0.7, top_p=0.6, stream=False, use_history=True)
_servers = ["sequential_thinking"]

# Define the basic instruction for the sequential thinking agent
SEQUENTIAL_THINKING_INSTRUCTION = """You are an AI assistant that uses sequential thinking to solve problems step by step.

When thinking through a problem:
1. Break it down into manageable steps
2. Consider different perspectives
3. Use your reasoning process to build toward a solution
4. Provide clear explanations of your thinking at each step
5. Draw a conclusion once you've worked through the problem

When using the sequential_thinking tool, follow the parameters provided in the message.
"""


# Define the agent function using FastAgent decorator
@fast.agent(
    name="default_agent",  # Explicitly name the agent for easier access
    instruction=SEQUENTIAL_THINKING_INSTRUCTION,
    servers=_servers,
    request_params=_request_params,
)
async def main():
    """Set up the sequential thinking demo."""

    # Create problem to solve
    problem = (
        "How might we implement a distributed system for real-time collaborative document editing?"
    )

    async with fast.run() as agent_app:
        # Get the default agent using the AgentApp's access methods
        default_agent = agent_app["default_agent"]

        # Create a configuration for our specialized sequential thinking agent
        config = AgentConfig(
            name="sequential_thinker",
            instruction=SEQUENTIAL_THINKING_INSTRUCTION,
            servers=_servers,
            use_history=True,
        )

        # Create our specialized sequential thinking agent using the same LLM
        sequential_agent = SequentialThinkingAgent(config=config, max_thoughts=6)

        # Attach the same LLM to our agent
        await sequential_agent.attach_llm(
            llm_factory=default_agent._llm.__class__,
            model=getattr(default_agent._llm, "request_params", {}).get("model"),
            request_params=_request_params,
        )

        # Initialize the connections
        await sequential_agent.initialize()

        try:
            print(f"\nProblem: {problem}\n")

            # Send the initial prompt with the problem
            response = await sequential_agent.send(
                f"I need to solve this problem using sequential thinking: {problem}"
            )
            print(f"Step {sequential_agent.state['thought_number']}:\n{response}\n")

            # Continue the sequential thinking process until complete
            while sequential_agent.state["next_thought_needed"]:
                # User prompt to continue
                input("Press Enter to continue to the next thought...")

                # Send the next prompt
                response = await sequential_agent.send(
                    "Continue the sequential thinking process based on your previous thoughts."
                )
                print(f"\nStep {sequential_agent.state['thought_number']}:\n{response}\n")

            # Final summary
            print("\n=== Final Summary ===")
            conclusion = await sequential_agent.send(
                "Now that you've completed the sequential thinking process, please provide a final summary of the solution."
            )
            print(f"{conclusion}\n")

        finally:
            # Clean up resources
            await sequential_agent.shutdown()


if __name__ == "__main__":
    # Fix asyncio.run issue by ensuring main returns a coroutine
    asyncio.run(main())
