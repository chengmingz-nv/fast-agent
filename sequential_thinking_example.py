import asyncio
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams

# Create the application
fast = FastAgent("sequential thinking demo")

_request_params = RequestParams(temperature=0.7, top_p=0.6, stream=False, use_history=True)
_servers = ["sequential_thinking"]


@fast.agent(
    instruction="""You are an AI assistant that uses sequential thinking to solve problems step by step.""",
    servers=_servers,
    request_params=_request_params,
)
async def main():
    # Create a simple sequential thinking state tracker
    state = {
        "thought_number": 1,
        "total_thoughts": 5,  # Initial estimate
        "next_thought_needed": True,
    }

    problem = (
        "How might we implement a distributed system for real-time collaborative document editing?"
    )

    async with fast.run() as agent:
        print(f"Problem: {problem}\n")

        # Send the initial prompt with the problem
        response = await agent.send(
            f"""I need to solve this problem using sequential thinking: {problem}
            
            Let me break this down step by step using the sequential_thinking tool.
            I'll set thoughtNumber={state["thought_number"]}, totalThoughts={state["total_thoughts"]}, and nextThoughtNeeded={str(state["next_thought_needed"]).lower()}.
            """
        )

        print(f"Step {state['thought_number']}:\n{response}\n")

        # Continue the sequential thinking process until complete
        while state["next_thought_needed"]:
            # Update state for next thought
            state["thought_number"] += 1

            # For demonstration, we'll say we're done after 5 thoughts
            if state["thought_number"] >= state["total_thoughts"]:
                state["next_thought_needed"] = False

            # Send the next prompt with updated state information
            response = await agent.send(
                f"""Continue the sequential thinking process.
                
                For this next thought, use these parameters:
                - thoughtNumber: {state["thought_number"]}
                - totalThoughts: {state["total_thoughts"]}
                - nextThoughtNeeded: {str(state["next_thought_needed"]).lower()}
                """
            )

            print(f"Step {state['thought_number']}:\n{response}\n")

        # Final summary
        conclusion = await agent.send(
            "Now that I've completed the sequential thinking process, please provide a final summary of the solution."
        )
        print(f"Conclusion:\n{conclusion}")


if __name__ == "__main__":
    asyncio.run(main())
