import asyncio
import json
import re
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams

# Create the application
fast = FastAgent("advanced sequential thinking demo")

_request_params = RequestParams(temperature=0.7, top_p=0.6, stream=False, use_history=True)
_servers = ["sequential_thinking"]


@fast.agent(
    instruction="""You are an AI assistant that uses sequential thinking to solve complex problems.
    
When using the sequential_thinking tool:
1. Always show your reasoning process clearly
2. Ensure each thought builds on previous ones
3. For complex problems, revise earlier thoughts if needed
4. State your parameters clearly in each response with this exact format:
   [TOOL_PARAMS] {"thoughtNumber": N, "totalThoughts": M, "nextThoughtNeeded": true/false} [/TOOL_PARAMS]
   
The tool itself will keep track of your thought history.""",
    servers=_servers,
    request_params=_request_params,
)
async def main():
    # Initialize state with default values
    state = {
        "thought_number": 1,
        "total_thoughts": 5,  # Initial estimate
        "next_thought_needed": True,
    }

    problem = "Design a system for autonomous drone delivery in urban environments, considering safety, regulations, technical challenges, and customer experience."

    async with fast.run() as agent:
        print(f"Problem: {problem}\n")

        # First thought
        response = await agent.send(
            f"""I need to solve this complex problem using sequential thinking: {problem}
            
            Please use the sequential_thinking tool with these parameters:
            - thoughtNumber: 1 (first thought)
            - Estimate an appropriate totalThoughts
            - nextThoughtNeeded: true
            
            Remember to include the parameters in your response with the exact format:
            [TOOL_PARAMS] {{...}} [/TOOL_PARAMS]
            """
        )

        print(f"Step {state['thought_number']}:\n{response}\n")

        # Extract state from response
        updated_state = extract_tool_params(response)
        if updated_state:
            state.update(updated_state)

        # Continue the sequential thinking process until complete
        while state["next_thought_needed"]:
            # Send the next prompt with updated state information
            response = await agent.send(
                f"""Continue the sequential thinking process about the drone delivery system.
                
                Please use the sequential_thinking tool with these parameters:
                - thoughtNumber: {state["thought_number"] + 1}
                - totalThoughts: {state["total_thoughts"]}
                - nextThoughtNeeded: Should be determined by you based on progress
                
                Remember to include the parameters in your response with the exact format:
                [TOOL_PARAMS] {{...}} [/TOOL_PARAMS]
                """
            )

            # Extract updated state from response
            updated_state = extract_tool_params(response)
            if updated_state:
                state.update(updated_state)
                print(f"Step {state['thought_number']}:\n{response}\n")
                print(f"Updated state: {state}\n")
            else:
                print("Warning: Could not extract tool parameters from response.")
                # Default to incrementing thought number and continuing
                state["thought_number"] += 1
                if state["thought_number"] >= state["total_thoughts"]:
                    state["next_thought_needed"] = False

        # Final summary after sequential thinking is complete
        conclusion = await agent.send(
            """Now that I've completed the sequential thinking process about the drone delivery system, 
            please provide a final comprehensive summary of the solution, key insights, and next steps."""
        )
        print(f"Conclusion:\n{conclusion}")


def extract_tool_params(response):
    """Extract tool parameters from the response using regex."""
    param_match = re.search(r"\[TOOL_PARAMS\]\s*(\{.*?\})\s*\[/TOOL_PARAMS\]", response, re.DOTALL)
    if param_match:
        try:
            params = json.loads(param_match.group(1))
            return {
                "thought_number": params.get("thoughtNumber", 1),
                "total_thoughts": params.get("totalThoughts", 5),
                "next_thought_needed": params.get("nextThoughtNeeded", False),
            }
        except json.JSONDecodeError:
            print("Warning: Invalid JSON in tool parameters.")
    return None


if __name__ == "__main__":
    asyncio.run(main())
