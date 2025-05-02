import asyncio
import json
import re
from typing import Dict, Any, Optional, List, Union

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.core.prompt import Prompt


class SequentialThinkingAgent:
    """
    A wrapper around FastAgent that manages the state for sequential thinking.
    This ensures proper continuity between sequential thinking steps.
    """

    def __init__(
        self,
        name: str = "Sequential Thinking Agent",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_thoughts: int = 10,
    ):
        """
        Initialize the Sequential Thinking Agent.

        Args:
            name: Name of the agent
            model: LLM model to use (defaults to FastAgent default)
            temperature: Temperature for LLM generation
            max_thoughts: Maximum number of thoughts to generate
        """
        # Configure request parameters
        self.request_params = RequestParams(temperature=temperature, stream=False, use_history=True)

        # Set up the FastAgent with sequential thinking
        self.fast = FastAgent(name)
        self.model = model
        self.max_thoughts = max_thoughts

        # Initialize state
        self.reset_state()

        # Define the agent with appropriate instructions
        @self.fast.agent(
            instruction="""You are an AI assistant that uses sequential thinking to solve complex problems step by step.
            
When using the sequential_thinking tool, follow these guidelines:
1. Break complex problems into manageable thoughts
2. Ensure each thought builds logically on previous ones
3. Track your progress with appropriate parameters
4. Revise earlier thoughts if needed
5. Provide clear conclusions when the sequence is complete

Your thinking process should be well-structured and show clear reasoning.""",
            servers=["sequential_thinking"],
            request_params=self.request_params,
            model=self.model,
        )
        async def sequential_thinking_agent():
            """Define the agent function for FastAgent."""
            pass

        # Store the agent function for later use
        self.agent_func = sequential_thinking_agent

    def reset_state(self):
        """Reset the sequential thinking state."""
        self.state = {
            "thought_number": 1,
            "total_thoughts": 5,  # Initial estimate
            "next_thought_needed": True,
            "branches": [],
            "branch_from_thought": None,
            "branch_id": None,
            "is_revision": False,
            "revises_thought": None,
            "needs_more_thoughts": False,
        }

    async def think(self, problem: str) -> List[str]:
        """
        Solve a problem using sequential thinking.

        Args:
            problem: The problem to solve

        Returns:
            A list of thought responses
        """
        # Reset state for new problem
        self.reset_state()

        # Store thought responses
        thoughts = []

        async with self.fast.run() as agent:
            # Initial prompt
            response = await self._send_thought_prompt(
                agent,
                f"""I need to solve this problem using sequential thinking: {problem}
                
                I'll begin by breaking down this problem and analyzing it step by step.""",
            )
            thoughts.append(response)

            # Continue thinking until complete or max thoughts reached
            while (
                self.state["next_thought_needed"]
                and self.state["thought_number"] < self.max_thoughts
            ):
                # Generate the next thought
                response = await self._send_thought_prompt(
                    agent,
                    """Continue the sequential thinking process based on previous thoughts.""",
                )
                thoughts.append(response)

            # If we reached max thoughts but still need more
            if self.state["next_thought_needed"]:
                # Final wrap-up thought
                response = await agent.send(
                    """We've reached the maximum number of thoughts, so please provide a final 
                    conclusion based on the thinking so far, even if it's not complete."""
                )
                thoughts.append(response)

        return thoughts

    async def interactive(self):
        """Start an interactive sequential thinking session."""
        try:
            self.reset_state()
            async with self.fast.run() as agent:
                # Welcome message
                print("\n=== Sequential Thinking Agent ===")
                print("Type your problem to start thinking sequentially.")
                print("Type 'exit' or 'quit' to end the session.\n")

                # Get the problem
                problem = input("Problem: ")
                if problem.lower() in ["exit", "quit"]:
                    return

                # Initial thought
                response = await self._send_thought_prompt(
                    agent,
                    f"""I need to solve this problem using sequential thinking: {problem}
                    
                    I'll begin by breaking down this problem and analyzing it step by step.""",
                )
                print(f"\nThought {self.state['thought_number']}:\n{response}\n")

                # Continue thinking until complete
                while self.state["next_thought_needed"]:
                    # Check if user wants to continue
                    user_input = input("Press Enter to continue, or type a new direction: ")

                    if user_input.lower() in ["exit", "quit"]:
                        break

                    # Use user input if provided, otherwise generic continuation
                    prompt = (
                        f"""Continue the sequential thinking process, considering: {user_input}"""
                        if user_input.strip()
                        else """Continue the sequential thinking process based on previous thoughts."""
                    )

                    # Generate the next thought
                    response = await self._send_thought_prompt(agent, prompt)
                    print(f"\nThought {self.state['thought_number']}:\n{response}\n")

                # Final summary
                if not self.state["next_thought_needed"]:
                    print("\n=== Final Summary ===")
                    conclusion = await agent.send(
                        """Now that I've completed the sequential thinking process, 
                        please provide a final comprehensive summary of the solution."""
                    )
                    print(f"{conclusion}\n")

        except KeyboardInterrupt:
            print("\nSequential thinking session ended by user.")
        except Exception as e:
            print(f"\nError in sequential thinking session: {str(e)}")

    async def _send_thought_prompt(self, agent, prompt: str) -> str:
        """
        Send a thought prompt and update state from the response.

        Args:
            agent: The agent to use
            prompt: The prompt to send

        Returns:
            The agent's response
        """
        # Create the full prompt with state parameters
        full_prompt = self._create_sequential_prompt(prompt)

        # Send the prompt
        response = await agent.send(full_prompt)

        # Extract and update state
        self._update_state_from_response(response)

        return response

    def _create_sequential_prompt(self, prompt: str) -> str:
        """
        Create a prompt with sequential thinking parameters.

        Args:
            prompt: The base prompt

        Returns:
            The prompt with sequential thinking parameters
        """
        # Format the state parameters for the prompt
        params_str = ", ".join(
            [
                f"thoughtNumber: {self.state['thought_number']}",
                f"totalThoughts: {self.state['total_thoughts']}",
                f"nextThoughtNeeded: {str(self.state['next_thought_needed']).lower()}",
            ]
        )

        # Add optional parameters if they're being used
        if self.state["is_revision"]:
            params_str += f", isRevision: true, revisesThought: {self.state['revises_thought']}"

        if self.state["branch_from_thought"] is not None:
            params_str += f", branchFromThought: {self.state['branch_from_thought']}"
            if self.state["branch_id"]:
                params_str += f', branchId: "{self.state["branch_id"]}"'

        if self.state["needs_more_thoughts"]:
            params_str += ", needsMoreThoughts: true"

        # Construct the full prompt
        return f"""{prompt}

Use the sequential_thinking tool with these parameters:
{params_str}

Think through the problem step by step, showing your reasoning clearly.
"""

    def _update_state_from_response(self, response: str) -> None:
        """
        Update the state based on the response from the sequential thinking tool.

        Args:
            response: The response from the agent
        """
        # Look for tool call results in the response
        match = re.search(r"Tool: sequentialthinking, Result: (\{.*?\})", response, re.DOTALL)
        if not match:
            # No tool result found, increment thought number as fallback
            self.state["thought_number"] += 1
            return

        try:
            # Parse the tool result
            result = json.loads(match.group(1))

            # Update the state with the tool result
            self.state["thought_number"] = result.get(
                "thoughtNumber", self.state["thought_number"] + 1
            )
            self.state["total_thoughts"] = result.get("totalThoughts", self.state["total_thoughts"])
            self.state["next_thought_needed"] = result.get("nextThoughtNeeded", False)

            # Update optional parameters if present
            if "branches" in result:
                self.state["branches"] = result["branches"]

            if "thoughtHistoryLength" in result:
                # Sanity check - thought number should match history length
                if result["thoughtHistoryLength"] != self.state["thought_number"]:
                    print(
                        f"Warning: Thought number mismatch. Expected {result['thoughtHistoryLength']}, got {self.state['thought_number']}"
                    )

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing sequential thinking result: {str(e)}")
            # Increment thought number as fallback
            self.state["thought_number"] += 1


async def main():
    # Create the sequential thinking agent
    agent = SequentialThinkingAgent(name="Sequential Thinker", temperature=0.7, max_thoughts=7)

    # Run in interactive mode
    await agent.interactive()

    # Alternatively, solve a specific problem:
    # thoughts = await agent.think(
    #     "How might we design a sustainable urban transportation system for the future?"
    # )
    # for i, thought in enumerate(thoughts, 1):
    #     print(f"\nThought {i}:\n{thought}\n")


if __name__ == "__main__":
    asyncio.run(main())
