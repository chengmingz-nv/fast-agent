import asyncio
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams

# Create the application
fast = FastAgent("bamboo-ai poc demo")

_request_params = RequestParams(temperature=0.7, top_p=0.6, stream=False, use_history=True)
_servers = ["sequential_thinking"]


# Define the agent
@fast.agent(
    instruction="""You are a helpful AI Agent that uses sequential thinking to solve problems.

When thinking through a problem, follow these guidelines:
1. Use the sequential thinking tool for step-by-step reasoning
2. ALWAYS maintain state between interactions by tracking:
   - thoughtNumber: The current thought number
   - totalThoughts: Your estimate of how many thoughts you'll need
   - nextThoughtNeeded: Whether you need another thought (true/false)

3. For your first thought:
   - Set thoughtNumber=1
   - Estimate totalThoughts (typically 3-7)
   - Set nextThoughtNeeded=true

4. For subsequent thoughts:
   - Increment thoughtNumber
   - Update totalThoughts if needed
   - Set nextThoughtNeeded=true until your final thought

5. For your final thought:
   - Set nextThoughtNeeded=false
   - Provide a conclusion

Remember that you MUST use all these parameters correctly with each tool call or the sequential thinking process will break.
""",
    servers=_servers,
    request_params=_request_params,
)
async def main():
    # use the --model command line switch or agent arguments to change model
    for i in range(10):
        async with fast.run() as agent:
            await agent.interactive()


if __name__ == "__main__":
    asyncio.run(main())
