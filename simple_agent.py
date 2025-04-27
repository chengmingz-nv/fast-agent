import asyncio
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams

# Create the application
fast = FastAgent("bamboo-ai poc demo")

_request_params = RequestParams(temperature=0.7, top_p=0.6, stream=False, use_history=False)
_servers = ["filesystem"]


# Define the agent
@fast.agent(
    instruction="You are a helpful AI Agent", servers=_servers, request_params=_request_params
)
async def main():
    # use the --model command line switch or agent arguments to change model
    for i in range(10):
        async with fast.run() as agent:
            await agent.interactive()


if __name__ == "__main__":
    asyncio.run(main())
