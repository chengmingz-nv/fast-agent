"""
Example MCP Agent application showing router workflow with decorator syntax.
Demonstrates router's ability to either:
1. Use tools directly to handle requests
2. Delegate requests to specialized agents
"""

import asyncio
import sys

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams

# Create the application
fast = FastAgent(
    "DL SWQA E2E POC Demo",
)

# Sample requests demonstrating direct tool use vs agent delegation
TEST_STAGES = [
    "clone chengmingz/dummy_tests",  # Router handles directly with fetch
    "run the python script under dummy_tests",  # Delegated to code expert
    "read the log under dummy_tests and report",  # Delegated to general assistant
]
_request_params = RequestParams(temperature=0.7, top_p=0.6, stream=False, use_history=False)


@fast.agent(
    name="deploy_agent",
    request_params=_request_params,
    instruction="""You are an agent, with a tool enabling you to clone or pull files from GitLab.""",
    servers=["nv_gitlab"],
)
@fast.agent(
    name="execution_agent",
    request_params=_request_params,
    instruction="""You are a QA engineer, you need to execute python script""",
    servers=["filesystem", "python_run"],
)
@fast.agent(
    name="reporting_agent",
    request_params=_request_params,
    instruction="""You are a QA engineer, read the log and report the results.""",
    servers=["filesystem"],
)
@fast.router(
    name="route",
    request_params=_request_params,
    agents=["deploy_agent", "execution_agent", "reporting_agent"],
)
async def main() -> None:
    async with fast.run() as agent:
        for stage in TEST_STAGES:
            await agent.route(stage)


if __name__ == "__main__":
    asyncio.run(main())
