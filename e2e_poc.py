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

task_prompt = r"""
Drive-LLM Release Test Suite:
1 test_vlm_benchmark.py
2 test_vlm_accuracy.py
3 test_vlm_chat.py

Host for Deployment:
Host: 10.191.151.221
GPU: thor

Available Agents:
deploy agent
    capabilities:
        - clone files from GitLab repository to a specific IP address
        - setup environment for test
test agent
    capabilities:
        - execute python script
report agent
    capabilities:
        - read the log and report the results

based on above information, convert the input test plan into a series of instruction which can be handled by agents.
requirements:
- each instruction should be from the perspective of agent to clearly tell agent the action to take
- each instruction should be straightforward and concise
- each instruction strictly follows "subject action object" format, explicity specify the subject, action and object, no ambiguity, no extra info.
- each instruction should be atomic and independent, agent should be able to handle one instruction at a time
- one instruction should be one action, one object
- the action should be based on the agent's capabilities

"""


@fast.agent(
    name="task",
    instruction=task_prompt,
)
@fast.agent(
    name="deploy_agent",
    request_params=_request_params,
    instruction="""You are an agent, with a tool enabling you to clone or pull files from GitLab.""",
    servers=["nv_gitlab", "fileio"],
)
@fast.agent(
    name="execution_agent",
    request_params=_request_params,
    instruction="""You are a QA engineer, you need to execute python script""",
    servers=["python_run"],
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
@fast.chain(
    name="entry",
    sequence=["task", "route"],
)
async def main() -> None:
    async with fast.run() as agent:
        # for stage in TEST_STAGES:
        #     await agent.route(stage)
        _prompt = task_prompt + "\n\nrun drive-llm release test"
        _tasks = await agent.task._send_internal(_prompt)
        _tasks = _tasks.split("\n")[2:]
        for _instruction in _tasks:
            await agent.route(_instruction)
            # await agent.route(TEST_STAGES[0])
        await agent.interactive(agent="entry")
    """
    1. Deploy the Drive-LLM test suite using the deployment agent.
2. Run the test suite on thor GPU hosted at 10.191.151.221 using test agent.
3. Report the results and create reports for quality assurance using report agent.
4. Verify if all quality assurance issues are covered in the report or additional action is needed.
5. Document and save changes made as part of the Drive-LLM test suite.
    """


if __name__ == "__main__":
    asyncio.run(main())
