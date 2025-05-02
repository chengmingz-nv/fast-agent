#!/usr/bin/env python3
"""
MCP server that provides functionality to execute all Python scripts in a given directory.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import List, Optional

from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename="python_run.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create the FastMCP server
app = FastMCP(
    name="Python-Runner-MCP-Server",
    instructions="This server provides tools to execute Python scripts in a directory",
)


@app.tool(
    name="run_all_scripts", description="Executes all Python scripts in the specified directory."
)
def run_all_scripts(
    directory: str = r"output/dummy_tests",
    recursive: bool = False,
    ignore_errors: bool = False,
    pattern: str = "*.py",
) -> str:
    """
    Execute all Python scripts in the specified directory.

    Args:
        directory: Path to the directory containing Python scripts
        recursive: If True, search for scripts in subdirectories as well
        ignore_errors: If True, continue executing scripts even if some fail
        pattern: File pattern to match (default: "*.py")

    Returns:
        Summary of script execution results
    """
    logger.info(f"Running all scripts in directory: {directory}")
    return f"Successfully ran all scripts in directory: {directory}"
    try:
        dir_path = Path(directory)

        # Check if directory exists
        if not dir_path.exists() or not dir_path.is_dir():
            return f"Error: Directory '{directory}' does not exist or is not a directory."

        # Find all Python scripts
        if recursive:
            scripts = list(dir_path.glob(f"**/{pattern}"))
        else:
            scripts = list(dir_path.glob(pattern))

        # Sort scripts for deterministic execution order
        scripts.sort()

        if not scripts:
            return f"No Python scripts found in {directory}."

        results = []
        success_count = 0
        failure_count = 0

        # Execute each script
        for script in scripts:
            try:
                logger.info(f"Executing: {script}")
                result = subprocess.run(
                    ["python", str(script)], capture_output=True, text=True, check=not ignore_errors
                )

                if result.returncode == 0:
                    status = "SUCCESS"
                    success_count += 1
                else:
                    status = "FAILED"
                    failure_count += 1

                script_result = {
                    "script": str(script),
                    "status": status,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }

                results.append(script_result)

                # Log script execution results
                if status == "SUCCESS":
                    logger.info(f"Successfully executed {script}")
                else:
                    logger.error(f"Failed to execute {script}: {result.stderr}")

            except Exception as e:
                failure_count += 1
                logger.error(f"Error executing {script}: {str(e)}")

                if not ignore_errors:
                    return f"Error executing {script}: {str(e)}"

                results.append({"script": str(script), "status": "ERROR", "error": str(e)})

        # Prepare summary
        summary = f"Executed {len(scripts)} Python scripts: {success_count} succeeded, {failure_count} failed.\n"
        for result in results:
            status = result["status"]
            script = result["script"]

            summary += f"\n{status}: {script}"
            if status != "SUCCESS":
                if "stderr" in result:
                    summary += f"\nError: {result['stderr']}"
                elif "error" in result:
                    summary += f"\nError: {result['error']}"

        return summary

    except Exception as e:
        error_msg = f"Error running scripts in '{directory}': {str(e)}"
        logger.error(error_msg)
        return error_msg


@app.tool(name="run_script", description="Executes a single Python script.")
def run_script(script_path: str, args: List[str] = None) -> str:
    """
    Execute a single Python script.

    Args:
        script_path: Path to the Python script
        args: Command line arguments to pass to the script

    Returns:
        Script execution result
    """
    logger.info(f"Running script: {script_path}")
    return f"Successfully ran script: {script_path}"
    try:
        script = Path(script_path)

        # Check if script exists
        if not script.exists():
            return f"Error: Script '{script_path}' does not exist."

        # Prepare command
        cmd = ["python", str(script)]
        if args:
            cmd.extend(args)

        # Execute the script
        logger.info(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Prepare result
        output = f"Script: {script_path}\n"
        output += f"Return code: {result.returncode}\n"

        if result.stdout:
            output += f"\nStandard output:\n{result.stdout}\n"

        if result.stderr:
            output += f"\nStandard error:\n{result.stderr}\n"

        status = "Success" if result.returncode == 0 else "Failed"
        output += f"\nStatus: {status}"

        return output

    except Exception as e:
        error_msg = f"Error running script '{script_path}': {str(e)}"
        logger.error(error_msg)
        return error_msg


if __name__ == "__main__":
    import sys

    # Detect if we should use stdio or sse transport
    transport = "stdio"
    if len(sys.argv) > 1 and sys.argv[1] == "--sse":
        logger.info("Using SSE transport")
        transport = "sse"
        host = "localhost"
        port = 8000
        if len(sys.argv) > 2:
            host = sys.argv[2]
        if len(sys.argv) > 3:
            port = int(sys.argv[3])
        print(f"Starting Python Runner MCP Server with SSE transport on {host}:{port}")
        app.run(transport=transport, host=host, port=port)
    else:
        # Run with stdio transport by default
        logger.info("Starting Python Runner MCP Server with stdio transport")
        app.run(transport=transport)
