#!/usr/bin/env python3
"""
MCP server that provides file I/O operations: create_file, read_file, and write_file.
This server allows tools for basic file system operations and GitLab repository operations.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename="nv_gitlab.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create the FastMCP server
app = FastMCP(
    name="FileIO-GitLab-MCP-Server",
    instructions="This server provides tools for basic file I/O operations and GitLab repository operations",
)


@app.tool(
    name="gitlab_clone",
    description="clone test suite/scripts/codes from GitLab repository",
)
def gitlab_clone() -> str:
    """
    Clone a GitLab repository to the specified destination.

    Returns:
        Success or error message
    """
    logger.info(r"Cloning repository via GitLab MCP Server")
    return r"Cloning repository via GitLab MCP Server"


@app.tool(
    name="gitlab_pull",
    description="Pulls latest changes from a GitLab repository.",
)
def gitlab_pull(
    repo_path: str,
    branch: str = "main",
) -> str:
    """
    Pull the latest changes from a GitLab repository.

    Args:
        repo_path: Path to the local repository
        branch: Branch to pull (default: main)

    Returns:
        Success or error message
    """
    logger.info(f"Pulling latest changes from GitLab repository via GitLab MCP Server: {repo_path}")
    return f"Successfully pulled latest changes from gitlab mcp server: {repo_path}"
    try:
        # Check if the repository exists
        if not Path(repo_path).exists():
            return f"Error: Repository path '{repo_path}' does not exist."

        # Change to the repository directory
        cwd = os.getcwd()
        os.chdir(repo_path)

        try:
            # Fetch the latest changes
            fetch_cmd = ["git", "fetch", "origin"]
            subprocess.run(fetch_cmd, capture_output=True, text=True, check=True)

            # Checkout the specified branch
            checkout_cmd = ["git", "checkout", branch]
            subprocess.run(checkout_cmd, capture_output=True, text=True, check=True)

            # Pull the latest changes
            pull_cmd = ["git", "pull", "origin", branch]
            result = subprocess.run(pull_cmd, capture_output=True, text=True, check=True)

            logger.info(f"Successfully pulled latest changes for branch '{branch}' in {repo_path}")
            return f"Successfully pulled latest changes for branch '{branch}' in {repo_path}"
        finally:
            # Return to the original directory
            os.chdir(cwd)

    except subprocess.CalledProcessError as e:
        error_msg = f"Error pulling repository at '{repo_path}': {e.stderr}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error pulling repository at '{repo_path}': {str(e)}"
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
        print(f"Starting GitLab MCP Server with SSE transport on {host}:{port}")
        app.run(transport=transport, host=host, port=port)
    else:
        # Run with stdio transport by default
        logger.info("Starting GitLab MCP Server with stdio transport")
        app.run(transport=transport)
