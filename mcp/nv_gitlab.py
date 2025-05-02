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
    name="create_file",
    description="Creates a new file with optional content. If the file already exists, it will not be modified unless force is set to True.",
)
def create_file(
    path: str,
    content: str = "",
    force: bool = False,
) -> str:
    """
    Create a new file with optional content.

    Args:
        path: Path to the file to create
        content: Optional content to write to the file
        force: If True, overwrites the file if it exists

    Returns:
        Success or error message
    """
    logger.info(f"Creating file via GitLab MCP Server: {path}")
    return f"Successfully created file via gitlab mcp server: {path}"
    try:
        file_path = Path(path)

        # Check if file exists and force is not enabled
        if file_path.exists() and not force:
            return f"Error: File '{path}' already exists. Use force=True to overwrite."

        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write content to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Successfully created file: {path}"
    except Exception as e:
        error_msg = f"Error creating file '{path}': {str(e)}"
        logger.error(error_msg)
        return error_msg


@app.tool(
    name="read_file",
    description="Reads content from a file.",
)
def read_file(
    path: str,
    encoding: str = "utf-8",
) -> str:
    """
    Read content from a file.

    Args:
        path: Path to the file to read
        encoding: File encoding (default: utf-8)

    Returns:
        File content or error message
    """
    logger.info(f"Reading file via GitLab MCP Server: {path}")
    return f"Successfully read file via gitlab mcp server: {path}"
    try:
        file_path = Path(path)

        # Check if file exists
        if not file_path.exists():
            return f"Error: File '{path}' does not exist."

        # Read and return file content
        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()

        return content
    except Exception as e:
        error_msg = f"Error reading file '{path}': {str(e)}"
        logger.error(error_msg)
        return error_msg


@app.tool(
    name="write_file",
    description="Writes content to a file, overwriting existing content or appending to it.",
)
def write_file(
    path: str,
    content: str,
    append: bool = False,
    create_dirs: bool = True,
    encoding: str = "utf-8",
) -> str:
    """
    Write content to a file.

    Args:
        path: Path to the file to write
        content: Content to write to the file
        append: If True, append content to the file instead of overwriting
        create_dirs: If True, create parent directories if they don't exist
        encoding: File encoding (default: utf-8)

    Returns:
        Success or error message
    """
    logger.info(f"Writing to file via GitLab MCP Server: {path}")
    return f"Successfully wrote to file via gitlab mcp server: {path}"
    try:
        print(f"Writing to file: {path}")
        file_path = Path(path)

        # Create parent directories if they don't exist and create_dirs is enabled
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write or append content to file
        mode = "a" if append else "w"
        with open(file_path, mode, encoding=encoding) as f:
            f.write(content)

        print(f"Create file: {path}")
        action = "appended to" if append else "written to"
        print(f"Successfully {action} file: {path}")
        return f"Successfully {action} file: {path}"
    except Exception as e:
        error_msg = f"Error writing to file '{path}': {str(e)}"
        logger.error(error_msg)
        return error_msg


@app.tool(
    name="gitlab_clone",
    description="Clones a GitLab repository to the specified destination path.",
)
def gitlab_clone(
    repo_name: str,
    destination: str = "output",
    branch: str = "main",
) -> str:
    """
    Clone a GitLab repository to the specified destination.

    Args:
        repo_name: Name of the repository (e.g., 'group/project')
        destination: Path where the repository's parent directory should be
        branch: Branch to clone (default: main)

    Returns:
        Success or error message
    """
    logger.info(f"Cloning repository via GitLab MCP Server: {repo_name}")
    return f"Successfully cloned repository via gitlab mcp server: {repo_name}"
    try:
        # Construct the GitLab repository URL
        repo_url = f"https://gitlab-master.nvidia.com/{repo_name}.git"

        # Create the parent destination directory if it doesn't exist
        dest_path = Path(destination)
        dest_path.mkdir(parents=True, exist_ok=True)

        # Extract the repository name from the full path (e.g., 'project' from 'group/project')
        repo_dir_name = repo_name.split("/")[-1]

        # Execute git clone command to a subdirectory of dest_path
        cmd = ["git", "clone", "--branch", branch, repo_url, str(dest_path / repo_dir_name)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        logger.info(f"Successfully cloned repository: {repo_name} to {dest_path / repo_dir_name}")
        return f"Successfully cloned repository: {repo_name} to {dest_path / repo_dir_name}"
    except subprocess.CalledProcessError as e:
        error_msg = f"Error cloning repository '{repo_name}': {e.stderr}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error cloning repository '{repo_name}': {str(e)}"
        logger.error(error_msg)
        return error_msg


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
