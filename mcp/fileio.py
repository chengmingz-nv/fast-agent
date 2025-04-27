#!/usr/bin/env python3
"""
MCP server that provides file I/O operations: create_file, read_file, and write_file.
This server allows tools for basic file system operations.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastMCP server
app = FastMCP(
    name="FileIO-MCP-Server",
    instructions="This server provides tools for basic file I/O operations",
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


if __name__ == "__main__":
    import sys

    # Detect if we should use stdio or sse transport
    transport = "stdio"
    if len(sys.argv) > 1 and sys.argv[1] == "--sse":
        transport = "sse"
        host = "localhost"
        port = 8000
        if len(sys.argv) > 2:
            host = sys.argv[2]
        if len(sys.argv) > 3:
            port = int(sys.argv[3])
        print(f"Starting FileIO MCP Server with SSE transport on {host}:{port}")
        app.run(transport=transport, host=host, port=port)
    else:
        # Run with stdio transport by default
        app.run(transport=transport)
